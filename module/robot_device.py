"""Device contract, virtual controller and simulation-only line protocol client.

No serial port is opened here. An injected nonblocking byte transport allows offline
protocol verification; this release refuses peers capable of physical output.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .executable_plan import ExecutionStep
from .gait import GROUPS, GaitModel

VERSION = 1
BAUDRATE = 115200
HEARTBEAT_INTERVAL = 0.2
HEARTBEAT_TIMEOUT = 1.0
ACK_TIMEOUT = 0.5
SIMULATION_PROFILE = "robotpal-virtual-v1"


class JointProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    leg: Literal["R1", "R2", "R3", "R4", "L1", "L2", "L3", "L4"]
    role: str = Field(min_length=1)
    servo_model: str = Field(min_length=1)
    address: Literal[0x40, 0x41]
    channel: StrictInt = Field(ge=0, le=15)
    direction: Literal[-1, 1]
    minimum_us: StrictInt = Field(gt=0)
    neutral_us: StrictInt = Field(gt=0)
    maximum_us: StrictInt = Field(gt=0)
    max_change_us_per_second: float = Field(gt=0)

    @model_validator(mode="after")
    def check_range(self):
        if not self.minimum_us < self.neutral_us < self.maximum_us:
            raise ValueError("관절 중립값은 허용 범위 안에 있어야 합니다.")
        return self


class HardwareProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    simulation_only: bool
    calibrated: bool
    support_verified: bool
    fault_policy: Literal["unverified", "hold", "disable"]
    joints: list[JointProfile] = Field(min_length=8, max_length=32)

    @model_validator(mode="after")
    def check_layout(self):
        if {j.leg for j in self.joints} != set(sum(GROUPS.values(), ())):
            raise ValueError("여덟 다리 모두의 관절 배치가 필요합니다.")
        channels = [(j.address, j.channel) for j in self.joints]
        roles = [(j.leg, j.role) for j in self.joints]
        if len(set(channels)) != len(channels) or len(set(roles)) != len(roles):
            raise ValueError("중복 채널 또는 다리별 중복 관절 역할입니다.")
        return self

    @property
    def fingerprint(self):
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def require_hardware_ready(self):
        checked = HardwareProfile.model_validate(self.model_dump())
        if (checked.simulation_only or not checked.calibrated or not checked.support_verified
                or checked.fault_policy == "unverified"):
            raise ValueError("실제 구동에 사용할 수 없는 미검증 하드웨어 프로필입니다.")


class Device(Protocol):
    idle: bool
    ready: bool
    def prepare(self, now: float) -> None: ...
    def move(self, step: ExecutionStep, command_id: str, now: float) -> None: ...
    def stop(self, now: float) -> None: ...
    def heartbeat(self, now: float) -> None: ...
    def tick(self, now: float) -> list[dict]: ...
    def snapshot(self) -> dict: ...


class VirtualDevice:
    """Deterministic logical controller. Caller supplies monotonic time."""
    def __init__(self, *, gait=None, accept_delay=0.0):
        if not math.isfinite(accept_delay) or accept_delay < 0:
            raise ValueError("지연 시간은 유한한 음이 아닌 값이어야 합니다.")
        self.gait = gait or GaitModel()
        self.accept_delay = accept_delay
        self.session = uuid.uuid4().hex
        self.connected = True
        self.prepared = False
        self.last_heartbeat = 0.0
        self.command_id = None
        self.pending = None
        self.events = []
        self.seen = set()
        self.last_snapshot = None
        self.fault = None

    @property
    def idle(self):
        return not self.gait.active and self.pending is None

    @property
    def ready(self):
        return self.prepared and self.connected and self.fault is None

    def prepare(self, now):
        if not self.connected or self.fault:
            raise RuntimeError("가상 장치 연결 또는 고장 상태를 확인하세요.")
        self.prepared = True
        self.last_heartbeat = now

    def heartbeat(self, now):
        if not self.connected:
            raise ConnectionError("가상 USB 연결이 끊겼습니다.")
        if self.prepared and now - self.last_heartbeat >= HEARTBEAT_TIMEOUT:
            self.inject_fault("생존 신호 만료", now)
            raise TimeoutError("만료 후 도착한 생존 신호로 동작을 재개하지 않습니다.")
        self.last_heartbeat = now

    def move(self, step, command_id, now):
        step = ExecutionStep.model_validate(step.model_dump())
        if step.action == "stop":
            raise ValueError("정지는 stop() 경로로 요청하세요.")
        if not self.prepared or self.fault or not self.connected or not self.idle:
            raise RuntimeError("가상 장치가 이동 준비 상태가 아닙니다.")
        if command_id in self.seen:
            raise RuntimeError("중복 명령 ID입니다. 이동은 재전송하지 않습니다.")
        self.seen.add(command_id)
        self.command_id = command_id
        self.pending = (step, now + self.accept_delay)
        self.events.append({"event": "accepted", "request_id": command_id})

    def stop(self, now):
        self.pending = None
        self.gait.stop(now)

    def inject_fault(self, reason, now):
        self.fault = reason
        self.prepared = False
        self.stop(now)
        self.events.append({"event": "error", "error": reason, "request_id": self.command_id})

    def disconnect(self, now):
        self.connected = False
        self.inject_fault("연결 단절", now)

    def reconnect(self):
        if not self.idle:
            raise RuntimeError("정지 전환이 끝나기 전에는 재연결할 수 없습니다.")
        self.session = uuid.uuid4().hex
        self.connected, self.prepared = True, False
        # A controller fault remains latched. Construct a new controller after inspection.

    def tick(self, now):
        if self.prepared and now - self.last_heartbeat >= HEARTBEAT_TIMEOUT:
            self.inject_fault("생존 신호 만료", now)
        if self.pending is not None and now >= self.pending[1]:
            step, _ = self.pending
            self.pending = None
            self.gait.start(step.action, step.duration_ms, now)
        self.gait.advance(now)
        snapshot = self.gait.snapshot()
        if snapshot != self.last_snapshot:
            self.events.append({"event": "gait", "request_id": self.command_id, **snapshot})
            self.last_snapshot = snapshot
        if self.command_id and self.idle:
            self.events.append({"event": "completed", "request_id": self.command_id})
            self.command_id = None
        result, self.events = self.events, []
        return result

    def snapshot(self):
        return {**self.gait.snapshot(), "session": self.session, "profile": SIMULATION_PROFILE,
                "fault": self.fault, "connected": self.connected}


class ByteTransport(Protocol):
    """Injected transport must be nonblocking; read returns currently available bytes."""
    def write(self, data: bytes) -> int: ...
    def read(self, maximum: int) -> bytes: ...


class SerialDevice:
    """NDJSON protocol adapter for an injected simulation peer only; no port factory."""
    def __init__(self, transport: ByteTransport, *, profile=SIMULATION_PROFILE):
        self.transport, self.profile = transport, profile
        self.session = uuid.uuid4().hex
        self.idle = True
        self.ready = False
        self.fault = None
        self.pending = {}
        self.buffer = b""
        self.stage = "new"
        self.current = None
        self.seen_ids = set()
        self._operations = {}
        self.stop_id = None
        self.completed_ids = set()
        self.cancelled_ids = set()
        self.completion_deadline = None

    def _send(self, op, now, *, request_id=None, **payload):
        request_id = request_id or uuid.uuid4().hex
        if request_id in self.seen_ids:
            raise RuntimeError("중복 요청은 재전송하지 않습니다.")
        self.seen_ids.add(request_id)
        self._operations[request_id] = op
        packet = {"version": VERSION, "session": self.session, "profile": self.profile,
                  "request_id": request_id, "op": op, **payload}
        data = (json.dumps(packet, separators=(",", ":")) + "\n").encode("ascii")
        if self.transport.write(data) != len(data):
            raise ConnectionError("직렬 전송이 완료되지 않았습니다.")
        self.pending[request_id] = (op, now + ACK_TIMEOUT)
        return request_id

    def prepare(self, now):
        if self.stage != "new" or self.fault:
            raise RuntimeError("재연결은 새 세션과 새 어댑터로 시작해야 합니다.")
        self.stage = "hello"
        self._send("hello", now)

    def move(self, step, command_id, now):
        step = ExecutionStep.model_validate(step.model_dump())
        if not self.ready or not self.idle or self.fault or step.action == "stop":
            raise RuntimeError("직렬 가상 장치가 이동 준비 상태가 아닙니다.")
        self._send("move", now, request_id=command_id, **step.model_dump())
        self.current, self.idle = command_id, False
        self.completion_deadline = now + step.duration_ms / 1000 + 2.0

    def stop(self, now):
        if self.ready and not self.fault:
            if self.stop_id is not None:
                return  # Repeated replacement requests share the ongoing stop.
            self.stop_id = self._send("stop", now)
            if self.current:
                self.cancelled_ids.add(self.current)
            self.idle = False
            self.completion_deadline = now + 2.0

    def heartbeat(self, now):
        if self.ready and not self.fault:
            self._send("heartbeat", now)

    def status(self, now):
        if self.ready and not self.fault:
            self._send("status", now)

    def tick(self, now):
        if self.fault:
            return []
        events = []
        try:
            self.buffer += self.transport.read(4096)
            if len(self.buffer) > 8192:
                raise ValueError("직렬 수신 버퍼 초과")
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                p = json.loads(line)
                if (type(p.get("version")) is not int or p["version"] != VERSION
                        or p.get("session") != self.session or p.get("profile") != self.profile
                        or p.get("simulation") is not True):
                    raise ValueError("프로토콜·세션·프로필 불일치 또는 실제 출력 장치입니다.")
                rid, event = p.get("request_id"), p.get("event")
                if not isinstance(rid, str) or rid not in self.seen_ids:
                    raise ValueError("알 수 없는 응답 ID")
                if event == "error":
                    raise RuntimeError(str(p.get("error", "장치 오류")))
                if event == "accepted":
                    if rid not in self.pending:
                        raise ValueError("중복 접수 응답")
                    op, deadline = self.pending.pop(rid)
                    if now >= deadline:
                        raise TimeoutError("늦게 도착한 접수 응답")
                    if op == "hello":
                        self.stage = "ready"
                        self._send("ready", now)
                    elif op == "ready":
                        self.stage, self.ready = "prepared", True
                    if op in {"hello", "ready", "heartbeat"}:
                        self.seen_ids.discard(rid)
                        self._operations.pop(rid)
                elif event == "completed":
                    # Completion is authoritative only after acceptance of move/stop.
                    if (rid in self.pending or rid in self.completed_ids
                            or self.operations[rid] not in {"move", "stop"}):
                        raise ValueError("잘못된 완료 응답")
                    if self.operations[rid] == "stop" and rid != self.stop_id:
                        raise ValueError("만료된 정지 완료 응답")
                    if rid != self.current and self.operations[rid] == "move" and rid not in self.cancelled_ids:
                        raise ValueError("만료된 동작 완료 응답")
                    self.completed_ids.add(rid)
                    if rid == self.stop_id or (rid == self.current and self.stop_id is None):
                        self.idle, self.current, self.completion_deadline = True, None, None
                        self.stop_id = None
                elif event == "status":
                    if self.operations[rid] != "status":
                        raise ValueError("상태 조회에 대응하지 않는 응답")
                    if rid in self.pending:
                        _, deadline = self.pending.pop(rid)
                        if now >= deadline:
                            raise TimeoutError("늦게 도착한 상태 응답")
                    self.seen_ids.discard(rid)
                    self._operations.pop(rid)
                else:
                    raise ValueError("알 수 없는 장치 응답")
                events.append(p)
            if any(now >= deadline for _, deadline in self.pending.values()):
                raise TimeoutError("접수 응답 만료: 이동을 재전송하지 않습니다.")
            if self.completion_deadline is not None and now >= self.completion_deadline:
                raise TimeoutError("정지 또는 이동 완료 응답 만료")
        except Exception as exc:
            # Best-effort stop before latching fault; no reconnect or movement retry.
            try:
                self.stop(now)
            except Exception:
                pass
            self.fault, self.ready = str(exc), False
            events.append({"event": "error", "error": self.fault})
        return events

    def snapshot(self):
        return {"simulation": True, "fault": self.fault, "session": self.session,
                "profile": self.profile, "idle": self.idle, "ready": self.ready}

    @property
    def operations(self):
        return self._operations
