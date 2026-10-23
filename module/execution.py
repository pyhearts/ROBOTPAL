"""Latest-request execution with explicit cancellation and logical device events."""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass

from .executable_plan import ExecutablePlan
from .robot_device import Device, HEARTBEAT_INTERVAL, VirtualDevice
from .session_log import record_event

REQUEST_TTL = 10.0
STATE_NAMES = {"ready": "준비", "running": "실행 중", "stopping": "정지 중", "fault": "고장"}


@dataclass(frozen=True)
class ExecutionRequest:
    generation: int
    command_id: str
    received_at: float


class ExecutionManager:
    """Single-owner state machine; the runtime serializes access under its lock."""
    def __init__(self, device: Device | None = None, *, clock=time.monotonic, emit=None):
        self.device = device or VirtualDevice()
        self.clock = clock
        self.emit = emit or (lambda event: None)
        self.state = "ready"
        self.generation = 0
        self.request = None
        self.awaiting = False
        self.steps = deque()
        self.step_number = 0
        self.started = False
        self.last_heartbeat = clock()
        self.closed = False
        self.outcome = None
        self.device.prepare(self.last_heartbeat)

    def _event(self, event, **data):
        payload = {"event": event, "state": self.state, "simulation": True,
                   "command_id": self.request.command_id if self.request else None, **data}
        record_event("execution_" + event, **{k: v for k, v in payload.items() if k != "event"})
        self.emit(payload)

    def _state(self, value):
        if self.state != value:
            self.state = value
            self._event("state", label=STATE_NAMES[value])

    def _stop(self, now):
        self.steps.clear()
        self.awaiting = False
        if not self.device.idle:
            self.device.stop(now)
            if self.state != "fault":
                self._state("stopping")
        elif self.state != "fault":
            self._state("ready")

    def begin(self, *, command_id=None, received_at=None):
        if self.closed or self.state == "fault":
            raise RuntimeError("종료 또는 고장 상태에서는 새 명령을 받을 수 없습니다.")
        now = self.clock()
        self._stop(now)
        self.generation += 1
        self.request = ExecutionRequest(self.generation, command_id or uuid.uuid4().hex,
                                        now if received_at is None else received_at)
        self.awaiting, self.started, self.step_number = True, False, 0
        self.outcome = None
        self._event("received")
        return self.request

    def is_current(self, request):
        return not self.closed and self.state != "fault" and self.request == request and self.generation == request.generation

    def offer(self, request, plan):
        if not self.is_current(request) or not self.awaiting:
            self._event("discarded", discarded_command_id=request.command_id)
            return False
        if self.clock() - request.received_at >= REQUEST_TTL:
            self.awaiting = False
            self.outcome = "expired"
            self._event("expired")
            return False
        try:
            # Revalidate at execution boundary, including mutable caller-owned lists.
            checked = ExecutablePlan.model_validate(plan.model_dump())
            if checked.status != "plan":
                raise ValueError(checked.error)
        except (ValueError, TypeError, AttributeError) as exc:
            self.reject(request, str(exc))
            return False
        self.steps.extend(checked.actions)
        self.awaiting = False
        self._event("plan", plan=checked.model_dump(mode="json"))
        return True

    def reject(self, request, reason):
        if self.is_current(request):
            self.awaiting = False
            self.steps.clear()
            self.outcome = "rejected"
            self._event("rejected", reason=reason)

    def stop(self, reason="사용자 정지"):
        if self.closed:
            return
        self.generation += 1
        self._stop(self.clock())
        self._event("cancelled", reason=reason)

    def tick(self):
        now = self.clock()
        try:
            if self.state != "fault" and now - self.last_heartbeat >= HEARTBEAT_INTERVAL:
                self.device.heartbeat(now)
                self.last_heartbeat = now
            events = self.device.tick(now)
            for event in events:
                device_id = event.get("request_id")
                self._event("device", device_event=event,
                            command_id=device_id.rsplit(":", 1)[0] if device_id else
                            (self.request.command_id if self.request else None))
                if event.get("event") == "error":
                    raise RuntimeError(event.get("error", "장치 오류"))
            if self.state == "fault":
                return  # Keep ticking the virtual landing, never resume a faulted controller.
            if self.request and not self.started and (self.awaiting or self.steps):
                if now - self.request.received_at >= REQUEST_TTL:
                    self.awaiting = False
                    self.steps.clear()
                    self.outcome = "expired"
                    self._event("expired")
            if not self.device.idle:
                return
            if self.state == "stopping":
                self._state("ready")
                self._event("stop_completed", contact_confirmed=False)
            if self.steps:
                if not self.device.ready:
                    return  # Asynchronous hello/ready must finish before sending movement.
                step = self.steps.popleft()
                self.step_number += 1
                self.started = True
                if step.action == "stop":
                    self.device.stop(now)
                    self._state("running")
                else:
                    self.device.move(step, f"{self.request.command_id}:{self.step_number}", now)
                    self._state("running")
                self._event("step", action=step.action, duration_ms=step.duration_ms)
            elif self.state == "running":
                self._state("ready")
                self.outcome = "completed"
                self._event("completed", contact_confirmed=False)
        except Exception as exc:
            self.steps.clear()
            self.awaiting = False
            self.generation += 1
            try:
                self.device.stop(now)
            except Exception:
                pass
            self._state("fault")
            self._event("error", error=str(exc))

    @property
    def settled(self):
        return (self.state in {"ready", "fault"} and not self.awaiting
                and not self.steps and self.device.idle)

    def close(self):
        self.stop("프로그램 종료")
        self.closed = True
