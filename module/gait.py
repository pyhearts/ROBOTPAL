"""Logical alternating tetrapod gait, with no geometry or motor output."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .executable_plan import MAX_ACTION_DURATION_MS

GROUPS = {"A": ("R1", "R3", "L2", "L4"), "B": ("R2", "R4", "L1", "L3")}
PHASES = ("support", "A_lift", "A_transfer", "A_lower",
          "support", "B_lift", "B_transfer", "B_lower")
INTENTS = {"forward": "전방으로 발 이동 / 지지 그룹은 전진 추진",
           "backward": "후방으로 발 이동 / 지지 그룹은 후진 추진",
           "turn_left": "좌회전용 발 이동 의도", "turn_right": "우회전용 발 이동 의도"}


@dataclass(frozen=True)
class GaitTiming:
    support: float = 0.1
    lift: float = 0.15
    transfer: float = 0.2
    lower: float = 0.15
    simulation_only: bool = True

    def __post_init__(self):
        if self.simulation_only is not True:
            raise ValueError("이 보행 시간 설정은 가상 실행 전용입니다.")
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in (self.support, self.lift, self.transfer, self.lower)):
            raise ValueError("보행 단계 시간은 유한한 양수여야 합니다.")


class GaitModel:
    def __init__(self, timing: GaitTiming | None = None):
        self.timing = timing or GaitTiming()
        self.index = 0
        self.phase = "support"
        self.direction = None
        self.active = False
        self.stopping = False
        self.deadline = 0.0
        self.phase_end = 0.0

    def start(self, direction: str, duration_ms: int, now: float):
        if direction not in INTENTS or type(duration_ms) is not int or not 1 <= duration_ms <= MAX_ACTION_DURATION_MS:
            raise ValueError("지원 방향과 1~10000ms의 시간이 필요합니다.")
        if self.active:
            raise RuntimeError("전체 지지 전환 후 새 보행을 시작해야 합니다.")
        self.index, self.phase = 0, "support"
        self.direction, self.active, self.stopping = direction, True, False
        self.deadline = now + duration_ms / 1000
        self.phase_end = now + self.timing.support

    def stop(self, now: float):
        if not self.active or self.stopping:
            return
        self.stopping = True
        if self.phase == "support":
            self.active = False
        else:
            # Lower the currently airborne group; never lift the opposite group.
            if not self.phase.endswith("lower"):
                self.phase = self.phase[0] + "_lower"
                self.phase_end = now + self.timing.lower

    def advance(self, now: float):
        # Check duration before advancing: a stalled host must not replay missed lifts.
        if self.active and now >= self.deadline:
            self.stop(now)
        # One physical-time transition per tick; never compress a landing into a catch-up loop.
        if not self.active or now < self.phase_end:
            return
        if self.stopping:
            if self.phase != "support":
                self.phase = "support"
                self.phase_end = now + self.timing.support
            else:
                self.active = False
            return
        self.index = (self.index + 1) % len(PHASES)
        self.phase = PHASES[self.index]
        kind = self.phase.split("_")[-1]
        self.phase_end = now + getattr(self.timing, kind)

    def snapshot(self):
        airborne = self.phase[0] if self.active and self.phase != "support" else None
        return {"phase": self.phase, "active": self.active, "stopping": self.stopping,
                "direction": self.direction, "intent": INTENTS.get(self.direction, "정지"),
                "groups": {group: (self.phase.split("_")[-1] if group == airborne else "support")
                           for group in GROUPS}, "simulation": True,
                "contact_confirmed": False}
