"""Deterministic hand-pose rules built from MediaPipe's 21 landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


EXTENDED = "extended"
FOLDED = "folded"
UNCERTAIN = "uncertain"
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class RuleResult:
    gesture: str | None
    finger_states: tuple[str, str, str, str, str]
    confidence: float = 1.0


def _point(landmark: object) -> tuple[float, float, float]:
    values = (
        float(getattr(landmark, "x")),
        float(getattr(landmark, "y")),
        float(getattr(landmark, "z", 0.0)),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("관절 좌표가 유한수가 아닙니다.")
    return values


def _vector(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(end[index] - start[index] for index in range(3))  # type: ignore[return-value]


def _distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _angle(
    first: tuple[float, float, float],
    vertex: tuple[float, float, float],
    last: tuple[float, float, float],
) -> float:
    left = _vector(vertex, first)
    right = _vector(vertex, last)
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    if left_length <= 1e-8 or right_length <= 1e-8:
        raise ValueError("겹친 관절 좌표로 각도를 계산할 수 없습니다.")
    cosine = sum(left[index] * right[index] for index in range(3)) / (
        left_length * right_length
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _finger_state(
    points: Sequence[tuple[float, float, float]],
    mcp: int,
    pip: int,
    dip: int,
    tip: int,
) -> str:
    pip_angle = _angle(points[mcp], points[pip], points[dip])
    dip_angle = _angle(points[pip], points[dip], points[tip])
    pip_distance = _distance(points[0], points[pip])
    if pip_distance <= 1e-8:
        return UNCERTAIN
    radial_ratio = _distance(points[0], points[tip]) / pip_distance
    if pip_angle >= 155.0 and dip_angle >= 155.0 and radial_ratio >= 1.1:
        return EXTENDED
    if pip_angle <= 130.0 or radial_ratio <= 1.0:
        return FOLDED
    return UNCERTAIN


def _thumb_state(points: Sequence[tuple[float, float, float]]) -> str:
    mcp_angle = _angle(points[1], points[2], points[3])
    ip_angle = _angle(points[2], points[3], points[4])
    ip_distance = _distance(points[0], points[3])
    if ip_distance <= 1e-8:
        return UNCERTAIN
    radial_ratio = _distance(points[0], points[4]) / ip_distance
    if mcp_angle >= 145.0 and ip_angle >= 150.0 and radial_ratio >= 1.08:
        return EXTENDED
    if ip_angle <= 125.0 or radial_ratio <= 1.0:
        return FOLDED
    return UNCERTAIN


def classify_landmarks(landmarks: Sequence[object]) -> RuleResult | None:
    """Return a number/OK rule result, or None for unusable coordinates."""
    if len(landmarks) != 21:
        return None
    try:
        points = tuple(_point(landmark) for landmark in landmarks)
        palm_scale = _distance(points[0], points[9])
        if palm_scale <= 1e-8:
            return None
        states = (
            _thumb_state(points),
            _finger_state(points, 5, 6, 7, 8),
            _finger_state(points, 9, 10, 11, 12),
            _finger_state(points, 13, 14, 15, 16),
            _finger_state(points, 17, 18, 19, 20),
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None

    # OK takes priority over number patterns. The touching fingers are allowed
    # to be folded or geometrically uncertain; the other three must be open.
    ok_distance = _distance(points[4], points[8]) / palm_scale
    if ok_distance <= 0.25 and states[2:] == (EXTENDED, EXTENDED, EXTENDED):
        return RuleResult("OK", states)

    if UNCERTAIN in states:
        return RuleResult(None, states)
    extended = tuple(state == EXTENDED for state in states)
    patterns = {
        (False, True, False, False, False): "Number_1",
        (False, True, True, False, False): "Number_2",
        (False, True, True, True, False): "Number_3",
        (True, True, True, False, False): "Number_3",
        (False, True, True, True, True): "Number_4",
        (True, True, True, True, True): "Number_5",
    }
    return RuleResult(patterns.get(extended), states)
