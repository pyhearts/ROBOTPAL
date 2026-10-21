"""Webcam gesture recognition and event stabilization."""

from __future__ import annotations

import queue
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .gesture_rules import FINGER_NAMES, RuleResult, classify_landmarks

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
MODEL_PATH = Path("models/gesture_recognizer.task")
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 15
SCORE_THRESHOLD = 0.7
STABLE_MILLISECONDS = 400
RELEASE_MILLISECONDS = 300
MAX_READ_FAILURES = 5
MAX_INFERENCE_FAILURES = 10

GESTURES = frozenset(
    {
        "Closed_Fist",
        "Open_Palm",
        "Pointing_Up",
        "Thumb_Down",
        "Thumb_Up",
        "Victory",
        "ILoveYou",
    }
)
GESTURE_LABELS = {
    "Closed_Fist": "주먹",
    "Open_Palm": "손바닥",
    "Pointing_Up": "검지",
    "Thumb_Down": "엄지 아래",
    "Thumb_Up": "엄지 위",
    "Victory": "브이",
    "ILoveYou": "아이 러브 유",
    "Number_1": "숫자 1",
    "Number_2": "숫자 2",
    "Number_3": "숫자 3",
    "Number_4": "숫자 4",
    "Number_5": "숫자 5",
    "OK": "OK",
}
HAND_LABELS = {"Left": "왼손", "Right": "오른손", "Unknown": "손"}
RULE_SETS = ("basic", "numbers")
NUMBER_MODE_SUPPRESSED = frozenset({"Pointing_Up", "Victory", "Open_Palm"})

# MediaPipe's 21-point hand skeleton. Kept local so preview rendering does not
# depend on a second MediaPipe drawing API.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass(frozen=True)
class GestureObservation:
    handedness: str
    gesture: str
    confidence: float


@dataclass(frozen=True)
class PreviewSnapshot:
    observation: GestureObservation | None
    landmarks: tuple[tuple[float, float], ...]
    status: str
    rule_set: str = "basic"
    finger_states: tuple[str, ...] = ()


class GestureStabilizer:
    """Turn timestamped predictions into one-shot, stable gesture events."""

    def __init__(
        self,
        stable_milliseconds: int = STABLE_MILLISECONDS,
        release_milliseconds: int = RELEASE_MILLISECONDS,
    ) -> None:
        self.stable_milliseconds = stable_milliseconds
        self.release_milliseconds = release_milliseconds
        self.candidate: GestureObservation | None = None
        self.candidate_since: int | None = None
        self.latched_key: tuple[str, str] | None = None
        self.missing_since: int | None = None
        self.last_timestamp: int | None = None

    def _reset_after_release(self) -> None:
        self.candidate = None
        self.candidate_since = None
        self.latched_key = None

    def update(
        self, timestamp_ms: int, observation: GestureObservation | None
    ) -> GestureObservation | None:
        if self.last_timestamp is not None and timestamp_ms < self.last_timestamp:
            raise ValueError("손 모양 결과 타임스탬프는 단조 증가해야 합니다.")
        self.last_timestamp = timestamp_ms

        if observation is None:
            if self.missing_since is None:
                self.missing_since = timestamp_ms
            elif timestamp_ms - self.missing_since >= self.release_milliseconds:
                self._reset_after_release()
            return None

        self.missing_since = None
        key = (observation.handedness, observation.gesture)
        candidate_key = None
        if self.candidate is not None:
            candidate_key = (self.candidate.handedness, self.candidate.gesture)
        if candidate_key != key:
            self.candidate = observation
            self.candidate_since = timestamp_ms
            return None

        # Keep the freshest confidence for the event while timing the same pose.
        self.candidate = observation
        assert self.candidate_since is not None
        if (
            timestamp_ms - self.candidate_since >= self.stable_milliseconds
            and self.latched_key != key
        ):
            self.latched_key = key
            return observation
        return None

    @property
    def status(self) -> str:
        if self.candidate is None:
            return "NO HAND"
        key = (self.candidate.handedness, self.candidate.gesture)
        return "ACTIVE" if self.latched_key == key else "STABILIZING"


def _category_name(category: object) -> str:
    return str(
        getattr(category, "category_name", None)
        or getattr(category, "categoryName", None)
        or ""
    )


def _category_score(category: object) -> float:
    try:
        return float(getattr(category, "score"))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def extract_observation(
    result: object,
    score_threshold: float = SCORE_THRESHOLD,
    rule_set: str = "basic",
) -> GestureObservation | None:
    """Extract the strongest allowed gesture from a MediaPipe-like result."""
    observation, _, _ = _interpret_result(result, score_threshold, rule_set)
    return observation


def _handedness(result: object, index: int) -> str:
    handedness = getattr(result, "handedness", None) or []
    if index < len(handedness) and handedness[index]:
        candidate = _category_name(handedness[index][0])
        if candidate in {"Left", "Right"}:
            return candidate
    return "Unknown"


def _usable_landmarks(result: object, index: int) -> Sequence[object]:
    world = getattr(result, "hand_world_landmarks", None) or []
    normalized = getattr(result, "hand_landmarks", None) or []
    if index < len(world) and len(world[index]) == 21:
        return world[index]
    if index < len(normalized) and len(normalized[index]) == 21:
        return normalized[index]
    return ()


def _interpret_result(
    result: object,
    score_threshold: float,
    rule_set: str,
) -> tuple[GestureObservation | None, RuleResult | None, int]:
    if rule_set not in RULE_SETS:
        raise ValueError(f"지원하지 않는 손 모양 규칙 모드입니다: {rule_set}")
    gestures = getattr(result, "gestures", None) or []
    hand_count = max(
        len(gestures),
        len(getattr(result, "hand_landmarks", None) or []),
        len(getattr(result, "hand_world_landmarks", None) or []),
    )
    rule_preview: RuleResult | None = None
    if rule_set == "numbers":
        for index in range(hand_count):
            rule_result = classify_landmarks(_usable_landmarks(result, index))
            if index == 0:
                rule_preview = rule_result
            if rule_result is not None and rule_result.gesture is not None:
                return (
                    GestureObservation(
                        _handedness(result, index),
                        rule_result.gesture,
                        rule_result.confidence,
                    ),
                    rule_result,
                    index,
                )

    best: GestureObservation | None = None
    best_index = 0
    for index, categories in enumerate(gestures):
        for category in categories or []:
            name = _category_name(category)
            score = _category_score(category)
            if name not in GESTURES or score + 1e-9 < score_threshold:
                continue
            if rule_set == "numbers" and name in NUMBER_MODE_SUPPRESSED:
                continue
            observation = GestureObservation(_handedness(result, index), name, score)
            if best is None or observation.confidence > best.confidence:
                best = observation
                best_index = index
    return best, rule_preview, best_index


def _extract_landmarks(result: object, hand_index: int = 0) -> tuple[tuple[float, float], ...]:
    hands = getattr(result, "hand_landmarks", None) or []
    if hand_index >= len(hands):
        return ()
    points: list[tuple[float, float]] = []
    for landmark in hands[hand_index]:
        try:
            points.append((float(landmark.x), float(landmark.y)))
        except (AttributeError, TypeError, ValueError):
            return ()
    return tuple(points)


class _ResultState:
    def __init__(self, rule_set: str = "basic") -> None:
        if rule_set not in RULE_SETS:
            raise ValueError(f"지원하지 않는 손 모양 규칙 모드입니다: {rule_set}")
        self.rule_set = rule_set
        self.stabilizer = GestureStabilizer()
        self.events: "queue.SimpleQueue[GestureObservation]" = queue.SimpleQueue()
        self.lock = threading.Lock()
        self.latest = PreviewSnapshot(None, (), "NO HAND", rule_set)
        self.consecutive_failures = 0
        self.fatal_error: RuntimeError | None = None

    def receive(self, result: object, output_image: object, timestamp_ms: int) -> None:
        del output_image
        try:
            observation, rule_result, hand_index = _interpret_result(
                result, SCORE_THRESHOLD, self.rule_set
            )
            with self.lock:
                event = self.stabilizer.update(timestamp_ms, observation)
                self.latest = PreviewSnapshot(
                    observation,
                    _extract_landmarks(result, hand_index),
                    self.stabilizer.status,
                    self.rule_set,
                    rule_result.finger_states if rule_result is not None else (),
                )
                self.consecutive_failures = 0
            if event is not None:
                self.events.put(event)
        except Exception as exc:
            self.record_failure(exc)

    def record_failure(self, exc: Exception) -> None:
        with self.lock:
            self.consecutive_failures += 1
            count = self.consecutive_failures
            if count >= MAX_INFERENCE_FAILURES:
                self.fatal_error = RuntimeError(
                    f"손 모양 인식이 {count}회 연속 실패했습니다: {exc}"
                )
        print(f"손 모양 프레임 인식 오류 ({count}/{MAX_INFERENCE_FAILURES}): {exc}", file=sys.stderr)

    def snapshot(self) -> PreviewSnapshot:
        with self.lock:
            return self.latest


def handle_gesture(handedness: str, gesture: str, confidence: float) -> None:
    """Receive a stable gesture event; connect robot commands here later."""
    hand = HAND_LABELS.get(handedness, handedness)
    label = GESTURE_LABELS.get(gesture, gesture)
    print(f"[{hand}] {label} ({confidence:.2f})", flush=True)


def ensure_gesture_model(path: Path = MODEL_PATH) -> Path:
    """Download the official task model once, using an atomic replacement."""
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    print("MediaPipe 손 모양 인식 모델을 내려받습니다.")
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "ROBOTPAL/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if temporary.stat().st_size == 0:
            raise OSError("다운로드한 모델 파일이 비어 있습니다.")
        temporary.replace(path)
        return path
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "손 모양 모델을 내려받지 못했습니다. 인터넷 연결을 확인하거나 다음 주소의 파일을 "
            f"{path}에 직접 저장해 주세요.\n{MODEL_URL}\n상세 정보: {exc}"
        ) from exc


def _load_dependencies():
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            f"필요한 패키지({exc.name or '필수 패키지'})를 찾을 수 없습니다. 다음 명령으로 설치해 주세요.\n"
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc
    return cv2, mp


def _drain_events(
    state: _ResultState,
    on_gesture: Callable[[str, str, float], None],
) -> None:
    while True:
        try:
            event = state.events.get_nowait()
        except queue.Empty:
            return
        try:
            on_gesture(event.handedness, event.gesture, event.confidence)
        except Exception as exc:
            print(f"손 모양 결과 전달 중 오류가 발생했습니다: {exc}", file=sys.stderr)


def _draw_preview(cv2: object, frame: object, snapshot: PreviewSnapshot) -> object:
    mirrored = cv2.flip(frame, 1)
    height, width = mirrored.shape[:2]
    points = [
        (int((1.0 - x) * width), int(y * height))
        for x, y in snapshot.landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(mirrored, points[start], points[end], (0, 220, 0), 2)
    for point in points:
        cv2.circle(mirrored, point, 3, (0, 255, 255), -1)
    if snapshot.observation is None:
        text = f"{snapshot.rule_set} {snapshot.status}"
    else:
        observation = snapshot.observation
        text = (
            f"{observation.handedness} {observation.gesture} "
            f"{observation.confidence:.2f} {snapshot.status} {snapshot.rule_set}"
        )
    cv2.putText(mirrored, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    if snapshot.finger_states:
        fingers = " ".join(
            f"{name[0].upper()}:{state[0].upper()}"
            for name, state in zip(FINGER_NAMES, snapshot.finger_states)
        )
        cv2.putText(
            mirrored,
            fingers,
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
    return mirrored


def _camera_loop(
    capture: object,
    recognizer: object,
    state: _ResultState,
    cv2: object,
    mp: object,
    on_gesture: Callable[[str, str, float], None],
    preview: bool,
) -> None:
    started = time.monotonic()
    next_submission = started
    last_timestamp = -1
    read_failures = 0
    submit_failures = 0
    while True:
        success, frame = capture.read()
        if not success or frame is None:
            read_failures += 1
            if read_failures >= MAX_READ_FAILURES:
                raise RuntimeError(
                    f"카메라 프레임을 {read_failures}회 연속 읽지 못했습니다. 카메라 연결과 권한을 확인해 주세요."
                )
            time.sleep(0.02)
            continue
        read_failures = 0
        now = time.monotonic()
        if now >= next_submission:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = max(last_timestamp + 1, int((now - started) * 1000))
            last_timestamp = timestamp
            try:
                recognizer.recognize_async(image, timestamp)
                submit_failures = 0
            except Exception as exc:
                submit_failures += 1
                print(
                    f"손 모양 프레임 제출 오류 ({submit_failures}/{MAX_INFERENCE_FAILURES}): {exc}",
                    file=sys.stderr,
                )
                if submit_failures >= MAX_INFERENCE_FAILURES:
                    raise RuntimeError(
                        f"손 모양 인식 요청이 {submit_failures}회 연속 실패했습니다: {exc}"
                    ) from exc
            next_submission = now + 1.0 / TARGET_FPS

        _drain_events(state, on_gesture)
        if state.fatal_error is not None:
            raise state.fatal_error
        if preview:
            cv2.imshow("ROBOTPAL Gesture", _draw_preview(cv2, frame, state.snapshot()))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return


def run_gesture_recognition(
    on_gesture: Callable[[str, str, float], None] = handle_gesture,
    camera_index: int = 0,
    preview: bool = False,
    rule_set: str = "basic",
) -> None:
    """Run standalone, one-hand gesture recognition until interrupted."""
    if rule_set not in RULE_SETS:
        raise ValueError(f"rule_set은 {' 또는 '.join(RULE_SETS)}이어야 합니다.")
    cv2, mp = _load_dependencies()
    model_path = ensure_gesture_model()
    state = _ResultState(rule_set)
    classifier_options = mp.tasks.components.processors.ClassifierOptions(
        score_threshold=SCORE_THRESHOLD,
        category_allowlist=sorted(GESTURES),
    )
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        canned_gesture_classifier_options=classifier_options,
        result_callback=state.receive,
    )
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"카메라 {camera_index}번을 열 수 없습니다. 장치 번호와 카메라 권한을 확인해 주세요."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    print(
        f"카메라 {camera_index}번에서 손 모양을 인식합니다 "
        f"(규칙 모드: {rule_set}). 종료하려면 Ctrl+C를 누르세요."
    )
    try:
        with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
            _camera_loop(capture, recognizer, state, cv2, mp, on_gesture, preview)
    except KeyboardInterrupt:
        print("\n손 모양 인식을 종료합니다.")
    finally:
        # Closing the recognizer can complete one last asynchronous result.
        _drain_events(state, on_gesture)
        capture.release()
        if preview:
            cv2.destroyAllWindows()
