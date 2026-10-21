"""Join all transcribed commands with current camera frames for the OpenAI API."""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .action_planner import (
    ActionPlan,
    create_action_plan,
    create_api_client,
    print_action_plan,
    selected_model,
)
from .speech import run_live_transcription

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
MAX_FRAME_AGE_SECONDS = 1.0
FIRST_FRAME_TIMEOUT_SECONDS = 5.0
MAX_READ_FAILURES = 5
_STOP = object()


@dataclass(frozen=True)
class CameraSnapshot:
    jpeg: bytes
    captured_at: float


class LatestFrameCamera:
    """Own one camera and retain only its latest unmirrored frame in memory."""

    def __init__(self, camera_index: int = 0, *, cv2_module: Any = None) -> None:
        if camera_index < 0:
            raise ValueError("카메라 번호는 0 이상이어야 합니다.")
        self.camera_index = camera_index
        self._cv2 = cv2_module
        self._capture: Any = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._frame: Any = None
        self._captured_at = 0.0
        self._error: str | None = None

    def start(self) -> None:
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("카메라 객체는 한 번만 시작할 수 있습니다.")
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("OpenCV 패키지가 없습니다. requirements.txt를 설치해 주세요.") from exc
            self._cv2 = cv2
        capture = self._cv2.VideoCapture(self.camera_index)
        self._capture = capture
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"카메라 {self.camera_index}번을 열 수 없습니다. 장치 번호와 권한을 확인해 주세요."
            )
        try:
            capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            capture.set(self._cv2.CAP_PROP_FPS, CAMERA_FPS)
            thread = threading.Thread(target=self._read_frames, name="assistant-camera", daemon=True)
            thread.start()
            self._thread = thread
        except BaseException:
            capture.release()
            self._capture = None
            raise
        if not self._ready.wait(FIRST_FRAME_TIMEOUT_SECONDS):
            self.close()
            raise RuntimeError("카메라에서 첫 프레임을 받지 못했습니다.")
        with self._lock:
            error = self._error
        if error:
            self.close()
            raise RuntimeError(error)

    def _read_frames(self) -> None:
        failures = 0
        try:
            while not self._stop.is_set():
                success, frame = self._capture.read()
                if not success or frame is None:
                    failures += 1
                    # Do not reuse the last good frame after a failed read.
                    with self._lock:
                        self._frame = None
                    if failures >= MAX_READ_FAILURES:
                        raise RuntimeError("카메라 연결이 끊겼거나 프레임을 읽지 못했습니다.")
                    self._stop.wait(1.0 / CAMERA_FPS)
                else:
                    failures = 0
                    with self._lock:
                        self._frame = frame
                        self._captured_at = time.monotonic()
                    self._ready.set()
                    # Drain successful reads at the camera's rate so backend buffers
                    # cannot accumulate old frames if the FPS setting is ignored.
        except Exception as exc:
            with self._lock:
                self._error = f"카메라 오류: {exc}"
                self._frame = None
            if not self._stop.is_set():
                print(self._error, file=sys.stderr, flush=True)
        finally:
            self._ready.set()
            self._capture.release()

    def snapshot(self) -> CameraSnapshot:
        """Freeze a fresh frame at callback time, not when inference begins."""
        with self._lock:
            if self._error:
                raise RuntimeError(self._error)
            if self._stop.is_set() or self._frame is None:
                raise RuntimeError("사용할 카메라 프레임이 없습니다. 이 명령은 AI에 보내지 않습니다.")
            captured_at = self._captured_at
            if time.monotonic() - captured_at > MAX_FRAME_AGE_SECONDS:
                raise RuntimeError("최신 카메라 프레임이 오래되었습니다. 이 명령은 AI에 보내지 않습니다.")
            frame = self._frame.copy()
        success, encoded = self._cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("카메라 프레임을 JPEG로 변환하지 못했습니다.")
        return CameraSnapshot(jpeg=encoded.tobytes(), captured_at=captured_at)

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._frame = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                # Some camera drivers block read(); releasing the handle unblocks them.
                self._capture.release()
                self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                print("카메라 종료가 지연되고 있습니다.", file=sys.stderr, flush=True)
        elif self._capture is not None:
            self._capture.release()


@dataclass(frozen=True)
class VoiceCommand:
    number: int
    speaker: str
    text: str
    snapshot: CameraSnapshot


class VoiceVisionAssistant:
    """Keep the transcription callback short and plan accepted commands in FIFO order."""

    def __init__(
        self,
        camera: LatestFrameCamera,
        *,
        client: Any = None,
        model: str | None = None,
        planner: Callable[..., ActionPlan] = create_action_plan,
    ) -> None:
        self._camera = camera
        self._client = client
        self._model = model
        self._planner = planner
        self._commands: queue.Queue[VoiceCommand | object] = queue.Queue()
        self._lock = threading.Lock()
        self._accepting = False
        self._closed = False
        self._next_number = 1
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None or self._closed:
                raise RuntimeError("음성·영상 AI는 한 번만 시작할 수 있습니다.")
            self._thread = threading.Thread(target=self._plan_commands, name="assistant-planner", daemon=True)
            self._thread.start()
            self._accepting = True

    def on_transcription(self, speaker: str, text: str) -> None:
        if not text.strip():
            return
        with self._lock:
            if not self._accepting:
                return
        try:
            snapshot = self._camera.snapshot()
        except Exception as exc:
            print(f"[{speaker}] 명령 처리 실패: {exc}", file=sys.stderr, flush=True)
            return
        with self._lock:
            if not self._accepting:
                return
            command = VoiceCommand(self._next_number, speaker, text.strip(), snapshot)
            self._next_number += 1
            print(f"[명령 {command.number} / {speaker}] 접수: {command.text}", flush=True)
            self._commands.put(command)

    def _plan_commands(self) -> None:
        while True:
            item = self._commands.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, VoiceCommand)
                try:
                    print(f"[명령 {item.number} / {item.speaker}] OpenAI API 처리 중...", flush=True)
                    plan = self._planner(
                        item.text, image=item.snapshot.jpeg, client=self._client, model=self._model
                    )
                    if plan.status == "invalid":
                        print(f"[명령 {item.number}] 오류: {plan.error}", file=sys.stderr, flush=True)
                    else:
                        print(f"[명령 {item.number} / {item.speaker}] 결과", flush=True)
                        print_action_plan(plan)
                except Exception as exc:
                    print(f"[명령 {item.number}] 계획 생성 실패: {exc}", file=sys.stderr, flush=True)
            finally:
                self._commands.task_done()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._accepting = False
                if self._thread is not None:
                    self._commands.put(_STOP)
        if self._thread is not None:
            if self._thread.is_alive():
                print("접수된 AI 명령을 순서대로 마무리합니다.", flush=True)
            self._thread.join()


def run_voice_vision_assistant(camera_index: int = 0) -> None:
    """Accept every voice and send text + JPEG to OpenAI until Ctrl+C."""
    client = create_api_client()
    model = selected_model()
    camera = LatestFrameCamera(camera_index)
    assistant = VoiceVisionAssistant(camera, client=client, model=model)
    try:
        camera.start()
        assistant.start()
        print(f"음성·영상 AI 준비 완료: {model} / 화자 등록 없이 모든 음성을 처리합니다.", flush=True)
        print("전사된 명령과 카메라 JPEG 한 장을 OpenAI API로 전송합니다.", flush=True)
        run_live_transcription(on_transcription=assistant.on_transcription, identify_speakers=False)
    finally:
        try:
            camera.close()
        finally:
            try:
                assistant.close()
            finally:
                client.close()
