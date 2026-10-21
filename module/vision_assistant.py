"""Join all transcribed commands with current camera frames for the OpenAI API."""

from __future__ import annotations

import queue
import sys
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .action_planner import (
    ActionPlan,
    create_action_plan,
)
from .api_common import create_api_client, selected_model
from .speech import run_live_transcription
from .decision_router import classify_command, process_command, print_command_result
from .session_log import record_event, save_command_image
from .speech_quality import SpeechOptions

from .camera import CameraSnapshot, LatestFrameCamera

_STOP = object()


@dataclass(frozen=True)
class VoiceCommand:
    number: int
    speaker: str
    text: str
    snapshot: CameraSnapshot | None
    command_id: str = ""


class VoiceVisionAssistant:
    """Keep the transcription callback short and plan accepted commands in FIFO order."""

    def __init__(
        self,
        camera: LatestFrameCamera,
        *,
        client: Any = None,
        model: str | None = None,
        planner: Callable[..., ActionPlan] = create_action_plan,
        classifier: Callable[..., Any] = classify_command,
    ) -> None:
        self._camera = camera
        self._client = client
        self._model = model
        self._planner = planner
        self._classifier = classifier
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
            snapshot = None
            record_event("camera_snapshot_error", speaker=speaker, error_type=type(exc).__name__)
        with self._lock:
            if not self._accepting:
                return
            command = VoiceCommand(self._next_number, speaker, text.strip(), snapshot, uuid.uuid4().hex)
            self._next_number += 1
            image_path = None
            if snapshot is not None:
                try:
                    image_path = save_command_image(snapshot.jpeg, command_id=command.command_id,
                                                    captured_at=snapshot.captured_at)
                except OSError as exc:
                    record_event("command_image_save_error", command_id=command.command_id,
                                 error_type=type(exc).__name__, error=str(exc))
                    print(f"[명령 {command.number}] 이미지 로그 저장 실패: {exc}", file=sys.stderr, flush=True)
            print(f"[명령 {command.number} / {speaker}] 최종 인식 접수: {command.text}", flush=True)
            record_event("voice_command_received", number=command.number, speaker=speaker,
                         command_id=command.command_id,
                         text=command.text, has_image=snapshot is not None,
                         image_path=image_path,
                         captured_at=snapshot.captured_at if snapshot is not None else None)
            self._commands.put(command)

    def _plan_commands(self) -> None:
        while True:
            item = self._commands.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, VoiceCommand)
                try:
                    print(f"[명령 {item.number} / {item.speaker}] Decisions 분류 중...", flush=True)
                    result = process_command(
                        item.text, image=item.snapshot.jpeg if item.snapshot is not None else None,
                        client=self._client, model=self._model, require_image=True,
                        planner=self._planner, classifier=self._classifier,
                        command_id=item.command_id,
                    )
                    print(f"[명령 {item.number} / {item.speaker}] 결과", flush=True)
                    print_command_result(result)
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


def run_voice_vision_assistant(camera_index: int = 0, *, options: SpeechOptions | None = None, offline: bool = False) -> None:
    """Accept every voice and send text + JPEG to OpenAI until Ctrl+C."""
    client = create_api_client()
    camera = None
    assistant = None
    try:
        model = selected_model()
        camera = LatestFrameCamera(camera_index)
        assistant = VoiceVisionAssistant(camera, client=client, model=model)
        try:
            camera.start()
        except RuntimeError as exc:
            print(f"카메라 준비 실패: {exc}", file=sys.stderr, flush=True)
            print("영상 없이 일반 대화와 정지를 처리합니다. 장면 질문과 영상이 필요한 행동 계획은 안내만 출력합니다.", flush=True)
        assistant.start()
        print(f"음성·영상 AI 준비 완료: {model} / 화자 등록 없이 모든 음성을 처리합니다.", flush=True)
        print("Decisions에는 텍스트만 전송합니다. 일반 대화는 텍스트로 답하고 장면 질문·행동 명령에만 사진을 전달합니다.", flush=True)
        speech_kwargs = {}
        if options is not None:
            speech_kwargs["options"] = options
        if offline:
            speech_kwargs["offline"] = True
        run_live_transcription(on_transcription=assistant.on_transcription, identify_speakers=False, **speech_kwargs)
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            try:
                if assistant is not None:
                    assistant.close()
            finally:
                client.close()
