"""Direct microphone capture, transcription and shutdown."""
from __future__ import annotations
import queue
import sys
import threading
import time
from typing import Callable
import numpy as np
from .session_log import record_event
from .speech_quality import SpeechOptions, SileroProbability, InputMeter, prepare_microphone
from .speech_pipeline import (
    SAMPLE_RATE, BLOCK_SIZE, _STOP, MonoUtteranceDetector, LiveTranscriptionQueue,
    _transcription_worker, create_speech_transcriber, handle_transcription,
)

def _load_transcription_dependencies():
    """Import only the packages needed for direct microphone transcription."""
    try:
        import sounddevice as sd
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            f"필요한 패키지({exc.name or '필수 패키지'})가 없습니다. requirements.txt를 설치해 주세요."
        ) from exc
    return sd, WhisperModel


def run_live_transcription(
    on_transcription: Callable[[str, str], None] = handle_transcription,
    *,
    offline: bool = False,
    identify_speakers: bool = False,
    options: SpeechOptions | None = None,
) -> None:
    """Pass mono mic audio straight to VAD and Whisper without SepFormer.

    Explicit legacy identification remains available for library callers only.
    """
    if identify_speakers:
        if options is not None:
            raise ValueError("새 음성 설정은 직접 마이크 입력 모드에서만 사용할 수 있습니다.")
        run_separated_transcription(on_transcription, offline=offline, identify_speakers=True)
        return
    sd, WhisperModel = _load_transcription_dependencies()
    options = options or SpeechOptions()
    prepare_microphone(sd, options.input_device)
    probability = SileroProbability()
    detector = MonoUtteranceDetector(probability=probability, options=options, status_updates=True)
    meter = InputMeter()
    captured: queue.Queue[tuple[np.ndarray, float, str]] = queue.Queue()
    preview_enabled = options.live_preview and options.backend == "local"
    transcript_tasks = LiveTranscriptionQueue(_STOP) if preview_enabled else queue.Queue()
    whisper = create_speech_transcriber(options, offline=offline, WhisperModel=WhisperModel)

    def capture_audio(indata, frames, time_info, status) -> None:
        del frames, time_info
        received_at = time.monotonic()
        start_time = received_at - len(indata) / SAMPLE_RATE
        audio = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
        item = (audio, start_time, str(status) if status else "")
        captured.put(item)

    def process_block(item: tuple[np.ndarray, float, str]) -> None:
        audio, start_time, status = item
        if status:
            print(f"마이크 상태: {status}", file=sys.stderr)
            record_event("microphone_status", status=status)
        meter.update(audio, show=options.show_levels)
        for task in detector.process(audio, start_time):
            transcript_tasks.put(task)
        if preview_enabled:
            transcript_tasks.activate(detector.state.utterance_id if detector.state.active else None)
            preview = detector.preview_snapshot()
            if preview is not None:
                transcript_tasks.offer_preview(preview)

    try:
        transcript_thread = threading.Thread(
            target=_transcription_worker,
            args=(whisper, None, transcript_tasks, on_transcription),
            name="speech-transcriber", daemon=True,
        )
        transcript_thread.start()
    except BaseException:
        whisper.close()
        raise
    interrupted = False
    try:
        print("직접 마이크 입력 모드: Silero 음성 감지 후 전사합니다 (SepFormer 미사용).")
        if preview_enabled:
            print("임시 인식은 화면과 로그에 표시합니다. 최종 확인된 문장만 명령으로 전달합니다.")
        print("마이크를 듣고 있습니다. 종료하려면 Ctrl+C를 누르세요.")
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=BLOCK_SIZE, callback=capture_audio, device=options.input_device,
        ):
            while True:
                item = captured.get()
                try:
                    process_block(item)
                finally:
                    captured.task_done()
    except KeyboardInterrupt:
        interrupted = True
        print("\n마이크 입력을 종료하고 남은 발화를 처리합니다.")
    except Exception as exc:
        raise RuntimeError(
            f"마이크 입력 처리에 실패했습니다. 기본 입력 장치와 마이크 권한을 확인해 주세요. 상세 정보: {exc}"
        ) from exc
    finally:
        # InputStream is closed before draining, so callbacks cannot add more audio.
        try:
            while True:
                try:
                    item = captured.get_nowait()
                except queue.Empty:
                    break
                try:
                    process_block(item)
                finally:
                    captured.task_done()
            for task in detector.flush():
                transcript_tasks.put(task)
        finally:
            try:
                transcript_tasks.put(_STOP)
                transcript_tasks.join()
                transcript_thread.join()
            finally:
                whisper.close()
        if interrupted:
            print("음성 인식을 종료했습니다.")


def enroll_speakers() -> None:
    from .legacy_speech import enroll_speakers as enroll
    return enroll()


def load_speaker_profiles():
    from .legacy_speech import load_speaker_profiles as load
    return load()


def run_separated_transcription(on_transcription=handle_transcription, *, offline=False, identify_speakers=False) -> None:
    from .legacy_speech import run_separated_transcription as run
    return run(on_transcription, offline=offline, identify_speakers=identify_speakers)
