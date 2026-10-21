"""Shared transcription model loading and final/preview worker."""
from __future__ import annotations
import queue
import sys
import time
from pathlib import Path
from typing import Any, Callable
import numpy as np
from .session_log import record_event
from .speech_quality import SpeechOptions, SpeechTranscriber, TranscriptionResult, RETRY_MESSAGE, options_dict
from .utterance import TranscriptionTask
from .live_transcription import PreviewTask

_STOP = object()
BLOCK_SIZE = 1_024

def handle_transcription(speaker: str, text: str) -> None:
    print(f"[최종 인식 / {speaker}] {text}", flush=True)


def transcribe_utterance(model: object, audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    if isinstance(model, SpeechTranscriber):
        result = model.transcribe_audio(audio)
        return result.text if result.accepted else ""
    segments, _ = model.transcribe(audio, language="ko", beam_size=1, vad_filter=True)  # type: ignore[attr-defined]
    return " ".join(segment.text.strip() for segment in segments).strip()


def _load_whisper_model(WhisperModel, *, offline: bool = False, options: SpeechOptions | None = None):
    options = options or SpeechOptions()
    print(f"한국어 Whisper 모델({options.model})을 준비합니다 ({options.device}, {options.compute_type}).")
    try:
        if options.device == "cuda":
            import ctranslate2
            if ctranslate2.get_cuda_device_count() == 0:
                raise RuntimeError("사용 가능한 CUDA GPU가 없습니다. --stt-device cpu를 사용해 주세요.")
        whisper_source = options.model
        if offline:
            from faster_whisper.utils import download_model

            whisper_source = download_model(options.model, local_files_only=True)
            # Whisper otherwise falls back to an online tokenizer even with local weights.
            if not (Path(whisper_source) / "tokenizer.json").is_file():
                raise RuntimeError("로컬 Whisper tokenizer.json이 없습니다.")
        return WhisperModel(
            whisper_source, device=options.device, compute_type=options.compute_type, local_files_only=offline
        )
    except Exception as exc:
        if offline:
            raise RuntimeError(
                "로컬 Whisper 모델을 준비하지 못했습니다. 인터넷을 사용할 수 있을 때 "
                f"기본 음성 전사 모드(main.py)를 한 번 실행해 주세요. 상세 정보: {exc}"
            ) from exc
        raise RuntimeError(f"Whisper 모델을 준비하지 못했습니다: {exc}") from exc


def create_speech_transcriber(options: SpeechOptions, *, offline: bool = False, WhisperModel: Any = None) -> SpeechTranscriber:
    record_event("stt_configuration", **options_dict(options))
    if options.backend == "openai":
        if offline:
            raise ValueError("offline 모드에서는 API 전사를 사용할 수 없습니다.")
        from .api_common import create_api_client
        print("API 전사 선택: 발화 오디오를 OpenAI gpt-transcribe로 전송합니다. 사용료가 발생합니다.", flush=True)
        return SpeechTranscriber(None, options, client=create_api_client())
    if WhisperModel is None:
        from faster_whisper import WhisperModel
    return SpeechTranscriber(_load_whisper_model(WhisperModel, offline=offline, options=options), options)


def _transcription_worker(whisper: object, identifier: Any, tasks: "queue.Queue[TranscriptionTask | object]", on_transcription: Callable[[str, str], None]) -> None:
    partial_text: dict[str, list[str]] = {}
    rejected: set[str] = set()
    last_preview: tuple[str, str] | None = None
    while True:
        task = tasks.get()
        try:
            if task is _STOP:
                return
            if isinstance(task, PreviewTask):
                if not isinstance(whisper, SpeechTranscriber) or not tasks.preview_is_current(task):
                    continue
                try:
                    result = whisper.transcribe_audio(task.audio, preview=True)
                    if result.accepted and tasks.preview_is_current(task):
                        key = (task.utterance_id, result.text)
                        if key != last_preview:
                            print(f"[임시 인식] {result.text}", flush=True)
                            record_event("stt_preview_displayed", utterance_id=task.utterance_id,
                                         text=result.text, audio_end_time=task.end_time)
                            last_preview = key
                    else:
                        record_event("stt_preview_discarded", utterance_id=task.utterance_id,
                                     reason=result.reason or "final_or_new_utterance_arrived")
                except Exception as exc:
                    record_event("stt_preview_error", error_type=type(exc).__name__)
                continue
            assert isinstance(task, TranscriptionTask)
            started = time.monotonic()
            try:
                text = transcribe_utterance(whisper, task.audio)
                quality = getattr(whisper, "last_result", None)
                if isinstance(quality, TranscriptionResult) and not quality.accepted:
                    rejected.add(task.utterance_id)
            except Exception as exc:
                print(f"음성 구간을 인식하지 못했습니다: {exc}", file=sys.stderr)
                text = ""
                rejected.add(task.utterance_id)
            record_event("stt_chunk", utterance_id=task.utterance_id, final=task.final,
                         text=text, queue_wait_seconds=max(0, started - task.end_time),
                         processing_seconds=time.monotonic() - started)
            texts = partial_text.setdefault(task.utterance_id, [])
            if text:
                texts.append(text)
            if task.final:
                complete = " ".join(texts).strip()
                partial_text.pop(task.utterance_id, None)
                if task.utterance_id in rejected:
                    rejected.discard(task.utterance_id)
                    complete = ""
                    print(RETRY_MESSAGE, flush=True)
                record_event("stt_utterance", utterance_id=task.utterance_id, text=complete,
                             accepted=bool(complete),
                             end_to_transcript_seconds=max(0, time.monotonic() - (
                                 task.end_time if task.speech_end_time is None else task.speech_end_time)))
                if complete:
                    try:
                        speaker = (
                            identifier.identify(task.identity_audio)
                            if identifier is not None
                            else ("음성 입력" if task.stream_id < 0 else f"음성 스트림 {task.stream_id + 1}")
                        )
                    except Exception as exc:
                        print(f"화자 식별에 실패해 기타로 처리합니다: {exc}", file=sys.stderr)
                        speaker = "기타"
                    try:
                        on_transcription(speaker, complete)
                    except Exception as exc:
                        print(f"인식 결과 전달 중 오류가 발생했습니다: {exc}", file=sys.stderr)
                delay = time.monotonic() - task.end_time
                if delay > 5.0:
                    print(f"경고: 전사 처리 지연이 {delay:.1f}초 누적되었습니다.", file=sys.stderr)
        finally:
            tasks.task_done()


