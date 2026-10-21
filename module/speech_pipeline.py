"""Shared speech pipeline: utterances, final/preview queue and transcription."""
from __future__ import annotations

import math
import queue
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .session_log import record_event
from .speech_quality import (
    SpeechOptions, SpeechTranscriber, TranscriptionResult, RETRY_MESSAGE, options_dict,
)


SAMPLE_RATE = 16_000
SILENCE_SECONDS = 0.7
MIN_SPEECH_SECONDS = 0.25
PARTIAL_SECONDS = 20.0
PRE_ROLL_SECONDS = 0.2
VAD_BLOCK_SAMPLES = 1_024
VAD_THRESHOLD = 0.012
_STOP = object()
BLOCK_SIZE = 1_024


@dataclass(frozen=True)
class PreviewTask:
    utterance_id: str
    audio: np.ndarray
    end_time: float


@dataclass
class TranscriptionTask:
    utterance_id: str
    stream_id: int
    audio: np.ndarray
    final: bool
    end_time: float
    identity_audio: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    speech_end_time: float | None = None


class LiveTranscriptionQueue:
    """Keep all final/chunk jobs; replace previews instead of building a backlog.

    Inference is serialized on one model. An already-running decoder finishes its
    bounded generation, but its stale preview is suppressed when a final arrives.
    """

    def __init__(self, stop_token: object) -> None:
        self.stop_token = stop_token
        self._condition = threading.Condition()
        self._finals: deque[Any] = deque()
        self._preview: PreviewTask | None = None
        self._active_id: str | None = None
        self._closed = False
        self._busy_final = False
        self.unfinished_tasks = 0

    def _discard_preview(self) -> None:
        if self._preview is not None:
            self._preview = None
            self.unfinished_tasks -= 1

    def activate(self, utterance_id: str | None) -> None:
        with self._condition:
            if self._active_id != utterance_id:
                self._discard_preview()
            self._active_id = utterance_id
            self._condition.notify_all()

    def offer_preview(self, preview: PreviewTask) -> bool:
        with self._condition:
            if self._closed or preview.utterance_id != self._active_id or self._finals or self._busy_final:
                return False
            if self._preview is None:
                self.unfinished_tasks += 1
            self._preview = preview
            self._condition.notify_all()
            return True

    def preview_is_current(self, preview: PreviewTask) -> bool:
        with self._condition:
            return not self._closed and not self._finals and preview.utterance_id == self._active_id

    def put(self, task: Any) -> None:
        with self._condition:
            if task is self.stop_token:
                if not self._closed:
                    self._closed = True
                    self._active_id = None
                    self._discard_preview()
                    self.unfinished_tasks += 1
            else:
                if self._closed:
                    raise RuntimeError("종료된 전사 큐에는 작업을 추가할 수 없습니다.")
                self._finals.append(task)
                self.unfinished_tasks += 1
                self._discard_preview()
                if task.final and self._active_id == task.utterance_id:
                    self._active_id = None
            self._condition.notify_all()

    def get(self) -> Any:
        with self._condition:
            while True:
                if self._finals:
                    self._busy_final = True
                    return self._finals.popleft()
                if self._preview is not None:
                    preview, self._preview = self._preview, None
                    return preview
                if self._closed:
                    return self.stop_token
                self._condition.wait()

    def task_done(self) -> None:
        with self._condition:
            if self.unfinished_tasks <= 0:
                raise ValueError("전사 작업 완료 횟수가 올바르지 않습니다.")
            self.unfinished_tasks -= 1
            self._busy_final = False
            self._condition.notify_all()

    def join(self) -> None:
        with self._condition:
            while self.unfinished_tasks:
                self._condition.wait()


class _StreamState:
    def __init__(self, stream_id: int, sample_rate: int, *, collect_identity: bool = True,
                 pre_roll: float = PRE_ROLL_SECONDS, silence: float = SILENCE_SECONDS,
                 min_speech: float = MIN_SPEECH_SECONDS) -> None:
        self.stream_id = stream_id
        self.sample_rate = sample_rate
        self.collect_identity = collect_identity
        self.silence_seconds, self.min_speech_seconds = silence, min_speech
        self.pre_roll: deque[np.ndarray] = deque()
        self.pre_roll_limit = round(pre_roll * sample_rate)
        self.active = False
        self.parts: list[np.ndarray] = []
        self.identity_parts: list[np.ndarray] = []
        self.identity_samples = 0
        self.buffered_samples = 0
        self.speech_samples = 0
        self.silence_samples = 0
        self.utterance_id = ""
        self.last_voice_end: float | None = None

    def _clear(self) -> None:
        self.active = False
        self.parts.clear()
        self.identity_parts.clear()
        self.identity_samples = self.buffered_samples = self.speech_samples = self.silence_samples = 0
        self.utterance_id = ""
        self.last_voice_end = None
        self.pre_roll.clear()

    def _audio(self) -> np.ndarray:
        return np.concatenate(self.parts) if self.parts else np.empty(0, dtype=np.float32)

    def _identity_audio(self) -> np.ndarray:
        return np.concatenate(self.identity_parts) if self.identity_parts else np.empty(0, dtype=np.float32)

    def feed(self, block: np.ndarray, end_time: float, *, voiced: bool | None = None) -> list[TranscriptionTask]:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if voiced is None:
            voiced = _rms(block) >= VAD_THRESHOLD
        tasks: list[TranscriptionTask] = []
        if voiced:
            self.last_voice_end = end_time
        if not self.active:
            self.pre_roll.append(block)
            total = sum(len(part) for part in self.pre_roll)
            while len(self.pre_roll) > 1 and total - len(self.pre_roll[0]) >= self.pre_roll_limit:
                total -= len(self.pre_roll.popleft())
            if not voiced:
                return tasks
            self.active = True
            self.utterance_id = uuid.uuid4().hex
            self.parts = list(self.pre_roll)
            self.buffered_samples = sum(len(item) for item in self.parts)
            self.speech_samples = len(block)
            self.silence_samples = 0
        else:
            self.parts.append(block)
            self.buffered_samples += len(block)
            if voiced:
                self.speech_samples += len(block)
                self.silence_samples = 0
            else:
                self.silence_samples += len(block)
        if self.collect_identity and voiced and self.identity_samples < 8 * self.sample_rate:
            remaining = 8 * self.sample_rate - self.identity_samples
            kept = block[:remaining].copy()
            self.identity_parts.append(kept)
            self.identity_samples += len(kept)
        if self.buffered_samples >= round(PARTIAL_SECONDS * self.sample_rate):
            tasks.append(TranscriptionTask(self.utterance_id, self.stream_id, self._audio(), False, end_time,
                                          speech_end_time=self.last_voice_end))
            self.parts.clear()
            self.buffered_samples = 0
        if self.silence_samples >= round(self.silence_seconds * self.sample_rate):
            if self.speech_samples >= round(self.min_speech_seconds * self.sample_rate):
                tasks.append(TranscriptionTask(self.utterance_id, self.stream_id, self._audio(), True, end_time,
                                              self._identity_audio(), self.last_voice_end))
            else:
                record_event("vad_short_utterance_discarded", speech_seconds=self.speech_samples / self.sample_rate)
            self._clear()
        return tasks

    def flush(self, end_time: float) -> list[TranscriptionTask]:
        if not self.active or self.speech_samples < round(self.min_speech_seconds * self.sample_rate):
            self._clear()
            return []
        task = TranscriptionTask(self.utterance_id, self.stream_id, self._audio(), True, end_time,
                                 self._identity_audio(), self.last_voice_end)
        self._clear()
        return [task]


class MonoUtteranceDetector:
    """Detect utterances directly in mono microphone blocks, without separation."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, *, probability: Any = None,
                 options: SpeechOptions | None = None, status_updates: bool = False) -> None:
        self.sample_rate = sample_rate
        self.options = options or (SpeechOptions() if probability is not None else None)
        if probability is not None and sample_rate != SAMPLE_RATE:
            raise ValueError("스트리밍 Silero 입력은 16000 Hz여야 합니다.")
        self.probability = probability
        kwargs = {} if self.options is None else {"pre_roll": self.options.pre_roll, "silence": self.options.silence,
                                                 "min_speech": self.options.min_speech}
        self.state = _StreamState(-1, sample_rate, collect_identity=False, **kwargs)
        self.last_end_time: float | None = None
        self.pending = np.empty(0, dtype=np.float32)
        self.pending_start: float | None = None
        self.status_updates = status_updates
        self._preview_id: str | None = None
        self._preview_at = -math.inf


    def process(self, mono: np.ndarray, start_time: float) -> list[TranscriptionTask]:
        mono = np.asarray(mono, dtype=np.float32)
        if mono.ndim != 1:
            raise ValueError("모노 마이크 오디오가 필요합니다.")
        if not np.all(np.isfinite(mono)):
            raise ValueError("마이크 오디오에 유효하지 않은 수치가 있습니다.")
        if self.probability is not None:
            return self._process_silero(mono, start_time)
        tasks: list[TranscriptionTask] = []
        for offset in range(0, len(mono), VAD_BLOCK_SAMPLES):
            block_end = min(offset + VAD_BLOCK_SAMPLES, len(mono))
            self.last_end_time = start_time + block_end / self.sample_rate
            tasks.extend(self.state.feed(mono[offset:block_end], self.last_end_time))
        return tasks

    def flush(self) -> list[TranscriptionTask]:
        tasks: list[TranscriptionTask] = []
        if self.probability is not None and self.pending.size:
            padded = np.pad(self.pending, (0, 512 - len(self.pending)))
            score = self.probability(padded)
            end = self.pending_start + len(self.pending) / self.sample_rate
            tasks.extend(self.state.feed(self.pending, end, voiced=score >= self.options.vad_threshold))
            self.last_end_time = end
            self.pending = np.empty(0, dtype=np.float32)
            self.pending_start = None
        if self.last_end_time is None:
            return []
        return tasks + self.state.flush(self.last_end_time)

    def _process_silero(self, mono: np.ndarray, start_time: float) -> list[TranscriptionTask]:
        if self.pending.size == 0:
            self.pending_start = start_time
        self.pending = np.concatenate([self.pending, mono])
        tasks = []
        while len(self.pending) >= 512:
            frame = self.pending[:512].copy()
            self.pending = self.pending[512:]
            self.last_end_time = self.pending_start + 512 / self.sample_rate
            score = self.probability(frame)
            was_active = self.state.active
            emitted = self.state.feed(frame, self.last_end_time, voiced=score >= self.options.vad_threshold)
            if self.status_updates and not was_active and self.state.active:
                print("[음성 감지] 듣고 있습니다...", flush=True)
                record_event("speech_started", utterance_id=self.state.utterance_id)
            if self.status_updates and any(task.final for task in emitted):
                print("[전사 중] 발화가 끝나 최종 문장을 확인합니다...", flush=True)
            tasks.extend(emitted)
            for task in emitted:
                record_event("vad_utterance_chunk", utterance_id=task.utterance_id, final=task.final,
                             duration_seconds=len(task.audio) / self.sample_rate,
                             final_frame_probability=score, threshold=self.options.vad_threshold,
                             speech_end_time=task.speech_end_time, detected_at=task.end_time)
            self.pending_start = self.last_end_time
        return tasks

    def preview_snapshot(self) -> PreviewTask | None:
        """Copy only at the preview interval; a preview never creates a final task."""
        if self.options is None or not self.state.active or self.last_end_time is None:
            return None
        if self._preview_id != self.state.utterance_id:
            self._preview_id, self._preview_at = self.state.utterance_id, -math.inf
        if (self.state.speech_samples < round(0.4 * self.sample_rate)
                or self.state.buffered_samples < round(0.8 * self.sample_rate)
                or self.state.silence_samples > round(0.128 * self.sample_rate)
                or self.last_end_time - self._preview_at < self.options.preview_interval):
            return None
        self._preview_at = self.last_end_time
        samples = self.state._audio()[-round(self.options.preview_window * self.sample_rate):].copy()
        return PreviewTask(self.state.utterance_id, samples, self.last_end_time)


class PerStreamUtteranceDetector:
    """Independent VAD and end-of-utterance state for two separated streams."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.states = [_StreamState(0, sample_rate), _StreamState(1, sample_rate)]

    def process(self, streams: np.ndarray, start_time: float) -> list[TranscriptionTask]:
        streams = np.asarray(streams, dtype=np.float32)
        if streams.ndim != 2 or streams.shape[0] != 2:
            raise ValueError("두 개의 분리 스트림이 필요합니다.")
        tasks: list[TranscriptionTask] = []
        for offset in range(0, streams.shape[1], VAD_BLOCK_SAMPLES):
            block_end = min(offset + VAD_BLOCK_SAMPLES, streams.shape[1])
            end_time = start_time + block_end / self.sample_rate
            block_tasks: list[TranscriptionTask] = []
            for stream_id, state in enumerate(self.states):
                block_tasks.extend(state.feed(streams[stream_id, offset:block_end], end_time))
            tasks.extend(sorted(block_tasks, key=lambda task: task.end_time))
        return tasks

    def flush(self, end_time: float | None = None) -> list[TranscriptionTask]:
        when = time.monotonic() if end_time is None else end_time
        tasks = [task for state in self.states for task in state.flush(when)]
        return sorted(tasks, key=lambda task: task.end_time)


def _rms(audio: np.ndarray) -> float:
    value = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(value)))) if value.size else 0.0


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
        from .ai_api import create_api_client
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
