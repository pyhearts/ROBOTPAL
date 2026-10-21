"""Shared utterance state and detection for direct and separated audio."""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .session_log import record_event
from .speech_quality import SpeechOptions
from .live_transcription import PreviewTask

SAMPLE_RATE = 16_000
SILENCE_SECONDS = 0.7
MIN_SPEECH_SECONDS = 0.25
PARTIAL_SECONDS = 20.0
PRE_ROLL_SECONDS = 0.2
VAD_BLOCK_SAMPLES = 1_024
VAD_THRESHOLD = 0.012

def _rms(audio: np.ndarray) -> float:
    value = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(value)))) if value.size else 0.0


@dataclass
class TranscriptionTask:
    utterance_id: str
    stream_id: int
    audio: np.ndarray
    final: bool
    end_time: float
    identity_audio: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    speech_end_time: float | None = None


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


