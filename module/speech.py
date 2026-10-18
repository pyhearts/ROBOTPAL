"""Direct microphone VAD and transcription, with legacy separation kept for reuse."""

from __future__ import annotations

import importlib.metadata
import json
import math
import queue
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from .session_log import record_event
from .speech_quality import (SpeechOptions, SileroProbability, SpeechTranscriber,
                             TranscriptionResult, InputMeter, RETRY_MESSAGE,
                             prepare_microphone, options_dict)

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 4.0
HOP_SECONDS = 3.0
OVERLAP_SECONDS = WINDOW_SECONDS - HOP_SECONDS
SILENCE_SECONDS = 0.7
MIN_SPEECH_SECONDS = 0.25
PARTIAL_SECONDS = 20.0
PRE_ROLL_SECONDS = 0.2
VAD_BLOCK_SAMPLES = 1_024
VAD_THRESHOLD = 0.012
UNKNOWN_SPEAKER = "기타"


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("빈 음성 지문입니다.")
    return value / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_unit(left), _unit(right)))


@dataclass(frozen=True)
class SpeakerProfile:
    name: str
    embedding: np.ndarray


class ProfileStore:
    """Read and atomically write voiceprints without storing source audio."""

    @staticmethod
    def save(path: Path, profiles: Sequence[SpeakerProfile], model_name: str, model_version: str) -> None:
        if len(profiles) != 2:
            raise ValueError("정확히 두 명의 화자 지문이 필요합니다.")
        names = [profile.name.strip() for profile in profiles]
        if any(not name for name in names) or len(set(names)) != 2:
            raise ValueError("서로 다른 두 화자 이름이 필요합니다.")
        if UNKNOWN_SPEAKER in names:
            raise ValueError("'기타'는 미등록 화자를 위한 예약 이름입니다.")
        payload = {
            "schema_version": 1,
            "model": model_name,
            "model_version": model_version,
            "speakers": [
                {"name": profile.name, "embedding": _unit(profile.embedding).tolist()}
                for profile in profiles
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def load(path: Path, expected_model: str) -> list[SpeakerProfile]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1:
                raise ValueError("지원하지 않는 등록 파일 형식")
            if payload.get("model") != expected_model:
                raise ValueError("등록에 사용한 지문 모델이 현재 설정과 다름")
            speakers = payload["speakers"]
            if not isinstance(speakers, list) or len(speakers) != 2:
                raise ValueError("화자 수가 2명이 아님")
            profiles = [
                SpeakerProfile(str(item["name"]).strip(), _unit(item["embedding"]))
                for item in speakers
            ]
            if any(not profile.name for profile in profiles):
                raise ValueError("빈 화자 이름")
            if len({profile.name for profile in profiles}) != 2:
                raise ValueError("중복 화자 이름")
            if any(profile.name == UNKNOWN_SPEAKER for profile in profiles):
                raise ValueError("'기타'는 화자 이름으로 사용할 수 없음")
            if len({len(profile.embedding) for profile in profiles}) != 1:
                raise ValueError("지문 차원이 서로 다름")
            return profiles
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"화자 등록 파일을 읽을 수 없습니다: {exc}") from exc


class RegisteredSpeakerIdentifier:
    def __init__(self, profiles: Sequence[SpeakerProfile], embedder: Callable[[np.ndarray], np.ndarray], threshold: float = 0.45, margin: float = 0.08) -> None:
        if len(profiles) != 2:
            raise ValueError("두 개의 등록 화자 지문이 필요합니다.")
        self.profiles = list(profiles)
        self.embedder = embedder
        self.threshold = threshold
        self.margin = margin

    def identify_embedding(self, embedding: np.ndarray) -> str:
        scores = [cosine_similarity(profile.embedding, embedding) for profile in self.profiles]
        order = np.argsort(scores)[::-1]
        best, second = int(order[0]), int(order[1])
        # Cosine scores are float32 in production; tolerate only representation
        # error so the documented inclusive boundaries remain inclusive.
        epsilon = 1e-6
        if scores[best] + epsilon >= self.threshold and scores[best] - scores[second] + epsilon >= self.margin:
            return self.profiles[best].name
        return UNKNOWN_SPEAKER

    def identify(self, audio: np.ndarray) -> str:
        return self.identify_embedding(self.embedder(np.asarray(audio, dtype=np.float32)))


def _wave_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    left, right = left[:size], right[:size]
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-8 else 0.0


def _rms(audio: np.ndarray) -> float:
    value = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(value)))) if value.size else 0.0


class ChannelLinker:
    """Keep SepFormer output channels stable across overlapping windows."""

    def __init__(self, overlap_samples: int, embedder: Callable[[np.ndarray], np.ndarray] | None = None, waveform_weight: float = 0.7) -> None:
        self.overlap_samples = overlap_samples
        self.embedder = embedder
        self.waveform_weight = waveform_weight
        self.previous_tail: np.ndarray | None = None
        self.previous_embeddings: list[np.ndarray | None] = [None, None]

    def reset(self) -> None:
        self.previous_tail = None
        self.previous_embeddings = [None, None]

    def _embedding(self, audio: np.ndarray) -> np.ndarray | None:
        if self.embedder is None or _rms(audio) < 0.002:
            return None
        try:
            return _unit(self.embedder(audio))
        except Exception:
            return None

    def link(self, channels: np.ndarray) -> np.ndarray:
        channels = np.asarray(channels, dtype=np.float32)
        if channels.ndim != 2 or channels.shape[0] != 2:
            raise ValueError(f"분리 결과는 (2, samples)여야 합니다: {channels.shape}")
        current_embeddings = [self._embedding(channels[index]) for index in range(2)]
        order = (0, 1)
        if self.previous_tail is not None:
            scores = np.zeros((2, 2), dtype=np.float32)
            for previous in range(2):
                for current in range(2):
                    wave = _wave_correlation(self.previous_tail[previous], channels[current, : self.overlap_samples])
                    voice = 0.0
                    if self.previous_embeddings[previous] is not None and current_embeddings[current] is not None:
                        voice = cosine_similarity(self.previous_embeddings[previous], current_embeddings[current])
                    scores[previous, current] = self.waveform_weight * wave + (1.0 - self.waveform_weight) * voice
            if float(scores[0, 1] + scores[1, 0]) > float(scores[0, 0] + scores[1, 1]):
                order = (1, 0)
        linked = channels[list(order)].copy()
        self.previous_tail = linked[:, -self.overlap_samples :].copy()
        self.previous_embeddings = [current_embeddings[index] for index in order]
        return linked


def suppress_quiet_leakage(channels: np.ndarray, absolute_threshold: float = 0.0025, relative_threshold: float = 0.12) -> np.ndarray:
    result = np.asarray(channels, dtype=np.float32).copy()
    levels = [_rms(result[index]) for index in range(2)]
    loud = max(levels)
    if loud == 0.0:
        return result
    quiet = int(np.argmin(levels))
    if levels[quiet] < absolute_threshold or levels[quiet] / loud < relative_threshold:
        result[quiet].fill(0.0)
    return result


class StreamingSeparator:
    """4 s windows / 3 s hops with permutation linking and overlap-add."""

    def __init__(self, separate: Callable[[np.ndarray], np.ndarray], embedder: Callable[[np.ndarray], np.ndarray] | None = None, sample_rate: int = SAMPLE_RATE) -> None:
        self.separate = separate
        self.sample_rate = sample_rate
        self.window_samples = round(WINDOW_SECONDS * sample_rate)
        self.hop_samples = round(HOP_SECONDS * sample_rate)
        self.overlap_samples = self.window_samples - self.hop_samples
        self.linker = ChannelLinker(self.overlap_samples, embedder)
        self.buffer = np.empty(0, dtype=np.float32)
        self.held_tail: np.ndarray | None = None
        self.output_position = 0

    def _run_model(self, window: np.ndarray) -> np.ndarray:
        failure: Exception | None = None
        for _ in range(2):
            try:
                result = np.asarray(self.separate(window), dtype=np.float32)
                if result.shape == (self.window_samples, 2):
                    result = result.T
                if result.shape != (2, self.window_samples):
                    raise ValueError(f"예상하지 못한 분리 결과 크기: {result.shape}")
                return result
            except Exception as exc:
                failure = exc
        assert failure is not None
        raise failure

    def push(self, mono: np.ndarray) -> list[tuple[int, np.ndarray]]:
        self.buffer = np.concatenate([self.buffer, np.asarray(mono, dtype=np.float32).reshape(-1)])
        outputs: list[tuple[int, np.ndarray]] = []
        while len(self.buffer) >= self.window_samples:
            window = self.buffer[: self.window_samples]
            try:
                linked = suppress_quiet_leakage(self.linker.link(self._run_model(window)))
            except Exception as exc:
                print(f"분리 창 처리에 두 번 실패해 건너뜁니다: {exc}")
                self.linker.reset()
                self.held_tail = None
                # Emit silence rather than a mixture. Besides preserving the
                # timeline, this lets each stream's VAD close an active phrase.
                outputs.append(
                    (
                        self.output_position,
                        np.zeros((2, self.hop_samples), dtype=np.float32),
                    )
                )
                self.buffer = self.buffer[self.hop_samples :]
                self.output_position += self.hop_samples
                continue
            if self.held_tail is None:
                output = linked[:, : self.hop_samples]
            else:
                fade = np.linspace(0.0, 1.0, self.overlap_samples, dtype=np.float32)
                blended = self.held_tail * (1.0 - fade) + linked[:, : self.overlap_samples] * fade
                output = np.concatenate([blended, linked[:, self.overlap_samples : self.hop_samples]], axis=1)
            outputs.append((self.output_position, output))
            self.held_tail = linked[:, self.hop_samples :].copy()
            self.buffer = self.buffer[self.hop_samples :]
            self.output_position += self.hop_samples
        return outputs

    def flush(self) -> list[tuple[int, np.ndarray]]:
        actual = len(self.buffer)
        if actual == 0:
            return []
        start = self.output_position
        if self.held_tail is not None and actual <= self.overlap_samples:
            output = self.held_tail[:, :actual]
        else:
            padded = np.pad(self.buffer, (0, max(0, self.window_samples - actual)))
            try:
                linked = suppress_quiet_leakage(self.linker.link(self._run_model(padded)))
            except Exception as exc:
                print(f"마지막 분리 창 처리에 두 번 실패해 건너뜁니다: {exc}")
                self.reset()
                return []
            if self.held_tail is None:
                output = linked[:, :actual]
            else:
                overlap = min(self.overlap_samples, actual)
                fade = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                blended = self.held_tail[:, :overlap] * (1.0 - fade) + linked[:, :overlap] * fade
                output = np.concatenate([blended, linked[:, overlap:actual]], axis=1)
        self.reset()
        return [(start, output)]

    def reset(self) -> None:
        self.buffer = np.empty(0, dtype=np.float32)
        self.held_tail = None
        self.linker.reset()


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
                 options: SpeechOptions | None = None) -> None:
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
            emitted = self.state.feed(frame, self.last_end_time, voiced=score >= self.options.vad_threshold)
            tasks.extend(emitted)
            for task in emitted:
                record_event("vad_utterance_chunk", utterance_id=task.utterance_id, final=task.final,
                             duration_seconds=len(task.audio) / self.sample_rate,
                             final_frame_probability=score, threshold=self.options.vad_threshold,
                             speech_end_time=task.speech_end_time, detected_at=task.end_time)
            self.pending_start = self.last_end_time
        return tasks


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


def average_chunk_embeddings(audio: np.ndarray, embedder: Callable[[np.ndarray], np.ndarray], sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    chunk_samples = 2 * sample_rate
    embeddings: list[np.ndarray] = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) >= sample_rate and _rms(chunk) >= 0.005:
            embeddings.append(_unit(embedder(chunk)))
    if not embeddings:
        raise ValueError("지문을 만들 수 있는 유효 음성이 없습니다.")
    return _unit(np.mean(embeddings, axis=0))


def validate_enrollment_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[bool, str]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if _rms(audio) < 0.005:
        return False, "입력 음량이 너무 작습니다. 마이크에 조금 더 가까이 말해 주세요."
    if float(np.mean(np.abs(audio) >= 0.98)) > 0.002:
        return False, "입력이 클리핑되었습니다. 마이크에서 조금 떨어져 다시 말해 주세요."
    voiced = 0
    for start in range(0, len(audio), VAD_BLOCK_SAMPLES):
        block = audio[start : start + VAD_BLOCK_SAMPLES]
        if _rms(block) >= VAD_THRESHOLD:
            voiced += len(block)
    if voiced < round(6.0 * sample_rate):
        return False, "유효 음성이 6초보다 짧습니다. 쉬지 말고 약 8초간 말해 주세요."
    return True, ""


BLOCK_SIZE = 1_024
WHISPER_MODEL_NAME = "small"
SEPARATION_MODEL_NAME = "speechbrain/sepformer-whamr16k"
SPEAKER_MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
PROFILE_PATH = Path("data/speaker_profiles.json")
MODEL_ROOT = Path("models")
_STOP = object()


def handle_transcription(speaker: str, text: str) -> None:
    print(f"[{speaker}] {text}", flush=True)


def transcribe_utterance(model: object, audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    if isinstance(model, SpeechTranscriber):
        result = model.transcribe_audio(audio)
        return result.text if result.accepted else ""
    segments, _ = model.transcribe(audio, language="ko", beam_size=1, vad_filter=True)  # type: ignore[attr-defined]
    return " ".join(segment.text.strip() for segment in segments).strip()


class EcapaEmbedder:
    def __init__(self, model: object, torch_module: object) -> None:
        self.model, self.torch = model, torch_module
        self._lock = threading.Lock()

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        minimum = round(0.8 * SAMPLE_RATE)
        if samples.size == 0:
            raise ValueError("빈 오디오는 지문화할 수 없습니다.")
        if len(samples) < minimum:
            samples = np.tile(samples, int(np.ceil(minimum / len(samples))))[:minimum]
        waveform = self.torch.from_numpy(samples).unsqueeze(0)  # type: ignore[attr-defined]
        # The same encoder is shared by separation linking and final speaker
        # identification. Serialize inference so backend state is never raced.
        with self._lock, self.torch.inference_mode():  # type: ignore[attr-defined]
            encoded = self.model.encode_batch(waveform, normalize=True)  # type: ignore[attr-defined]
        vector = encoded.squeeze().detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("빈 음성 지문이 생성되었습니다.")
        return vector / norm


class SepformerAdapter:
    def __init__(self, model: object, torch_module: object) -> None:
        self.model, self.torch = model, torch_module

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        waveform = self.torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)  # type: ignore[attr-defined]
        with self.torch.inference_mode():  # type: ignore[attr-defined]
            separated = self.model.separate_batch(waveform)  # type: ignore[attr-defined]
        result = separated.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return result.T if result.shape[0] == len(audio) else result


def _load_runtime_dependencies():
    """Legacy separation/enrollment dependencies; unused by direct mic input."""
    try:
        import sounddevice as sd
        import torch
        from faster_whisper import WhisperModel
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.utils.fetching import LocalStrategy
    except ImportError as exc:
        raise RuntimeError(
            f"필요한 패키지({exc.name or '필수 패키지'})를 찾을 수 없습니다. 다음 명령으로 설치해 주세요.\n"
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc
    return sd, torch, WhisperModel, SepformerSeparation, SpeakerRecognition, LocalStrategy


def _load_voice_models(torch, SepformerSeparation, SpeakerRecognition, LocalStrategy, *, offline=False, identify_speakers=False):
    try:
        fetch_options = {}
        if offline:
            from speechbrain.utils.fetching import FetchConfig

            fetch_options["fetch_config"] = FetchConfig(allow_network=False)
        embedder = None
        if identify_speakers:
            speaker_model = SpeakerRecognition.from_hparams(
                source=SPEAKER_MODEL_NAME, savedir=str(MODEL_ROOT / "spkrec-ecapa-voxceleb"),
                run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY,
                **fetch_options,
            )
            embedder = EcapaEmbedder(speaker_model, torch)
        separator_model = SepformerSeparation.from_hparams(
            source=SEPARATION_MODEL_NAME, savedir=str(MODEL_ROOT / "sepformer-whamr16k"),
            run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY,
            **fetch_options,
        )
        return embedder, SepformerAdapter(separator_model, torch)
    except Exception as exc:
        if offline:
            raise RuntimeError(
                "로컬 음성 모델을 준비하지 못했습니다. "
                "인터넷을 사용할 수 있을 때 기본 음성 전사 모드(main.py)를 한 번 실행해 "
                f"모델을 준비해 주세요. offline=True에서는 다운로드하지 않습니다. 상세 정보: {exc}"
            ) from exc
        raise RuntimeError(
            "음성 모델을 준비하지 못했습니다. 첫 실행에는 인터넷 연결과 "
            f"모델 저장 공간이 필요합니다. 상세 정보: {exc}"
        ) from exc


def enroll_speakers() -> None:
    sd, torch, _, _, SpeakerRecognition, LocalStrategy = _load_runtime_dependencies()
    print("화자 지문 모델을 준비합니다.")
    try:
        speaker_model = SpeakerRecognition.from_hparams(
            source=SPEAKER_MODEL_NAME, savedir=str(MODEL_ROOT / "spkrec-ecapa-voxceleb"),
            run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY,
        )
    except Exception as exc:
        raise RuntimeError(f"화자 지문 모델을 준비하지 못했습니다: {exc}") from exc
    embedder = EcapaEmbedder(speaker_model, torch)
    profiles: list[SpeakerProfile] = []
    used_names: set[str] = set()
    for number in range(1, 3):
        while True:
            name = input(f"{number}번째 화자 이름: ").strip()
            if name and name not in used_names and name != UNKNOWN_SPEAKER:
                break
            print("'기타'를 제외한 비어 있지 않은 서로 다른 이름을 입력해 주세요.")
        while True:
            input(f"{name} 님: 준비되면 Enter를 누르고 약 8초 동안 계속 말해 주세요. ")
            print("녹음 중...")
            recording = sd.rec(round(8.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            audio = np.asarray(recording, dtype=np.float32).reshape(-1)
            valid, reason = validate_enrollment_audio(audio)
            if valid:
                break
            print(f"재녹음이 필요합니다: {reason}")
        profiles.append(SpeakerProfile(name, average_chunk_embeddings(audio, embedder)))
        used_names.add(name)
        print(f"{name} 님의 지문을 만들었습니다. 원본 녹음은 저장하지 않습니다.")
    try:
        version = importlib.metadata.version("speechbrain")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    ProfileStore.save(PROFILE_PATH, profiles, SPEAKER_MODEL_NAME, version)
    print(f"두 명의 화자 등록을 마쳤습니다: {PROFILE_PATH}")


def _transcription_worker(whisper: object, identifier: RegisteredSpeakerIdentifier | None, tasks: "queue.Queue[TranscriptionTask | object]", on_transcription: Callable[[str, str], None]) -> None:
    partial_text: dict[str, list[str]] = {}
    rejected: set[str] = set()
    while True:
        task = tasks.get()
        try:
            if task is _STOP:
                return
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


def _separation_worker(separator: StreamingSeparator, captured: "queue.Queue[np.ndarray | object]", separated: "queue.Queue[tuple[int, np.ndarray] | object]") -> None:
    while True:
        item = captured.get()
        try:
            if item is _STOP:
                for output in separator.flush():
                    separated.put(output)
                separated.put(_STOP)
                return
            assert isinstance(item, np.ndarray)
            for output in separator.push(item):
                separated.put(output)
        finally:
            captured.task_done()


def load_speaker_profiles() -> list[SpeakerProfile]:
    """Legacy opt-in identification only; never used for AI authorization."""
    if not PROFILE_PATH.exists():
        raise RuntimeError("화자 식별을 명시적으로 켰지만 등록 지문이 없습니다. module.speech.enroll_speakers()로 등록하거나 identify_speakers=False를 사용해 주세요.")
    try:
        return ProfileStore.load(PROFILE_PATH, SPEAKER_MODEL_NAME)
    except ValueError as exc:
        raise RuntimeError(f"화자 등록 파일이 없거나 손상되었습니다: {exc}. 지문을 다시 등록하거나 identify_speakers=False를 사용해 주세요.") from exc


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
        from .action_planner import create_api_client
        print("API 전사 선택: 발화 오디오를 OpenAI gpt-transcribe로 전송합니다. 사용료가 발생합니다.", flush=True)
        return SpeechTranscriber(None, options, client=create_api_client())
    if WhisperModel is None:
        from faster_whisper import WhisperModel
    return SpeechTranscriber(_load_whisper_model(WhisperModel, offline=offline, options=options), options)


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
    detector = MonoUtteranceDetector(probability=probability, options=options)
    whisper = create_speech_transcriber(options, offline=offline, WhisperModel=WhisperModel)
    meter = InputMeter()
    captured: queue.Queue[tuple[np.ndarray, float, str]] = queue.Queue()
    transcript_tasks: queue.Queue[TranscriptionTask | object] = queue.Queue()
    transcript_thread = threading.Thread(
        target=_transcription_worker,
        args=(whisper, None, transcript_tasks, on_transcription),
        name="speech-transcriber", daemon=True,
    )

    def capture_audio(indata, frames, time_info, status) -> None:
        del frames, time_info
        received_at = time.monotonic()
        audio = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
        captured.put((audio, received_at - len(audio) / SAMPLE_RATE, str(status) if status else ""))

    def process_block(item: tuple[np.ndarray, float, str]) -> None:
        audio, start_time, status = item
        if status:
            print(f"마이크 상태: {status}", file=sys.stderr)
            record_event("microphone_status", status=status)
        meter.update(audio, show=options.show_levels)
        for task in detector.process(audio, start_time):
            transcript_tasks.put(task)

    transcript_thread.start()
    print("직접 마이크 입력 모드: Silero 음성 감지 후 전사합니다 (SepFormer 미사용).")
    print("마이크를 듣고 있습니다. 종료하려면 Ctrl+C를 누르세요.")
    interrupted = False
    try:
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


def run_separated_transcription(
    on_transcription: Callable[[str, str], None] = handle_transcription,
    *,
    offline: bool = False,
    identify_speakers: bool = False,
) -> None:
    """Legacy SepFormer path; never used by the default CLI or assistant."""
    profiles = load_speaker_profiles() if identify_speakers else None
    sd, torch, WhisperModel, SepformerSeparation, SpeakerRecognition, LocalStrategy = _load_runtime_dependencies()
    print("음성 분리 모델을 준비합니다. 화자 등록 없이 모든 음성을 처리합니다." if not identify_speakers else "음성 분리 모델과 화자 지문 모델을 준비합니다.")
    embedder, separate = _load_voice_models(
        torch, SepformerSeparation, SpeakerRecognition, LocalStrategy,
        offline=offline, identify_speakers=identify_speakers
    )
    whisper = _load_whisper_model(WhisperModel, offline=offline)
    started = time.perf_counter()
    separate(np.zeros(round(4.0 * SAMPLE_RATE), dtype=np.float32))
    elapsed = time.perf_counter() - started
    print(f"CPU 분리 속도: 4초 창당 {elapsed:.2f}초")
    if elapsed > HOP_SECONDS:
        print(f"경고: 분리 시간이 3초 이동 간격보다 길어 실시간 처리가 어렵습니다 ({elapsed:.2f}초).", file=sys.stderr)

    identifier = RegisteredSpeakerIdentifier(profiles, embedder) if profiles is not None else None
    detector = PerStreamUtteranceDetector()
    captured: "queue.Queue[np.ndarray | object]" = queue.Queue()
    separated: "queue.Queue[tuple[int, np.ndarray] | object]" = queue.Queue()
    transcript_tasks: "queue.Queue[TranscriptionTask | object]" = queue.Queue()
    separator_thread = threading.Thread(target=_separation_worker, args=(StreamingSeparator(separate, embedder), captured, separated), name="speech-separator", daemon=True)
    transcript_thread = threading.Thread(target=_transcription_worker, args=(whisper, identifier, transcript_tasks, on_transcription), name="speech-transcriber", daemon=True)
    separator_thread.start()
    transcript_thread.start()
    session_started = time.monotonic()

    def capture_audio(indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            print(f"마이크 상태: {status}", file=sys.stderr)
        captured.put(np.asarray(indata, dtype=np.float32).reshape(-1).copy())

    print("마이크를 듣고 있습니다. 종료하려면 Ctrl+C를 누르세요.")
    interrupted = False
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE, callback=capture_audio):
            while True:
                item = separated.get()
                try:
                    if item is _STOP:
                        break
                    position, streams = item  # type: ignore[misc]
                    for task in detector.process(streams, session_started + position / SAMPLE_RATE):
                        transcript_tasks.put(task)
                finally:
                    separated.task_done()
    except KeyboardInterrupt:
        interrupted = True
        print("\n마이크 입력을 종료하고 남은 발화를 처리합니다.")
    except Exception as exc:
        raise RuntimeError("마이크를 열거나 오디오를 읽지 못했습니다. 기본 입력 장치와 " f"마이크 권한을 확인해 주세요. 상세 정보: {exc}") from exc
    finally:
        captured.put(_STOP)
        while separator_thread.is_alive() or not separated.empty():
            try:
                item = separated.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if item is not _STOP:
                    position, streams = item  # type: ignore[misc]
                    for task in detector.process(streams, session_started + position / SAMPLE_RATE):
                        transcript_tasks.put(task)
            finally:
                separated.task_done()
        for task in detector.flush():
            transcript_tasks.put(task)
        captured.join()
        separator_thread.join()
        transcript_tasks.put(_STOP)
        transcript_tasks.join()
        transcript_thread.join()
        if interrupted:
            print("음성 인식을 종료했습니다.")
