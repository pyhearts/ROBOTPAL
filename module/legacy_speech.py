"""Optional speaker enrollment, identification and SepFormer transcription."""

from __future__ import annotations

import importlib.metadata
import json
import math
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from .utterance import SAMPLE_RATE, VAD_BLOCK_SAMPLES, VAD_THRESHOLD, TranscriptionTask, PerStreamUtteranceDetector, _rms
from .transcription import BLOCK_SIZE, _STOP, _load_whisper_model, _transcription_worker, handle_transcription

WINDOW_SECONDS = 4.0
HOP_SECONDS = 3.0
OVERLAP_SECONDS = WINDOW_SECONDS - HOP_SECONDS
UNKNOWN_SPEAKER = "기타"
SEPARATION_MODEL_NAME = "speechbrain/sepformer-whamr16k"
SPEAKER_MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
PROFILE_PATH = Path("data/speaker_profiles.json")
MODEL_ROOT = Path("models")

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
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-legacy-speech.txt"
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
                "인터넷을 사용할 수 있을 때 run_separated_transcription(offline=False)을 실행해 "
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


