"""Configurable speech recognition, streaming Silero VAD, and input diagnostics."""

from __future__ import annotations

import io
import math
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .session_log import as_json, record_event

SAMPLE_RATE = 16_000
RETRY_MESSAGE = "잘 듣지 못했습니다. 구체적인 명령을 다시 말해 주세요."
VOCABULARY = "로봇, 전진, 후진, 좌회전, 우회전, 정지, 센티미터, 미터, 도, 초"


@dataclass(frozen=True)
class SpeechOptions:
    input_device: int | None = None
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 3
    backend: str = "local"
    vad_threshold: float = 0.5
    pre_roll: float = 0.4
    silence: float = 0.45
    min_speech: float = 0.12
    internal_vad: bool = False
    min_logprob: float = -1.2
    max_no_speech: float = 0.7
    vocabulary: str = VOCABULARY
    show_levels: bool = False
    live_preview: bool = True
    preview_interval: float = 1.0
    preview_window: float = 4.0
    max_output_tokens: int = 256

    def __post_init__(self) -> None:
        if self.input_device is not None and self.input_device < 0:
            raise ValueError("마이크 번호는 0 이상이어야 합니다.")
        if not self.model.strip() or self.device not in {"cpu", "cuda"}:
            raise ValueError("전사 모델 이름과 실행 장치(cpu/cuda)를 확인해 주세요.")
        if self.backend not in {"local", "openai"} or not 1 <= self.beam_size <= 10:
            raise ValueError("전사 방식(local/openai)과 beam 크기(1~10)를 확인해 주세요.")
        if not 0 < self.vad_threshold < 1 or not 0 <= self.pre_roll <= 2:
            raise ValueError("VAD 기준은 0~1 사이, 시작 전 보존 시간은 0~2초여야 합니다.")
        if not 0.2 <= self.silence <= 3 or not 0.03 <= self.min_speech <= 1:
            raise ValueError("종료 침묵은 0.2~3초, 최소 음성은 0.03~1초여야 합니다.")
        numbers = [self.vad_threshold, self.pre_roll, self.silence, self.min_speech,
                   self.min_logprob, self.max_no_speech, self.preview_interval, self.preview_window]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("음성 설정에는 유한한 수치를 사용해 주세요.")
        if not -10 <= self.min_logprob < 0 or not 0 < self.max_no_speech <= 1:
            raise ValueError("전사 품질 기준값의 범위를 확인해 주세요.")
        if not 0.5 <= self.preview_interval <= 5 or not 1 <= self.preview_window <= 8:
            raise ValueError("임시 전사 간격은 0.5~5초, 창 길이는 1~8초여야 합니다.")
        if not 32 <= self.max_output_tokens <= 256:
            raise ValueError("최대 전사 출력 토큰은 32~256이어야 합니다.")


class SileroProbability:
    """Carry Silero's recurrent state across 32ms frames; never reset per block.

    Uses the packaged faster-whisper 1.2.1 Silero v6 ONNX model, without downloads.
    Each live microphone has independent h/c/context despite a shared ONNX session.
    """
    frame_samples = 512

    def __init__(self) -> None:
        try:
            from faster_whisper.vad import get_vad_model

            self.session = get_vad_model().session
            inputs = {item.name for item in self.session.get_inputs()}
            if inputs != {"input", "h", "c"}:
                raise RuntimeError("지원하지 않는 Silero ONNX 형식")
        except Exception as exc:
            raise RuntimeError(
                "Silero VAD를 준비하지 못했습니다. requirements.txt를 설치해 주세요 "
                f"(faster-whisper>=1.2.1, onnxruntime 필요). 상세 정보: {exc}"
            ) from exc
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros_like(self.h)
        self.context = np.zeros(64, dtype=np.float32)

    def reset(self) -> None:
        self.h.fill(0)
        self.c.fill(0)
        self.context.fill(0)

    def __call__(self, frame: np.ndarray) -> float:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (self.frame_samples,):
            raise ValueError("Silero에는 512샘플 프레임이 필요합니다.")
        combined = np.concatenate([self.context, frame]).reshape(1, -1)
        probability, self.h, self.c = self.session.run(
            None, {"input": combined, "h": self.h, "c": self.c}
        )
        self.context = frame[-64:].copy()
        value = float(np.asarray(probability).reshape(-1)[0])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise RuntimeError("Silero가 유효하지 않은 음성 확률을 반환했습니다.")
        return value


def audio_metrics(audio: np.ndarray) -> dict[str, float]:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return {"rms": 0.0, "peak": 0.0, "clipping_ratio": 0.0, "duration": 0.0}
    return {"rms": float(np.sqrt(np.mean(samples * samples))),
            "peak": float(np.max(np.abs(samples))),
            "clipping_ratio": float(np.mean(np.abs(samples) >= 0.98)),
            "duration": len(samples) / SAMPLE_RATE}


class InputMeter:
    """Emit one aggregate per second without inference or logging in the callback."""
    def __init__(self) -> None:
        self.count = 0
        self.square_sum = 0.0
        self.peak = 0.0
        self.clipped = 0

    def update(self, audio: np.ndarray, *, show: bool = False) -> None:
        self.count += len(audio)
        self.square_sum += float(np.dot(audio, audio))
        self.peak = max(self.peak, float(np.max(np.abs(audio))) if len(audio) else 0)
        self.clipped += int(np.count_nonzero(np.abs(audio) >= 0.98))
        if self.count < SAMPLE_RATE:
            return
        rms = math.sqrt(self.square_sum / self.count)
        clipping = self.clipped / self.count
        record_event("microphone_level", rms=rms, peak=self.peak, clipping_ratio=clipping,
                     duration=self.count / SAMPLE_RATE)
        if show or clipping > 0.002:
            note = " 입력이 찌그러집니다. 마이크 감도를 낮추세요." if clipping > 0.002 else ""
            print(f"마이크 음량: RMS={rms:.4f}, 최대={self.peak:.3f}, 클리핑={clipping:.1%}{note}", flush=True)
        self.count = self.clipped = 0
        self.square_sum = self.peak = 0.0


def list_microphones(sd: Any = None) -> None:
    if sd is None:
        import sounddevice as sd
    default = sd.default.device[0]
    print("사용 가능한 입력 장치:")
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            marker = " (기본)" if index == default else ""
            print(f"{index}: {info['name']}{marker} / 기본 샘플레이트 {info['default_samplerate']:.0f} Hz")


def prepare_microphone(sd: Any, device: int | None) -> dict[str, Any]:
    try:
        info = dict(sd.query_devices(device, "input"))
        sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=SAMPLE_RATE)
    except Exception as exc:
        raise RuntimeError(f"마이크 입력 설정을 확인해 주세요 (--list-mics, --input-device). 상세 정보: {exc}") from exc
    record_event("microphone_selected", device=device, name=info.get("name"),
                 sample_rate=SAMPLE_RATE, default_sample_rate=info.get("default_samplerate"))
    print(f"마이크: {info.get('name', device)} / 16000 Hz 모노", flush=True)
    return info


def capture_sample(*, device: int | None, seconds: float, destination: Path | None = None) -> None:
    """Diagnostic capture; persist PCM only when the user explicitly supplies a path."""
    if not math.isfinite(seconds) or not 1 <= seconds <= 30:
        raise ValueError("진단·평가 녹음은 1~30초 사이여야 합니다.")
    if destination is not None and destination.exists():
        raise ValueError("이미 존재하는 음성 파일은 덮어쓰지 않습니다.")
    import sounddevice as sd
    prepare_microphone(sd, device)
    print(f"{seconds:g}초 동안 평소처럼 말해 주세요. 입력 상태를 측정합니다.", flush=True)
    try:
        recording = sd.rec(round(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=1, dtype="float32", device=device)
        status = sd.wait()
        if status:
            record_event("microphone_status", status=str(status))
            print(f"마이크 상태: {status}")
    except Exception as exc:
        raise RuntimeError(f"마이크 진단에 실패했습니다: {exc}") from exc
    samples = np.asarray(recording, dtype=np.float32).reshape(-1)
    metrics = audio_metrics(samples)
    record_event("microphone_diagnostic", **metrics)
    print(f"평균 음량={metrics['rms']:.4f}, 최대={metrics['peak']:.3f}, 클리핑={metrics['clipping_ratio']:.1%}")
    if metrics["rms"] < 0.005:
        print("입력이 작습니다. 장치 선택·입력 감도·마이크 거리를 확인해 주세요.")
    if metrics["clipping_ratio"] > 0.002:
        print("입력이 찌그러집니다. 감도를 낮추거나 마이크에서 조금 떨어져 주세요.")
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as target:
            target.write(encode_wav(samples))
        print(f"평가용 원본 녹음 저장: {destination.resolve()}")
        record_event("evaluation_audio_saved", path=str(destination.resolve()), **metrics)


def encode_wav(samples: np.ndarray) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return output.getvalue()


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    accepted: bool
    reason: str
    metrics: dict[str, Any]


class SpeechTranscriber:
    def __init__(self, model: Any, options: SpeechOptions, *, client: Any = None) -> None:
        self.model, self.options, self.client = model, options, client
        self.last_result: TranscriptionResult | None = None

    def transcribe_audio(self, audio: np.ndarray, *, preview: bool = False) -> TranscriptionResult:
        self.last_result = None
        started = time.monotonic()
        metrics: dict[str, Any] = audio_metrics(audio)
        if preview and self.options.backend != "local":
            raise ValueError("임시 전사는 추가 API 호출 없이 로컬에서만 수행합니다.")
        try:
            if not np.all(np.isfinite(audio)) or metrics["peak"] == 0:
                self.last_result = TranscriptionResult("", False, "invalid_or_silent_audio", metrics)
                record_event("stt_preview_result" if preview else "stt_result", text="", accepted=False, reason=self.last_result.reason,
                             backend=self.options.backend, model=self.options.model, **metrics)
                return self.last_result
            if self.options.backend == "openai":
                text = self._api(audio)
                quality: list[dict[str, Any]] = []
            else:
                token_budget = min(40 if preview else self.options.max_output_tokens,
                                   max(24 if preview else 32, math.ceil(metrics["duration"] * 12) + 16))
                segments, info = self.model.transcribe(
                    audio, language="ko", beam_size=1 if preview else self.options.beam_size,
                    vad_filter=self.options.internal_vad, temperature=0,
                    condition_on_previous_text=False, hotwords=self.options.vocabulary,
                    max_new_tokens=token_budget, repetition_penalty=1.1,
                    no_repeat_ngram_size=6, without_timestamps=True,
                )
                segments = list(segments)  # One bounded decoding window for each <=20s chunk.
                text = " ".join(segment.text.strip() for segment in segments).strip()
                quality = [{"text": segment.text, "start": segment.start, "end": segment.end,
                            "avg_logprob": segment.avg_logprob, "no_speech_prob": segment.no_speech_prob,
                            "compression_ratio": segment.compression_ratio} for segment in segments]
                metrics["language_probability"] = getattr(info, "language_probability", None)
                metrics["token_budget"] = token_budget
                metrics["generated_tokens"] = sum(len(getattr(segment, "tokens", [])) for segment in segments)
            metrics["segments"] = quality
            reason = self._quality_reason(text, quality, metrics)
            if metrics.get("generated_tokens", 0) >= metrics.get("token_budget", math.inf) - 2:
                reason = "output_token_limit"
            metrics["transcription_seconds"] = time.monotonic() - started
            metrics["realtime_factor"] = metrics["transcription_seconds"] / max(metrics["duration"], 0.001)
            self.last_result = TranscriptionResult(text, not reason, reason, metrics)
            record_event("stt_preview_result" if preview else "stt_result", text=text, accepted=not reason, reason=reason,
                         backend=self.options.backend, model="gpt-transcribe" if self.options.backend == "openai" else self.options.model,
                         beam_size=1 if preview else self.options.beam_size, **metrics)
            return self.last_result
        except Exception as exc:
            record_event("stt_error", error_type=type(exc).__name__, backend=self.options.backend,
                         elapsed_seconds=time.monotonic() - started)
            raise

    def _quality_reason(self, text: str, segments: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
        if not text:
            return "empty_transcript"
        if not np.isfinite(metrics["rms"]) or metrics["peak"] == 0:
            return "invalid_or_silent_audio"
        for segment in segments:
            values = [segment["avg_logprob"], segment["no_speech_prob"], segment["compression_ratio"]]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                return "invalid_quality_metrics"
            if segment["avg_logprob"] < self.options.min_logprob:
                return "low_log_probability"
            if segment["no_speech_prob"] > self.options.max_no_speech:
                return "high_no_speech_probability"
            if segment["compression_ratio"] > 2.4:
                return "repetitive_transcript"
        return ""

    def _api(self, audio: np.ndarray) -> str:
        from .action_planner import PlannerConnectionError, _friendly_api_error
        try:
            response = self.client.audio.transcriptions.create(
                model="gpt-transcribe", file=("utterance.wav", encode_wav(audio), "audio/wav"),
                languages=["ko"], keywords=[term.strip() for term in self.options.vocabulary.split(",")],
                response_format="json", timeout=60.0,
            )
        except Exception as exc:
            raise PlannerConnectionError(_friendly_api_error(exc, model="gpt-transcribe")) from exc
        record_event("stt_api_response", model="gpt-transcribe", text=response.text,
                     usage=as_json(getattr(response, "usage", None)))
        return response.text.strip()

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        self.model = None


def options_dict(options: SpeechOptions) -> dict[str, Any]:
    return asdict(options)
