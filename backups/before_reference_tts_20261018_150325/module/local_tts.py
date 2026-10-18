"""Optional local synthesis, bounded playback, caching, and result selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from typing import Any

from .audio_gate import PlaybackGate
from .session_log import command_context, record_event, save_command_audio

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = ROOT / ".tts-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


@dataclass(frozen=True)
class TTSOptions:
    python: Path = DEFAULT_PYTHON
    models: Path = ROOT / "models/tts"
    voice: Path = ROOT / "data/tts/voice.json"
    cache: Path = ROOT / "data/tts/cache"
    output_device: int | None = None
    speed: float = 1.0
    style: str = "standard"
    threads: int = 2
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.speed) or not .5 <= self.speed <= 2:
            raise ValueError("TTS 속도는 0.5~2여야 합니다.")
        if not 1 <= self.threads <= 8 or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("TTS 스레드 또는 제한 시간이 올바르지 않습니다.")
        if self.output_device is not None and self.output_device < 0:
            raise ValueError("스피커 번호는 0 이상이어야 합니다.")
        if self.style not in {"standard", "natural"}:
            raise ValueError("TTS 말투는 standard 또는 natural이어야 합니다.")


def spoken_text(result: Any) -> str:
    """Speak results, never status messages or unconfirmed execution claims."""
    if result.decision.category == "not_command":
        return ""
    if result.message:
        text = result.message
    elif result.plan is not None:
        if result.plan.status == "invalid":
            text = result.plan.error
        elif result.decision.category == "stop_command":
            text = "정지 계획을 접수했습니다."
        else:
            text = "행동 계획입니다. " + result.plan.summary
    else:
        return ""
    text = re.sub(r"[`*_#]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= 300:
        return text
    sentences = re.split(r"(?<=[.!?。！？])\s+|(?<=다\.)", text)
    selected = []
    for sentence in sentences:
        if len(" ".join(selected + [sentence])) > 300:
            break
        selected.append(sentence)
    return " ".join(selected) or "답변이 길어 화면에 표시했습니다. 화면을 확인해 주세요."


class WorkerClient:
    def __init__(self, options: TTSOptions) -> None:
        self.options = options
        self.process = None
        self._responses: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = threading.Event()

    def start(self) -> None:
        if self._closed.is_set():
            raise RuntimeError("TTS 작업자가 종료되었습니다.")
        if self.process is not None:
            return
        if not self.options.python.is_file():
            raise RuntimeError("로컬 TTS 환경이 없습니다. .venv의 Python으로 setup_tts.py를 실행하세요.")
        if not (self.options.models / "manifest.json").is_file():
            raise RuntimeError("TTS 모델이 없습니다. setup_tts.py를 먼저 실행하세요.")
        self._responses = queue.Queue()
        responses = self._responses
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", OMP_NUM_THREADS=str(self.options.threads))
        # The synthesis process has no reason to receive cloud credentials.
        env.pop("OPENAI_API_KEY", None)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            [str(self.options.python), "-u", str(Path(__file__).with_name("local_tts_worker.py")),
             "--models", str(self.options.models.resolve()), "--threads", str(self.options.threads)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", env=env, creationflags=flags,
        )
        self.process = process

        def read_responses():
            try:
                for line in process.stdout:
                    try:
                        responses.put(json.loads(line))
                    except ValueError:
                        responses.put({"ok": False, "error": "TTS 작업자 응답 형식 오류"})
            finally:
                responses.put({"ok": False, "error": "TTS 작업자가 종료되었습니다."})

        def read_errors():
            for line in process.stderr:
                record_event("tts_worker_output", text=line.rstrip())

        threading.Thread(target=read_responses, name="tts-protocol", daemon=True).start()
        threading.Thread(target=read_errors, name="tts-diagnostics", daemon=True).start()
        try:
            ready = responses.get(timeout=120)
            if not ready.get("ready"):
                raise RuntimeError(ready.get("error", "TTS 초기화 실패"))
        except queue.Empty as exc:
            self._terminate()
            raise RuntimeError("TTS 모델 준비 제한 시간(120초)을 초과했습니다.") from exc
        except BaseException:
            self._terminate()
            raise

    def request(self, action: str, **data) -> dict:
        with self._lock:
            self.start()
            try:
                self.process.stdin.write(json.dumps({"action": action, **data}, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
                response = self._responses.get(timeout=self.options.timeout)
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "TTS 작업 실패"))
                return response["result"]
            except queue.Empty as exc:
                self._terminate()
                raise RuntimeError("TTS 합성 제한 시간을 초과했습니다.") from exc
            except (BrokenPipeError, OSError) as exc:
                self._terminate()
                raise RuntimeError("TTS 작업자 연결이 끊겼습니다.") from exc

    def _terminate(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            for pipe in [process.stdin, process.stdout, process.stderr]:
                if pipe is not None:
                    pipe.close()

    def close(self) -> None:
        self._closed.set()
        self._terminate()
        self._responses.put({"ok": False, "error": "TTS 종료"})


def wav_info(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getnframes() == 0:
            raise ValueError("TTS 출력은 비어 있지 않은 16비트 모노 WAV여야 합니다.")
        return audio.getnframes() / audio.getframerate(), audio.getframerate()


class LocalTTS:
    def __init__(self, options: TTSOptions, *, gate: PlaybackGate | None = None,
                 worker: Any = None, sound: Any = None) -> None:
        self.options = options
        self.gate = gate or PlaybackGate()
        self.worker = worker or WorkerClient(options)
        self.sound = sound
        self._jobs: queue.Queue = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._thread = None
        self._voice_version = ""
        self._model_version = ""
        self._state_lock = threading.Lock()
        self.failed_jobs = 0

    def start(self) -> None:
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("TTS는 한 번만 시작할 수 있습니다.")
        if not self.options.voice.is_file():
            raise RuntimeError("기준 목소리가 준비되지 않았습니다. --prepare-tts --tts-reference 경로를 실행하세요.")
        metadata = json.loads(self.options.voice.read_text(encoding="utf-8"))
        self._voice_version = metadata["voice_version"]
        self._model_version = metadata["model_version"]
        self.options.cache.mkdir(parents=True, exist_ok=True)
        if self.sound is None:
            import sounddevice
            self.sound = sounddevice
        self.sound.query_devices(self.options.output_device, "output")
        self.worker.start()
        self._thread = threading.Thread(target=self._run, name="tts-playback", daemon=True)
        self._thread.start()
        record_event("tts_ready", voice_version=self._voice_version, model_version=self._model_version,
                     output_device=self.options.output_device, speed=self.options.speed, style=self.options.style)
        print("로컬 TTS 준비 완료. 음성 재생 중에는 사용자 음성을 인식하지 않습니다.", flush=True)

    def submit_result(self, result: Any, command_id: str) -> bool:
        return self.submit(spoken_text(result), command_id)

    def submit(self, text: str, command_id: str) -> bool:
        with self._state_lock:
            if not text or self._stop.is_set():
                return False
            try:
                self._jobs.put_nowait((text, command_id, time.monotonic()))
                record_event("tts_queued", command_id=command_id, text=text)
                return True
            except queue.Full:
                record_event("tts_skipped", command_id=command_id, reason="queue_full", text=text)
                return False

    def synthesize(self, text: str) -> tuple[Path, bool, float]:
        settings = {"text": text, "voice": self._voice_version,
            "model": self._model_version, "speed": self.options.speed,
            "pipeline": 1}
        if self.options.style != "standard":
            settings["style"] = self.options.style
            settings["style_version"] = 1
        key = hashlib.sha256(json.dumps(settings, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = self.options.cache / f"{key}.wav"
        if path.is_file():
            try:
                wav_info(path)
                return path, True, 0.0
            except (OSError, ValueError, EOFError, wave.Error):
                record_event("tts_cache_invalid", path=str(path))
        started = time.monotonic()
        temporary = self.options.cache / f"{key}.{uuid.uuid4().hex}.wav"
        try:
            profile = self.worker.request("synthesize", text=text, output=str(temporary.resolve()),
                                          voice=str(self.options.voice.resolve()), speed=self.options.speed,
                                          style=self.options.style)
            if isinstance(profile, dict):
                record_event("tts_synthesis_profile", **{k: profile[k] for k in
                    ["base_synthesis_seconds", "voice_conversion_seconds"] if k in profile})
            wav_info(temporary)
            try:
                temporary.replace(path)
            except PermissionError:
                # Some Windows sandbox/antivirus filters reject rename although writes work.
                # Invalid/interrupted cache entries are validated and regenerated on use.
                with temporary.open("rb") as source, path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return path, False, time.monotonic() - started

    def _play(self, path: Path) -> None:
        import numpy as np
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype("float32") / 32768
        with self._state_lock:
            if self._stop.is_set():
                return
            self.gate.begin()
            try:
                self.sound.play(samples, rate, device=self.options.output_device, blocking=False)
            except BaseException:
                self.gate.end()
                raise
        try:
            self.sound.wait()
        finally:
            self.gate.end()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text, command_id, queued_at = self._jobs.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                with command_context(command_id):
                    try:
                        path, cached, elapsed = self.synthesize(text)
                        if self._stop.is_set():
                            continue
                        duration, rate = wav_info(path)
                        saved = save_command_audio(path, command_id=command_id)
                        record_event("tts_audio_ready", text=text, audio_path=saved, cache_path=str(path),
                                     cache_hit=cached, synthesis_seconds=elapsed, audio_seconds=duration,
                                     sample_rate=rate, voice_version=self._voice_version,
                                     model_version=self._model_version, speed=self.options.speed,
                                     style=self.options.style,
                                     queue_wait_seconds=max(0, time.monotonic() - queued_at - elapsed))
                        record_event("tts_playback_start")
                        self._play(path)
                        record_event("tts_playback_end", cancelled=self._stop.is_set())
                    except Exception as exc:
                        self.failed_jobs += 1
                        record_event("tts_error", error_type=type(exc).__name__, error=str(exc))
                        if not self._stop.is_set():
                            print(f"음성 출력 실패: {exc}", flush=True)
            finally:
                self._jobs.task_done()

    def wait(self) -> None:
        self._jobs.join()
        if self.failed_jobs:
            raise RuntimeError(f"{self.failed_jobs}개 TTS 음성 출력에 실패했습니다. 로그를 확인하세요.")

    def close(self) -> None:
        with self._state_lock:
            self._stop.set()
            if self.sound is not None:
                try:
                    self.sound.stop()
                except Exception as exc:
                    record_event("tts_stop_error", error_type=type(exc).__name__)
        self.worker.close()
        if self._thread is not None:
            self._thread.join(timeout=6)
        while True:
            try:
                _, command_id, _ = self._jobs.get_nowait()
            except queue.Empty:
                break
            record_event("tts_skipped", command_id=command_id, reason="shutdown")
            self._jobs.task_done()


def list_speakers() -> None:
    import sounddevice as sd
    for index, device in enumerate(sd.query_devices()):
        if device["max_output_channels"]:
            print(f"{index}: {device['name']}")


def import_reference(reference: Path, directory: Path, *, always_convert: bool = False) -> Path:
    """Decode common recorder formats to a new PCM WAV without changing originals."""
    if not always_convert and reference.suffix.lower() in {".wav", ".flac"}:
        return reference
    import av
    chunks = []
    frame_count = 0
    converter = av.AudioResampler(format="s16", layout="mono", rate=24000)
    with av.open(str(reference)) as source:
        for decoded in source.decode(audio=0):
            for frame in converter.resample(decoded):
                chunks.append(frame.to_ndarray().tobytes())
                frame_count += frame.samples
                if frame_count > 60 * 24000:
                    raise ValueError("기준 녹음은 60초 이하여야 합니다.")
        for frame in converter.resample(None):
            chunks.append(frame.to_ndarray().tobytes())
            frame_count += frame.samples
    if not 5 * 24000 <= frame_count <= 60 * 24000:
        raise ValueError("기준 녹음은 5~60초여야 합니다.")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"reference_{uuid.uuid4().hex}.wav"
    with target.open("xb") as output, wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"".join(chunks))
    record_event("tts_reference_imported", original=str(reference.resolve()),
                 wav_path=str(target.resolve()), original_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
    return target


def combine_references(references: list[Path], directory: Path) -> Path:
    """Use equal central excerpts, capped at 50s total, preserving all originals."""
    per_file_frames = int(50 * 24000 / len(references))
    if per_file_frames < 5 * 24000:
        raise ValueError("기준 녹음은 최대 10개까지 함께 사용할 수 있습니다.")
    parts = []
    for reference in references:
        converted = import_reference(reference, directory, always_convert=True)
        with wave.open(str(converted), "rb") as audio:
            count = min(per_file_frames, audio.getnframes())
            start = max(0, (audio.getnframes() - count) // 2)
            audio.setpos(start)
            parts.append(audio.readframes(count))
        record_event("tts_reference_excerpt", original=str(reference.resolve()),
                     start_seconds=start / 24000, duration_seconds=count / 24000)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"combined_{uuid.uuid4().hex}.wav"
    with target.open("xb") as output, wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"".join(parts))
    return target


def prepare_voice(options: TTSOptions, reference: Path | list[Path]) -> None:
    originals = [reference] if isinstance(reference, Path) else list(reference)
    if not originals or not all(path.is_file() for path in originals):
        raise ValueError("기준 녹음 파일을 찾을 수 없습니다.")
    if options.voice.exists():
        raise ValueError("기존 목소리 설정을 덮어쓰지 않습니다. --tts-voice로 새 파일을 지정하세요.")
    directory = options.voice.parent / "imports"
    reference = (import_reference(originals[0], directory) if len(originals) == 1
                 else combine_references(originals, directory))
    worker = WorkerClient(options)
    try:
        metadata = worker.request("prepare", reference=str(reference.resolve()), voice=str(options.voice.resolve()))
        metadata["reference_originals"] = [str(path.resolve()) for path in originals]
        metadata["original_sha256"] = [hashlib.sha256(path.read_bytes()).hexdigest() for path in originals]
        options.voice.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"목소리 특성 준비 완료: {options.voice}", flush=True)
        # Keep the same loaded model for embedding extraction and fixed guidance.
        from .decision_router import FIXED_MESSAGES
        tts = LocalTTS(options, worker=worker)
        metadata = json.loads(options.voice.read_text(encoding="utf-8"))
        tts._voice_version, tts._model_version = metadata["voice_version"], metadata["model_version"]
        options.cache.mkdir(parents=True, exist_ok=True)
        messages = [v for k, v in FIXED_MESSAGES.items() if k != "not_command"] + ["정지 계획을 접수했습니다."]
        for index, text in enumerate(messages, 1):
            print(f"고정 안내 준비 중: {index}/{len(messages)}", flush=True)
            path, _, elapsed = tts.synthesize(text)
            record_event("tts_fixed_prepared", text=text, path=str(path), synthesis_seconds=elapsed)
    finally:
        worker.close()
    print(f"기준 목소리와 고정 안내 준비 완료: {options.voice}", flush=True)
