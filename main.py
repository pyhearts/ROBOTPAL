"""Separate up to two microphone speakers and transcribe Korean speech."""

from __future__ import annotations

import argparse
import importlib.metadata
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from speaker_pipeline import (
    HOP_SECONDS, SAMPLE_RATE, PerStreamUtteranceDetector, ProfileStore,
    RegisteredSpeakerIdentifier, SpeakerProfile, StreamingSeparator,
    TranscriptionTask, average_chunk_embeddings, validate_enrollment_audio,
)

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


def _load_voice_models(torch, SepformerSeparation, SpeakerRecognition, LocalStrategy):
    try:
        speaker_model = SpeakerRecognition.from_hparams(
            source=SPEAKER_MODEL_NAME, savedir=str(MODEL_ROOT / "spkrec-ecapa-voxceleb"),
            run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY,
        )
        separator_model = SepformerSeparation.from_hparams(
            source=SEPARATION_MODEL_NAME, savedir=str(MODEL_ROOT / "sepformer-whamr16k"),
            run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY,
        )
        return EcapaEmbedder(speaker_model, torch), SepformerAdapter(separator_model, torch)
    except Exception as exc:
        raise RuntimeError(
            "음성 분리 또는 화자 지문 모델을 준비하지 못했습니다. 첫 실행에는 인터넷 연결과 "
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
            if name and name not in used_names:
                break
            print("비어 있지 않은 서로 다른 이름을 입력해 주세요.")
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


def _transcription_worker(whisper: object, identifier: RegisteredSpeakerIdentifier, tasks: "queue.Queue[TranscriptionTask | object]", on_transcription: Callable[[str, str], None]) -> None:
    partial_text: dict[str, list[str]] = {}
    while True:
        task = tasks.get()
        try:
            if task is _STOP:
                return
            assert isinstance(task, TranscriptionTask)
            try:
                text = transcribe_utterance(whisper, task.audio)
            except Exception as exc:
                print(f"음성 구간을 인식하지 못했습니다: {exc}", file=sys.stderr)
                text = ""
            texts = partial_text.setdefault(task.utterance_id, [])
            if text:
                texts.append(text)
            if task.final:
                complete = " ".join(texts).strip()
                partial_text.pop(task.utterance_id, None)
                if complete:
                    try:
                        speaker = identifier.identify(task.identity_audio)
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


def run_live_transcription(on_transcription: Callable[[str, str], None] = handle_transcription) -> None:
    if not PROFILE_PATH.exists():
        raise RuntimeError("등록된 화자 지문이 없습니다. 마이크 실행 전에 다음 명령을 실행해 주세요.\n.\\.venv\\Scripts\\python.exe .\\main.py --enroll")
    try:
        profiles = ProfileStore.load(PROFILE_PATH, SPEAKER_MODEL_NAME)
    except ValueError as exc:
        raise RuntimeError(f"화자 등록 파일이 없거나 손상되었습니다: {exc}\n.\\.venv\\Scripts\\python.exe .\\main.py --enroll") from exc
    sd, torch, WhisperModel, SepformerSeparation, SpeakerRecognition, LocalStrategy = _load_runtime_dependencies()
    print("음성 분리 모델과 화자 지문 모델을 준비합니다.")
    embedder, separate = _load_voice_models(torch, SepformerSeparation, SpeakerRecognition, LocalStrategy)
    print(f"한국어 Whisper 모델({WHISPER_MODEL_NAME})을 준비합니다.")
    try:
        whisper = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
    except Exception as exc:
        raise RuntimeError(f"Whisper 모델을 준비하지 못했습니다: {exc}") from exc
    started = time.perf_counter()
    separate(np.zeros(round(4.0 * SAMPLE_RATE), dtype=np.float32))
    elapsed = time.perf_counter() - started
    print(f"CPU 분리 속도: 4초 창당 {elapsed:.2f}초")
    if elapsed > HOP_SECONDS:
        print(f"경고: 분리 시간이 3초 이동 간격보다 길어 실시간 처리가 어렵습니다 ({elapsed:.2f}초).", file=sys.stderr)

    identifier = RegisteredSpeakerIdentifier(profiles, embedder)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enroll", action="store_true", help="두 명의 화자를 새로 등록합니다.")
    args = parser.parse_args(argv)
    try:
        enroll_speakers() if args.enroll else run_live_transcription()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
