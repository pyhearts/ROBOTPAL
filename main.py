"""Transcribe Korean speech from the default microphone on this computer."""

from __future__ import annotations

import queue
import sys
import threading
from collections import deque
from typing import Callable


SAMPLE_RATE = 16_000
BLOCK_SIZE = 1_024
VAD_THRESHOLD = 0.012
END_OF_SPEECH_SECONDS = 0.7
MIN_SPEECH_SECONDS = 0.25
MAX_UTTERANCE_SECONDS = 12.0
PRE_ROLL_SECONDS = 0.2
MODEL_NAME = "small"


def transcribe_utterance(model: object, audio: "object") -> str:
    """Return the recognized text for one audio segment, or an empty string."""
    segments, _ = model.transcribe(  # type: ignore[attr-defined]
        audio,
        language="ko",
        beam_size=1,
        vad_filter=True,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def handle_transcription(text: str) -> None:
    """Deliver a final transcript; connect robot command handling here later."""
    print(f"인식: {text}", flush=True)


def _transcription_worker(
    model: object,
    utterances: "queue.Queue[object]",
    on_transcription: Callable[[str], None],
) -> None:
    while True:
        audio = utterances.get()
        try:
            if audio is None:
                return
            try:
                text = transcribe_utterance(model, audio)
            except Exception as exc:  # Keep listening if one utterance fails.
                print(f"음성 구간을 인식하지 못했습니다: {exc}", file=sys.stderr)
                continue
            if text:
                try:
                    on_transcription(text)
                except Exception as exc:
                    print(f"인식 결과 전달 중 오류가 발생했습니다: {exc}", file=sys.stderr)
        finally:
            utterances.task_done()


def run_live_transcription(
    on_transcription: Callable[[str], None] = handle_transcription,
) -> None:
    """Listen continuously and send each finalized Korean phrase to a callback."""
    try:
        import numpy as np
        import sounddevice as sd
        from faster_whisper import WhisperModel
    except ImportError as exc:
        missing = exc.name or "required package"
        raise RuntimeError(
            f"필요한 패키지({missing})를 찾을 수 없습니다. "
            "PowerShell에서 아래 명령으로 프로젝트 의존성을 설치해 주세요.\n"
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
            "설치 후 실행: .\\.venv\\Scripts\\python.exe main.py"
        ) from exc

    print(f"한국어 음성 인식 모델({MODEL_NAME})을 준비합니다.")
    try:
        model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    except Exception as exc:
        raise RuntimeError(
            "음성 인식 모델을 준비하지 못했습니다. 첫 실행에는 인터넷 연결과 "
            f"모델 저장 공간이 필요합니다. 상세 정보: {exc}"
        ) from exc

    audio_blocks: "queue.Queue[np.ndarray]" = queue.Queue()
    utterances: "queue.Queue[object]" = queue.Queue()
    worker = threading.Thread(
        target=_transcription_worker,
        args=(model, utterances, on_transcription),
        name="speech-transcriber",
        daemon=True,
    )
    worker.start()

    def capture_audio(
        indata: "np.ndarray", frames: int, time_info: object, status: object
    ) -> None:
        if status:
            print(f"마이크 상태: {status}", file=sys.stderr)
        audio_blocks.put(indata.copy())

    block_seconds = BLOCK_SIZE / SAMPLE_RATE
    pre_roll: deque["np.ndarray"] = deque(
        maxlen=max(1, round(PRE_ROLL_SECONDS / block_seconds))
    )
    current: list["np.ndarray"] = []
    speech_seconds = 0.0
    silence_seconds = 0.0

    print("마이크를 듣고 있습니다. 종료하려면 Ctrl+C를 누르세요.")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=capture_audio,
        ):
            while True:
                block = audio_blocks.get()
                level = float(np.sqrt(np.mean(np.square(block))))

                if not current:
                    pre_roll.append(block)
                    if level >= VAD_THRESHOLD:
                        current = list(pre_roll)
                        speech_seconds = block_seconds
                        silence_seconds = 0.0
                    continue

                current.append(block)
                if level >= VAD_THRESHOLD:
                    speech_seconds += block_seconds
                    silence_seconds = 0.0
                else:
                    silence_seconds += block_seconds

                utterance_seconds = len(current) * block_seconds
                if (
                    silence_seconds >= END_OF_SPEECH_SECONDS
                    or utterance_seconds >= MAX_UTTERANCE_SECONDS
                ):
                    if speech_seconds >= MIN_SPEECH_SECONDS:
                        audio = np.concatenate(current).reshape(-1)
                        utterances.put(audio)
                    current.clear()
                    pre_roll.clear()
                    speech_seconds = 0.0
                    silence_seconds = 0.0
    except KeyboardInterrupt:
        print("\n음성 인식을 종료합니다.")
    except Exception as exc:
        raise RuntimeError(
            "마이크를 열거나 오디오를 읽지 못했습니다. 기본 입력 장치와 "
            f"마이크 권한을 확인해 주세요. 상세 정보: {exc}"
        ) from exc
    finally:
        if current and speech_seconds >= MIN_SPEECH_SECONDS:
            utterances.put(np.concatenate(current).reshape(-1))
        utterances.put(None)
        utterances.join()
        worker.join()


def main() -> int:
    try:
        run_live_transcription()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
