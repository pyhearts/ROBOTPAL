"""Isolated OpenVoice/MeloTTS CPU worker; JSON lines are the only stdout output."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
from pathlib import Path
import socket
import sys
import tempfile
import uuid
import time


def deny_network() -> None:
    def blocked(*args, **kwargs):
        raise OSError("로컬 TTS 실행 중 네트워크 접근은 비활성화되어 있습니다.")
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked


def split_phrases(text: str) -> list[str]:
    """Keep decimal numbers intact while separating complete sentences."""
    return [part.strip() for part in re.split(r"(?<=[!?。！？])\s*|(?<=\.)\s+", text) if part.strip()]


class OpenVoiceBackend:
    def __init__(self, root: Path, threads: int = 2) -> None:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.hub.set_dir(str(root / "torchhub"))
        # g2pkk's old Windows branch tries to install eunjeon at runtime. Use the
        # bundled modern MeCab wheel explicitly and never spawn pip during speech.
        from g2pkk import G2p
        from mecab import MeCab
        G2p.check_mecab = lambda self: None
        G2p.get_mecab = lambda self: MeCab()
        from melo.api import TTS
        from openvoice.api import ToneColorConverter
        self.torch = torch
        self.root = root
        self.converter = ToneColorConverter(str(root / "checkpoints_v2/converter/config.json"), device="cpu")
        self.converter.load_ckpt(str(root / "checkpoints_v2/converter/checkpoint.pth"))
        self.model = TTS(language="KR", device="cpu", config_path=str(root / "melo/config.json"),
                         ckpt_path=str(root / "melo/checkpoint.pth"))
        speakers = self.model.hps.data.spk2id
        key = next(iter(speakers.keys()))
        self.speaker_id = speakers[key]
        source = root / "checkpoints_v2/base_speakers/ses" / (key.lower().replace("_", "-") + ".pth")
        self.source_se = torch.load(str(source), map_location="cpu", weights_only=True)
        self._voice = None
        self._target = None

    def prepare(self, reference: Path, voice_path: Path) -> dict:
        import numpy as np
        import soundfile as sf
        import librosa
        if voice_path.exists():
            raise ValueError("목소리 설정 파일이 이미 있습니다. 새 --tts-voice 경로를 지정하세요.")
        raw, rate = sf.read(str(reference), dtype="float32", always_2d=True)
        audio = raw.mean(axis=1)
        duration = len(audio) / rate
        if not 5 <= duration <= 60 or not np.all(np.isfinite(audio)):
            raise ValueError("기준 음성은 5~60초의 정상 WAV/FLAC 파일이어야 합니다.")
        if np.max(np.abs(audio)) > 1.01 or np.mean(audio ** 2) < 1e-7:
            raise ValueError("기준 음성이 무음이거나 진폭 범위를 벗어났습니다.")
        rate_out = self.converter.hps.data.sampling_rate
        audio = librosa.resample(audio, orig_sr=rate, target_sr=rate_out)
        # Trim only leading/trailing quiet; do not import an extra Whisper/VAD model.
        audio, _ = librosa.effects.trim(audio, top_db=35)
        if len(audio) / rate_out < 5:
            raise ValueError("앞뒤 무음을 제외한 기준 음성이 5초 미만입니다.")
        digest = hashlib.sha256(reference.read_bytes()).hexdigest()
        voice_dir = voice_path.parent / (voice_path.stem + "_" + digest[:12] + "_" + uuid.uuid4().hex[:6])
        voice_dir.mkdir(parents=True, exist_ok=False)
        prepared = voice_dir / "reference.wav"
        sf.write(str(prepared), audio, rate_out, subtype="PCM_16")
        target = voice_dir / "speaker.pth"
        self.converter.extract_se(str(prepared), se_save_path=str(target))
        version = hashlib.sha256(target.read_bytes()).hexdigest()
        metadata = {"version": 1, "reference_sha256": digest,
                    "reference_original": str(reference.resolve()), "embedding": str(target.resolve()),
                    "voice_version": version, "prepared_reference": str(prepared.resolve()),
                    "model_version": self.model_version()}
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        with voice_path.open("x", encoding="utf-8") as out:
            json.dump(metadata, out, ensure_ascii=False, indent=2)
        return metadata

    def model_version(self) -> str:
        manifest = self.root / "manifest.json"
        if not manifest.is_file():
            raise ValueError("TTS 모델 manifest가 없습니다. setup_tts.py를 실행하세요.")
        return hashlib.sha256(manifest.read_bytes()).hexdigest()

    def synthesize(self, text: str, output: Path, voice: Path, speed: float, style: str = "standard") -> dict:
        metadata = json.loads(voice.read_text(encoding="utf-8"))
        if metadata["model_version"] != self.model_version():
            raise ValueError("모델이 바뀌었습니다. 새 목소리 설정을 준비하세요.")
        embedding = Path(metadata["embedding"])
        if hashlib.sha256(embedding.read_bytes()).hexdigest() != metadata["voice_version"]:
            raise ValueError("목소리 특성 파일이 변경되었습니다. 새 설정을 준비하세요.")
        if self._voice != str(voice):
            self._target = self.torch.load(str(embedding), map_location="cpu", weights_only=True)
            self._voice = str(voice)
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
            source = Path(tmp) / "base.wav"
            started = time.monotonic()
            if style == "natural":
                import numpy as np
                import soundfile as sf
                # Keep punctuation as a prosody cue and avoid merging a short
                # greeting into the next sentence (Melo's default min_len=10).
                phrases = split_phrases(text)
                audio_parts = []
                rate = self.model.hps.data.sampling_rate
                for index, phrase in enumerate(phrases):
                    audio_parts.append(self.model.tts_to_file(
                        phrase, self.speaker_id, speed=speed, quiet=True,
                        sdp_ratio=0.55, noise_scale=0.6, noise_scale_w=0.9,
                    ))
                    if index < len(phrases) - 1:
                        pause = 0.23 if phrase.endswith(("?", "？")) else 0.18
                        audio_parts.append(np.zeros(round(rate * pause), dtype=np.float32))
                sf.write(str(source), np.concatenate(audio_parts), rate)
            elif style == "standard":
                self.model.tts_to_file(text, self.speaker_id, str(source), speed=speed, quiet=True)
            else:
                raise ValueError("지원하지 않는 TTS 말투입니다.")
            base_seconds = time.monotonic() - started
            started = time.monotonic()
            self.converter.convert(str(source), self.source_se, self._target,
                                   output_path=str(output), message="@MyShell")
            return {"base_synthesis_seconds": base_seconds,
                    "voice_conversion_seconds": time.monotonic() - started}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    os.environ["HF_HOME"] = str(args.models / "hf")
    os.environ["NLTK_DATA"] = str(args.models / "nltk")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    deny_network()
    protocol = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            backend = OpenVoiceBackend(args.models, args.threads)
        protocol.write(json.dumps({"ready": True}) + "\n")
        protocol.flush()
    except Exception as exc:
        protocol.write(json.dumps({"ready": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n")
        protocol.flush()
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request["action"] == "prepare":
                    result = backend.prepare(Path(request["reference"]), Path(request["voice"]))
                else:
                    output = Path(request["output"])
                    timings = backend.synthesize(request["text"], output, Path(request["voice"]), request["speed"],
                                                 request.get("style", "standard"))
                    result = {"output": str(output), **timings}
            response = {"ok": True, "result": result}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        protocol.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
