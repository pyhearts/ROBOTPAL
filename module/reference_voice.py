"""Prepare intact reference phrases and local ASR transcripts for reference TTS."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import uuid
import wave

import numpy as np

from .session_log import record_event


def _read_32k(path: Path) -> np.ndarray:
    import av
    frames = []
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=32000)
    with av.open(str(path)) as source:
        for frame in source.decode(audio=0):
            frames.extend(item.to_ndarray().reshape(-1).copy() for item in resampler.resample(frame))
        frames.extend(item.to_ndarray().reshape(-1).copy() for item in resampler.resample(None))
    return np.concatenate(frames).astype("float32") if frames else np.empty(0, dtype="float32")


def _save_wav(path: Path, samples: np.ndarray, rate=32000):
    with path.open("xb") as output, wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def curate_references(paths: list[Path], destination: Path, *, whisper=None, detector=None) -> dict:
    from .speech_evaluation import load_evaluation_audio
    from .speech_quality import SpeechOptions
    if not paths or not all(path.is_file() for path in paths):
        raise ValueError("기준 녹음 파일을 확인하세요.")
    if detector is None:
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        def detector(samples):
            return get_speech_timestamps(samples, VadOptions(
                min_speech_duration_ms=400, min_silence_duration_ms=350,
                speech_pad_ms=120, max_speech_duration_s=8.5,
            ), sampling_rate=16000)
    if whisper is None:
        from .speech import _load_whisper_model
        from faster_whisper import WhisperModel
        whisper = _load_whisper_model(WhisperModel, offline=True,
                                     options=SpeechOptions(beam_size=5, live_preview=False))
    destination.mkdir(parents=True, exist_ok=False)
    dataset = {"version": 1, "originals": [], "references": [], "transcripts_reviewed": False}
    for path in paths:
        original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        dataset["originals"].append({"path": str(path.resolve()), "sha256": original_hash})
        samples16 = load_evaluation_audio(path)
        samples32 = _read_32k(path)
        candidates = []
        for bounds in detector(samples16):
            start, end = bounds["start"] / 16000, bounds["end"] / 16000
            duration = end - start
            if not 3.2 <= duration <= 9.5:
                continue
            audio = samples16[bounds["start"]:bounds["end"]]
            rms = float(np.sqrt(np.mean(audio ** 2)))
            clipping = float(np.mean(np.abs(audio) >= .98))
            if rms < .008 or clipping > .005:
                continue
            candidates.append({"start": start, "end": end,
                "audio": audio, "score": 1 - abs(duration - 6) / 6 - clipping * 20})
        # Limit local decoding work while keeping a few alternatives per recording.
        candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:3]
        accepted = []
        for item in candidates:
            segments, info = whisper.transcribe(item["audio"], language="ko", beam_size=5,
                condition_on_previous_text=False, temperature=0, vad_filter=False,
                max_new_tokens=160, repetition_penalty=1.1, no_repeat_ngram_size=6)
            segments = list(segments)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if not segments or len(text) < 8:
                continue
            logprob = sum(seg.avg_logprob for seg in segments) / len(segments)
            if logprob < -.8 or any(seg.no_speech_prob > .6 or seg.compression_ratio > 2.4 for seg in segments):
                record_event("reference_clip_rejected", source=str(path), start=item["start"], text=text)
                continue
            identifier = uuid.uuid4().hex[:12]
            target = destination / f"reference_{identifier}.wav"
            # Keep original pace and pitch. Only add a short boundary fade to avoid clicks.
            clip = samples32[round(item["start"] * 32000):round(item["end"] * 32000)].copy()
            fade = min(160, len(clip) // 2)
            clip[:fade] *= np.linspace(0, 1, fade)
            clip[-fade:] *= np.linspace(1, 0, fade)
            _save_wav(target, clip)
            category = "question" if re.search(r"[?？]|(?:나요|까요|니)[.!]?$", text) else "neutral"
            reference = {"id": identifier, "audio": str(target.resolve()), "text": text,
                "source": str(path.resolve()), "start_seconds": item["start"],
                "duration_seconds": len(clip) / 32000, "category": category,
                "quality_score": item["score"] + logprob,
                "transcript_source": "local_whisper_small_beam5", "avg_logprob": logprob,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            accepted.append(reference)
            record_event("reference_clip_prepared", **reference)
        dataset["references"].extend(sorted(accepted, key=lambda item: item["quality_score"], reverse=True)[:2])
    if not dataset["references"]:
        raise ValueError("깨끗한 3~10초 발화 구간을 찾지 못했습니다. 새 기준 녹음을 준비하세요.")
    dataset_path = destination / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"참조 발화 {len(dataset['references'])}개 준비 완료: {dataset_path}", flush=True)
    return dataset


def finalize_voice(dataset: dict, models: Path, voice_path: Path) -> dict:
    if voice_path.exists():
        raise ValueError("기존 참조 목소리 설정은 덮어쓰지 않습니다. --tts-voice로 새 경로를 지정하세요.")
    manifest = models / "manifest.json"
    if not manifest.is_file():
        raise ValueError("참조 TTS 모델이 없습니다. setup_reference_tts.py를 실행하세요.")
    profile = {**dataset, "engine": "gpt-sovits", "model_version": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    canonical = json.dumps(profile, ensure_ascii=False, sort_keys=True).encode()
    profile["voice_version"] = hashlib.sha256(canonical).hexdigest()
    from .reference_tts_backend import validate_profile
    validate_profile(profile, models)
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    with voice_path.open("x", encoding="utf-8") as output:
        json.dump(profile, output, ensure_ascii=False, indent=2)
    record_event("reference_voice_ready", path=str(voice_path.resolve()), voice_version=profile["voice_version"])
    return profile
