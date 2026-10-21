"""Korean-only GPT-SoVITS adapter for the isolated JSON worker."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import types


def validate_profile(profile: dict, root: Path) -> None:
    if profile.get("engine") != "gpt-sovits":
        raise ValueError("GPT-SoVITS용 참조 목소리 설정을 지정하세요.")
    canonical = {key: value for key, value in profile.items() if key != "voice_version"}
    digest = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if digest != profile.get("voice_version"):
        raise ValueError("참조 대본이나 설정이 변경되었습니다. 새 목소리 설정을 준비하세요.")
    version = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    if profile["model_version"] != version:
        raise ValueError("모델 버전이 바뀌었습니다. 참조 목소리를 다시 준비하세요.")
    if not profile.get("references"):
        raise ValueError("참조 음성 구간이 없습니다.")
    for item in profile["references"]:
        if not item.get("text", "").strip() or not 3 <= item["duration_seconds"] <= 10:
            raise ValueError("참조 대본 또는 음성 길이가 올바르지 않습니다.")
        if hashlib.sha256(Path(item["audio"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("참조 음성 파일이 변경되었습니다.")


def choose_reference(profile: dict, text: str) -> dict:
    references = profile["references"]
    if not references:
        raise ValueError("참조 음성 구간이 없습니다.")
    category = "question" if re.search(r"[?？]|(?:나요|까요|니|을까|는가)[.!。]?$", text.strip()) else "neutral"
    matches = [item for item in references if item.get("category") == category]
    return max(matches or references, key=lambda item: item.get("quality_score", 0))


class ReferenceTTSBackend:
    def __init__(self, root: Path, source: Path, threads: int = 2) -> None:
        import torch
        self.torch = torch
        self.root = root.resolve()
        source = source.resolve()
        if not (source / "GPT_SoVITS/TTS_infer_pack/TTS.py").is_file():
            raise ValueError("GPT-SoVITS 소스가 없습니다. setup_reference_tts.py를 실행하세요.")
        os.chdir(source)
        sys.path[:0] = [str(source / "GPT_SoVITS"), str(source)]
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        from g2pk2 import G2p
        from mecab import MeCab
        G2p.check_mecab = lambda self: None
        G2p.get_mecab = lambda self: MeCab()
        self._install_korean_only_adapters()
        from TTS_infer_pack.TTS import TTS
        import sv
        sv.sv_path = str(self.root / "sv/pretrained_eres2netv2w24s4ep4.ckpt")

        class KoreanTTS(TTS):
            def init_bert_weights(self, base_path):
                # Korean uses zero BERT features upstream; do not load a 1.2GB Chinese model.
                self.bert_model = self.bert_tokenizer = None

        unused = self.root / "unused_chinese_bert"
        unused.mkdir(exist_ok=True)
        config = {"version": "v2", "custom": {
            "device": "cpu", "is_half": False, "version": "v2",
            "t2s_weights_path": str(self.root / "s1v3.ckpt"),
            "vits_weights_path": str(self.root / "v2Pro/s2Gv2ProPlus.pth"),
            "cnhuhbert_base_path": str(self.root / "chinese-hubert-base"),
            "bert_base_path": str(unused),
        }}
        self.pipeline = KoreanTTS(config)
        self._profile_path = None
        self._profile = None

    @staticmethod
    def _install_korean_only_adapters() -> None:
        # Explicitly unsupported paths rather than downloading automatic language detectors.
        chinese = types.ModuleType("text.chinese")
        sys.modules["text.chinese"] = chinese
        segmenter = types.ModuleType("text.LangSegmenter")
        class KoreanOnlySegmenter:
            @staticmethod
            def getTexts(text):
                raise ValueError("참조 TTS는 all_ko 한국어 모드만 지원합니다.")
        segmenter.LangSegmenter = KoreanOnlySegmenter
        sys.modules[segmenter.__name__] = segmenter
        audio_module = types.ModuleType("tools.my_utils")
        def load_audio(path, rate):
            import soundfile as sf
            import numpy as np
            from scipy.signal import resample_poly
            from math import gcd
            samples, original_rate = sf.read(str(path), dtype="float32", always_2d=True)
            samples = samples.mean(axis=1)
            if original_rate != rate:
                divisor = gcd(original_rate, rate)
                samples = resample_poly(samples, rate // divisor, original_rate // divisor)
            return np.asarray(samples, dtype="float32")
        audio_module.load_audio = load_audio
        sys.modules[audio_module.__name__] = audio_module

    def synthesize(self, text: str, output: Path, voice: Path, speed: float, style="natural") -> dict:
        import numpy as np
        import soundfile as sf
        if self._profile_path != str(voice):
            profile = json.loads(voice.read_text(encoding="utf-8"))
            validate_profile(profile, self.root)
            self._profile, self._profile_path = profile, str(voice)
        reference = choose_reference(self._profile, text)
        # Keep one intact phrase for rhythm; use the best phrase from each other
        # recording as additional timbre evidence rather than splicing their speech.
        other_sources = {}
        for item in self._profile["references"]:
            source = item.get("source", item["audio"])
            if source == reference.get("source", reference["audio"]):
                continue
            if source not in other_sources or item.get("quality_score", 0) > other_sources[source].get("quality_score", 0):
                other_sources[source] = item
        auxiliary = list(other_sources.values())[:3]
        request = {
            "text": text, "text_lang": "all_ko", "ref_audio_path": reference["audio"],
            "aux_ref_audio_paths": [item["audio"] for item in auxiliary],
            "prompt_text": reference["text"], "prompt_lang": "all_ko",
            "top_k": 15, "top_p": 0.9, "temperature": 0.85,
            "text_split_method": "cut0", "batch_size": 1, "split_bucket": False,
            "speed_factor": speed, "fragment_interval": 0.22, "return_fragment": False,
            "parallel_infer": True, "repetition_penalty": 1.35, "seed": 42,
            "super_sampling": False,
        }
        started = time.monotonic()
        results = list(self.pipeline.run(request))
        if not results:
            raise ValueError("참조 TTS의 음성 출력이 비어 있습니다.")
        rate = results[0][0]
        if any(sr != rate for sr, _ in results):
            raise ValueError("합성 중 샘플레이트가 변경되었습니다.")
        audio = np.concatenate([waveform for _, waveform in results])
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            raise ValueError("참조 TTS가 정상 음성을 생성하지 못했습니다.")
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype("float32") / 32768
        if np.max(np.abs(audio)) < .0001:
            raise ValueError("참조 TTS 출력이 무음입니다.")
        sf.write(str(output), audio, rate, subtype="PCM_16")
        return {"base_synthesis_seconds": time.monotonic() - started,
                "voice_conversion_seconds": 0, "reference_id": reference["id"],
                "auxiliary_reference_ids": [item["id"] for item in auxiliary],
                "engine": "gpt-sovits"}
