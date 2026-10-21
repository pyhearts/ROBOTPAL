"""Reference curation and synthesis contracts without model downloads or playback."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from module.local_tts import LocalTTS, TTSOptions
from module.reference_tts_backend import ReferenceTTSBackend, choose_reference, validate_profile
from module.reference_voice import curate_references, finalize_voice, _save_wav


class ReferenceTests(unittest.TestCase):
    def test_engine_defaults_keep_old_environment_available(self):
        current, old = TTSOptions(), TTSOptions(engine="openvoice")
        self.assertEqual(current.engine, "gpt-sovits")
        self.assertEqual(current.voice.name, "reference_voice.json")
        self.assertEqual(old.voice.name, "voice.json")
        self.assertNotEqual(current.python, old.python)
        self.assertEqual(current.timeout, 240)

    def test_reference_selection_matches_question_and_falls_back(self):
        neutral = {"category": "neutral", "quality_score": 1}
        question = {"category": "question", "quality_score": 0}
        self.assertIs(choose_reference({"references": [neutral, question]}, "잘 들리시나요?"), question)
        self.assertIs(choose_reference({"references": [neutral, question]}, "감사합니다."), neutral)
        self.assertIs(choose_reference({"references": [neutral]}, "잘 들리나요?"), neutral)

    def make_profile(self, root):
        (root / "manifest.json").write_text('{"engine":"gpt-sovits"}')
        references = []
        for index in range(2):
            audio = root / f"ref{index}.wav"
            _save_wav(audio, np.full(4 * 32000, .1, dtype="float32"))
            references.append({"id": str(index), "audio": str(audio), "source": str(index),
                "text": "안녕하세요. 반갑습니다.", "duration_seconds": 4,
                "quality_score": 1 - index, "category": "neutral",
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest()})
        voice = root / "voice.json"
        profile = finalize_voice({"references": references}, root, voice)
        return voice, profile

    def test_profile_rejects_transcript_audio_and_model_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            voice, profile = self.make_profile(root)
            validate_profile(profile, root)
            altered = json.loads(voice.read_text(encoding="utf-8"))
            altered["references"][0]["text"] = "다른 대본"
            with self.assertRaisesRegex(ValueError, "대본"):
                validate_profile(altered, root)
            audio = Path(profile["references"][0]["audio"])
            original = audio.read_bytes()
            audio.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "음성 파일"):
                validate_profile(profile, root)
            audio.write_bytes(original)
            (root / "manifest.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "모델 버전"):
                validate_profile(profile, root)
            with self.assertRaisesRegex(ValueError, "덮어쓰지"):
                finalize_voice({}, root, voice)

    def test_changed_transcript_cannot_reuse_cached_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            voice, profile = self.make_profile(root)
            profile["references"][0]["text"] = "수정한 대본"
            voice.write_text(json.dumps(profile), encoding="utf-8")
            worker = Mock()
            tts = LocalTTS(TTSOptions(voice=voice, models=root, cache=root), worker=worker, sound=Mock())
            with self.assertRaisesRegex(ValueError, "대본"):
                tts.start()
            worker.start.assert_not_called()

    def test_valid_reference_profile_uses_normal_playback_lifecycle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            voice, profile = self.make_profile(root)
            worker, sound = Mock(), Mock()
            tts = LocalTTS(TTSOptions(voice=voice, models=root, cache=root), worker=worker, sound=sound)
            try:
                tts.start()
                self.assertEqual(tts._voice_version, profile["voice_version"])
                worker.start.assert_called_once()
            finally:
                tts.close()

    def test_backend_passes_intact_transcript_and_both_recordings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            voice, profile = self.make_profile(root)
            backend = ReferenceTTSBackend.__new__(ReferenceTTSBackend)
            backend.root = root
            backend._profile_path = None
            requests = []
            def run(request):
                requests.append(request)
                return [(32000, np.full(3200, 1000, dtype="int16"))]
            backend.pipeline = SimpleNamespace(run=run)
            result = backend.synthesize("안녕하세요.", root / "out.wav", voice, 1)
            request = requests[0]
            self.assertEqual(request["ref_audio_path"], profile["references"][0]["audio"])
            self.assertEqual(request["prompt_text"], profile["references"][0]["text"])
            self.assertEqual(request["aux_ref_audio_paths"], [profile["references"][1]["audio"]])
            self.assertEqual(request["text_lang"], "all_ko")
            self.assertEqual(request["speed_factor"], 1)
            self.assertEqual(result["auxiliary_reference_ids"], ["1"])
            backend.pipeline.run = lambda _: [(32000, np.zeros(100, dtype="float32"))]
            with self.assertRaisesRegex(ValueError, "무음"):
                backend.synthesize("안녕하세요.", root / "bad.wav", voice, 1)

    def test_curation_retains_timing_and_rejects_poor_transcript(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "original.wav"
            waveform = np.full(10 * 32000, .1, dtype="float32")
            _save_wav(original, waveform)
            before = original.read_bytes()
            good = SimpleNamespace(text="안녕하세요. 반갑습니다.", avg_logprob=-.1, no_speech_prob=.01, compression_ratio=1)
            poor = SimpleNamespace(text="이 구간은 제외합니다.", avg_logprob=-2, no_speech_prob=.01, compression_ratio=1)
            calls = iter([([good], None), ([poor], None)])
            whisper = SimpleNamespace(transcribe=lambda *a, **kw: next(calls))
            detector = lambda _: [{"start": 0, "end": 5 * 16000}, {"start": 5 * 16000, "end": 10 * 16000}]
            with patch("module.reference_voice._read_32k", return_value=waveform):
                data = curate_references([original], root / "dataset", whisper=whisper, detector=detector)
            self.assertEqual(original.read_bytes(), before)
            self.assertEqual(len(data["references"]), 1)
            self.assertEqual(data["references"][0]["duration_seconds"], 5)
            self.assertFalse(data["transcripts_reviewed"])


if __name__ == "__main__":
    unittest.main()
