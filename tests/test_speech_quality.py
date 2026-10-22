"""Quality gates, streaming VAD, microphone setup, and reproducible comparisons."""

import contextlib
import io
import json
import queue
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from module import cli, speech, speech_quality, vision_assistant, speech_pipeline
from module import speech_quality as quality
from module import speech_evaluation as evaluation


def audio(seconds, level=0.001):
    return np.full(round(seconds * 16000), level, dtype=np.float32)


def segment(text="앞으로 30cm 이동해", **kwargs):
    values = dict(text=text, start=0, end=1, avg_logprob=-0.3, no_speech_prob=0.01, compression_ratio=1)
    values.update(kwargs)
    return SimpleNamespace(**values)


def recognizer(segments=None, **kwargs):
    model = Mock()
    model.transcribe.return_value = (iter(segments or [segment()]), SimpleNamespace(language_probability=1))
    return quality.SpeechTranscriber(model, quality.SpeechOptions(**kwargs))


class StreamingVadTests(unittest.TestCase):
    def detector(self, probability=None):
        return speech.MonoUtteranceDetector(
            options=quality.SpeechOptions(),
            probability=probability or (lambda block: float(np.max(np.abs(block)) > 0)),
        )

    def test_soft_short_command_is_not_discarded_by_old_rms_threshold(self):
        detector = self.detector()
        tasks = detector.process(np.concatenate([audio(0.16), audio(0.8, 0)]), 0)
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0].final)
        self.assertLess(np.max(tasks[0].audio), speech_pipeline.VAD_THRESHOLD)
        self.assertAlmostEqual(tasks[0].speech_end_time, 0.16)

    def test_loud_noise_without_voice_probability_is_ignored(self):
        detector = self.detector(lambda block: 0.0)
        self.assertEqual(detector.process(audio(2, 0.8), 0), [])
        self.assertEqual(detector.flush(), [])

    def test_arbitrary_callback_sizes_preserve_audio_and_end_time(self):
        samples = np.concatenate([audio(0.6, 0), audio(0.32), audio(0.9, 0)])
        baseline = None
        for block_size in [100, 512, 1024, 3277]:
            detector = self.detector()
            tasks = []
            for offset in range(0, len(samples), block_size):
                tasks += detector.process(samples[offset:offset + block_size], offset / 16000)
            tasks += detector.flush()
            self.assertEqual(len(tasks), 1)
            if baseline is None:
                baseline = tasks[0]
            else:
                np.testing.assert_array_equal(tasks[0].audio, baseline.audio)
                self.assertAlmostEqual(tasks[0].end_time, baseline.end_time)
            first_voice = int(np.flatnonzero(tasks[0].audio)[0])
            self.assertGreaterEqual(first_voice / 16000, 0.35)

    def test_flush_preserves_partial_silero_frame(self):
        detector = self.detector()
        samples = np.full(512 * 5 + 27, 0.001, dtype=np.float32)
        self.assertEqual(detector.process(samples, 3), [])
        tasks = detector.flush()
        self.assertEqual(len(tasks), 1)
        np.testing.assert_array_equal(tasks[0].audio, samples)
        self.assertAlmostEqual(tasks[0].end_time, 3 + len(samples) / 16000)
        self.assertEqual(detector.flush(), [])

    def test_invalid_audio_is_rejected(self):
        with self.assertRaises(ValueError):
            self.detector().process(np.array([np.nan], dtype=np.float32), 0)

    def test_real_packaged_silero_runs_without_download_or_mutating_audio(self):
        model = quality.SileroProbability()
        frame = np.zeros(512, dtype=np.float32)
        for _ in range(10):
            probability = model(frame)
            self.assertTrue(0 <= probability <= 1)
        self.assertLess(probability, 0.5)
        np.testing.assert_array_equal(frame, np.zeros(512))
        with self.assertRaises(ValueError):
            model(np.zeros(1024))


class RecognitionQualityTests(unittest.TestCase):
    def test_local_settings_and_quality_metadata(self):
        transcriber = recognizer()
        result = transcriber.transcribe_audio(audio(1))
        self.assertTrue(result.accepted)
        kwargs = transcriber.model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["beam_size"], 3)
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(kwargs["language"], "ko")
        self.assertFalse(kwargs["vad_filter"])
        self.assertFalse(kwargs["condition_on_previous_text"])
        self.assertIn("센티미터", kwargs["hotwords"])
        self.assertIn("realtime_factor", result.metrics)

    def test_low_quality_is_rejected_without_changing_number_or_direction(self):
        for patch_values in [dict(avg_logprob=-2), dict(no_speech_prob=0.9),
                             dict(compression_ratio=3), dict(avg_logprob=float("nan"))]:
            with self.subTest(patch_values=patch_values):
                transcriber = recognizer([segment("오른쪽 300cm", **patch_values)])
                result = transcriber.transcribe_audio(audio(1))
                self.assertFalse(result.accepted)
                self.assertEqual(result.text, "오른쪽 300cm")

    def test_bad_chunk_prevents_partial_command_callback(self):
        transcriber = recognizer()
        transcriber.model.transcribe.side_effect = [
            (iter([segment("앞으로")]), SimpleNamespace()),
            (iter([segment("30cm", avg_logprob=-2)]), SimpleNamespace()),
        ]
        jobs, received = queue.Queue(), []
        jobs.put(speech_pipeline.TranscriptionTask("same", -1, audio(1), False, 0))
        jobs.put(speech_pipeline.TranscriptionTask("same", -1, audio(1), True, 0))
        jobs.put(speech._STOP)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            speech._transcription_worker(transcriber, None, jobs, lambda *args: received.append(args))
        self.assertEqual(received, [])
        self.assertEqual(output.getvalue().count(quality.RETRY_MESSAGE), 1)

    def test_api_is_opt_in_uses_memory_wav_and_returns_metadata(self):
        client = Mock()
        client.audio.transcriptions.create.return_value = SimpleNamespace(text="멈춰", usage={"total_tokens": 2})
        transcriber = quality.SpeechTranscriber(None, quality.SpeechOptions(backend="openai"), client=client)
        result = transcriber.transcribe_audio(audio(0.4))
        self.assertTrue(result.accepted)
        kwargs = client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-transcribe")
        self.assertEqual(kwargs["languages"], ["ko"])
        with wave.open(io.BytesIO(kwargs["file"][1])) as wav:
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnchannels(), 1)
        transcriber.close()
        client.close.assert_called_once()

    def test_silence_never_calls_model_or_paid_api(self):
        client = Mock()
        transcriber = quality.SpeechTranscriber(None, quality.SpeechOptions(backend="openai"), client=client)
        self.assertFalse(transcriber.transcribe_audio(audio(1, 0)).accepted)
        client.audio.transcriptions.create.assert_not_called()

    def test_invalid_settings_fail_early(self):
        for values in [dict(beam_size=0), dict(input_device=-1), dict(vad_threshold=float("nan")),
                       dict(silence=0), dict(min_speech=5), dict(backend="unknown")]:
            with self.assertRaises(ValueError):
                quality.SpeechOptions(**values)


class MicrophoneAndCliTests(unittest.TestCase):
    def test_diagnostics_do_not_store_audio_unless_explicit_path_is_given(self):
        sd = SimpleNamespace(rec=Mock(return_value=audio(1).reshape(-1, 1)), wait=Mock(return_value=None))
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", {"sounddevice": sd}), \
                patch.object(quality, "prepare_microphone"), contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            quality.capture_sample(device=1, seconds=1)
            self.assertEqual(list(root.iterdir()), [])
            destination = root / "explicit.wav"
            quality.capture_sample(device=1, seconds=1, destination=destination)
            self.assertTrue(destination.is_file())
            with wave.open(str(destination)) as wav:
                self.assertEqual(wav.getframerate(), 16000)
            with self.assertRaises(ValueError):
                quality.capture_sample(device=1, seconds=1, destination=destination)
            self.assertEqual(sd.rec.call_count, 2)

    def test_input_selection_checks_supported_settings_and_prints_device(self):
        sd = Mock()
        sd.query_devices.return_value = {"name": "선택한 마이크", "default_samplerate": 48000}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            quality.prepare_microphone(sd, 3)
        sd.check_input_settings.assert_called_once_with(device=3, channels=1, dtype="float32", samplerate=16000)
        self.assertIn("선택한 마이크", output.getvalue())

    def test_level_meter_accumulates_and_reports_clipping(self):
        meter = quality.InputMeter()
        output = io.StringIO()
        with patch.object(quality, "record_event") as log, contextlib.redirect_stdout(output):
            meter.update(audio(0.5, 1))
            log.assert_not_called()
            meter.update(audio(0.5, 1))
        self.assertEqual(log.call_args.kwargs["clipping_ratio"], 1)
        self.assertIn("찌그러집니다", output.getvalue())

    def test_cli_passes_selected_settings_to_default_and_assistant(self):
        with patch.object(vision_assistant, "run_voice_vision_assistant") as run:
            self.assertEqual(cli.run(["--input-device", "2", "--stt-beam", "5", "--show-levels"]), 0)
        options = run.call_args.kwargs["options"]
        self.assertEqual(options.input_device, 2)
        self.assertEqual(options.beam_size, 5)
        self.assertTrue(options.show_levels)
        with patch.object(vision_assistant, "run_voice_vision_assistant") as run:
            self.assertEqual(cli.run(["--assist", "--stt-backend", "openai"]), 0)
        self.assertEqual(run.call_args.kwargs["options"].backend, "openai")

    def test_bad_option_combinations_are_rejected(self):
        for args in [["--plan", "--stt-model", "medium"], ["--compare-beams", "5"],
                     ["--sample-seconds", "2"], ["--gesture", "--input-device", "0"]]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli.run(args)

    def test_microphone_listing_needs_no_model_or_api(self):
        with patch.object(speech_quality, "list_microphones") as listing, patch.object(speech, "run_live_transcription") as run:
            self.assertEqual(cli.run(["--list-mics"]), 0)
        listing.assert_called_once()
        run.assert_not_called()


class EvaluationTests(unittest.TestCase):
    def test_non_16khz_stereo_wav_is_resampled_with_installed_pyav(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stereo.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(np.zeros(4800 * 2, dtype="<i2").tobytes())
            samples = evaluation.load_evaluation_audio(path)
        self.assertEqual(len(samples), 1600)
        self.assertEqual(samples.dtype, np.float32)

    def test_missing_audio_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "bad.jsonl"
            manifest.write_text(json.dumps({"audio": "missing.wav", "text": "멈춰"}), encoding="utf-8")
            with patch.object(speech_pipeline, "create_speech_transcriber") as create:
                with self.assertRaises(ValueError):
                    evaluation.compare_transcription_settings(manifest, options=quality.SpeechOptions(), models=["small"], beams=[3])
            create.assert_not_called()

    def test_character_metrics_and_number_unit_equivalence(self):
        self.assertEqual(evaluation.edit_distance("정지", "전진"), 2)
        self.assertEqual(evaluation.normalized_text("정지!"), "정지")
        self.assertEqual(evaluation.command_fields("앞으로 삼십 센티미터 이동해")["distance_cm"], 30)
        self.assertEqual(evaluation.command_fields("뒤로 0.3m 이동해")["distance_cm"], 30)
        self.assertEqual(evaluation.command_fields("왼쪽으로 구십 도 돌아")["angle_deg"], 90)
        self.assertTrue(evaluation.command_fields("멈추지 마")["negated"])

    def test_same_file_is_compared_and_report_contains_metrics(self):
        fake = Mock()
        fake.transcribe_audio.return_value = quality.TranscriptionResult("앞으로 30cm 이동해", True, "", {})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wav = root / "sample.wav"
            wav.write_bytes(quality.encode_wav(audio(0.4)))
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"audio": "sample.wav", "text": "앞으로 30cm 이동해",
                                           "fields": {"direction": "forward", "distance_cm": 30}}, ensure_ascii=False), encoding="utf-8")
            with patch.object(speech_pipeline, "create_speech_transcriber", return_value=fake), \
                    patch.object(evaluation, "SileroProbability", return_value=lambda frame: float(np.max(np.abs(frame)) > 0)), \
                    contextlib.redirect_stdout(io.StringIO()):
                report_path = evaluation.compare_transcription_settings(
                    manifest, options=quality.SpeechOptions(), models=["small"], beams=[1, 3])
            self.addCleanup(report_path.unlink, missing_ok=True)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(report["profiles"]), 2)
            self.assertEqual(report["profiles"][0]["field_accuracy"], 1)
            self.assertEqual(report["profiles"][0]["character_error_rate"], 0)
            self.assertEqual(fake.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
