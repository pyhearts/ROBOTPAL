"""Exercise direct mono VAD and microphone lifecycle without real devices."""

import contextlib
import io
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from module import speech


def audio(seconds, level=0.1):
    return np.full(round(seconds * speech.SAMPLE_RATE), level, dtype=np.float32)


class DirectDetectorTests(unittest.TestCase):
    def test_short_command_completes_without_waiting_for_four_seconds(self):
        detector = speech.MonoUtteranceDetector()
        self.assertEqual(detector.process(audio(0.4), 10), [])
        tasks = detector.process(audio(0.8, 0), 10.4)
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0].final)
        self.assertLess(tasks[0].end_time - 10, 1.3)
        self.assertEqual(tasks[0].stream_id, -1)
        self.assertEqual(tasks[0].identity_audio.size, 0)
        self.assertTrue(np.all(tasks[0].audio[:6400] == np.float32(0.1)))
        self.assertEqual(detector.flush(), [])

    def test_silence_and_too_short_sound_do_not_create_commands(self):
        detector = speech.MonoUtteranceDetector()
        self.assertEqual(detector.process(audio(1, 0), 0), [])
        self.assertEqual(detector.process(audio(0.1), 1), [])
        self.assertEqual(detector.process(audio(1, 0), 1.1), [])
        self.assertEqual(detector.flush(), [])

    def test_two_utterances_have_separate_ids(self):
        detector = speech.MonoUtteranceDetector()
        tasks = detector.process(np.concatenate([audio(0.4), audio(0.8, 0),
                                                  audio(0.4), audio(0.8, 0)]), 0)
        self.assertEqual(len(tasks), 2)
        self.assertNotEqual(tasks[0].utterance_id, tasks[1].utterance_id)

    def test_flush_keeps_unfinished_command_and_capture_timestamp(self):
        detector = speech.MonoUtteranceDetector()
        detector.process(audio(0.4), 20)
        tasks = detector.flush()
        self.assertEqual(len(tasks), 1)
        self.assertAlmostEqual(tasks[0].end_time, 20.4)
        self.assertTrue(tasks[0].final)
        self.assertEqual(detector.flush(), [])

    def test_long_utterance_parts_still_deliver_one_callback(self):
        detector = speech.MonoUtteranceDetector()
        tasks = detector.process(np.concatenate([audio(20.2), audio(0.8, 0)]), 0)
        self.assertEqual(len(tasks), 2)
        self.assertFalse(tasks[0].final)
        self.assertTrue(tasks[1].final)
        self.assertEqual(tasks[0].utterance_id, tasks[1].utterance_id)
        jobs, results = queue.Queue(), []
        for task in tasks:
            jobs.put(task)
        jobs.put(speech._STOP)
        with patch.object(speech, "transcribe_utterance", side_effect=["첫 부분", "마지막 부분"]), \
                contextlib.redirect_stderr(io.StringIO()):
            speech._transcription_worker(None, None, jobs, lambda *args: results.append(args))
        self.assertEqual(results, [("음성 입력", "첫 부분 마지막 부분")])


class DirectRuntimeTests(unittest.TestCase):
    def setUp(self):
        microphone = patch.object(speech, "prepare_microphone", return_value={"name": "test"})
        vad = patch.object(speech, "SileroProbability", return_value=lambda frame: float(np.max(np.abs(frame)) > 0.05))
        microphone.start()
        vad.start()
        self.addCleanup(microphone.stop)
        self.addCleanup(vad.stop)

    def test_microphone_audio_reaches_whisper_unchanged_and_drains_on_stop(self):
        # Stop the main loop after the simulated microphone queue is consumed.
        class StopWhenEmpty(queue.Queue):
            def get(self, block=True, timeout=None):
                if block and self.empty():
                    raise KeyboardInterrupt
                return super().get(block, timeout)

        captured, jobs, received, transcribed = StopWhenEmpty(), queue.Queue(), [], []
        samples = np.concatenate([audio(0.4), audio(0.8, 0), audio(0.4, 0.2)])
        stream_settings = {}

        class Microphone:
            def __init__(self, **kwargs):
                stream_settings.update(kwargs)

            def __enter__(self):
                callback = stream_settings["callback"]
                for offset in range(0, len(samples), speech.BLOCK_SIZE):
                    block = samples[offset:offset + speech.BLOCK_SIZE].reshape(-1, 1).copy()
                    callback(block, len(block), None, None)
                    # Verify callback audio owns its copy rather than driver memory.
                    block.fill(-1)
                return self

            def __exit__(self, *args):
                stream_settings["closed"] = True

        def transcribe(model, samples):
            transcribed.append(samples.copy())
            return "명령"

        output = io.StringIO()
        with patch.object(speech, "_load_transcription_dependencies",
                          return_value=(SimpleNamespace(InputStream=Microphone), Mock())), \
                patch.object(speech, "_load_whisper_model", return_value=object()), \
                patch.object(speech, "_load_runtime_dependencies", side_effect=AssertionError("legacy loader")), \
                patch.object(speech, "_load_voice_models", side_effect=AssertionError("SepFormer")), \
                patch.object(speech, "_separation_worker", side_effect=AssertionError("separator worker")), \
                patch.object(speech, "load_speaker_profiles", side_effect=AssertionError("registration")), \
                patch.object(speech.queue, "Queue", side_effect=[captured, jobs]), \
                patch.object(speech, "transcribe_utterance", side_effect=transcribe), \
                contextlib.redirect_stdout(output):
            speech.run_live_transcription(lambda *args: received.append(args))
        self.assertEqual(received, [("음성 입력", "명령"), ("음성 입력", "명령")])
        self.assertEqual(len(transcribed), 2)
        self.assertTrue(np.any(transcribed[0] == np.float32(0.1)))
        self.assertTrue(np.any(transcribed[1] == np.float32(0.2)))
        self.assertFalse(any(np.any(chunk == -1) for chunk in transcribed))
        self.assertTrue(stream_settings["closed"])
        self.assertEqual(stream_settings["channels"], 1)
        self.assertNotIn("CPU 분리 속도", output.getvalue())
        self.assertEqual(jobs.unfinished_tasks, 0)
        self.assertEqual(captured.unfinished_tasks, 0)

    def test_stop_during_open_drains_already_captured_audio(self):
        samples = audio(0.4).reshape(-1, 1)

        def open_and_stop(**kwargs):
            kwargs["callback"](samples, len(samples), None, None)
            raise KeyboardInterrupt

        received = []
        with patch.object(speech, "_load_transcription_dependencies",
                          return_value=(SimpleNamespace(InputStream=open_and_stop), Mock())), \
                patch.object(speech, "_load_whisper_model", return_value=object()), \
                patch.object(speech, "transcribe_utterance", return_value="멈춰"), \
                contextlib.redirect_stdout(io.StringIO()):
            speech.run_live_transcription(lambda *args: received.append(args))
        self.assertEqual(received, [("음성 입력", "멈춰")])

    def test_failed_microphone_open_closes_transcription_worker(self):
        jobs = queue.Queue()
        with patch.object(speech, "_load_transcription_dependencies", return_value=(
                SimpleNamespace(InputStream=Mock(side_effect=OSError("no mic"))), Mock())), \
                patch.object(speech, "_load_whisper_model", return_value=object()), \
                patch.object(speech.queue, "Queue", side_effect=[queue.Queue(), jobs]), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "마이크"):
                speech.run_live_transcription()
        self.assertEqual(jobs.unfinished_tasks, 0)


if __name__ == "__main__":
    unittest.main()
