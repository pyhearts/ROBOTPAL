import tempfile
import queue
import threading
import time
import unittest
from pathlib import Path

import numpy as np

import main
from speaker_pipeline import ChannelLinker, PerStreamUtteranceDetector, ProfileStore, RegisteredSpeakerIdentifier, SpeakerProfile, StreamingSeparator, TranscriptionTask, UNKNOWN_SPEAKER, validate_enrollment_audio


class ChannelLinkerTests(unittest.TestCase):
    def test_swapped_model_outputs_keep_internal_streams(self):
        overlap = 100
        linker = ChannelLinker(overlap)
        rng = np.random.default_rng(7)
        first_a = rng.normal(size=300).astype(np.float32)
        first_b = rng.normal(size=300).astype(np.float32)
        linker.link(np.stack([first_a, first_b]))
        second_a = np.concatenate([first_a[-overlap:], rng.normal(size=200)]).astype(np.float32)
        second_b = np.concatenate([first_b[-overlap:], rng.normal(size=200)]).astype(np.float32)
        linked = linker.link(np.stack([second_b, second_a]))
        self.assertGreater(np.corrcoef(first_a[-overlap:], linked[0, :overlap])[0, 1], 0.99)
        self.assertGreater(np.corrcoef(first_b[-overlap:], linked[1, :overlap])[0, 1], 0.99)


class StreamingSeparatorTests(unittest.TestCase):
    def test_overlap_add_preserves_full_timeline(self):
        def separate(window):
            return np.stack([window, -window])

        separator = StreamingSeparator(separate)
        audio = np.random.default_rng(3).normal(0, 0.1, 7 * 16_000).astype(np.float32)
        outputs = separator.push(audio[: 4 * 16_000])
        outputs += separator.push(audio[4 * 16_000 :])
        outputs += separator.flush()
        self.assertEqual([position for position, _ in outputs], [0, 48_000, 96_000])
        self.assertEqual(sum(streams.shape[1] for _, streams in outputs), len(audio))

    def test_model_failure_retries_once(self):
        calls = 0

        def separate(window):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary")
            return np.stack([window, -window])

        outputs = StreamingSeparator(separate).push(np.ones(4 * 16_000, dtype=np.float32))
        self.assertEqual(calls, 2)
        self.assertEqual(len(outputs), 1)


class SpeakerIdentificationTests(unittest.TestCase):
    def setUp(self):
        self.profiles = [SpeakerProfile("민수", np.array([1.0, 0.0], dtype=np.float32)), SpeakerProfile("지영", np.array([0.0, 1.0], dtype=np.float32))]
        self.identifier = RegisteredSpeakerIdentifier(self.profiles, lambda audio: audio)

    def test_registered_and_unknown_speakers(self):
        self.assertEqual(self.identifier.identify_embedding(np.array([0.9, 0.1])), "민수")
        self.assertEqual(self.identifier.identify_embedding(np.array([0.1, 0.9])), "지영")
        self.assertEqual(self.identifier.identify_embedding(np.array([-1.0, -1.0])), UNKNOWN_SPEAKER)
        self.assertEqual(self.identifier.identify_embedding(np.array([1.0, 1.0])), UNKNOWN_SPEAKER)

    def test_threshold_and_margin_boundaries_are_inclusive(self):
        profiles = [SpeakerProfile("A", np.array([1.0, 0.0, 0.0])), SpeakerProfile("B", np.array([0.0, 1.0, 0.0]))]
        identifier = RegisteredSpeakerIdentifier(profiles, lambda audio: audio)
        vector = np.array([0.45, 0.37, np.sqrt(1 - 0.45**2 - 0.37**2)])
        self.assertEqual(identifier.identify_embedding(vector), "A")

    def test_profile_round_trip_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            ProfileStore.save(path, self.profiles, "model", "1")
            self.assertEqual([p.name for p in ProfileStore.load(path, "model")], ["민수", "지영"])
            path.write_text("not-json", encoding="utf-8")
            with self.assertRaises(ValueError):
                ProfileStore.load(path, "model")


class EnrollmentValidationTests(unittest.TestCase):
    def test_requires_six_seconds_and_rejects_clipping(self):
        self.assertTrue(validate_enrollment_audio(np.full(8 * 16_000, 0.1, dtype=np.float32))[0])
        self.assertFalse(validate_enrollment_audio(np.full(5 * 16_000, 0.1, dtype=np.float32))[0])
        self.assertFalse(validate_enrollment_audio(np.ones(8 * 16_000, dtype=np.float32))[0])


class UtteranceDetectorTests(unittest.TestCase):
    def _stereo(self, seconds, left=0.0, right=0.0):
        samples = round(seconds * 16_000)
        return np.stack([np.full(samples, left, dtype=np.float32), np.full(samples, right, dtype=np.float32)])

    def test_short_pause_stays_in_one_utterance(self):
        detector = PerStreamUtteranceDetector()
        tasks = detector.process(self._stereo(0.5, left=0.1), 0.0)
        tasks += detector.process(self._stereo(0.5), 0.5)
        tasks += detector.process(self._stereo(0.5, left=0.1), 1.0)
        tasks += detector.process(self._stereo(0.8), 1.5)
        finals = [task for task in tasks if task.final]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0].stream_id, 0)

    def test_each_stream_finishes_once_at_different_times(self):
        detector = PerStreamUtteranceDetector()
        tasks = detector.process(self._stereo(0.6, 0.1, 0.1), 0.0)
        tasks += detector.process(self._stereo(0.8, 0.0, 0.1), 0.6)
        tasks += detector.process(self._stereo(0.8), 1.4)
        finals = [task for task in tasks if task.final]
        self.assertEqual([task.stream_id for task in finals], [0, 1])
        self.assertEqual(len({task.utterance_id for task in finals}), 2)

    def test_long_utterance_is_partial_but_has_one_final(self):
        detector = PerStreamUtteranceDetector()
        tasks = detector.process(self._stereo(20.5, left=0.1), 0.0)
        tasks += detector.process(self._stereo(0.8), 20.5)
        stream_tasks = [task for task in tasks if task.stream_id == 0]
        self.assertEqual(sum(not task.final for task in stream_tasks), 1)
        self.assertEqual(sum(task.final for task in stream_tasks), 1)
        self.assertEqual(len({task.utterance_id for task in stream_tasks}), 1)


class TranscriptionWorkerTests(unittest.TestCase):
    def test_partial_chunks_produce_exactly_one_callback(self):
        class Segment:
            def __init__(self, text):
                self.text = text

        class Whisper:
            def transcribe(self, audio, **kwargs):
                del audio, kwargs
                return [Segment("문장")], None

        class Identifier:
            def identify(self, audio):
                del audio
                return "민수"

        task_queue = queue.Queue()
        calls = []
        worker = threading.Thread(
            target=main._transcription_worker,
            args=(Whisper(), Identifier(), task_queue, lambda speaker, text: calls.append((speaker, text))),
        )
        worker.start()
        utterance_id = "one"
        task_queue.put(TranscriptionTask(utterance_id, 0, np.ones(100), False, time.monotonic()))
        task_queue.put(TranscriptionTask(utterance_id, 0, np.ones(100), True, time.monotonic(), np.ones(100)))
        task_queue.put(main._STOP)
        task_queue.join()
        worker.join()
        self.assertEqual(calls, [("민수", "문장 문장")])


if __name__ == "__main__":
    unittest.main()
