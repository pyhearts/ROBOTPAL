"""Consolidated regression tests for the modular ROBOTPAL application."""

import tempfile
import queue
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from module import speech
from module.speech import ChannelLinker, PerStreamUtteranceDetector, ProfileStore, RegisteredSpeakerIdentifier, SpeakerProfile, StreamingSeparator, TranscriptionTask, UNKNOWN_SPEAKER, validate_enrollment_audio


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
            target=speech._transcription_worker,
            args=(Whisper(), Identifier(), task_queue, lambda speaker, text: calls.append((speaker, text))),
        )
        worker.start()
        utterance_id = "one"
        task_queue.put(TranscriptionTask(utterance_id, 0, np.ones(100), False, time.monotonic()))
        task_queue.put(TranscriptionTask(utterance_id, 0, np.ones(100), True, time.monotonic(), np.ones(100)))
        task_queue.put(speech._STOP)
        task_queue.join()
        worker.join()
        self.assertEqual(calls, [("민수", "문장 문장")])


import types
import io
import tempfile
import unittest
from unittest import mock

import numpy as np

from module import gesture


def category(name, score):
    return types.SimpleNamespace(category_name=name, score=score)


def result(name=None, score=0.0, hand="Right", landmarks=()):
    gestures = [] if name is None else [[category(name, score)]]
    handedness = [] if name is None else [[category(hand, 1.0)]]
    return types.SimpleNamespace(
        gestures=gestures,
        handedness=handedness,
        hand_landmarks=[landmarks] if landmarks else [],
        hand_world_landmarks=[],
    )


class ObservationTests(unittest.TestCase):
    def test_all_seven_gestures_and_inclusive_threshold(self):
        for name in gesture.GESTURES:
            with self.subTest(name=name):
                observation = gesture.extract_observation(result(name, 0.7))
                self.assertEqual(observation, gesture.GestureObservation("Right", name, 0.7))

    def test_none_empty_unknown_and_low_score_are_ignored(self):
        self.assertIsNone(gesture.extract_observation(result()))
        self.assertIsNone(gesture.extract_observation(result("None", 0.99)))
        self.assertIsNone(gesture.extract_observation(result("Wave", 0.99)))
        self.assertIsNone(gesture.extract_observation(result("Thumb_Up", 0.699)))

    def test_strongest_hand_is_selected(self):
        value = types.SimpleNamespace(
            gestures=[[category("Victory", 0.8)], [category("Thumb_Up", 0.9)]],
            handedness=[[category("Left", 1.0)], [category("Right", 1.0)]],
            hand_landmarks=[],
        )
        self.assertEqual(gesture.extract_observation(value).gesture, "Thumb_Up")


class StabilizerTests(unittest.TestCase):
    def setUp(self):
        self.stabilizer = gesture.GestureStabilizer()
        self.thumb = gesture.GestureObservation("Right", "Thumb_Up", 0.9)
        self.victory = gesture.GestureObservation("Right", "Victory", 0.8)

    def test_emits_once_after_400ms(self):
        self.assertIsNone(self.stabilizer.update(0, self.thumb))
        self.assertIsNone(self.stabilizer.update(399, self.thumb))
        self.assertEqual(self.stabilizer.update(400, self.thumb), self.thumb)
        self.assertIsNone(self.stabilizer.update(900, self.thumb))

    def test_short_loss_does_not_rearm_but_300ms_loss_does(self):
        self.stabilizer.update(0, self.thumb)
        self.stabilizer.update(400, self.thumb)
        self.stabilizer.update(500, None)
        self.stabilizer.update(799, None)
        self.stabilizer.update(800, self.thumb)
        self.assertIsNone(self.stabilizer.update(1200, self.thumb))
        self.stabilizer.update(1300, None)
        self.stabilizer.update(1600, None)
        self.stabilizer.update(1700, self.thumb)
        self.assertEqual(self.stabilizer.update(2100, self.thumb), self.thumb)

    def test_new_pose_can_emit_without_release(self):
        self.stabilizer.update(0, self.thumb)
        self.stabilizer.update(400, self.thumb)
        self.stabilizer.update(500, self.victory)
        self.assertEqual(self.stabilizer.update(900, self.victory), self.victory)

    def test_backward_timestamp_is_rejected(self):
        self.stabilizer.update(10, self.thumb)
        with self.assertRaises(ValueError):
            self.stabilizer.update(9, self.thumb)


class ResultAndCallbackTests(unittest.TestCase):
    def test_result_callback_stabilizes_and_queues_one_event(self):
        state = gesture._ResultState()
        state.receive(result("Open_Palm", 0.8), None, 0)
        state.receive(result("Open_Palm", 0.9), None, 400)
        state.receive(result("Open_Palm", 0.9), None, 800)
        calls = []
        gesture._drain_events(state, lambda *args: calls.append(args))
        self.assertEqual(calls, [("Right", "Open_Palm", 0.9)])

    def test_user_callback_exception_is_contained(self):
        state = gesture._ResultState()
        state.events.put(gesture.GestureObservation("Left", "Victory", 0.9))
        with mock.patch("sys.stderr"):
            gesture._drain_events(state, lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertTrue(state.events.empty())


class CameraLoopTests(unittest.TestCase):
    class Cv2:
        COLOR_BGR2RGB = 1

        @staticmethod
        def cvtColor(frame, code):
            assert code == 1
            return frame[:, :, ::-1]

    class Mp:
        class ImageFormat:
            SRGB = "srgb"

        class Image:
            def __init__(self, image_format, data):
                self.image_format = image_format
                self.data = data

    def test_frame_is_converted_and_submitted(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)

        class Capture:
            def __init__(self):
                self.calls = 0

            def read(self):
                self.calls += 1
                if self.calls == 1:
                    return True, frame
                raise KeyboardInterrupt

        timestamps = []
        recognizer = types.SimpleNamespace(
            recognize_async=lambda image, timestamp: timestamps.append(timestamp)
        )
        with self.assertRaises(KeyboardInterrupt):
            gesture._camera_loop(
                Capture(), recognizer, gesture._ResultState(), self.Cv2, self.Mp,
                lambda *args: None, False,
            )
        self.assertEqual(len(timestamps), 1)

    def test_five_camera_failures_raise_clear_error(self):
        capture = types.SimpleNamespace(read=lambda: (False, None))
        with mock.patch("module.gesture.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "5회 연속"):
                gesture._camera_loop(
                    capture, object(), gesture._ResultState(), self.Cv2, self.Mp,
                    lambda *args: None, False,
                )


class ModelTests(unittest.TestCase):
    def test_existing_model_is_reused(self):
        with mock.patch.object(gesture.Path, "is_file", return_value=True), mock.patch.object(
            gesture.Path, "stat", return_value=types.SimpleNamespace(st_size=10)
        ), mock.patch("module.gesture.urllib.request.urlopen") as urlopen:
            self.assertEqual(gesture.ensure_gesture_model(gesture.Path("model.task")), gesture.Path("model.task"))
            urlopen.assert_not_called()

    def test_download_is_written_to_final_path(self):
        response = io.BytesIO(b"model-data")
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "module.gesture.urllib.request.urlopen", return_value=response
        ):
            path = gesture.Path(directory) / "gesture.task"
            self.assertEqual(gesture.ensure_gesture_model(path), path)
            self.assertEqual(path.read_bytes(), b"model-data")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_failed_download_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "module.gesture.urllib.request.urlopen", side_effect=OSError("offline")
        ):
            path = gesture.Path(directory) / "gesture.task"
            with self.assertRaisesRegex(RuntimeError, "직접 저장"):
                gesture.ensure_gesture_model(path)
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()), [])


import math
import types
import unittest

from module import gesture
from module.gesture_rules import EXTENDED, FOLDED, UNCERTAIN, classify_landmarks


def landmark(point):
    return types.SimpleNamespace(x=point[0], y=point[1], z=point[2])


def hand_pose(extended=(), ok=False):
    """Create a synthetic but anatomically indexed 21-point hand."""
    points = [(0.0, 0.0, 0.0) for _ in range(21)]
    points[0] = (0.0, 0.0, 0.0)

    if "thumb" in extended:
        points[1:5] = [
            (-0.3, 0.3, 0.0),
            (-0.6, 0.6, 0.0),
            (-0.9, 0.9, 0.0),
            (-1.2, 1.2, 0.0),
        ]
    else:
        points[1:5] = [
            (-0.3, 0.3, 0.0),
            (-0.5, 0.5, 0.0),
            (-0.4, 0.65, 0.0),
            (-0.1, 0.5, 0.0),
        ]

    fingers = {
        "index": (5, (-0.35, 0.8, 0.0)),
        "middle": (9, (0.0, 1.0, 0.0)),
        "ring": (13, (0.32, 0.85, 0.0)),
        "pinky": (17, (0.58, 0.7, 0.0)),
    }
    for name, (mcp, base) in fingers.items():
        x, y, z = base
        points[mcp] = base
        if name in extended:
            points[mcp + 1] = (x, y + 0.45, z)
            points[mcp + 2] = (x, y + 0.85, z)
            points[mcp + 3] = (x, y + 1.2, z)
        else:
            points[mcp + 1] = (x, y + 0.4, z)
            points[mcp + 2] = (x + 0.3, y + 0.2, z)
            points[mcp + 3] = (x + 0.45, y, z)

    if ok:
        # Touch thumb and index tips without changing the three open fingers.
        points[4] = (-0.25, 1.15, 0.0)
        points[8] = (-0.25, 1.15, 0.0)
    return [landmark(point) for point in points]


def transform(landmarks, scale=1.0, angle_degrees=0.0, mirror=False):
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    result = []
    for item in landmarks:
        x = -item.x if mirror else item.x
        y = item.y
        result.append(
            landmark(
                (
                    scale * (x * cosine - y * sine),
                    scale * (x * sine + y * cosine),
                    scale * item.z,
                )
            )
        )
    return result


class RuleClassifierTests(unittest.TestCase):
    CASES = {
        "Number_1": ("index",),
        "Number_2": ("index", "middle"),
        "Number_3": ("index", "middle", "ring"),
        "Number_4": ("index", "middle", "ring", "pinky"),
        "Number_5": ("thumb", "index", "middle", "ring", "pinky"),
    }

    def test_numbers_one_through_five(self):
        for expected, fingers in self.CASES.items():
            with self.subTest(expected=expected):
                result = classify_landmarks(hand_pose(fingers))
                self.assertIsNotNone(result)
                self.assertEqual(result.gesture, expected)
                self.assertEqual(result.confidence, 1.0)

    def test_both_number_three_forms(self):
        regular = classify_landmarks(hand_pose(("index", "middle", "ring")))
        alternate = classify_landmarks(hand_pose(("thumb", "index", "middle")))
        self.assertEqual(regular.gesture, "Number_3")
        self.assertEqual(alternate.gesture, "Number_3")

    def test_ok_has_priority(self):
        value = classify_landmarks(hand_pose(("middle", "ring", "pinky"), ok=True))
        self.assertEqual(value.gesture, "OK")

    def test_ok_distance_boundary_is_inclusive(self):
        pose = hand_pose(("middle", "ring", "pinky"))
        # Palm scale is exactly 1.0 in this fixture.
        pose[4] = landmark((-0.25, 1.15, 0.0))
        pose[8] = landmark((0.0, 1.15, 0.0))
        self.assertEqual(classify_landmarks(pose).gesture, "OK")
        pose[8] = landmark((0.001, 1.15, 0.0))
        self.assertNotEqual(classify_landmarks(pose).gesture, "OK")

    def test_rotation_scale_and_mirroring_do_not_change_result(self):
        original = hand_pose(("index", "middle"))
        for scale, angle, mirror in ((0.5, 70, False), (2.4, -125, True), (1.3, 180, False)):
            with self.subTest(scale=scale, angle=angle, mirror=mirror):
                value = classify_landmarks(transform(original, scale, angle, mirror))
                self.assertEqual(value.gesture, "Number_2")

    def test_invalid_and_zero_size_hands_are_rejected(self):
        self.assertIsNone(classify_landmarks(hand_pose(("index",))[:20]))
        zero = [landmark((0.0, 0.0, 0.0)) for _ in range(21)]
        self.assertIsNone(classify_landmarks(zero))

    def test_ambiguous_finger_blocks_number(self):
        pose = hand_pose(("index",))
        # Straight angles, but a radial ratio between folded and extended.
        pose[8] = landmark((pose[6].x, pose[6].y * 1.05, 0.0))
        value = classify_landmarks(pose)
        self.assertIsNotNone(value)
        self.assertIsNone(value.gesture)
        self.assertIn(UNCERTAIN, value.finger_states)


class RuleIntegrationTests(unittest.TestCase):
    @staticmethod
    def media_result(canned, landmarks, world_landmarks=None):
        category = types.SimpleNamespace(category_name=canned, score=0.95)
        hand = types.SimpleNamespace(category_name="Right", score=1.0)
        return types.SimpleNamespace(
            gestures=[[category]],
            handedness=[[hand]],
            hand_landmarks=[landmarks],
            hand_world_landmarks=[] if world_landmarks is None else [world_landmarks],
        )

    def test_basic_mode_is_unchanged(self):
        value = self.media_result("Victory", hand_pose(("index", "middle")))
        self.assertEqual(gesture.extract_observation(value, rule_set="basic").gesture, "Victory")

    def test_numbers_mode_prioritizes_landmark_rule(self):
        value = self.media_result("Victory", hand_pose(("index", "middle")))
        observation = gesture.extract_observation(value, rule_set="numbers")
        self.assertEqual(observation, gesture.GestureObservation("Right", "Number_2", 1.0))

    def test_numbers_mode_suppresses_overlapping_canned_fallback(self):
        ambiguous = hand_pose(("index",))
        ambiguous[8] = landmark((ambiguous[6].x, ambiguous[6].y * 1.05, 0.0))
        value = self.media_result("Pointing_Up", ambiguous)
        self.assertIsNone(gesture.extract_observation(value, rule_set="numbers"))

    def test_numbers_mode_keeps_non_overlapping_fallback(self):
        value = self.media_result("Thumb_Up", [])
        self.assertEqual(
            gesture.extract_observation(value, rule_set="numbers").gesture,
            "Thumb_Up",
        )

    def test_world_landmarks_take_priority(self):
        normalized = hand_pose(("index",))
        world = hand_pose(("index", "middle", "ring", "pinky"))
        value = self.media_result("None", normalized, world)
        self.assertEqual(
            gesture.extract_observation(value, rule_set="numbers").gesture,
            "Number_4",
        )

    def test_rule_result_uses_existing_stabilizer(self):
        value = self.media_result("Victory", hand_pose(("index", "middle")))
        state = gesture._ResultState("numbers")
        state.receive(value, None, 0)
        state.receive(value, None, 399)
        self.assertTrue(state.events.empty())
        state.receive(value, None, 400)
        event = state.events.get_nowait()
        self.assertEqual(event.gesture, "Number_2")


class MainInterfaceTests(unittest.TestCase):
    def test_main_reexports_public_callbacks_and_runners(self):
        import main as app_main
        from module import gesture as gesture_module
        from module import speech as speech_module

        self.assertIs(app_main.handle_transcription, speech_module.handle_transcription)
        self.assertIs(app_main.run_live_transcription, speech_module.run_live_transcription)
        self.assertIs(app_main.handle_gesture, gesture_module.handle_gesture)
        self.assertIs(app_main.run_gesture_recognition, gesture_module.run_gesture_recognition)

    def test_default_command_dispatches_to_speech(self):
        import main as app_main

        with mock.patch.object(app_main, "run_live_transcription") as run:
            self.assertEqual(app_main.main([]), 0)
        run.assert_called_once_with()

    def test_enroll_command_dispatches_to_enrollment(self):
        import main as app_main

        with mock.patch.object(app_main, "enroll_speakers") as enroll:
            self.assertEqual(app_main.main(["--enroll"]), 0)
        enroll.assert_called_once_with()

    def test_gesture_options_are_forwarded(self):
        import main as app_main

        with mock.patch.object(app_main, "run_gesture_recognition") as run:
            result_code = app_main.main(
                ["--gesture", "--preview", "--camera", "1", "--rule-set", "numbers"]
            )
        self.assertEqual(result_code, 0)
        run.assert_called_once_with(camera_index=1, preview=True, rule_set="numbers")

    def test_camera_options_require_gesture_mode(self):
        import main as app_main

        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                app_main.main(["--camera", "1"])

    def test_mutually_exclusive_modes_and_negative_camera_are_rejected(self):
        import main as app_main

        invalid_commands = (
            ["--enroll", "--gesture"],
            ["--gesture", "--camera", "-1"],
        )
        for command in invalid_commands:
            with self.subTest(command=command):
                with mock.patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit):
                        app_main.main(command)


if __name__ == "__main__":
    unittest.main()
