import types
import io
import tempfile
import unittest
from unittest import mock

import numpy as np

import gesture


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
        with mock.patch("gesture.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "5회 연속"):
                gesture._camera_loop(
                    capture, object(), gesture._ResultState(), self.Cv2, self.Mp,
                    lambda *args: None, False,
                )


class ModelTests(unittest.TestCase):
    def test_existing_model_is_reused(self):
        with mock.patch.object(gesture.Path, "is_file", return_value=True), mock.patch.object(
            gesture.Path, "stat", return_value=types.SimpleNamespace(st_size=10)
        ), mock.patch("gesture.urllib.request.urlopen") as urlopen:
            self.assertEqual(gesture.ensure_gesture_model(gesture.Path("model.task")), gesture.Path("model.task"))
            urlopen.assert_not_called()

    def test_download_is_written_to_final_path(self):
        response = io.BytesIO(b"model-data")
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "gesture.urllib.request.urlopen", return_value=response
        ):
            path = gesture.Path(directory) / "gesture.task"
            self.assertEqual(gesture.ensure_gesture_model(path), path)
            self.assertEqual(path.read_bytes(), b"model-data")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_failed_download_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "gesture.urllib.request.urlopen", side_effect=OSError("offline")
        ):
            path = gesture.Path(directory) / "gesture.task"
            with self.assertRaisesRegex(RuntimeError, "직접 저장"):
                gesture.ensure_gesture_model(path)
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
