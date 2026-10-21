"""Image persistence and command linkage without camera or API access."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from module import session_log, vision_assistant as vision
from module.action_planner import ActionPlan
from module.decision_router import CommandDecision


class ImageLogTests(unittest.TestCase):
    def run_assistant(self, camera, classifier, planner):
        assistant = vision.VoiceVisionAssistant(camera, client=Mock(),
                                               classifier=classifier, planner=planner)
        assistant.start()
        try:
            assistant.on_transcription("음성 입력", "앞으로 이동해")
            assistant.on_transcription("음성 입력", "안녕")
        finally:
            assistant.close()

    def test_saved_images_match_selected_bytes_and_link_to_results(self):
        camera = Mock()
        camera.snapshot.side_effect = [vision.CameraSnapshot(b"jpeg-one", 12.0),
                                       vision.CameraSnapshot(b"jpeg-two", 13.0)]
        classifier = Mock(side_effect=[CommandDecision("concrete_command", 1),
                                       CommandDecision("not_command", 1)])
        planner = Mock(return_value=ActionPlan(status="plan", summary="이동",
                                              steps=["이동한다."], error=""))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with session_log.capture_output(Path(folder)) as log:
                self.run_assistant(camera, classifier, planner)
            events = [json.loads(line) for line in log.events_path.read_text(encoding="utf-8").splitlines()]
            commands = [e for e in events if e["event"] == "voice_command_received"]
            images = [e for e in events if e["event"] == "command_image_saved"]
            results = [e for e in events if e["event"] == "command_result"]
            self.assertEqual(len(images), 2)
            self.assertEqual(len(results), 2)
            self.assertNotEqual(commands[0]["image_path"], commands[1]["image_path"])
            for command, image, result, expected in zip(commands, images, results, [b"jpeg-one", b"jpeg-two"]):
                self.assertEqual((Path(folder) / command["image_path"]).read_bytes(), expected)
                self.assertEqual(command["command_id"], image["command_id"])
                self.assertEqual(command["command_id"], result["command_id"])
            self.assertEqual(planner.call_args.kwargs["image"], b"jpeg-one")
            planner.assert_called_once()

    def test_disk_failure_does_not_drop_commands(self):
        camera = Mock()
        camera.snapshot.return_value = vision.CameraSnapshot(b"jpeg", 12.0)
        classifier = Mock(return_value=CommandDecision("not_command", 1))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with session_log.capture_output(Path(folder)) as log, \
                    patch.object(session_log.SessionLog, "save_image", side_effect=OSError("disk full")):
                self.run_assistant(camera, classifier, Mock())
            events = [json.loads(line) for line in log.events_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(e["event"] == "command_image_save_error" for e in events), 2)
            self.assertEqual(sum(e["event"] == "command_result" for e in events), 2)
            self.assertEqual(classifier.call_count, 2)

    def test_missing_camera_creates_no_image_files(self):
        camera = Mock()
        camera.snapshot.side_effect = RuntimeError("no frame")
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with session_log.capture_output(Path(folder)) as log:
                self.run_assistant(camera, Mock(return_value=CommandDecision("not_command", 1)), Mock())
            self.assertFalse(log.images_path.exists())
            events = [json.loads(line) for line in log.events_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(e["image_path"] is None for e in events if e["event"] == "voice_command_received"))


if __name__ == "__main__":
    unittest.main()
