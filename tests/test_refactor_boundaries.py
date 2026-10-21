"""Refactor contracts: archives, optional imports, compatibility and shutdown."""

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import Mock, patch

from module import cli, legacy_speech, speech, vision_assistant as vision


ROOT = Path(__file__).resolve().parent.parent


class ArchiveTests(unittest.TestCase):
    def test_tts_sources_match_manifest_and_are_outside_active_tree(self):
        archive = ROOT / "archive/tts"
        manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), 22)
        for item in manifest["files"]:
            with self.subTest(path=item["original_path"]):
                target = archive / item["archive_path"]
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), item["sha256"])
                if item["archive_path"].startswith("source/") or item["original_path"] == "TTS_GUIDE.md":
                    self.assertFalse((ROOT / item["original_path"]).exists())

    def test_all_baseline_test_identities_are_retained_or_archived(self):
        archive = ROOT / "archive/tts"
        baseline = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))["baseline_tests"]
        total = archived = 0
        for filename, identities in baseline.items():
            path = ROOT / "tests" / filename
            if not path.exists():
                path = archive / "source/tests" / filename
                archived += len(identities)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            current = {f"{node.name}.{method.name}" for node in tree.body if isinstance(node, ast.ClassDef)
                       for method in node.body if isinstance(method, ast.FunctionDef) and method.name.startswith("test_")}
            self.assertTrue(set(identities) <= current, filename)
            total += len(identities)
        self.assertEqual((total, archived), (119, 31))


class OptionalImportTests(unittest.TestCase):
    def run_isolated(self, body):
        completed = subprocess.run([sys.executable, "-c", textwrap.dedent(body)], cwd=ROOT,
                                   text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_text_entrypoints_need_only_core_dependencies(self):
        self.run_isolated('''
            import importlib.abc
            import sys
            class BlockOptional(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'numpy', 'sounddevice', 'faster_whisper', 'cv2', 'mediapipe', 'speechbrain', 'torch'}:
                        raise AssertionError('Optional import: ' + fullname)
            sys.meta_path.insert(0, BlockOptional())
            import main, ai_action_planner
            from module import cli, session_log
            from module.decision_router import CommandDecision, CommandResult
            from pathlib import Path
            from tempfile import TemporaryDirectory
            from unittest.mock import patch
            cli.process_command = lambda text: CommandResult(CommandDecision('conversation', 1), message='hello')
            with TemporaryDirectory() as folder:
                session_log.LOG_ROOT = Path(folder)
                assert main.main(['--plan', '--command', 'hello']) == 0
                with patch('builtins.input', return_value='hello'):
                    assert ai_action_planner.main() == 0
            assert 'module.speech' not in sys.modules
            assert 'module.gesture' not in sys.modules
            assert 'module.camera' not in sys.modules
        ''')

    def test_direct_speech_does_not_import_legacy_or_tts(self):
        self.run_isolated('''
            import importlib.abc
            import sys
            class BlockLegacy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in {'module.legacy_speech', 'module.local_tts'} or fullname.split('.')[0] in {'torch', 'speechbrain'}:
                        raise AssertionError('Legacy import: ' + fullname)
            sys.meta_path.insert(0, BlockLegacy())
            from module import speech
            from unittest.mock import patch
            with patch.object(speech, '_load_transcription_dependencies', side_effect=RuntimeError('stop before device')):
                try:
                    speech.run_live_transcription()
                except RuntimeError as exc:
                    assert str(exc) == 'stop before device'
                else:
                    raise AssertionError('Expected stop')
            assert 'module.legacy_speech' not in sys.modules
        ''')

    def test_tts_flags_fail_before_dispatch(self):
        flags = ['--tts', '--tts-engine=gpt-sovits', '--tts-future', '--prepare-tts',
                 '--test-tts', '--list-speakers', '--output-device=1']
        for flag in flags:
            with self.subTest(flag=flag), patch.object(cli, "_dispatch") as dispatch, \
                    contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as caught:
                    cli.run([flag])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("archive/tts/RESTORE.md", error.getvalue())
                dispatch.assert_not_called()
        help_text = cli._build_parser().format_help()
        self.assertNotIn("--tts", help_text)
        self.assertNotIn("--output-device", help_text)


class LegacyCompatibilityTests(unittest.TestCase):
    def test_speaker_profiles_round_trip_after_module_move(self):
        import numpy as np
        profiles = [legacy_speech.SpeakerProfile("first", np.array([1, 0], dtype=np.float32)),
                    legacy_speech.SpeakerProfile("second", np.array([0, 1], dtype=np.float32))]
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "profiles.json"
            legacy_speech.ProfileStore.save(path, profiles, "model", "version")
            loaded = legacy_speech.ProfileStore.load(path, "model")
            self.assertEqual([p.name for p in loaded], ["first", "second"])
            for expected, actual in zip(profiles, loaded):
                np.testing.assert_array_equal(expected.embedding, actual.embedding)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_public_functions_delegate_with_original_arguments(self):
        callback = Mock()
        with patch.object(legacy_speech, "enroll_speakers") as enroll, \
                patch.object(legacy_speech, "load_speaker_profiles", return_value=["profile"]) as profiles, \
                patch.object(legacy_speech, "run_separated_transcription") as run:
            speech.enroll_speakers()
            self.assertEqual(speech.load_speaker_profiles(), ["profile"])
            speech.run_separated_transcription(callback, offline=True, identify_speakers=False)
            run.assert_called_once_with(callback, offline=True, identify_speakers=False)
            run.reset_mock()
            speech.run_live_transcription(callback, offline=True, identify_speakers=True)
            run.assert_called_once_with(callback, offline=True, identify_speakers=True)
        enroll.assert_called_once_with()
        profiles.assert_called_once_with()

    def test_legacy_runtime_starts_and_drains_without_live_device(self):
        import numpy as np
        device = Mock()
        device.InputStream.return_value.__enter__ = Mock(side_effect=KeyboardInterrupt)
        device.InputStream.return_value.__exit__ = Mock(return_value=False)
        separator = Mock(return_value=np.zeros((2, 64000), dtype=np.float32))
        with patch.object(legacy_speech, "_load_runtime_dependencies", return_value=(device, Mock(), Mock(), Mock(), Mock(), Mock())), \
                patch.object(legacy_speech, "_load_voice_models", return_value=(None, separator)), \
                patch.object(legacy_speech, "_load_whisper_model", return_value=Mock()), \
                patch.object(legacy_speech, "load_speaker_profiles") as profiles, \
                contextlib.redirect_stdout(io.StringIO()):
            legacy_speech.run_separated_transcription()
        separator.assert_called_once()
        profiles.assert_not_called()


class ShutdownTests(unittest.TestCase):
    def test_transcription_worker_start_failure_closes_model(self):
        recognizer = Mock()
        with patch.object(speech, "_load_transcription_dependencies", return_value=(Mock(), Mock())), \
                patch.object(speech, "prepare_microphone"), \
                patch.object(speech, "SileroProbability"), \
                patch.object(speech, "create_speech_transcriber", return_value=recognizer), \
                patch.object(speech.threading, "Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("cannot start worker")
            with self.assertRaisesRegex(RuntimeError, "cannot start worker"):
                speech.run_live_transcription()
        recognizer.close.assert_called_once_with()

    def run_with_resources(self, fail_at=None, camera_failure=False, close_failure=False):
        events = []
        client, camera, assistant = Mock(), Mock(), Mock()
        def operation(name, fail=False):
            def invoke(*args, **kwargs):
                events.append(name)
                if fail:
                    raise RuntimeError(name)
            return invoke
        client.close.side_effect = operation("client.close")
        camera.start.side_effect = operation("camera.start", camera_failure)
        camera.close.side_effect = operation("camera.close", close_failure)
        assistant.start.side_effect = operation("assistant.start", fail_at == "assistant.start")
        assistant.close.side_effect = operation("assistant.close")
        def camera_factory(*args):
            if fail_at == "camera.construct":
                raise RuntimeError("camera.construct")
            return camera
        def assistant_factory(*args, **kwargs):
            if fail_at == "assistant.construct":
                raise RuntimeError("assistant.construct")
            return assistant
        with patch.object(vision, "create_api_client", return_value=client), \
                patch.object(vision, "LatestFrameCamera", side_effect=camera_factory), \
                patch.object(vision, "VoiceVisionAssistant", side_effect=assistant_factory), \
                patch.object(vision, "run_live_transcription", side_effect=operation("speech.drained", fail_at == "speech")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if fail_at or close_failure:
                with self.assertRaises(RuntimeError):
                    vision.run_voice_vision_assistant()
            else:
                vision.run_voice_vision_assistant()
        return events

    def test_shutdown_order_after_final_transcription(self):
        self.assertEqual(self.run_with_resources(), ["camera.start", "assistant.start", "speech.drained",
                                                    "camera.close", "assistant.close", "client.close"])

    def test_camera_start_failure_still_allows_text_processing(self):
        self.assertIn("speech.drained", self.run_with_resources(camera_failure=True))

    def test_partial_initialization_and_close_failure_release_resources(self):
        expected = {
            "camera.construct": ["client.close"],
            "assistant.construct": ["camera.close", "client.close"],
            "assistant.start": ["camera.close", "assistant.close", "client.close"],
            "speech": ["camera.close", "assistant.close", "client.close"],
        }
        for fail_at, suffix in expected.items():
            with self.subTest(fail_at=fail_at):
                events = self.run_with_resources(fail_at)
                self.assertEqual(events[-len(suffix):], suffix)
        self.assertEqual(self.run_with_resources(close_failure=True)[-3:],
                         ["camera.close", "assistant.close", "client.close"])


class GestureCompatibilityTests(unittest.TestCase):
    def test_number_preview_draws_finger_states(self):
        import numpy as np
        from module import gesture
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        cv2 = Mock()
        cv2.flip.return_value = frame
        snapshot = gesture.PreviewSnapshot(None, (), "tracking", "numbers", ("open",) * 5)
        self.assertIs(gesture._draw_preview(cv2, frame, snapshot), frame)
        self.assertEqual(cv2.putText.call_count, 2)
        self.assertIn("T:O", cv2.putText.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
