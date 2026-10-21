"""Offline regression checks: no API charges, microphones, or cameras."""

import base64
import contextlib
import io
import json
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx2 as httpx
import numpy as np
from openai import OpenAI

import main
from module import action_planner as planner
from module import speech
from module import vision_assistant as vision
from module.decision_router import CommandDecision


PLAN = {"status": "plan", "summary": "정지", "steps": ["정지한다."], "error": ""}


def api_response(text=None, *, status="completed", refusal=False):
    part = (
        {"type": "refusal", "refusal": "refused"}
        if refusal else
        {"type": "output_text", "text": text if text is not None else json.dumps(PLAN), "annotations": []}
    )
    return {
        "id": "resp_test", "object": "response", "created_at": 0,
        "model": planner.DEFAULT_MODEL, "status": status,
        "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                    "status": "completed", "content": [part]}],
    }


class PlannerTests(unittest.TestCase):
    def make_client(self, response=None, status=200):
        self.requests = []

        def respond(request):
            self.requests.append(request)
            return httpx.Response(status, json=response or api_response())

        client = OpenAI(api_key="test-not-a-real-key", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.addCleanup(client.close)
        return client

    def test_real_sdk_text_request_and_response(self):
        client = self.make_client()
        result = planner.create_action_plan("  멈춰  ", client=client)
        self.assertEqual(result.steps, ["정지한다."])
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://api.openai.com/v1/responses")
        body = json.loads(request.content)
        self.assertEqual(body["input"][0]["content"], [{"type": "input_text", "text": "멈춰"}])
        self.assertFalse(body["store"])
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertFalse(body["text"]["format"]["schema"]["additionalProperties"])
        self.assertFalse(client.is_closed())

    def test_jpeg_round_trip_and_model_selection(self):
        client = self.make_client()
        jpeg = b"\xff\xd8\xff\xd9"
        with patch.dict("os.environ", {"OPENAI_MODEL": "configured-model"}):
            planner.create_action_plan("멈춰", image=jpeg, client=client)
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["model"], "configured-model")
        image = body["input"][0]["content"][1]
        self.assertEqual(image["type"], "input_image")
        prefix, encoded = image["image_url"].split(",", 1)
        self.assertEqual(prefix, "data:image/jpeg;base64")
        self.assertEqual(base64.b64decode(encoded), jpeg)

    def test_invalid_command_response(self):
        invalid = {"status": "invalid", "summary": "", "steps": [], "error": "거리가 필요합니다."}
        client = self.make_client(api_response(json.dumps(invalid)))
        self.assertEqual(planner.create_action_plan("앞으로 가", client=client).status, "invalid")

    def test_refusal(self):
        client = self.make_client(api_response(refusal=True))
        self.assertEqual(planner.create_action_plan("명령", client=client).status, "invalid")

    def test_malformed_inconsistent_and_incomplete_responses(self):
        invalid_plan = {**PLAN, "error": "unexpected"}
        for response in [api_response("not json"), api_response(json.dumps(invalid_plan)),
                         api_response(status="incomplete"), api_response("")]:
            with self.subTest(response=response):
                client = self.make_client(response)
                with self.assertRaises(planner.PlannerResponseError):
                    planner.create_action_plan("명령", client=client)

    def test_auth_and_limit_errors_do_not_echo_server_body(self):
        for status, message in [(401, "인증"), (429, "한도")]:
            with self.subTest(status=status):
                client = self.make_client({"error": {"message": "secret-value", "type": "test"}}, status)
                with self.assertRaises(planner.PlannerConnectionError) as error:
                    planner.create_action_plan("명령", client=client)
                self.assertIn(message, str(error.exception))
                self.assertNotIn("secret-value", str(error.exception))
                self.assertEqual(len(self.requests), 1)

    def test_empty_input_needs_no_key_and_missing_key_is_clear(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            self.assertEqual(planner.create_action_plan(" ").status, "invalid")
            with self.assertRaisesRegex(planner.PlannerConnectionError, "OPENAI_API_KEY"):
                planner.create_action_plan("멈춰")

    def test_bad_image_never_calls_api(self):
        client = Mock()
        for image in [b"", "image.jpg"]:
            with self.assertRaises(ValueError):
                planner.create_action_plan("멈춰", image=image, client=client)
        client.responses.create.assert_not_called()

    def test_owned_client_is_closed_on_success_and_failure(self):
        client = Mock()
        client.responses.create.return_value = {
            "status": "completed", "output": [], "output_text": json.dumps(PLAN),
        }
        with patch.object(planner, "create_api_client", return_value=client):
            planner.create_action_plan("멈춰")
        client.close.assert_called_once()
        client.reset_mock()
        client.responses.create.side_effect = ConnectionError()
        with patch.object(planner, "create_api_client", return_value=client):
            with self.assertRaises(planner.PlannerConnectionError):
                planner.create_action_plan("멈춰")
        client.close.assert_called_once()


class VoiceTests(unittest.TestCase):
    def test_all_speakers_repeats_and_snapshot_fifo(self):
        camera = Mock()
        camera.snapshot.side_effect = [vision.CameraSnapshot(str(i).encode(), 0) for i in range(3)]
        received = []

        def plan(text, **kwargs):
            received.append((text, kwargs["image"]))
            return planner.ActionPlan(**PLAN)

        assistant = vision.VoiceVisionAssistant(camera, client=Mock(), planner=plan,
            classifier=lambda *args, **kwargs: CommandDecision("concrete_command", 1.0))
        with contextlib.redirect_stdout(io.StringIO()):
            assistant.start()
            try:
                for name in ["기타", "등록하지 않은 사람", "음성 스트림 1"]:
                    assistant.on_transcription(name, "멈춰")
            finally:
                assistant.close()
            assistant.on_transcription("기타", "종료 후")
        self.assertEqual(received, [("멈춰", b"0"), ("멈춰", b"1"), ("멈춰", b"2")])
        self.assertEqual(camera.snapshot.call_count, 3)

    def test_empty_and_unavailable_camera_do_not_reach_planner(self):
        camera, plan = Mock(), Mock()
        camera.snapshot.side_effect = RuntimeError("no frame")
        assistant = vision.VoiceVisionAssistant(camera, client=Mock(), planner=plan,
            classifier=lambda *args, **kwargs: CommandDecision("concrete_command", 1.0))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            assistant.start()
            assistant.on_transcription("기타", "  ")
            assistant.on_transcription("기타", "명령")
            assistant.close()
        camera.snapshot.assert_called_once()
        plan.assert_not_called()

    def test_api_error_does_not_discard_next_command(self):
        camera = Mock()
        camera.snapshot.return_value = vision.CameraSnapshot(b"jpeg", 0)
        plan = Mock(side_effect=[RuntimeError("API failure"), planner.ActionPlan(**PLAN)])
        assistant = vision.VoiceVisionAssistant(camera, client=Mock(), planner=plan,
            classifier=lambda *args, **kwargs: CommandDecision("concrete_command", 1.0))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            assistant.start()
            assistant.on_transcription("기타", "첫째")
            assistant.on_transcription("기타", "둘째")
            assistant.close()
        self.assertEqual(plan.call_count, 2)

    def test_transcription_without_identifier_finishes_once(self):
        tasks, received = queue.Queue(), []
        for final in [False, True]:
            tasks.put(speech.TranscriptionTask("utterance", 1, np.ones(1024), final, 0))
        tasks.put(speech._STOP)
        with patch.object(speech, "transcribe_utterance", side_effect=["앞으로", "30cm 이동해"]), \
                contextlib.redirect_stderr(io.StringIO()):
            speech._transcription_worker(None, None, tasks, lambda *args: received.append(args))
        self.assertEqual(received, [("음성 스트림 2", "앞으로 30cm 이동해")])
        self.assertEqual(tasks.unfinished_tasks, 0)

    def test_default_voice_model_loader_never_loads_speaker_model(self):
        speaker, separator = Mock(), Mock()
        embedder, _ = speech._load_voice_models(Mock(), separator, speaker, Mock())
        self.assertIsNone(embedder)
        speaker.from_hparams.assert_not_called()
        separator.from_hparams.assert_called_once()

    def test_default_start_never_reads_registration(self):
        # Stop before loading heavy runtime packages; profile loading must not occur.
        with patch.object(speech, "load_speaker_profiles") as profiles, \
                patch.object(speech, "_load_transcription_dependencies", side_effect=RuntimeError("test stop")):
            with self.assertRaisesRegex(RuntimeError, "test stop"):
                speech.run_live_transcription()
        profiles.assert_not_called()

    def test_stale_camera_frame_rejected(self):
        camera = vision.LatestFrameCamera()
        camera._frame = np.zeros((2, 2, 3), dtype=np.uint8)
        camera._captured_at = 0
        with patch.object(vision.time, "monotonic", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "오래"):
                camera.snapshot()

    def test_enrollment_cli_disabled_and_default_is_transcription(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main._run_main(["--enroll"])
        with patch.object(main, "run_live_transcription") as run:
            self.assertEqual(main._run_main([]), 0)
        run.assert_called_once_with()

    def test_assistant_start_and_cleanup_use_no_profile(self):
        client, camera, assistant = Mock(), Mock(), Mock()
        with patch.object(vision, "create_api_client", return_value=client), \
                patch.object(vision, "LatestFrameCamera", return_value=camera), \
                patch.object(vision, "VoiceVisionAssistant", return_value=assistant), \
                patch.object(vision, "run_live_transcription") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            vision.run_voice_vision_assistant()
        run.assert_called_once_with(on_transcription=assistant.on_transcription, identify_speakers=False)
        camera.close.assert_called_once()
        assistant.close.assert_called_once()
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
