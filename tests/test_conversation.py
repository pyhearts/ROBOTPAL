"""Offline SDK and assistant integration checks for independent replies."""

import base64
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx2 as httpx
from openai import OpenAI

import main
import ai_action_planner
from module import conversation, decision_router as router, session_log
from module import vision_assistant as vision
from module.action_planner import selected_model


def reply_response(text="안녕하세요. 무엇을 도와드릴까요?", status="completed", refusal=False):
    content = {"type": "refusal", "refusal": "refused"} if refusal else {
        "type": "output_text", "text": text, "annotations": []}
    return {"id": "reply_test", "object": "response", "created_at": 0,
            "model": selected_model(), "status": status,
            "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                        "status": "completed", "content": [content]}],
            "usage": {"input_tokens": 40, "output_tokens": 20, "total_tokens": 60}}


def decision_response(category, confidence=1):
    return {"model": router.DECISION_MODEL, "answers": [
        {"name": router.QUESTION_NAME, "type": "choice", "choice": category,
         "confidence": confidence}], "usage": {"total_tokens": 10}}


class ConversationTests(unittest.TestCase):
    def client(self, decisions=None, replies=None, status=200):
        self.requests = []
        decisions = iter(decisions or [decision_response("conversation")])
        replies = iter(replies or [reply_response()])

        def respond(request):
            self.requests.append(request)
            if request.url.path.endswith("/decisions"):
                return httpx.Response(200, json=next(decisions))
            response = next(replies)
            if isinstance(response, Exception):
                raise response
            return httpx.Response(status, json=response)

        client = OpenAI(api_key="offline-test-key", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.addCleanup(client.close)
        return client

    def process(self, client, text="안녕", **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return router.process_command(text, client=client, **kwargs)

    def test_text_reply_drops_available_image_and_uses_no_history_or_tools(self):
        client = self.client()
        planner = Mock()
        result = self.process(client, image=b"must-not-be-sent", require_image=True, planner=planner)
        self.assertIsNone(result.plan)
        self.assertIn("안녕하세요", result.message)
        planner.assert_not_called()
        self.assertEqual([r.url.path for r in self.requests], ["/v1/decisions", "/v1/responses"])
        body = json.loads(self.requests[1].content)
        self.assertEqual(body["input"], [{"role": "user", "content": [{"type": "input_text", "text": "안녕"}]}])
        self.assertEqual(body["model"], selected_model())
        self.assertFalse(body["store"])
        self.assertEqual(body["max_output_tokens"], 1024)
        for field in ["previous_response_id", "conversation", "tools", "text"]:
            self.assertNotIn(field, body)
        self.assertEqual(self.requests[1].extensions["timeout"]["read"], 30)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            router.print_command_result(result)
        self.assertIn("[AI 답변] 안녕하세요", output.getvalue())

    def test_visual_reply_sends_exact_selected_jpeg(self):
        client = self.client([decision_response("visual_question")])
        result = self.process(client, "앞에 뭐가 있어?", image=b"selected-jpeg")
        self.assertIsNone(result.plan)
        self.assertEqual(len(self.requests), 2)
        content = json.loads(self.requests[1].content)["input"][0]["content"]
        self.assertEqual(base64.b64decode(content[1]["image_url"].split(",", 1)[1]), b"selected-jpeg")
        self.assertNotIn("image", self.requests[0].content.decode())

    def test_visual_without_image_skips_reply(self):
        client = self.client([decision_response("visual_question")])
        result = self.process(client)
        self.assertEqual(result.message, router.FIXED_MESSAGES["conversation_no_image"])
        self.assertEqual(len(self.requests), 1)

    def test_low_confidence_conversations_skip_reply(self):
        for category in ["conversation", "visual_question"]:
            with self.subTest(category=category):
                client = self.client([decision_response(category, .79)])
                result = self.process(client, image=b"jpeg")
                self.assertEqual(result.message, router.FIXED_MESSAGES["uncertain"])
                self.assertEqual(len(self.requests), 1)

    def test_failures_become_fixed_guidance_without_retries(self):
        cases = [(401, {"error": {"message": "unauthorized"}}),
                 (403, {"error": {"message": "forbidden"}}),
                 (429, {"error": {"message": "rate limited"}}),
                 (500, {"error": {"message": "server error"}}),
                 (200, reply_response("")), (200, reply_response(status="incomplete")),
                 (200, reply_response(refusal=True)),
                 (200, httpx.ReadTimeout("test timeout"))]
        for status, response in cases:
            with self.subTest(status=status, response=type(response).__name__):
                client = self.client(replies=[response], status=status)
                # The reply helper must override even a caller's retry policy.
                client.max_retries = 2
                result = self.process(client)
                self.assertEqual(result.message, router.FIXED_MESSAGES["conversation_error"])
                self.assertEqual(len(self.requests), 2)
                self.assertFalse(client.is_closed())

    def test_assistant_keeps_snapshot_and_log_ids_across_independent_turns(self):
        camera = Mock()
        camera.snapshot.side_effect = [vision.CameraSnapshot(b"first", 1), vision.CameraSnapshot(b"second", 2)]
        client = self.client([decision_response("conversation"), decision_response("visual_question")],
                             [reply_response("안녕하세요. 반갑습니다."), reply_response("빨간 물체가 보입니다.")])
        planner = Mock()
        assistant = vision.VoiceVisionAssistant(camera, client=client, planner=planner)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with session_log.capture_output(Path(folder)) as log:
                assistant.start()
                assistant.on_transcription("화자 A", "안녕")
                assistant.on_transcription("화자 B", "앞에 뭐가 있어?")
                assistant.close()
            events = [json.loads(line) for line in log.events_path.read_text(encoding="utf-8").splitlines()]
            received = [e for e in events if e["event"] == "voice_command_received"]
            responses = [e for e in events if e["event"] == "conversation_response"]
            results = [e for e in events if e["event"] == "command_result"]
            self.assertEqual(len(results), 2)
            self.assertNotEqual(results[0]["command_id"], results[1]["command_id"])
            for i, raw in enumerate([b"first", b"second"]):
                self.assertEqual(received[i]["command_id"], results[i]["command_id"])
                self.assertEqual(responses[i]["command_id"], results[i]["command_id"])
                self.assertEqual(responses[i]["usage"]["total_tokens"], 60)
                self.assertEqual(results[i]["image_sent"], bool(i))
                self.assertEqual((Path(folder) / received[i]["image_path"]).read_bytes(), raw)
            bodies = [json.loads(r.content) for r in self.requests if r.url.path.endswith("/responses")]
            self.assertEqual(len(bodies[0]["input"]), 1)
            self.assertEqual(len(bodies[1]["input"]), 1)
            self.assertEqual(bodies[1]["input"][0]["content"][0]["text"], "앞에 뭐가 있어?")
            self.assertEqual(base64.b64decode(bodies[1]["input"][0]["content"][1]["image_url"].split(",", 1)[1]), b"second")
            self.assertNotIn("data:image", log.events_path.read_text(encoding="utf-8"))
        self.assertEqual(camera.snapshot.call_count, 2)
        planner.assert_not_called()

    def test_assistant_recovers_after_reply_failure_without_camera(self):
        camera = Mock()
        camera.snapshot.side_effect = RuntimeError("no camera")
        client = self.client([decision_response("conversation"), decision_response("conversation")],
                             [reply_response(status="incomplete"), reply_response("다시 안녕하세요.")])
        assistant = vision.VoiceVisionAssistant(camera, client=client)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assistant.start()
            assistant.on_transcription("A", "안녕")
            assistant.on_transcription("A", "다시 안녕")
            assistant.close()
        self.assertIn(router.FIXED_MESSAGES["conversation_error"], output.getvalue())
        self.assertIn("[AI 답변] 다시 안녕하세요.", output.getvalue())
        self.assertEqual(len(self.requests), 4)

    def test_model_environment_override_and_owned_client_cleanup(self):
        client = self.client()
        with patch.dict("os.environ", {"OPENAI_MODEL": "configured-model"}), \
                patch.object(conversation, "create_api_client", return_value=client):
            reply = conversation.create_conversation_reply("안녕")
        self.assertTrue(reply)
        self.assertTrue(client.is_closed())
        self.assertEqual(json.loads(self.requests[0].content)["model"], "configured-model")

    def test_conversation_function_can_be_injected(self):
        reply = Mock(return_value="반갑습니다.")
        result = self.process(Mock(), classifier=lambda *a, **k: router.CommandDecision("conversation", 1),
                              conversation=reply, image=b"unused")
        self.assertEqual(result.message, "반갑습니다.")
        self.assertIsNone(reply.call_args.kwargs["image"])

    def test_both_text_entrypoints_generate_replies_and_default_stt_does_not(self):
        for legacy in [False, True]:
            with self.subTest(legacy=legacy):
                client = self.client()
                with tempfile.TemporaryDirectory() as folder, \
                        patch.object(session_log, "LOG_ROOT", Path(folder)), \
                        patch.object(router, "create_api_client", return_value=client), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    if legacy:
                        with patch("builtins.input", return_value="안녕"):
                            self.assertEqual(ai_action_planner.main(), 0)
                    else:
                        self.assertEqual(main.main(["--plan", "--command", "안녕"]), 0)
                    self.assertIn("[AI 답변]", output.getvalue())
                self.assertEqual(len(self.requests), 2)
        with patch.object(main, "run_live_transcription") as transcribe, \
                patch.object(router, "create_api_client") as create:
            self.assertEqual(main._run_main([]), 0)
        transcribe.assert_called_once_with()
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
