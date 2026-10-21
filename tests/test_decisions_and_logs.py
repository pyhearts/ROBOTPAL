"""Real SDK transport tests for classification, API savings, and durable output."""

import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx2 as httpx
from openai import OpenAI

import main
from module import cli
import ai_action_planner
from module import decision_router as router
from module import session_log
from module import vision_assistant as vision
from module.action_planner import ActionPlan


PLAN = {"status": "plan", "summary": "이동", "steps": ["앞으로 30cm 이동한다.", "정지한다."], "error": ""}


def decision_response(category="concrete_command", confidence=0.99):
    return {
        "model": router.DECISION_MODEL,
        "answers": [{"type": "choice", "name": router.QUESTION_NAME,
                     "choice": category, "confidence": confidence,
                     "probabilities": [{"value": category, "probability": confidence}]}],
        "usage": {"input_tokens": 100, "output_tokens": 0, "total_tokens": 100},
    }


def plan_response():
    return {"id": "resp_test", "object": "response", "created_at": 0,
            "model": "gpt-5.4-mini", "status": "completed",
            "output": [{"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(PLAN), "annotations": []}]}],
            "usage": {"input_tokens": 200, "output_tokens": 50, "total_tokens": 250}}


class DecisionsTests(unittest.TestCase):
    def client(self, decision=None, *, status=200):
        self.requests = []

        def respond(request):
            self.requests.append(request)
            if request.url.path.endswith("/decisions"):
                return httpx.Response(status, json=decision if decision is not None else decision_response())
            return httpx.Response(200, json=plan_response())

        client = OpenAI(api_key="test-not-a-real-key", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.addCleanup(client.close)
        return client

    def process(self, client, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return router.process_command("앞으로 30cm 이동해", client=client, **kwargs)

    def test_real_sdk_calls_decisions_then_responses_for_concrete_command(self):
        client = self.client()
        result = self.process(client, image=b"jpeg")
        self.assertEqual(result.plan.steps, PLAN["steps"])
        self.assertEqual([r.url.path for r in self.requests], ["/v1/decisions", "/v1/responses"])
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertEqual(body["questions"][0]["type"], "choice")
        self.assertIsInstance(body["input"], str)
        self.assertNotIn("image", json.dumps(body))
        self.assertNotIn("store", body)

    def test_non_commands_and_incomplete_or_unsupported_commands_skip_planner(self):
        for category in ["not_command", "needs_details", "unsupported", "uncertain"]:
            with self.subTest(category=category):
                client = self.client(decision_response(category))
                result = self.process(client)
                self.assertIsNone(result.plan)
                self.assertEqual(result.message, router.FIXED_MESSAGES[category])
                self.assertEqual(len(self.requests), 1)

    def test_low_confidence_and_refusal_skip_planner(self):
        refusal = decision_response()
        refusal["answers"] = [{"type": "refusal", "name": router.QUESTION_NAME}]
        for response in [decision_response(confidence=0.2), refusal]:
            with self.subTest(response=response):
                result = self.process(self.client(response))
                self.assertEqual(result.message, router.FIXED_MESSAGES["uncertain"])
                self.assertEqual(len(self.requests), 1)

    def test_stop_uses_fixed_plan_without_image_or_second_api_request(self):
        result = self.process(self.client(decision_response("stop_command")), require_image=True)
        self.assertEqual(result.plan.steps, ["진행 중인 동작을 정지한다."])
        self.assertEqual(len(self.requests), 1)

    def test_missing_image_does_not_request_plan_in_assistant(self):
        result = self.process(self.client(), require_image=True)
        self.assertEqual(result.message, router.FIXED_MESSAGES["no_image"])
        self.assertEqual(len(self.requests), 1)

    def test_invalid_decisions_never_fall_back_to_responses(self):
        responses = [{"model": router.DECISION_MODEL, "answers": [], "usage": {}},
                     decision_response("invented"), decision_response(confidence=2),
                     decision_response(confidence=float("nan"))]
        wrong_name = decision_response()
        wrong_name["answers"][0]["name"] = "wrong_question"
        responses.append(wrong_name)
        for response in responses:
            with self.subTest(response=response):
                # JSON cannot encode NaN; validate that case through a fake SDK result.
                if response["answers"] and response["answers"][0].get("confidence") != response["answers"][0].get("confidence"):
                    client = Mock()
                    client.decisions.create.return_value = response
                    result = self.process(client)
                    client.responses.create.assert_not_called()
                else:
                    result = self.process(self.client(response))
                    self.assertEqual(len(self.requests), 1)
                self.assertEqual(result.message, router.FIXED_MESSAGES["unavailable"])

    def test_api_errors_do_not_trigger_plan_or_retries(self):
        for status in [401, 403, 429, 500]:
            with self.subTest(status=status):
                result = self.process(self.client({"error": {"message": "do not log this secret"}}, status=status))
                self.assertEqual(result.message, router.FIXED_MESSAGES["unavailable"])
                self.assertEqual(len(self.requests), 1)

    def test_empty_input_uses_no_api_or_key(self):
        with patch.object(router, "create_api_client") as create, contextlib.redirect_stdout(io.StringIO()):
            result = router.process_command("   ")
        self.assertEqual(result.message, router.FIXED_MESSAGES["not_command"])
        create.assert_not_called()

    def test_camera_unavailable_still_classifies_and_prints_non_command(self):
        camera = Mock()
        camera.snapshot.side_effect = RuntimeError("no frame")
        planner = Mock()
        assistant = vision.VoiceVisionAssistant(camera, client=self.client(decision_response("not_command")), planner=planner)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assistant.start()
            assistant.on_transcription("음성 입력", "감사합니다")
            assistant.close()
        self.assertIn(router.FIXED_MESSAGES["not_command"], output.getvalue())
        planner.assert_not_called()

    def test_results_usage_and_all_outputs_are_saved_with_command_ids(self):
        client = self.client()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with session_log.capture_output(Path(folder)) as log:
                result = router.process_command("앞으로 30cm 이동해", client=client)
                router.print_command_result(result)
                print("오류 출력 예시", file=sys.stderr)
            events = [json.loads(line) for line in log.events_path.read_text(encoding="utf-8").splitlines()]
            by_kind = {event["event"]: event for event in events}
            self.assertEqual(by_kind["decisions_response"]["usage"]["total_tokens"], 100)
            self.assertEqual(by_kind["responses_result"]["usage"]["total_tokens"], 250)
            self.assertEqual(by_kind["command_result"]["plan"], PLAN)
            ids = {by_kind[kind]["command_id"] for kind in ["command_input", "decisions_response", "responses_result", "command_result"]}
            self.assertEqual(len(ids), 1)
            text = log.console_path.read_text(encoding="utf-8")
            self.assertIn("행동 계획", text)
            self.assertIn("앞으로 30cm 이동한다.", text)
            self.assertIn("오류 출력 예시", text)


class LogTests(unittest.TestCase):
    def test_threads_partial_lines_redaction_and_no_overwrite(self):
        original_stdout, original_stderr = sys.stdout, sys.stderr
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"OPENAI_API_KEY": "secret-test-key"}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            paths = []
            for _ in range(2):
                with session_log.capture_output(Path(folder)) as log:
                    paths.append(log.console_path)
                    print("한글 정상 출력")
                    print("secret-test-key", file=sys.stderr)
                    print("남은 조각", end="")
                    worker = threading.Thread(target=lambda: print("작업자 결과", flush=True))
                    worker.start()
                    worker.join()
                    session_log.record_event("sample", nested={"key": "secret-test-key"})
                text = log.console_path.read_text(encoding="utf-8")
                jsonl = log.events_path.read_text(encoding="utf-8")
                self.assertIn("한글 정상 출력", text)
                self.assertIn("남은 조각", text)
                self.assertIn("작업자 결과", text)
                self.assertNotIn("secret-test-key", text + jsonl)
                for line in jsonl.splitlines():
                    json.loads(line)
            self.assertNotEqual(paths[0], paths[1])
        self.assertIs(sys.stdout, original_stdout)
        self.assertIs(sys.stderr, original_stderr)

    def test_unhandled_exception_is_logged_and_streams_restored(self):
        stdout = sys.stdout
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "검증 실패"):
                with session_log.capture_output(Path(folder)) as log:
                    raise ValueError("검증 실패")
            self.assertIn("ValueError: 검증 실패", log.console_path.read_text(encoding="utf-8"))
        self.assertIs(sys.stdout, stdout)

    def test_cli_and_legacy_entrypoint_both_classify_and_log_fixed_output(self):
        fixed = router.CommandResult(router.CommandDecision("not_command", 1), message=router.FIXED_MESSAGES["not_command"])
        with tempfile.TemporaryDirectory() as folder, patch.object(session_log, "LOG_ROOT", Path(folder)), \
                contextlib.redirect_stdout(io.StringIO()):
            with patch.object(cli, "process_command", return_value=fixed) as process:
                self.assertEqual(main.main(["--plan", "--command", "안녕"]), 0)
                process.assert_called_once_with("안녕")
            with patch.object(cli, "process_command", return_value=fixed), \
                    patch("builtins.input", return_value="안녕"):
                self.assertEqual(ai_action_planner.main(), 0)
            logs = list(Path(folder).glob("*.log"))
            self.assertEqual(len(logs), 2)
            for path in logs:
                self.assertIn(router.FIXED_MESSAGES["not_command"], path.read_text(encoding="utf-8"))

    def test_argument_errors_are_logged(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(session_log, "LOG_ROOT", Path(folder)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main.main(["--camera", "-1"])
            text = next(Path(folder).glob("*.log")).read_text(encoding="utf-8")
            self.assertIn("error", text)


if __name__ == "__main__":
    unittest.main()
