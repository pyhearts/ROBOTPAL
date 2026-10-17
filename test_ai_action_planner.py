from __future__ import annotations

import io
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx

import ai_action_planner as planner


class FakeResponses:
    def __init__(self, output_parsed: object) -> None:
        self.output_parsed = output_parsed
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.output_parsed)


class FakeClient:
    def __init__(self, output_parsed: object) -> None:
        self.responses = FakeResponses(output_parsed)


class ActionPlanTests(unittest.TestCase):
    def test_valid_plan_uses_default_model_and_structured_schema(self):
        expected = planner.ActionPlan(
            status="plan",
            summary="책상 정리",
            steps=["쓰레기를 분리한다.", "물건을 제자리에 둔다."],
            error="",
        )
        client = FakeClient(expected)

        with mock.patch.dict(os.environ, {}, clear=True):
            actual = planner.create_action_plan("  책상을 정리해  ", client=client)

        self.assertEqual(actual, expected)
        call = client.responses.calls[0]
        self.assertEqual(call["model"], planner.DEFAULT_MODEL)
        self.assertEqual(call["input"], "책상을 정리해")
        self.assertIs(call["text_format"], planner.ActionPlan)
        self.assertIn("직접", str(call["instructions"]))

    def test_environment_model_and_explicit_model_precedence(self):
        result = planner.ActionPlan(
            status="plan", summary="시험", steps=["시험한다."], error=""
        )
        client = FakeClient(result)
        with mock.patch.dict(os.environ, {"OPENAI_MODEL": "environment-model"}):
            planner.create_action_plan("시험해", client=client)
            planner.create_action_plan("다시 시험해", client=client, model="explicit-model")

        self.assertEqual(client.responses.calls[0]["model"], "environment-model")
        self.assertEqual(client.responses.calls[1]["model"], "explicit-model")

    def test_empty_command_is_rejected_without_api_call(self):
        client = FakeClient(None)
        result = planner.create_action_plan("   ", client=client)

        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.steps, [])
        self.assertIn("비어", result.error)
        self.assertEqual(client.responses.calls, [])

    def test_model_invalid_result_is_preserved(self):
        expected = planner.ActionPlan(
            status="invalid",
            summary="",
            steps=[],
            error="무엇을 옮길지 알 수 없습니다.",
        )
        client = FakeClient(expected)

        self.assertEqual(planner.create_action_plan("그거 옮겨", client=client), expected)

    def test_missing_or_invalid_parsed_output_is_rejected(self):
        with self.subTest("missing output"):
            with self.assertRaises(planner.PlannerResponseError):
                planner.create_action_plan("정리해", client=FakeClient(None))

        malformed = planner.ActionPlan.model_construct(
            status="plan", summary="정리", steps=[], error=""
        )
        with self.subTest("empty steps"):
            with self.assertRaises(planner.PlannerResponseError):
                planner.create_action_plan("정리해", client=FakeClient(malformed))

    def test_missing_api_key_is_reported(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                planner.create_action_plan("책상을 정리해")

    def test_api_errors_have_user_friendly_messages(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        authentication_response = httpx.Response(401, request=request)
        rate_limit_response = httpx.Response(429, request=request)
        server_response = httpx.Response(500, request=request)
        cases = (
            (
                planner.AuthenticationError(
                    "unauthorized", response=authentication_response, body=None
                ),
                "인증",
            ),
            (
                planner.RateLimitError(
                    "rate limited", response=rate_limit_response, body=None
                ),
                "한도",
            ),
            (planner.APIConnectionError(request=request), "연결"),
            (
                planner.APIStatusError(
                    "server error", response=server_response, body=None
                ),
                "500",
            ),
        )

        for error, expected_text in cases:
            with self.subTest(error=type(error).__name__):
                self.assertIn(expected_text, planner._friendly_api_error(error))

    def test_print_action_plan_uses_numbered_korean_output(self):
        plan = planner.ActionPlan(
            status="plan",
            summary="책상 정리",
            steps=["쓰레기를 버린다.", "도구를 정리한다."],
            error="",
        )
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            planner.print_action_plan(plan)

        self.assertEqual(
            output.getvalue(),
            "행동 계획: 책상 정리\n1. 쓰레기를 버린다.\n2. 도구를 정리한다.\n",
        )

    def test_main_success_and_invalid_exit_codes(self):
        valid = planner.ActionPlan(
            status="plan", summary="정리", steps=["정리한다."], error=""
        )
        with mock.patch("builtins.input", return_value="정리해"), mock.patch.object(
            planner, "create_action_plan", return_value=valid
        ), mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(planner.main(), 0)
        self.assertIn("1. 정리한다.", stdout.getvalue())

        invalid = planner.ActionPlan(
            status="invalid", summary="", steps=[], error="명령이 모호합니다."
        )
        with mock.patch("builtins.input", return_value="그거 해"), mock.patch.object(
            planner, "create_action_plan", return_value=invalid
        ), mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(planner.main(), 1)
        self.assertIn("명령이 모호합니다.", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
