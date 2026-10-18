"""Backward-compatible command-line entry point for the action planner."""

from __future__ import annotations

import sys
from module.decision_router import process_command, print_command_result
from module.session_log import capture_output

from module.action_planner import (
    ActionPlan,
    DEFAULT_MODEL,
    PLANNER_INSTRUCTIONS,
    PlannerConnectionError,
    PlannerResponseError,
    _friendly_api_error,
    create_action_plan,
    print_action_plan,
)


def _run_main() -> int:
    try:
        command = input("명령을 입력하세요: ")
        result = process_command(command)
        print_command_result(result)
        if result.plan is not None and result.plan.status == "invalid":
            return 1
        if result.decision.category == "unavailable":
            return 1
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 취소되었습니다.", file=sys.stderr)
    except (PlannerResponseError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
    return 1


def main() -> int:
    with capture_output():
        return _run_main()


if __name__ == "__main__":
    raise SystemExit(main())
