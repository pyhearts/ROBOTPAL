"""Backward-compatible command-line entry point for the action planner."""

from __future__ import annotations

import sys

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


def main() -> int:
    try:
        command = input("명령을 입력하세요: ")
        plan = create_action_plan(command)
        if plan.status == "invalid":
            print(f"오류: {plan.error}", file=sys.stderr)
            return 1
        print_action_plan(plan)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 취소되었습니다.", file=sys.stderr)
    except (PlannerResponseError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
