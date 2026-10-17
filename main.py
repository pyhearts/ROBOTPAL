"""ROBOTPAL microphone and webcam command-line entry point."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from openai import OpenAIError

from module.action_planner import (
    ActionPlan,
    PlannerResponseError,
    _friendly_api_error,
    create_action_plan,
    print_action_plan,
)
from module.gesture import RULE_SETS, handle_gesture, run_gesture_recognition
from module.speech import enroll_speakers, handle_transcription, run_live_transcription

__all__ = [
    "handle_transcription",
    "run_live_transcription",
    "handle_gesture",
    "run_gesture_recognition",
    "ActionPlan",
    "create_action_plan",
    "print_action_plan",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--enroll", action="store_true", help="두 명의 화자를 새로 등록합니다.")
    mode.add_argument("--gesture", action="store_true", help="웹캠 손 모양 인식을 실행합니다.")
    mode.add_argument("--plan", action="store_true", help="텍스트 명령의 로봇 행동 계획을 생성합니다.")
    parser.add_argument("--command", help="--plan에 전달할 명령 (생략하면 입력을 요청합니다)")
    parser.add_argument("--camera", type=int, default=0, help="사용할 카메라 번호 (기본값: 0)")
    parser.add_argument("--preview", action="store_true", help="랜드마크 미리보기 창을 표시합니다.")
    parser.add_argument(
        "--rule-set",
        choices=RULE_SETS,
        default="basic",
        help="손 모양 규칙 모드 (기본값: basic)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    command_line = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(command_line)
    camera_flags = {
        option.split("=", 1)[0]
        for option in command_line
        if option.startswith("--")
    } & {"--camera", "--preview", "--rule-set"}
    if camera_flags and not args.gesture:
        parser.error("--camera, --preview, --rule-set은 --gesture와 함께 사용해야 합니다.")
    if args.command is not None and not args.plan:
        parser.error("--command는 --plan과 함께 사용해야 합니다.")
    if args.camera < 0:
        parser.error("--camera는 0 이상의 정수여야 합니다.")
    try:
        if args.enroll:
            enroll_speakers()
        elif args.gesture:
            run_gesture_recognition(
                camera_index=args.camera,
                preview=args.preview,
                rule_set=args.rule_set,
            )
        elif args.plan:
            command = args.command if args.command is not None else input("로봇 명령을 입력하세요: ")
            plan = create_action_plan(command)
            if plan.status == "invalid":
                print(f"오류: {plan.error}", file=sys.stderr)
                return 1
            print_action_plan(plan)
        else:
            run_live_transcription()
    except (RuntimeError, ValueError, PlannerResponseError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except OpenAIError as exc:
        print(_friendly_api_error(exc), file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 취소되었습니다.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
