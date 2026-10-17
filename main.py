"""ROBOTPAL microphone and webcam command-line entry point."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from module.gesture import RULE_SETS, handle_gesture, run_gesture_recognition
from module.speech import enroll_speakers, handle_transcription, run_live_transcription

__all__ = [
    "handle_transcription",
    "run_live_transcription",
    "handle_gesture",
    "run_gesture_recognition",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--enroll", action="store_true", help="두 명의 화자를 새로 등록합니다.")
    mode.add_argument("--gesture", action="store_true", help="웹캠 손 모양 인식을 실행합니다.")
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
        else:
            run_live_transcription()
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
