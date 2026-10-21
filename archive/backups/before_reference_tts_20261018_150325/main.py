"""ROBOTPAL speech, gesture, and OpenAI voice/vision entry point."""
#.\.venv\Scripts\python.exe .\main.py --assist
#$env:OPENAI_API_KEY = [Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "User")

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Sequence

from module.action_planner import (
    ActionPlan,
    create_action_plan,
    print_action_plan,
)
from module.gesture import RULE_SETS, handle_gesture, run_gesture_recognition
from module.speech import handle_transcription, run_live_transcription
from module.vision_assistant import run_voice_vision_assistant
from module.decision_router import process_command, print_command_result
from module.session_log import capture_output
from module.speech_quality import SpeechOptions, list_microphones, capture_sample

__all__ = [
    "handle_transcription",
    "run_live_transcription",
    "handle_gesture",
    "run_gesture_recognition",
    "ActionPlan",
    "create_action_plan",
    "print_action_plan",
    "run_voice_vision_assistant",
    "process_command",
    "print_command_result",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gesture", action="store_true", help="웹캠 손 모양 인식을 실행합니다.")
    mode.add_argument("--plan", action="store_true", help="텍스트 명령의 로봇 행동 계획을 생성합니다.")
    mode.add_argument("--assist", action="store_true", help="모든 화자의 음성과 최신 카메라 영상을 OpenAI API로 처리합니다.")
    mode.add_argument("--list-mics", action="store_true", help="마이크 장치 번호를 표시합니다.")
    mode.add_argument("--diagnose-mic", action="store_true", help="녹음 저장 없이 마이크 음량을 진단합니다.")
    mode.add_argument("--record-sample", type=Path, help="비교 평가용 WAV를 명시적으로 녹음·저장합니다.")
    mode.add_argument("--compare-stt", type=Path, help="정답 JSONL 목록으로 전사 설정을 비교합니다.")
    mode.add_argument("--list-speakers", action="store_true", help="스피커 장치 번호를 표시합니다.")
    mode.add_argument("--prepare-tts", action="store_true", help="기준 녹음으로 로컬 목소리와 고정 안내를 준비합니다.")
    mode.add_argument("--test-tts", metavar="TEXT", help="API 없이 로컬 TTS 문장을 시험 재생합니다.")
    parser.add_argument("--tts", action="store_true", help="--assist 또는 --plan 결과를 로컬 음성으로 읽습니다.")
    parser.add_argument("--tts-reference", type=Path, nargs="+", help="--prepare-tts에 사용할 기준 녹음 경로 (여러 개 가능)")
    parser.add_argument("--tts-voice", type=Path, help="준비된 목소리 JSON 경로")
    parser.add_argument("--tts-python", type=Path, help="TTS 전용 Python 경로 (기본 .tts-venv)")
    parser.add_argument("--tts-models", type=Path, help="TTS 모델 폴더 (기본 models/tts)")
    parser.add_argument("--tts-speed", type=float, default=1.0, help="음성 속도 0.5~2 (기본 1)")
    parser.add_argument("--tts-style", choices=("standard", "natural"), default="standard", help="말투: 기존 standard 또는 문장별 리듬 natural")
    parser.add_argument("--output-device", type=int, help="사용할 스피커 번호 (--list-speakers)")
    parser.add_argument("--command", help="--plan에 전달할 명령 (생략하면 입력을 요청합니다)")
    parser.add_argument("--camera", type=int, default=0, help="사용할 카메라 번호 (기본값: 0)")
    parser.add_argument("--preview", action="store_true", help="랜드마크 미리보기 창을 표시합니다.")
    parser.add_argument("--input-device", type=int, help="사용할 마이크 번호 (--list-mics로 확인)")
    parser.add_argument("--stt-backend", choices=("local", "openai"), default="local", help="전사 방식 (기본 local)")
    parser.add_argument("--stt-model", default="small", help="로컬 Whisper 모델 (예: small, medium, turbo)")
    parser.add_argument("--stt-device", choices=("cpu", "cuda"), default="cpu", help="로컬 전사 실행 장치")
    parser.add_argument("--stt-compute", choices=("int8", "float32", "float16", "int8_float16"), default="int8")
    parser.add_argument("--stt-beam", type=int, default=3, help="로컬 탐색 크기 1~10 (기본 3)")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero 음성 확률 기준 (기본 0.5)")
    parser.add_argument("--speech-pre-roll", type=float, default=0.4, help="시작 전 보존 시간, 초")
    parser.add_argument("--speech-silence", type=float, default=0.45, help="발화 종료 침묵 시간, 초 (기본 0.45)")
    parser.add_argument("--speech-min-duration", type=float, default=0.12, help="최소 음성 시간, 초")
    parser.add_argument("--whisper-vad", action="store_true", help="Whisper 내부 VAD도 함께 적용해 비교합니다.")
    parser.add_argument("--show-levels", action="store_true", help="매초 마이크 음량을 표시합니다.")
    parser.add_argument("--stt-offline", action="store_true", help="로컬에 준비된 Whisper 모델만 사용합니다.")
    parser.add_argument("--stt-min-logprob", type=float, default=-1.2, help="전사 품질 최소 평균 logprob")
    parser.add_argument("--stt-max-no-speech", type=float, default=0.7, help="허용할 최대 무음 확률")
    parser.add_argument("--no-live-preview", action="store_true", help="임시 전사 없이 최종 전사만 수행합니다.")
    parser.add_argument("--preview-interval", type=float, default=1.0, help="임시 전사 요청 간격, 초 (기본 1)")
    parser.add_argument("--preview-window", type=float, default=4.0, help="임시 전사의 최근 음성 길이, 초 (기본 4)")
    parser.add_argument("--stt-max-tokens", type=int, default=256, help="로컬 전사 최대 출력 토큰 32~256")
    parser.add_argument("--sample-seconds", type=float, default=5, help="진단·평가 녹음 길이 1~30초")
    parser.add_argument("--compare-models", help="비교할 로컬 모델 목록 (예: small,medium)")
    parser.add_argument("--compare-beams", help="비교할 beam 목록 (기본: 1,3,5)")
    parser.add_argument(
        "--rule-set",
        choices=RULE_SETS,
        default="basic",
        help="손 모양 규칙 모드 (기본값: basic)",
    )
    return parser


def _run_main(argv: Sequence[str] | None = None) -> int:
    command_line = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(command_line)
    supplied_flags = {
        option.split("=", 1)[0]
        for option in command_line
        if option.startswith("--")
    }
    tts_mode = args.tts or args.prepare_tts or args.test_tts is not None
    if args.tts and not (args.assist or args.plan):
        parser.error("--tts는 --assist 또는 --plan과 함께 사용합니다.")
    if args.prepare_tts and args.tts_reference is None:
        parser.error("--prepare-tts에는 --tts-reference가 필요합니다.")
    if args.tts_reference is not None and not args.prepare_tts:
        parser.error("--tts-reference는 --prepare-tts에서 사용합니다.")
    if supplied_flags & {"--tts-voice", "--tts-python", "--tts-models", "--tts-speed", "--tts-style", "--output-device"} and not tts_mode:
        parser.error("TTS 설정은 --tts, --prepare-tts 또는 --test-tts에서 사용합니다.")
    if args.test_tts is not None and not 1 <= len(args.test_tts.strip()) <= 300:
        parser.error("--test-tts 문장은 1~300자여야 합니다.")
    if "--camera" in supplied_flags and not (args.gesture or args.assist):
        parser.error("--camera는 --gesture 또는 --assist와 함께 사용해야 합니다.")
    if supplied_flags & {"--preview", "--rule-set"} and not args.gesture:
        parser.error("--preview, --rule-set은 --gesture와 함께 사용해야 합니다.")
    if args.command is not None and not args.plan:
        parser.error("--command는 --plan과 함께 사용해야 합니다.")
    if args.camera < 0:
        parser.error("--camera는 0 이상의 정수여야 합니다.")
    stt_flags = {"--stt-backend", "--stt-model", "--stt-device", "--stt-compute", "--stt-beam",
                 "--vad-threshold", "--speech-pre-roll", "--speech-silence", "--speech-min-duration",
                 "--whisper-vad", "--show-levels", "--stt-offline", "--stt-min-logprob", "--stt-max-no-speech",
                 "--no-live-preview", "--preview-interval", "--preview-window", "--stt-max-tokens"}
    if supplied_flags & stt_flags and (args.plan or args.gesture or args.list_mics or args.diagnose_mic or args.record_sample or args.list_speakers or args.prepare_tts or args.test_tts is not None):
        parser.error("전사 설정은 기본 음성 모드, --assist, --compare-stt에서만 사용합니다.")
    if "--input-device" in supplied_flags and (args.plan or args.gesture or args.list_mics or args.compare_stt):
        parser.error("--input-device는 실시간 음성 모드 또는 마이크 진단·녹음에서 사용합니다.")
    if supplied_flags & {"--compare-models", "--compare-beams"} and not args.compare_stt:
        parser.error("비교 설정은 --compare-stt와 함께 사용해야 합니다.")
    if "--sample-seconds" in supplied_flags and not (args.diagnose_mic or args.record_sample):
        parser.error("--sample-seconds는 마이크 진단·녹음에서만 사용합니다.")
    try:
        tts_options = None
        if tts_mode:
            from module.local_tts import TTSOptions
            tts_kwargs = {"speed": args.tts_speed, "output_device": args.output_device, "style": args.tts_style}
            for field, value in [("voice", args.tts_voice), ("python", args.tts_python), ("models", args.tts_models)]:
                if value is not None:
                    tts_kwargs[field] = value
            tts_options = TTSOptions(**tts_kwargs)
        if args.list_speakers:
            from module.local_tts import list_speakers
            list_speakers()
            return 0
        if args.prepare_tts:
            from module.local_tts import prepare_voice
            prepare_voice(tts_options, args.tts_reference)
            return 0
        if args.test_tts is not None:
            from module.local_tts import LocalTTS
            tts = LocalTTS(tts_options)
            try:
                tts.start()
                tts.submit(args.test_tts, uuid.uuid4().hex)
                tts.wait()
            finally:
                tts.close()
            return 0
        options = SpeechOptions(input_device=args.input_device, backend=args.stt_backend,
            model=args.stt_model, device=args.stt_device, compute_type=args.stt_compute,
            beam_size=args.stt_beam, vad_threshold=args.vad_threshold, pre_roll=args.speech_pre_roll,
            silence=args.speech_silence, min_speech=args.speech_min_duration,
            internal_vad=args.whisper_vad, show_levels=args.show_levels,
            min_logprob=args.stt_min_logprob, max_no_speech=args.stt_max_no_speech,
            live_preview=not args.no_live_preview, preview_interval=args.preview_interval,
            preview_window=args.preview_window, max_output_tokens=args.stt_max_tokens)
        speech_kwargs = {}
        if supplied_flags & (stt_flags | {"--input-device"}):
            speech_kwargs["options"] = options
        if args.stt_offline:
            speech_kwargs["offline"] = True
        if args.list_mics:
            list_microphones()
        elif args.diagnose_mic or args.record_sample:
            capture_sample(device=args.input_device, seconds=args.sample_seconds, destination=args.record_sample)
        elif args.compare_stt:
            from module.speech_evaluation import compare_transcription_settings
            models = args.compare_models or ("gpt-transcribe" if options.backend == "openai" else options.model)
            beams = args.compare_beams or (str(options.beam_size) if options.backend == "openai" else "1,3,5")
            compare_transcription_settings(args.compare_stt, options=options,
                models=[item.strip() for item in models.split(",")],
                beams=[int(item.strip()) for item in beams.split(",")], offline=args.stt_offline)
        elif args.assist:
            if tts_options is not None:
                speech_kwargs["tts_options"] = tts_options
            run_voice_vision_assistant(camera_index=args.camera, **speech_kwargs)
        elif args.gesture:
            run_gesture_recognition(
                camera_index=args.camera,
                preview=args.preview,
                rule_set=args.rule_set,
            )
        elif args.plan:
            command = args.command if args.command is not None else input("로봇 명령을 입력하세요: ")
            result_id = uuid.uuid4().hex if tts_options is not None else None
            result = process_command(command, **({"command_id": result_id} if result_id is not None else {}))
            print_command_result(result)
            if tts_options is not None:
                from module.local_tts import LocalTTS
                tts = LocalTTS(tts_options)
                try:
                    tts.start()
                    tts.submit_result(result, result_id)
                    tts.wait()
                except Exception as exc:
                    print(f"음성 출력 실패: {exc}", file=sys.stderr)
                finally:
                    tts.close()
            if result.plan is not None and result.plan.status == "invalid":
                return 1
            if result.decision.category == "unavailable":
                return 1
        else:
            run_live_transcription(**speech_kwargs)
    except (RuntimeError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 취소되었습니다.", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    with capture_output():
        return _run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
