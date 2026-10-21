"""CLI parsing and mode dispatch; optional runtimes are loaded per mode."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Sequence
from .decision_router import process_command, print_command_result

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
        choices=("basic", "numbers"),
        default="basic",
        help="손 모양 규칙 모드 (기본값: basic)",
    )
    return parser


def _validate_options(parser, args, supplied_flags):
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
    if supplied_flags & stt_flags and (args.plan or args.gesture or args.list_mics or args.diagnose_mic or args.record_sample):
        parser.error("전사 설정은 기본 음성 모드, --assist, --compare-stt에서만 사용합니다.")
    if "--input-device" in supplied_flags and (args.plan or args.gesture or args.list_mics or args.compare_stt):
        parser.error("--input-device는 실시간 음성 모드 또는 마이크 진단·녹음에서 사용합니다.")
    if supplied_flags & {"--compare-models", "--compare-beams"} and not args.compare_stt:
        parser.error("비교 설정은 --compare-stt와 함께 사용해야 합니다.")
    if "--sample-seconds" in supplied_flags and not (args.diagnose_mic or args.record_sample):
        parser.error("--sample-seconds는 마이크 진단·녹음에서만 사용합니다.")
    return stt_flags


def _speech_options(args):
    from .speech_quality import SpeechOptions
    options = SpeechOptions(input_device=args.input_device, backend=args.stt_backend,
        model=args.stt_model, device=args.stt_device, compute_type=args.stt_compute,
        beam_size=args.stt_beam, vad_threshold=args.vad_threshold, pre_roll=args.speech_pre_roll,
        silence=args.speech_silence, min_speech=args.speech_min_duration,
        internal_vad=args.whisper_vad, show_levels=args.show_levels,
        min_logprob=args.stt_min_logprob, max_no_speech=args.stt_max_no_speech,
        live_preview=not args.no_live_preview, preview_interval=args.preview_interval,
        preview_window=args.preview_window, max_output_tokens=args.stt_max_tokens)
    return options


def run_text_command(command: str) -> int:
    result = process_command(command)
    print_command_result(result)
    if result.plan is not None and result.plan.status == "invalid":
        return 1
    return 1 if result.decision.category == "unavailable" else 0


def _dispatch(args, supplied_flags, stt_flags):
    if args.plan:
        return run_text_command(args.command if args.command is not None else input("로봇 명령을 입력하세요: "))
    if args.gesture:
        from .gesture import run_gesture_recognition
        run_gesture_recognition(camera_index=args.camera, preview=args.preview, rule_set=args.rule_set)
        return 0
    if args.list_mics:
        from .speech_quality import list_microphones
        list_microphones()
        return 0
    if args.diagnose_mic or args.record_sample:
        from .speech_quality import capture_sample
        capture_sample(device=args.input_device, seconds=args.sample_seconds, destination=args.record_sample)
        return 0
    options = _speech_options(args)
    if args.compare_stt:
        from .speech_evaluation import compare_transcription_settings
        models = args.compare_models or ("gpt-transcribe" if options.backend == "openai" else options.model)
        beams = args.compare_beams or (str(options.beam_size) if options.backend == "openai" else "1,3,5")
        compare_transcription_settings(args.compare_stt, options=options,
            models=[item.strip() for item in models.split(",")],
            beams=[int(item.strip()) for item in beams.split(",")], offline=args.stt_offline)
        return 0
    speech_kwargs = {}
    if supplied_flags & (stt_flags | {"--input-device"}):
        speech_kwargs["options"] = options
    if args.stt_offline:
        speech_kwargs["offline"] = True
    if args.assist:
        from .vision_assistant import run_voice_vision_assistant
        run_voice_vision_assistant(camera_index=args.camera, **speech_kwargs)
    else:
        from .speech import run_live_transcription
        run_live_transcription(**speech_kwargs)
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    command_line = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    supplied_flags = {option.split("=", 1)[0] for option in command_line if option.startswith("--")}
    removed = {"--prepare-tts", "--test-tts", "--list-speakers", "--output-device"}
    if any(flag == "--tts" or flag.startswith("--tts-") or flag in removed for flag in supplied_flags):
        parser.error("TTS는 보관 상태이며 archive/tts/RESTORE.md를 참고하세요")
    args = parser.parse_args(command_line)
    stt_flags = _validate_options(parser, args, supplied_flags)
    try:
        return _dispatch(args, supplied_flags, stt_flags)
    except (RuntimeError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 취소되었습니다.", file=sys.stderr)
        return 1
