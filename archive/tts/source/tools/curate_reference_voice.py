"""Build local reference clips and transcripts without loading a TTS engine."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from module.reference_voice import curate_references, finalize_voice
from module.session_log import capture_output

parser = argparse.ArgumentParser()
parser.add_argument("recordings", nargs="*", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--finalize", type=Path, help="수정한 dataset.json을 새 목소리 설정으로 저장")
parser.add_argument("--voice", type=Path)
parser.add_argument("--models", type=Path, default=ROOT / "models/reference_tts")
args = parser.parse_args()
if args.finalize and (args.recordings or args.output or args.voice is None):
    parser.error("--finalize에는 --voice 새경로가 필요하며 녹음·--output을 함께 지정할 수 없습니다.")
if not args.finalize and (not args.recordings or args.voice):
    parser.error("녹음 경로를 지정하거나 --finalize와 --voice를 함께 사용하세요.")
with capture_output():
    if args.finalize:
        dataset = json.loads(args.finalize.read_text(encoding="utf-8"))
        finalize_voice(dataset, args.models, args.voice)
        print(f"참조 목소리 저장 완료: {args.voice}", flush=True)
    else:
        destination = args.output or ROOT / "data/tts" / ("reference_dataset_" + time.strftime("%Y%m%d_%H%M%S"))
        curate_references(args.recordings, destination)
