"""Online-only model download and Korean pronunciation warm-up, never live audio."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=Path, required=True)
args = parser.parse_args()
root = args.models.resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module"))

from huggingface_hub import hf_hub_download
import nltk

os.environ["HF_HOME"] = str(root / "hf")
os.environ["NLTK_DATA"] = str(root / "nltk")
for filename in ["converter/config.json", "converter/checkpoint.pth", "base_speakers/ses/kr.pth"]:
    source = hf_hub_download("myshell-ai/OpenVoiceV2", filename, resume_download=True)
    destination = root / "checkpoints_v2" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)

for filename in ["config.json", "checkpoint.pth"]:
    source = hf_hub_download("myshell-ai/MeloTTS-Korean", filename, resume_download=True)
    (root / "melo").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, root / "melo" / filename)
for resource in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict", "punkt", "punkt_tab"]:
    if not nltk.download(resource, download_dir=str(root / "nltk"), quiet=True):
        raise RuntimeError(f"발음 자료 준비 실패: {resource}")

from local_tts_worker import OpenVoiceBackend

# No personal reference is used. This prepares BERT, pronunciation, and watermark assets.
backend = OpenVoiceBackend(root)
backend.model.tts_to_file("안녕하세요. 왼쪽으로 구십 도, 앞으로 삼십 센티미터입니다.",
                          backend.speaker_id, str(root / "base_voice_test.wav"), quiet=True)
print("한국어 기본 음성 시험 생성 완료 (사용자 목소리 복제 아님).")
