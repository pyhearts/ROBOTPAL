"""Online download of official model files, followed by an offline load check."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
args = parser.parse_args()
root = args.models.resolve()
from huggingface_hub import HfApi, hf_hub_download
import nltk

api = HfApi()
repo = "lj1995/GPT-SoVITS"
info = api.model_info(repo)
revision = info.sha
files = ["s1v3.ckpt", "v2Pro/s2Gv2ProPlus.pth", "sv/pretrained_eres2netv2w24s4ep4.ckpt",
         "chinese-hubert-base/config.json", "chinese-hubert-base/preprocessor_config.json",
         "chinese-hubert-base/pytorch_model.bin"]
import shutil
checksums = {}
for name in files:
    print(f"모델 준비: {name}", flush=True)
    cached = hf_hub_download(repo, name, revision=revision)
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, target)
    checksums[name] = hashlib.sha256(target.read_bytes()).hexdigest()
for resource in ["cmudict", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    if not nltk.download(resource, download_dir=str(root / "nltk"), quiet=True):
        raise RuntimeError(f"발음 자료 준비 실패: {resource}")
manifest = {"engine": "gpt-sovits", "source_revision": "d7c2210da8c013e81a94bfc7b811a477c99fd506",
            "model_repository": repo, "model_revision": revision, "files": checksums,
            "korean_only_adapter": 1}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
worker = Path(__file__).resolve().parent.parent / "module/local_tts_worker.py"
check = subprocess.run([sys.executable, str(worker), "--engine", "gpt-sovits", "--models", str(root),
                        "--source", str(args.source.resolve())], input="", capture_output=True,
                       text=True, encoding="utf-8", timeout=240)
if check.returncode:
    print(check.stdout, check.stderr)
    raise RuntimeError("GPT-SoVITS 오프라인 로딩 확인 실패")
print("GPT-SoVITS 오프라인 준비 완료.", flush=True)
