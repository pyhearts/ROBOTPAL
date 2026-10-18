"""Install GPT-SoVITS v2ProPlus as an isolated, offline Korean TTS engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REVISION = "d7c2210da8c013e81a94bfc7b811a477c99fd506"


def main() -> int:
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / ".tts-tools"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               UV_PYTHON_INSTALL_DIR=str(ROOT / ".tts-python"),
               UV_PYTHON_BIN_DIR=str(ROOT / ".tts-tools/bin"),
               UV_CACHE_DIR=str(ROOT / "models/tts/uv-cache"), UV_NATIVE_TLS="1")
    def run(args):
        subprocess.run([str(a) for a in args], env=env, cwd=ROOT, check=True)
    if not (ROOT / ".tts-tools/uv").is_dir():
        run([sys.executable, "-m", "pip", "install", "--target", ROOT / ".tts-tools", "uv==0.9.3"])
    uv = [sys.executable, "-m", "uv", "--quiet"]
    run([*uv, "python", "install", "3.10"])
    target = ROOT / ".gsv-venv"
    if not target.exists():
        run([*uv, "venv", "--python", "3.10", "--seed", target])
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    pip = [*uv, "pip", "install", "--python", python]
    run([*pip, "torch==2.5.1+cpu", "torchaudio==2.5.1+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"])
    run([*pip, "-r", ROOT / "requirements-reference-tts.txt"])
    source = ROOT / ".gsv-sources/GPT-SoVITS"
    if not source.exists():
        run(["git", "clone", "--depth", "1", "--branch", "20250606v2pro",
             "https://github.com/RVC-Boss/GPT-SoVITS.git", source])
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise RuntimeError("GPT-SoVITS 소스 버전이 다릅니다. 기존 파일은 자동 변경하지 않습니다.")
    env.pop("PYTHONPATH", None)
    models = ROOT / "models/reference_tts"
    env.update(HF_HOME=str(models / "hf"), NLTK_DATA=str(models / "nltk"),
               HF_HUB_DOWNLOAD_TIMEOUT="120", HF_HUB_ETAG_TIMEOUT="30",
               HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    run([python, "-u", ROOT / "tools/prepare_reference_tts_models.py", "--models", models, "--source", source])
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True)
    (models / "installed-requirements.txt").write_text(freeze, encoding="utf-8")
    print("참조 음성 TTS 설치 완료. --prepare-tts로 녹음과 대본을 준비하세요.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f"참조 음성 TTS 설치 실패: {exc}", file=sys.stderr)
        raise SystemExit(1)
