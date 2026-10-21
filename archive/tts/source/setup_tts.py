"""Explicit online installer for an isolated, CPU-only Korean local TTS runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "OpenVoice": "74a1d147b17a8c3092dd5430504bd83ef6c7eb23",
    "MeloTTS": "209145371cff8fc3bd60d7be902ea69cbdb7965a",
}


def run(args, env):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, env=env, check=True)


def main() -> int:
    env = os.environ.copy()
    tools = ROOT / ".tts-tools"
    env.update(UV_PYTHON_INSTALL_DIR=str(ROOT / ".tts-python"),
               UV_PYTHON_BIN_DIR=str(tools / "bin"),
               UV_CACHE_DIR=str(ROOT / "models/tts/uv-cache"),
               UV_NO_MODIFY_PATH="1", UV_NATIVE_TLS="1", PYTHONUTF8="1")
    if not (tools / "uv").is_dir():
        run([sys.executable, "-m", "pip", "install", "--target", tools, "uv==0.9.3"], env)
    env["PYTHONPATH"] = str(tools)
    uv = [sys.executable, "-m", "uv", "--quiet"]
    run([*uv, "python", "install", "3.10"], env)
    target = ROOT / ".tts-venv"
    if not target.exists():
        run([*uv, "venv", "--python", "3.10", "--seed", target], env)
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    pip = [*uv, "pip", "install", "--python", python]
    run([*pip, "torch==2.2.2+cpu", "torchaudio==2.2.2+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"], env)
    run([*pip, "-r", ROOT / "requirements-tts.txt"], env)
    for name, revision in SOURCES.items():
        dest = ROOT / ".tts-sources" / name
        if not dest.exists():
            run(["git", "clone", "--no-checkout", f"https://github.com/myshell-ai/{name}.git", dest], env)
            run(["git", "-C", dest, "checkout", revision], env)
        actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"{name} 소스 버전이 다릅니다. 기존 소스는 자동 변경하지 않습니다.")
        if name == "MeloTTS":
            # Korean-only integration: avoid eager imports/downloads for six unused languages.
            cleaner = dest / "melo/text/cleaner.py"
            text = subprocess.check_output(["git", "-C", str(dest), "show", f"{revision}:melo/text/cleaner.py"], encoding="utf-8")
            text = text.replace("from . import chinese, japanese, english, chinese_mix, korean, french, spanish", "from . import korean")
            text = re.sub(r"language_module_map\s*=\s*\{.*?\}", 'language_module_map = {"KR": korean}', text, count=1, flags=re.S)
            cleaner.write_text(text, encoding="utf-8")
            init = dest / "melo/text/__init__.py"
            text = init.read_text(encoding="utf-8")
            start = text.index("def get_bert(")
            init.write_text(text[:start] + '''def get_bert(norm_text, word2ph, language, device):
    if language != "KR":
        raise ValueError("ROBOTPAL local TTS supports Korean only")
    from .korean import get_bert_feature
    return get_bert_feature(norm_text, word2ph, device)
''', encoding="utf-8")
        run([*pip, "--no-deps", "--no-build-isolation", "-e", dest], env)
    models = ROOT / "models/tts"
    env.pop("PYTHONPATH", None)
    env.update(HF_HOME=str(models / "hf"), NLTK_DATA=str(models / "nltk"),
               HF_HUB_DOWNLOAD_TIMEOUT="120", HF_HUB_ETAG_TIMEOUT="30",
               HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    run([python, ROOT / "tools/prepare_tts_models.py", "--models", models], env)
    files = [models / "checkpoints_v2/converter/config.json", models / "checkpoints_v2/converter/checkpoint.pth",
             models / "melo/config.json", models / "melo/checkpoint.pth",
             models / "checkpoints_v2/base_speakers/ses/kr.pth"]
    manifest = {"sources": SOURCES, "korean_only_adapter": 1,
                "files": {str(p.relative_to(models)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (models / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True, env=env)
    (models / "installed-requirements.txt").write_text(freeze, encoding="utf-8")
    # This only checks imports/model loading without a network connection; no speaker is cloned.
    process = subprocess.run([str(python), str(ROOT / "module/local_tts_worker.py"), "--models", str(models)],
                             input="", text=True, encoding="utf-8", capture_output=True, env=env, timeout=180)
    if process.returncode:
        print(process.stdout, process.stderr)
        raise RuntimeError("오프라인 TTS 준비 확인에 실패했습니다.")
    print("로컬 TTS 설치 완료. --prepare-tts --tts-reference 기준녹음.wav 로 목소리를 준비하세요.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        print(f"TTS 설치 실패: {exc}", file=sys.stderr)
        raise SystemExit(1)
