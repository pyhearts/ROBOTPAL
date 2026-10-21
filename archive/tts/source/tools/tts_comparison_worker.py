"""Isolated, network-denied TTS measurements; never uses a speaker or API."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "module"))
sys.path.insert(0, str(ROOT / ".gsv-sources/benchmark-dependencies"))
TEXT = "주변 모습을 설명해 드릴게요. 오른쪽에 파란색 컵이 보여요."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--phase", choices=("solo", "stream", "probe"), default="solo")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.voice = args.voice.resolve()
    is_fast = "CPUFast" in str(args.source)
    models = ROOT / "models/reference_tts"
    os.environ.update(HF_HOME=str(models / "hf"), NLTK_DATA=str(models / "nltk"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      MPLCONFIGDIR=str(models / "matplotlib"), TOKENIZERS_PARALLELISM="false",
                      OMP_NUM_THREADS=str(args.threads))
    os.environ.pop("OPENAI_API_KEY", None)
    from local_tts_worker import deny_network
    deny_network()
    from reference_tts_backend import ReferenceTTSBackend
    if is_fast:
        # CPUFast's all_ko frontend calls the segmenter even for pure Korean.
        # Keep this benchmark limited to Korean; no language detector downloads.
        install_adapters = ReferenceTTSBackend._install_korean_only_adapters
        def fast_adapters():
            import re
            install_adapters()
            def korean_segments(text, language=None):
                if language != "ko" or re.search(r"[A-Za-z\u3400-\u9fff\u3040-\u30ff]", text):
                    raise ValueError("CPUFast benchmark accepts pure Korean only")
                return [{"text": text, "lang": "ko"}]
            sys.modules["text.LangSegmenter"].LangSegmenter.getTexts = staticmethod(korean_segments)
        ReferenceTTSBackend._install_korean_only_adapters = staticmethod(fast_adapters)
    import numpy as np
    import soundfile as sf
    import torch

    protocol = sys.stdout
    def emit(event, **kwargs):
        protocol.write(json.dumps({"event": event, **kwargs}, ensure_ascii=False) + "\n")
        protocol.flush()

    report = {"source": str(args.source.resolve()), "threads": args.threads, "phase": args.phase,
              "voice": str(args.voice), "text": TEXT, "samples": [],
              "network_disabled": True, "playback_tested": False, "cache_used": False}
    report["torch_version"] = torch.__version__
    report["vits_parallel_infer"] = not is_fast
    report["voice_sha256"] = hashlib.sha256(args.voice.read_bytes()).hexdigest()
    report["model_manifest_sha256"] = hashlib.sha256((models / "manifest.json").read_bytes()).hexdigest()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            started = time.perf_counter()
            backend = ReferenceTTSBackend(models, args.source, args.threads)
            report["model_start_seconds"] = time.perf_counter() - started
            pipeline = backend.pipeline

            class TracedPipeline:
                def run(self, request):
                    request = dict(request)
                    if is_fast:
                        request["vits_parallel_infer"] = False
                    if args.phase == "stream" and job_id != "warmup":
                        request.update(text_split_method="cut5", return_fragment=True)
                    for index, (rate, audio) in enumerate(pipeline.run(request)):
                        if audio.size == 0 or not np.all(np.isfinite(audio)) or not np.any(audio):
                            raise ValueError("Empty, silent, or invalid fragment")
                        target = args.output / f"{job_id}_chunk{index}.wav"
                        sf.write(str(target), audio, rate, subtype="PCM_16")
                        chunks.append({"path": str(target), "ready_seconds": time.perf_counter() - job_start,
                                       "duration_seconds": len(audio) / rate, "sample_rate": rate})
                        yield rate, audio

            backend.pipeline = TracedPipeline()
            job_id, chunks = "warmup", []
            job_start = time.perf_counter()
            backend.synthesize("안녕하세요. 반갑습니다.", args.output / "warmup.wav", args.voice, 1)
            report["warmup_seconds"] = time.perf_counter() - job_start
            emit("ready", seconds=report["model_start_seconds"])
            for index in range(args.repeats):
                job_id, chunks = f"sample{index}", []
                emit("measure_start", index=index)
                job_start = time.perf_counter()
                target = args.output / f"{job_id}.wav"
                backend.synthesize(TEXT, target, args.voice, 1)
                elapsed = time.perf_counter() - job_start
                playhead = chunks[0]["ready_seconds"]
                gaps = []
                for chunk in chunks:
                    gaps.append(max(0, chunk["ready_seconds"] - playhead))
                    playhead = max(playhead, chunk["ready_seconds"]) + chunk["duration_seconds"]
                audio, rate = sf.read(str(target))
                item = {"index": index, "audio": str(target), "total_seconds": elapsed,
                        "first_audio_seconds": chunks[0]["ready_seconds"],
                        "audio_seconds": len(audio) / rate, "chunks": chunks,
                        "predicted_gap_seconds": sum(gaps), "predicted_playback_end_seconds": playhead,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
                report["samples"].append(item)
                emit("measure_end", **item)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
        emit("error", message=report["error"])
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if "error" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
