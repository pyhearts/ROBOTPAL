"""Compare CPU threads, CPUFast, and sentence-first audio without changing app defaults."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from module.session_log import capture_output, record_event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/tts_compare_20261018")
    parser.add_argument("--voice", type=Path, default=ROOT / "data/tts/female_voice.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    transcriber = None
    stt_samples = None
    report = {"voice": str(args.voice.resolve()), "repeats": args.repeats, "conditions": results,
              "playback_measured": False, "ai_response_streaming_measured": False,
              "subjective_voice_quality_verified": False, "application_defaults_changed": False}

    def run_case(engine, threads, phase="solo"):
        name = f"case_{engine}_t{threads}" + ("_" + phase if phase != "solo" else "")
        source = ROOT / ".gsv-sources" / ("GPT-SoVITS-CPUFast" if engine == "cpufast" else "GPT-SoVITS")
        output = folder / name
        result_file = output / "result.json"
        if result_file.exists():
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if ("error" not in result
                    and result.get("voice_sha256") == hashlib.sha256(args.voice.read_bytes()).hexdigest()
                    and result.get("model_manifest_sha256") == hashlib.sha256((ROOT / "models/reference_tts/manifest.json").read_bytes()).hexdigest()
                    and result.get("source_revision") == subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
                    and len(result["samples"]) == (2 if phase == "probe" else args.repeats)):
                result.update(engine=engine, name=name,
                              source_revision=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip())
                result["median_total_seconds"] = statistics.median(item["total_seconds"] for item in result["samples"])
                result["median_first_audio_seconds"] = statistics.median(item["first_audio_seconds"] for item in result["samples"])
                result["median_predicted_gap_seconds"] = statistics.median(item["predicted_gap_seconds"] for item in result["samples"])
                results.append(result)
                print(f"기존 측정 재사용: {name}", flush=True)
                return result
        command = [str(ROOT / ".gsv-venv/Scripts/python.exe"), "-u", str(ROOT / "tools/tts_comparison_worker.py"),
                   "--source", str(source), "--voice", str(args.voice.resolve()), "--output", str(output),
                   "--threads", str(threads), "--phase", "stream" if phase == "stream" else "probe" if phase == "probe" else "solo",
                   "--repeats", str(2 if phase == "probe" else args.repeats)]
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        env.pop("OPENAI_API_KEY", None)
        probe_results, probe_threads = [], []
        print(f"측정 시작: {name}", flush=True)
        with (folder / (name + ".log")).open("w", encoding="utf-8") as diagnostic:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=diagnostic,
                                       encoding="utf-8", env=env, creationflags=subprocess.CREATE_NO_WINDOW)
            timer = threading.Timer(600, process.kill)
            timer.start()
            try:
                for line in process.stdout:
                    event = json.loads(line)
                    record_event("tts_comparison_worker", condition=name, **{k: v for k, v in event.items() if k != "event"}, worker_event=event["event"])
                    if event["event"] == "measure_start" and phase == "probe":
                        def probe(index=event["index"]):
                            started = time.perf_counter()
                            recognized = transcriber.transcribe_audio(stt_samples)
                            probe_results.append({"index": index, "seconds": time.perf_counter() - started,
                                                  "accepted": recognized.accepted})
                        thread = threading.Thread(target=probe)
                        thread.start()
                        probe_threads.append(thread)
                    if event["event"] == "measure_end":
                        print(f"{name}: 합성 {event['total_seconds']:.2f}초 / 첫 음성 {event['first_audio_seconds']:.2f}초 / 예상 끊김 {event['predicted_gap_seconds']:.2f}초", flush=True)
                process.wait()
            finally:
                timer.cancel()
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()
        for thread in probe_threads:
            thread.join()
        result = json.loads(result_file.read_text(encoding="utf-8"))
        result.update(engine=engine, name=name,
                      source_revision=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip())
        if probe_results:
            result["stt_concurrent"] = sorted(probe_results, key=lambda item: item["index"])
        if "error" not in result:
            result["median_total_seconds"] = statistics.median(item["total_seconds"] for item in result["samples"])
            result["median_first_audio_seconds"] = statistics.median(item["first_audio_seconds"] for item in result["samples"])
            result["median_predicted_gap_seconds"] = statistics.median(item["predicted_gap_seconds"] for item in result["samples"])
        result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)
        if "error" in result:
            raise RuntimeError(f"{name}: {result['error']}")
        return result

    with capture_output():
        try:
            upstream = [run_case("upstream", threads) for threads in [2, 4, 6, 8]]
            # A pre-existing baseline may have been measured directly by the worker.
            for item in upstream:
                if "median_total_seconds" not in item:
                    item["median_total_seconds"] = statistics.median(sample["total_seconds"] for sample in item["samples"])
            best = min(upstream, key=lambda item: item["median_total_seconds"])["threads"]
            report["fastest_upstream_threads"] = best
            fast2 = run_case("cpufast", 2)
            fast_best = fast2 if best == 2 else run_case("cpufast", best)
            run_case("upstream", best, "stream")
            run_case("cpufast", best, "stream")
            from module.speech import create_speech_transcriber
            from module.speech_quality import SpeechOptions
            from module.speech_evaluation import load_evaluation_audio
            profile = json.loads(args.voice.read_text(encoding="utf-8"))
            stt_samples = load_evaluation_audio(Path(profile["references"][0]["audio"]))[:3 * 16000]
            transcriber = create_speech_transcriber(SpeechOptions(live_preview=False), offline=True)
            transcriber.transcribe_audio(stt_samples)
            baseline = []
            for _ in range(3):
                start = time.perf_counter()
                transcriber.transcribe_audio(stt_samples)
                baseline.append(time.perf_counter() - start)
            report["stt_baseline_seconds"] = baseline
            run_case("upstream", 2, "probe")
            if best != 2:
                run_case("upstream", best, "probe")
            run_case("cpufast", best, "probe")
            # Recognition is an intelligibility check, not a human voice-quality score.
            for item in list(results):
                if item["phase"] == "probe":
                    continue
                sample = item["samples"][0]
                recognized = transcriber.transcribe_audio(load_evaluation_audio(Path(sample["audio"])))
                item["verification_transcript"] = recognized.text
                item["verification_accepted"] = recognized.accepted
            report["original_hashes_unchanged"] = all(hashlib.sha256(Path(o["path"]).read_bytes()).hexdigest() == o["sha256"] for o in profile["originals"])
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if transcriber is not None:
                transcriber.close()
            (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"비교 결과: {folder / 'report.json'}")


if __name__ == "__main__":
    main()
