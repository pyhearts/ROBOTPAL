"""Offline synthesis/cache measurements; optionally compare STT during synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from module.local_tts import LocalTTS, TTSOptions, wav_info
from module.session_log import capture_output, command_context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", type=Path, default=ROOT / "data/tts/voice.json")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--compare-stt", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "reports" / ("tts_" + time.strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=False)
    options = TTSOptions(voice=args.voice, cache=folder, threads=args.threads)
    tts = LocalTTS(options)
    metadata = json.loads(args.voice.read_text(encoding="utf-8"))
    tts._voice_version, tts._model_version = metadata["voice_version"], metadata["model_version"]
    report = {"threads": args.threads, "voice": str(args.voice), "samples": [],
              "speaker_playback_tested": False, "network_disabled_in_worker": True}
    transcriber = None
    with capture_output():
        try:
            tts.worker.start()
            if args.compare_stt:
                from module.speech import create_speech_transcriber
                from module.speech_quality import SpeechOptions
                from module.speech_evaluation import load_evaluation_audio
                samples = load_evaluation_audio(Path(metadata["prepared_reference"]))[:3 * 16000]
                transcriber = create_speech_transcriber(SpeechOptions(live_preview=False), offline=True)
                # Warm-up is excluded. Do not publish the person's reference transcript.
                transcriber.transcribe_audio(samples)
                times = []
                for _ in range(2):
                    started = time.monotonic()
                    transcriber.transcribe_audio(samples)
                    times.append(time.monotonic() - started)
                report["stt_baseline_seconds"] = times
            for index, text in enumerate(["안녕하세요. 로봇팔 음성 안내입니다.",
                                          "행동 계획입니다. 왼쪽으로 구십 도 회전합니다.",
                                          "잘 듣지 못했습니다. 다시 말씀해 주세요."]):
                with command_context(uuid.uuid4().hex):
                    concurrent = []
                    if transcriber is not None:
                        def infer():
                            start = time.monotonic()
                            transcriber.transcribe_audio(samples)
                            concurrent.append(time.monotonic() - start)
                        thread = threading.Thread(target=infer)
                        thread.start()
                    path, _, elapsed = tts.synthesize(text)
                    if transcriber is not None:
                        thread.join()
                    start = time.monotonic()
                    _, cached, _ = tts.synthesize(text)
                    cache_seconds = time.monotonic() - start
                    duration, _ = wav_info(path)
                    item = {"text": text, "audio": str(path), "synthesis_seconds": elapsed,
                            "audio_seconds": duration, "real_time_factor": elapsed / duration,
                            "cache_seconds": cache_seconds, "cache_hit": cached,
                            "stt_concurrent_seconds": concurrent}
                    report["samples"].append(item)
                    print(json.dumps(item, ensure_ascii=False), flush=True)
        finally:
            if transcriber is not None:
                transcriber.close()
            tts.close()
            (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"측정 결과: {folder / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
