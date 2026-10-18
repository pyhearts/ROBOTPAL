"""Compare recognition settings against the same explicitly supplied audio set."""

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
import wave
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .session_log import KST, record_event
from .speech_quality import SpeechOptions, SileroProbability, options_dict


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def normalized_text(text: str) -> str:
    return "".join(char for char in text.lower()
                   if not char.isspace() and not unicodedata.category(char).startswith("P"))


def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        digits = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
        units = {"십": 10, "백": 100, "천": 1000}
        total = digit = 0
        for char in text:
            if char in digits:
                digit = digits[char]
            else:
                total += (digit or 1) * units[char]
                digit = 0
        return float(total + digit)


def command_fields(text: str) -> dict[str, Any]:
    """Transparent evaluation rules, never used to alter or execute live commands."""
    compact = re.sub(r"\s", "", text.lower())
    result: dict[str, Any] = {}
    result["negated"] = any(word in compact for word in ["하지마", "하지말", "추지마", "추지말"])
    if any(word in compact for word in ["멈", "정지", "그만"]):
        result["action"] = "stop"
    elif any(word in compact for word in ["회전", "돌아", "돌아라", "돌려"]):
        result["action"] = "turn"
    elif any(word in compact for word in ["이동", "전진", "후진", "앞으로", "뒤로"]):
        result["action"] = "move"
    for direction, words in [("forward", ["앞", "전진"]), ("backward", ["뒤", "후진"]),
                             ("left", ["왼", "좌회전"]), ("right", ["오른", "우회전"])]:
        if any(word in compact for word in words):
            result["direction"] = direction
            break
    for match in re.finditer(r"(\d+(?:\.\d+)?|[일이삼사오육칠팔구십백천]+)(센티미터|미터|cm|m|도|초)", compact):
        number, unit = _number(match[1]), match[2]
        if unit in {"미터", "m"}:
            result["distance_cm"] = number * 100
        elif unit in {"센티미터", "cm"}:
            result["distance_cm"] = number
        elif unit == "도":
            result["angle_deg"] = number
        else:
            result["duration_seconds"] = number
    return result


def load_evaluation_audio(path: Path) -> np.ndarray:
    """Read recorded PCM natively; use PyAV's current resampler for other formats."""
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() == 2 and wav.getframerate() == 16000 and wav.getcomptype() == "NONE":
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
                return samples.reshape(-1, wav.getnchannels()).mean(axis=1)
    except (wave.Error, EOFError):
        pass
    import av
    frames = []
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    with av.open(str(path), mode="r") as container:
        for decoded in container.decode(audio=0):
            frames.extend(frame.to_ndarray().reshape(-1).copy() for frame in resampler.resample(decoded))
        frames.extend(frame.to_ndarray().reshape(-1).copy() for frame in resampler.resample(None))
    return np.concatenate(frames).astype(np.float32) / 32768 if frames else np.empty(0, dtype=np.float32)


def compare_transcription_settings(manifest: Path, *, options: SpeechOptions,
                                   models: list[str], beams: list[int], offline: bool = False) -> Path:
    from .speech import MonoUtteranceDetector, create_speech_transcriber

    rows = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("audio"), str) or not isinstance(row.get("text"), str):
            raise ValueError(f"평가 목록 {number}행에는 audio 경로와 text 정답이 필요합니다.")
        if "fields" in row and not isinstance(row["fields"], dict):
            raise ValueError(f"평가 목록 {number}행의 fields는 객체여야 합니다.")
        row["path"] = (manifest.parent / row["audio"]).resolve()
        if not row["path"].is_file():
            raise ValueError(f"평가 음성 파일을 찾을 수 없습니다: {row['path']}")
        rows.append(row)
    if not rows:
        raise ValueError("평가 목록이 비어 있습니다.")
    if not models or not beams or any(not model.strip() for model in models) or any(not 1 <= beam <= 10 for beam in beams):
        raise ValueError("비교할 모델과 beam 크기(1~10)를 확인해 주세요.")
    if options.backend == "openai":
        if models != ["gpt-transcribe"] or len(beams) != 1:
            raise ValueError("API 평가는 중복 과금을 막기 위해 모델·beam 조합 하나씩만 실행합니다.")
        print("API 평가 선택: 평가 목록의 발화 오디오를 전송하며 사용료가 발생합니다.")
    report: dict[str, Any] = {"manifest": str(manifest.resolve()), "profiles": []}
    for model in models:
        for beam in beams:
            profile = replace(options, model=model, beam_size=beam)
            recognizer = create_speech_transcriber(profile, offline=offline)
            results = []
            try:
                for row in rows:
                    samples = load_evaluation_audio(row["path"])
                    started = time.monotonic()
                    # Run the same outer VAD as live capture, resetting state for each file.
                    detector = MonoUtteranceDetector(probability=SileroProbability(), options=profile)
                    padding = np.zeros(round((profile.silence + 0.064) * 16000), dtype=np.float32)
                    tasks = detector.process(np.concatenate([samples, padding]), 0)
                    tasks += detector.flush()
                    parts, failed = [], False
                    for task in tasks:
                        result = recognizer.transcribe_audio(task.audio)
                        failed = failed or not result.accepted
                        parts.append(result.text)
                    prediction = "" if failed else " ".join(parts).strip()
                    reference, normalized = normalized_text(row["text"]), normalized_text(prediction)
                    found = command_fields(prediction)
                    expected_fields = row.get("fields")
                    result_row = {"audio": row["audio"], "reference": row["text"], "prediction": prediction,
                                  "rejected": failed, "no_speech_detected": not tasks,
                                  "char_errors": edit_distance(reference, normalized),
                                  "reference_chars": len(reference), "exact_match": reference == normalized,
                                  "fields": found, "expected_fields": expected_fields,
                                  "fields_match": None if not expected_fields else all(found.get(k) == v for k, v in expected_fields.items()),
                                  "evaluation_seconds": time.monotonic() - started}
                    results.append(result_row)
                    record_event("stt_evaluation_sample", model=model, beam=beam, **result_row)
            finally:
                recognizer.close()
            annotated = [row for row in results if row["fields_match"] is not None]
            reference_chars = sum(row["reference_chars"] for row in results)
            errors = sum(row["char_errors"] for row in results)
            profile_report = {"options": options_dict(profile), "samples": results,
                              "character_error_rate": errors / reference_chars if reference_chars else None,
                              "exact_match_rate": sum(row["exact_match"] for row in results) / len(results),
                              "field_accuracy": sum(row["fields_match"] for row in annotated) / len(annotated) if annotated else None,
                              "noise_false_positives": sum(not row["reference"] and bool(row["prediction"]) for row in results),
                              "median_evaluation_seconds": float(np.median([row["evaluation_seconds"] for row in results])),
                              "p95_evaluation_seconds": float(np.percentile([row["evaluation_seconds"] for row in results], 95))}
            report["profiles"].append(profile_report)
            print(f"비교 결과: {options.backend}/{model} beam={beam} / 문장 일치 {profile_report['exact_match_rate']:.1%} / "
                  f"문자 오류율 {profile_report['character_error_rate']} / 핵심 조건 일치 {profile_report['field_accuracy']}")
    root = Path(__file__).resolve().parent.parent / "reports"
    root.mkdir(exist_ok=True)
    output = root / ("stt_" + datetime.now(KST).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + ".json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"설정 비교 보고서: {output}")
    record_event("stt_evaluation_report", path=str(output))
    return output
