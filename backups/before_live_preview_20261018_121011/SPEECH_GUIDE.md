# 음성 인식 설정과 비교 평가

기본 실행은 마이크 → Silero 음성 감지 → 로컬 Whisper small → 품질 검사 순서입니다. `--assist`에서는 품질 검사를 통과한 전사만 Decisions로 전달합니다. 실제 한국어 인식 정확도 개선 폭은 같은 녹음으로 비교해야 합니다.

## 먼저 장치와 음량 확인

```powershell
.\.venv\Scripts\python.exe .\main.py --list-mics
.\.venv\Scripts\python.exe .\main.py --diagnose-mic --input-device 1 --sample-seconds 5
.\.venv\Scripts\python.exe .\main.py --assist --input-device 1 --show-levels
```

`1`은 예시이며 목록의 실제 마이크 번호를 선택하세요. 2026-10-18 확인 당시 이 환경의 기본 입력은 1번이고 CUDA 장치는 검출되지 않았습니다. 장치 목록은 연결 상태에 따라 달라질 수 있습니다. 선택한 장치의 16 kHz 모노 지원 여부를 실행 전에 검사합니다.

진단 모드는 음량 RMS·최대 진폭·클리핑 비율을 출력하며 파일을 저장하지 않습니다. 말했는데 입력이 매우 작으면 마이크 선택·입력 감도·거리를 확인하세요. 클리핑 경고가 나면 감도를 낮추거나 조금 떨어져 말하세요. 음량 수치만으로 인식 정확도를 판단하거나 입력을 무조건 증폭하지는 않습니다.

## 기본 변경점

| 설정 | 이전 | 현재 기본 |
|---|---|---|
| 외부 발화 감지 | 고정 RMS 기준 | Silero v6 음성 확률, 0.5 기준 |
| 발화 시작 전 보존 | 약 0.2초 | 약 0.4초 |
| 최소 유효 음성 | 0.25초 | 0.12초 |
| 발화 종료 침묵 | 0.7초 | 0.7초 |
| Whisper 탐색 크기 | beam 1 | beam 3 |
| 내부 음성 필터 | 활성 | 기본 비활성, 비교 옵션 제공 |
| 용어 힌트 | 없음 | 방향·정지·거리·각도·시간 용어 |
| 전사 품질 검사 | 없음 | 점수·무음 확률·반복 검사 |

Silero ONNX 모델은 faster-whisper에 포함된 파일을 사용하므로 별도 음성 분리 모델 다운로드가 없습니다. 512샘플(32ms) 프레임 사이에서 모델 상태를 유지합니다. 마이크 콜백에서는 복사와 큐 전달만 하고, 감지·로그·전사는 콜백 밖에서 처리합니다. 작은 음량이라는 이유만으로 새 발화 감지에서 거절하지 않습니다.

## 로컬 모델과 설정 선택

```powershell
# 기본 로컬, 명령 해석 탐색 크기 비교
.\.venv\Scripts\python.exe .\main.py --stt-model small --stt-beam 3
.\.venv\Scripts\python.exe .\main.py --stt-model small --stt-beam 5

# 처리 시간이 길면 탐색 크기를 1로 비교
.\.venv\Scripts\python.exe .\main.py --stt-beam 1

# 더 큰 모델: 첫 실행에 다운로드가 필요할 수 있음
.\.venv\Scripts\python.exe .\main.py --stt-model medium --stt-beam 3

# CUDA GPU 및 관련 라이브러리가 준비된 환경에서만 사용
.\.venv\Scripts\python.exe .\main.py --stt-model turbo --stt-device cuda --stt-compute float16
```

현재 환경에서는 CPU small을 기본으로 유지합니다. 큰 모델이나 큰 beam이 항상 이 프로젝트의 한국어 명령을 더 정확히 인식한다고 보장하지 않습니다. 정확도와 지연을 함께 비교하세요. `--stt-offline`을 붙이면 이미 준비된 로컬 모델만 사용하며 자동으로 다른 모델이나 API로 전환하지 않습니다.

발화 감지와 내부 필터도 조절할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe .\main.py --vad-threshold 0.5 --speech-pre-roll 0.4 --speech-silence 0.7 --speech-min-duration 0.12

# Whisper 내부 VAD를 함께 적용한 경우와 비교
.\.venv\Scripts\python.exe .\main.py --whisper-vad
```

음성 확률 기준을 낮추면 약한 음성을 잡을 가능성과 잡음 오인의 가능성이 함께 달라집니다. 종료 침묵을 줄이면 반응은 빨라지지만 문장 중간의 짧은 쉼을 발화 종료로 볼 수 있습니다. 설정을 한 번에 모두 바꾸지 말고 비교 결과로 선택하세요.

## 품질 검사와 재발화

로컬 전사는 한국어 지정, temperature 0, 조각 사이의 이전 텍스트 조건 비활성화, 짧은 용어 힌트를 사용합니다. 특정 거리나 각도를 예시 답변으로 강제하지 않습니다.

다음 조건 중 하나면 원본 전사 내용과 거절 사유를 로그에 남기고 “잘 듣지 못했습니다. 구체적인 명령을 다시 말해 주세요.”를 출력합니다. Decisions와 계획 API는 호출하지 않습니다.

- 평균 log probability가 기본 `-1.2`보다 낮음
- 무음 확률이 기본 `0.7`보다 높음
- 반복을 나타내는 compression ratio가 `2.4`보다 높음
- 빈 결과·유효하지 않은 점수·전사 오류

20초 이상의 발화는 내부 조각을 전사하되 한 조각이라도 거절되면 나머지 조각만 명령으로 전달하지 않습니다. 숫자나 방향을 추측해서 고치지 않습니다. 위 기준은 초기 경험값이며 정답 확률이나 안전 인증이 아닙니다.

```powershell
# 점수 로그를 비교한 뒤 조정할 때 사용
.\.venv\Scripts\python.exe .\main.py --stt-min-logprob -1.2 --stt-max-no-speech 0.7
```

`logs/*.jsonl`에 입력 장치, 매초 음량·클리핑, 발화 감지, 전사 구간별 원문·점수·거절 사유·추론 시간·오디오 길이 대비 처리 시간 비율을 기록합니다. `stt_utterance`의 `end_to_transcript_seconds`는 감지된 마지막 음성 프레임부터 전사 완료까지의 시간으로, 발화 종료 침묵 대기를 포함합니다. Decisions의 분류 확신도와 전사의 품질 점수는 서로 다른 값입니다.

## 같은 녹음으로 설정 비교

일반 실행에서는 원본 음성을 저장하지 않습니다. 아래 명령처럼 경로를 명시한 경우에만 평가용 WAV를 저장합니다. 기존 파일은 덮어쓰지 않습니다.

```powershell
# 녹음 안내 후 해당 문장을 말하세요. 마지막 noise.wav에서는 조용히 기다리세요.
.\.venv\Scripts\python.exe .\main.py --record-sample evaluation_audio/forward30.wav --sample-seconds 3
.\.venv\Scripts\python.exe .\main.py --record-sample evaluation_audio/left90.wav --sample-seconds 3
.\.venv\Scripts\python.exe .\main.py --record-sample evaluation_audio/stop.wav --sample-seconds 3
.\.venv\Scripts\python.exe .\main.py --record-sample evaluation_audio/negative.wav --sample-seconds 3
.\.venv\Scripts\python.exe .\main.py --record-sample evaluation_audio/noise.wav --sample-seconds 3
```

`examples/stt_eval.jsonl`에는 위 파일에 맞는 정답과 핵심 조건 예시를 준비했습니다. 각각 “앞으로 30cm 이동해”, “왼쪽으로 90도 돌아”, “멈춰”, “멈추지 마”, 무음입니다. 실제 발화에 맞게 정답을 수정하고 다양한 화자·거리·잡음을 포함해 50~100개 정도로 확장할 수 있습니다.

```powershell
# 동일한 녹음들을 small의 beam 1,3,5로 비교
.\.venv\Scripts\python.exe .\main.py --compare-stt examples/stt_eval.jsonl --compare-models small --compare-beams 1,3,5

# 모델까지 비교하려면 (큰 모델의 CPU 지연·메모리에 유의)
.\.venv\Scripts\python.exe .\main.py --compare-stt examples/stt_eval.jsonl --compare-models small,medium --compare-beams 3
```

`reports/stt_*.json`에 각 파일의 정답·예측·거절 여부, 문자 오류율, 문장 일치율, 방향·거리·각도·정지·부정 표현의 명시된 핵심 조건 일치율, 무음 오인식 수, 처리 시간 중앙값·95백분위수를 기록합니다. 평가 시간에는 파일을 재생하며 기다리는 시간이 없으므로 실제 실시간 응답 시간과는 다릅니다. 모델 초기 로딩 시간은 평가 시간에서 제외합니다. 실제 응답 시간은 라이브 로그로 확인하세요.

핵심 조건 추출은 평가용으로 공개된 단순 규칙이며 실제 명령을 교정하거나 실행하지 않습니다. “삼십 센티미터”와 “30cm” 같은 일부 표기는 수치로 비교하지만 모든 한국어 표현을 처리하는 의미 분석기는 아닙니다. 잘못 분석한 경우 원문과 예측을 함께 검토하세요. 비교 기능은 Decisions나 행동 계획 API를 호출하지 않습니다.

## 선택적인 OpenAI 전사

```powershell
# 명시적으로 선택할 때만 마이크 발화 오디오를 외부 전사 API로 보냅니다.
.\.venv\Scripts\python.exe .\main.py --assist --stt-backend openai --input-device 1

# 준비한 동일 녹음의 API 전사 결과를 로컬 보고서와 비교
.\.venv\Scripts\python.exe .\main.py --compare-stt examples/stt_eval.jsonl --stt-backend openai
```

`gpt-transcribe`에 한국어 및 용어 힌트와 메모리에서 만든 WAV를 전달합니다. 외부 VAD로 음성 구간을 고른 후 호출하며, 파일은 디스크에 임시 저장하지 않습니다. 전사 API 사용료와 음성 전송이 발생합니다. 기본은 로컬이고, 로컬 실패 시 자동 API 재시도도 없습니다. API 평가에서는 중복 과금을 막기 위해 한 번에 한 조합만 허용합니다.

API 응답의 전사·사용량도 로그에 남기지만 로컬 Whisper의 log probability·무음 확률과 같은 품질 점수를 제공한다고 가정하지 않습니다. API 요청 제한 시간은 60초이고 자동 재시도는 없습니다. 원본 오디오와 이미지 데이터는 로그에 기록하지 않습니다.

공식 자료: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Silero 코드](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py), [OpenAI 전사 API](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create).
