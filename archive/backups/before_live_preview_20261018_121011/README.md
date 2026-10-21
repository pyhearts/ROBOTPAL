# ROBOTPAL 음성·영상 API 행동 계획

화자 등록 없이 마이크 음성을 한국어로 전사하고, OpenAI Decisions로 먼저 발화를 분류합니다. 구체적인 이동·회전 명령만 기존 Responses API로 보내 상위 행동 계획을 생성합니다. 일반 대화·정보 부족·미지원 요청은 정해진 안내문을 출력합니다. 명확한 정지는 고정된 정지 계획을 출력합니다. 실행 중 화면 출력·오류와 분류·계획·API 사용량을 로그에 보관합니다. 실제 로봇 제어로 연결되지는 않습니다.

## 설치

Python 3.12 환경에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

마이크의 16 kHz 모노 입력을 Silero VAD로 감지한 뒤 로컬 Whisper small로 전사합니다. 기본 탐색 크기는 beam 3, 시작 전 보존은 0.4초, 최소 음성은 0.12초입니다. Whisper 내부 VAD는 기본 비활성화해 외부 감지와 중복 절단을 피합니다. 기본 전사 모드와 `--assist`에서는 SepFormer를 불러오거나 실행하지 않고, 4초 창 대기·분리 속도 측정·분리 대기열도 사용하지 않습니다. 최초 실행에는 Whisper 모델 다운로드를 위한 인터넷 연결과 저장 공간이 필요합니다. ECAPA 화자 지문 모델이나 등록 파일도 사용하지 않습니다. Ollama 설치·실행은 필요하지 않습니다.

기존 분리·등록 코드를 나중에 재사용할 수 있도록 SpeechBrain·PyTorch 의존성과 기존 모델 파일은 보존했습니다. 기본 입력 경로는 해당 라이브러리를 직접 가져오지 않습니다.

## OpenAI API 설정

실행할 PowerShell 창에서 API 키를 환경 변수로 설정합니다. 아래 예시의 값을 본인의 키로 바꾸세요. 키를 코드나 채팅에 붙여 넣지 마세요.

```powershell
$env:OPENAI_API_KEY = "본인의_API_키"
$env:OPENAI_MODEL = "gpt-5.4-mini"
```

행동 계획 모델은 `gpt-5.4-mini`이며 `OPENAI_MODEL`은 생략할 수 있습니다. 분류 모델은 Decisions 전용 `gpt-6-luna`로 고정되어 있습니다. OpenAI SDK는 `3.26` 이상 `4` 미만이 필요하며, 현재 프로젝트 가상환경에는 `3.35.1`을 설치했습니다. 이미지 입력과 구조화된 응답을 지원하는 모델을 사용하세요. 위 설정은 현재 PowerShell 세션에 적용됩니다. `.env` 파일을 자동으로 읽지는 않습니다. 기존 `OLLAMA_HOST`, `OLLAMA_MODEL`은 사용하지 않습니다.

분류기는 공식 OpenAI 엔드포인트의 `/decisions`, 계획기는 `/responses`를 사용합니다. 기본 주소는 `https://api.openai.com/v1`입니다. Decisions에는 전사 텍스트만 보내고, 구체적 명령의 계획 생성 때만 텍스트와 `--assist`의 JPEG 한 장을 보냅니다. 비어 있지 않은 발화는 Decisions 사용료가 발생합니다. 일반 대화에서 줄어드는 것은 추가 Responses 호출이며, 전체 비용 감소율은 발화 구성에 따라 달라집니다. 기본 로컬 전사에서는 마이크 오디오를 API로 보내지 않습니다. `--stt-backend openai`를 명시하면 발화 오디오를 OpenAI 전사 API로 보내며 별도 사용료가 발생합니다. Responses 요청에는 `store=False`를 지정하며, 서비스의 별도 데이터 보관 정책까지 비활성화한다는 의미는 아닙니다.

## 실행

### 음성·영상으로 행동 계획 생성

```powershell
.\.venv\Scripts\python.exe .\main.py --assist

# 두 번째 카메라
.\.venv\Scripts\python.exe .\main.py --assist --camera 1
```

등록 여부와 관계없이 모든 비어 있지 않은 전사 결과를 분류 대상으로 받습니다. 출력 라벨은 `[음성 입력]`이며 화자를 나누거나 식별하지 않습니다. 여러 사람이 동시에 말하면 하나의 섞인 입력으로 전사되므로 한 번에 한 사람이 말하는 용도로 사용하세요.

카메라 작업자는 최신 프레임 하나를 메모리에 유지합니다. 전사 완료 콜백에서 프레임을 JPEG로 복사하고 명령과 함께 FIFO 큐에 저장합니다. 각 명령은 접수 당시 이미지를 유지하며 반복 발화도 별도로 처리합니다. 화자 이름·음성 지문은 API 요청에 포함하지 않습니다.

카메라에 오류가 있거나 프레임이 없거나 갱신된 지 1초가 넘으면 이미지가 없는 상태로 텍스트 분류만 수행합니다. 카메라가 시작되지 않아도 일반 대화 안내와 정지 계획은 처리할 수 있습니다. 구체적인 행동 요청인데 접수 당시 영상이 없으면 고정 오류를 출력하고 계획 API를 호출하지 않습니다. API 인증·모델 접근·요청 제한·응답 형식 오류를 표시하고 다음 발화를 계속 처리합니다. Decisions 요청 제한 시간은 20초, Responses 요청은 120초이며 자동 재시도는 없습니다. 분류 실패 시 Responses로 우회하지 않습니다. 처리 속도보다 발화가 많으면 대기가 늘어날 수 있습니다.

Ctrl+C 시 남은 음성 처리를 마치고, 카메라를 종료한 뒤 이미 접수된 AI 명령을 순서대로 처리하고 종료합니다. 영상·JPEG·원본 음성 파일은 디스크에 저장하지 않습니다.

### 텍스트로 행동 계획 생성

```powershell
.\.venv\Scripts\python.exe .\main.py --plan --command "앞으로 30cm 이동해"

# 대화형 입력
.\.venv\Scripts\python.exe .\main.py --plan
```

`main.py --plan`과 `ai_action_planner.py`도 Decisions 분류 후 필요한 경우에만 계획을 생성합니다. Python 코드에서는 다음의 고수준 함수를 사용하면 분류와 로그를 함께 적용할 수 있습니다.

```python
from module.decision_router import process_command, print_command_result
from module.session_log import capture_output

with capture_output():
    result = process_command("앞으로 30cm 이동해")
    print_command_result(result)
```

`CommandResult`에는 `decision`, 선택적인 `plan`, 고정 안내문인 `message`가 있습니다. 저수준 `module.action_planner.create_action_plan`은 분류를 생략하는 기존 계획 함수로 유지합니다. 비용 절감 경로를 쓰려면 `process_command`를 사용하세요.

### 분류 결과별 처리

고정 문구는 `module/decision_router.py`의 `FIXED_MESSAGES`에서 수정할 수 있습니다. 문구 생성용 API나 음성 합성은 호출하지 않으며 콘솔에 출력합니다.

| 분류 | 동작 및 고정 출력 |
|---|---|
| `concrete_command` | 방향·종료 조건이 명확한 이동·회전 명령만 Responses로 전달 |
| `stop_command` | 별도 Responses 호출 없이 정해진 정지 계획 출력 |
| `not_command` | 구체적인 로봇 명령이 아니어서 행동 계획을 생성하지 않습니다. |
| `needs_details` | 이동 거리나 시간, 회전 각도 등 구체적인 행동 조건을 알려주세요. |
| `unsupported` | 현재 지원하지 않는 행동입니다. 이동·회전·정지 명령을 사용해 주세요. |
| `uncertain` | 명령을 명확히 이해하지 못했습니다. 구체적으로 다시 말해 주세요. |

분류 확신도가 0.8 미만이거나 모델이 판단을 거절하면 불확실 안내를 출력합니다. 이 값은 프로젝트에서 선택한 초기 기준이며 실제 한국어 발화로 정확도를 검증한 값은 아닙니다. `멈추지 마`, 질문·인용, 복합 명령도 구분하도록 분류 지침을 제공합니다. 실제 정확도는 모델 판단과 전사 정확도에 영향을 받습니다. 정지 계획은 접수 순서대로 출력하며 실제 하드웨어 정지나 긴급 정지를 구현하지 않습니다.

### 로그 저장

`main.py`의 모든 모드와 `ai_action_planner.py`는 실행 시작부터 콘솔의 Python 표준 출력·표준 오류를 로그에도 저장합니다. 시작 시 절대 로그 경로를 출력합니다.

```text
logs/
  YYYYMMDD_HHMMSS_고유번호.log
  YYYYMMDD_HHMMSS_고유번호.jsonl
```

- `.log`: 시각·출력 종류와 함께 안내문, 전사, 분류, 계획, 경고, 오류를 기록합니다.
- `.jsonl`: 한 줄당 JSON 하나로 입력, 분류의 원본 선택지·확률·확신도, 계획 API 응답, 최종 결과, 각 API의 토큰 사용량과 소요 시간, 화면 출력을 기록합니다. 마이크 음량·클리핑, 발화 감지 결과, 전사 구간별 품질 점수·거절 사유·처리 시간도 포함합니다.
- 시각은 한국 시간 `+09:00`이고, 요청별 `command_id`로 입력·분류·계획·결과를 연결합니다.
- 실행마다 새 파일을 만들고 즉시 flush합니다. 자동 삭제하거나 덮어쓰지 않습니다.
- API 키는 로그에서 가립니다. 원본 오디오·JPEG·이미지 base64는 로그에 저장하지 않습니다.
- 로그에는 전사 내용이 남습니다. `logs/`는 Git에서 제외했습니다.
- 라이브러리를 별도로 호출하는 프로그램에서는 위 예시처럼 `capture_output()`으로 감싸야 출력 기록이 활성화됩니다. 다른 프로세스나 장치 펌웨어의 출력은 이 Python 로그에 포함되지 않습니다.

### 음성 전사만 실행

```powershell
.\.venv\Scripts\python.exe .\main.py
```

기본 로컬 전사는 화자 등록과 API 키 없이 마이크 음성을 Silero로 감지·전사해 출력합니다. 이 모드는 AI에 명령을 보내지 않습니다. `run_live_transcription`의 사용자 콜백도 사용할 수 있습니다.

### 음성 인식 설정 및 진단

```powershell
# 마이크 장치 번호 확인 (녹음 없음)
.\.venv\Scripts\python.exe .\main.py --list-mics

# 5초 진단: 음량·클리핑만 기록하며 원본 파일은 저장하지 않음
.\.venv\Scripts\python.exe .\main.py --diagnose-mic --input-device 1

# 마이크 선택 및 음량 표시 후 실행
.\.venv\Scripts\python.exe .\main.py --assist --input-device 1 --show-levels

# 로컬 전사 탐색 크기 변경
.\.venv\Scripts\python.exe .\main.py --stt-beam 5 --input-device 1
```

장치 번호는 `--list-mics` 결과에 맞게 바꾸세요. 전사 점수가 낮거나 무음으로 판단되거나 반복이 심하면 “잘 듣지 못했습니다. 구체적인 명령을 다시 말해 주세요.”를 출력하고 Decisions에 전달하지 않습니다. 이 품질 기준은 초기 경험값이며 실제 정답 확률을 보장하지 않습니다.

[음성 인식 설정·모델 비교 안내](SPEECH_GUIDE.md)에 모델 선택, 선택적 API 전사, 평가용 녹음 및 동일 음성 비교 방법을 정리했습니다. 설정 비교 보고서는 `reports/`에, 음량·발화·전사 품질은 기존 `logs/`에 기록합니다.

### 손 모양 인식

```powershell
.\.venv\Scripts\python.exe .\main.py --gesture --preview
.\.venv\Scripts\python.exe .\main.py --gesture --rule-set numbers --preview
```

손 모양 인식은 별도 모드이며 AI 행동 계획과 자동으로 연결되지 않습니다. 기본 모드는 주먹·손바닥·검지·엄지 아래/위·브이·아이 러브 유를 인식합니다. 숫자 모드는 21개 관절로 숫자 1~5와 OK를 우선 판정합니다. 같은 결과가 400ms 유지되면 한 번 전달하며, 결과가 300ms 이상 사라지면 중복 방지 상태를 해제합니다. 미리보기에서는 Q 또는 Esc, 그 외에는 Ctrl+C로 종료합니다.

## 원본 백업과 화자 등록 복원

수정 전 파일은 `backups/before_api_20261018_105455/`에 보관했습니다. Python 소스, Arduino 스케치, 의존성 목록, README, 기존 순서도와 `.gitignore`가 포함됩니다. `manifest.json`의 SHA-256으로 원본을 확인할 수 있습니다. 대용량 모델과 가상환경은 복제하지 않았고 기존 파일을 그대로 유지합니다.

명령행의 `--enroll`은 비활성화했습니다. 등록·식별 및 SepFormer 함수는 나중에 재사용할 수 있도록 라이브러리에 남겨두었고, 일반 실행 및 통합 모드에서는 호출하지 않습니다. 직접 마이크 입력으로 전환하기 직전의 API 버전은 `backups/before_direct_mic_20261018_110826/`에 추가로 백업했습니다. 해당 `manifest.json`에 원본 SHA-256을 기록했습니다. Decisions·로그 적용 직전 버전은 `backups/before_decisions_20261018_113345/`에 별도로 보관했습니다. Silero·음성 품질 개선 직전 버전은 `backups/before_stt_quality_20261018_115215/`에 보관했습니다.

기존 두 스트림 분리·화자 식별 방식을 명시적으로 다시 사용하려면 다음과 같이 실행할 수 있습니다. 이 경로에는 기존 CPU 분리 지연이 다시 발생할 수 있습니다.

```python
from module.speech import enroll_speakers, run_separated_transcription

enroll_speakers()  # 필요할 때 별도로 한 번 실행
run_separated_transcription(identify_speakers=True)
```

식별을 다시 켜는 것만으로 AI 명령 권한 제한이 복원되지는 않습니다. 이전의 등록자 전용 로컬 AI 방식 전체를 복원하려면 백업의 관련 Python 파일과 `requirements.txt`를 함께 복원해야 합니다. 복원 전에 현재 수정본도 별도로 보관하세요.

## 구성 및 동작 메모

- `main.py`: 음성 전사, 통합 API 계획, 텍스트 API 계획, 손 모양 인식의 진입점.
- `module/speech.py`: 16 kHz 모노 입력을 Silero 발화 감지 및 한국어 전사로 전달. 기존 분리 경로는 명시적 재사용용으로 보존.
- `module/speech_quality.py`: 입력 장치·음량 진단, Silero 스트리밍 확률, 전사 설정·품질 검사, 선택적 API 전사.
- `module/speech_evaluation.py`: 동일 녹음의 설정별 비교와 문자 오류·핵심 명령 조건 평가.
- `module/vision_assistant.py`: 모든 화자의 명령과 최신 JPEG를 묶어 순서대로 처리.
- `module/action_planner.py`: OpenAI Responses 호출 및 계획 응답 검증.
- `module/decision_router.py`: Decisions 분류, 고정 안내, 필요한 경우에만 계획 API 호출.
- `module/session_log.py`: 실행 출력과 구조화된 결과·사용량 로그.
- `module/gesture.py`, `module/gesture_rules.py`: 손 모양 인식과 숫자·OK 규칙.
- `sketch_oct17a/sketch_oct17a.ino`: 별도로 실행되는 PCA9685 서보 중앙 신호 설정.
- `FLOWCHART.md`: 변경된 구조의 순서도.

마이크 입력을 기본 1024샘플 블록(16 kHz에서 약 64ms)으로 받아 0.7초 침묵으로 발화 종료를 판단합니다. Silero 음성 확률로 감지한 구간이 기본 0.12초보다 짧으면 버리며 `--speech-min-duration`으로 조절할 수 있습니다. 20초 이상의 발화는 내부 전사 조각으로 나눠도 최종 콜백은 한 번 전달합니다. 전사는 별도 작업자에서 처리하며 Ctrl+C 시 수집된 입력과 끝나지 않은 발화를 마무리합니다. 긴 발화의 어느 한 조각이라도 품질 검사를 통과하지 못하면 나머지만 명령으로 보내지 않고 발화 전체를 보류합니다.

SepFormer 병목은 제거했지만 Whisper 전사나 API 응답이 느리면 대기할 수 있습니다. 대기열은 입력을 자동으로 버리지 않습니다. 실제 응답 속도는 CPU와 발화 길이에 따라 달라집니다.

## 검증

외부 요청이나 실제 장치 구동 없이 회귀 테스트를 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

API 모델·입력 형식 참고: [모델 문서](https://developers.openai.com/api/docs/models/gpt-5.4-mini), [구조화된 응답](https://developers.openai.com/api/docs/guides/structured-outputs), [이미지 입력](https://developers.openai.com/api/docs/guides/images-vision).

Decisions API 참고: [공식 안내](https://developers.openai.com/api/docs/guides/decisions), [Python API 명세](https://developers.openai.com/api/reference/python/resources/decisions/methods/create).
