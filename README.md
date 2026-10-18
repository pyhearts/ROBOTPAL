# ROBOTPAL 음성·영상 API 행동 계획

화자 등록 없이 모든 화자의 음성을 받아 한국어로 전사합니다. `--assist`에서는 전사된 명령과 최신 카메라 프레임 한 장을 OpenAI API로 보내 8족 로봇의 상위 행동 계획을 생성합니다. 생성된 계획은 출력하며 실제 로봇 제어로 연결되지는 않습니다.

## 설치

Python 3.12 환경에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

음성 분리는 로컬 SepFormer, 전사는 로컬 Whisper small을 사용합니다. 최초 실행에는 모델 다운로드를 위한 인터넷 연결과 저장 공간이 필요합니다. 기본 실행에서는 ECAPA 화자 지문 모델이나 등록 파일을 사용하지 않습니다. Ollama 설치·실행도 필요하지 않습니다.

## OpenAI API 설정

실행할 PowerShell 창에서 API 키를 환경 변수로 설정합니다. 아래 예시의 값을 본인의 키로 바꾸세요. 키를 코드나 채팅에 붙여 넣지 마세요.

```powershell
$env:OPENAI_API_KEY = "본인의_API_키"
$env:OPENAI_MODEL = "gpt-5.4-mini"
```

기본 모델은 `gpt-5.4-mini`이며 `OPENAI_MODEL`은 생략할 수 있습니다. 이미지 입력과 구조화된 응답을 지원하는 모델을 사용하세요. 위 설정은 현재 PowerShell 세션에 적용됩니다. `.env` 파일을 자동으로 읽지는 않습니다. 기존 `OLLAMA_HOST`, `OLLAMA_MODEL`은 사용하지 않습니다.

행동 계획기는 공식 OpenAI 엔드포인트 `https://api.openai.com/v1`의 Responses API를 사용합니다. 텍스트 명령과 `--assist`의 JPEG 한 장이 OpenAI로 전송되며 API 사용료가 발생합니다. 원본 마이크 오디오는 API로 보내지 않습니다. 요청에는 `store=False`를 지정하며, 서비스의 별도 데이터 보관 정책까지 비활성화한다는 의미는 아닙니다.

## 실행

### 음성·영상으로 행동 계획 생성

```powershell
.\.venv\Scripts\python.exe .\main.py --assist

# 두 번째 카메라
.\.venv\Scripts\python.exe .\main.py --assist --camera 1
```

등록 여부와 관계없이 모든 비어 있지 않은 전사 결과를 명령으로 받습니다. 화자 이름 대신 `음성 스트림 1`, `음성 스트림 2`를 표시합니다. 이 번호는 분리 채널이며 특정 사람의 고정 신원이 아닙니다.

카메라 작업자는 최신 프레임 하나를 메모리에 유지합니다. 전사 완료 콜백에서 프레임을 JPEG로 복사하고 명령과 함께 FIFO 큐에 저장합니다. 각 명령은 접수 당시 이미지를 유지하며 반복 발화도 별도로 처리합니다. 화자 이름·음성 지문은 API 요청에 포함하지 않습니다.

카메라에 오류가 있거나 프레임이 없거나 갱신된 지 1초가 넘으면 해당 명령은 보내지 않습니다. API 인증·모델 접근·요청 제한·응답 형식 오류를 표시하고 다음 명령을 계속 처리합니다. 요청별 제한 시간은 120초이고 자동 재시도는 끕니다. 처리 속도보다 명령이 많으면 대기가 늘어날 수 있습니다.

Ctrl+C 시 남은 음성 처리를 마치고, 카메라를 종료한 뒤 이미 접수된 AI 명령을 순서대로 처리하고 종료합니다. 영상·JPEG·원본 음성 파일은 디스크에 저장하지 않습니다.

### 텍스트로 행동 계획 생성

```powershell
.\.venv\Scripts\python.exe .\main.py --plan --command "앞으로 30cm 이동해"

# 대화형 입력
.\.venv\Scripts\python.exe .\main.py --plan
```

`ai_action_planner.py`도 동일한 API 계획기를 사용합니다. Python에서는 `from module.action_planner import create_action_plan`으로 가져오며, `create_action_plan(command, image=jpeg_bytes)`로 JPEG를 함께 전달할 수 있습니다. 결과는 `status`, `summary`, `steps`, `error`가 있는 `ActionPlan`입니다.

### 음성 전사만 실행

```powershell
.\.venv\Scripts\python.exe .\main.py
```

화자 등록과 API 키 없이 두 음성 스트림을 분리·전사해 출력합니다. 이 모드는 AI에 명령을 보내지 않습니다. `run_live_transcription`의 사용자 콜백도 사용할 수 있습니다.

### 손 모양 인식

```powershell
.\.venv\Scripts\python.exe .\main.py --gesture --preview
.\.venv\Scripts\python.exe .\main.py --gesture --rule-set numbers --preview
```

손 모양 인식은 별도 모드이며 AI 행동 계획과 자동으로 연결되지 않습니다. 기본 모드는 주먹·손바닥·검지·엄지 아래/위·브이·아이 러브 유를 인식합니다. 숫자 모드는 21개 관절로 숫자 1~5와 OK를 우선 판정합니다. 같은 결과가 400ms 유지되면 한 번 전달하며, 결과가 300ms 이상 사라지면 중복 방지 상태를 해제합니다. 미리보기에서는 Q 또는 Esc, 그 외에는 Ctrl+C로 종료합니다.

## 원본 백업과 화자 등록 복원

수정 전 파일은 `backups/before_api_20261018_105455/`에 보관했습니다. Python 소스, Arduino 스케치, 의존성 목록, README, 기존 순서도와 `.gitignore`가 포함됩니다. `manifest.json`의 SHA-256으로 원본을 확인할 수 있습니다. 대용량 모델과 가상환경은 복제하지 않았고 기존 파일을 그대로 유지합니다.

명령행의 `--enroll`은 비활성화했습니다. 등록·식별 함수는 나중에 재사용할 수 있도록 라이브러리에 남겨두었고, 일반 실행 및 통합 모드에서는 호출하지 않습니다. 나중에 화자 식별만 다시 사용하려면 다음과 같이 명시적으로 등록하고 켤 수 있습니다.

```python
from module.speech import enroll_speakers, run_live_transcription

enroll_speakers()  # 필요할 때 별도로 한 번 실행
run_live_transcription(identify_speakers=True)
```

식별을 다시 켜는 것만으로 AI 명령 권한 제한이 복원되지는 않습니다. 이전의 등록자 전용 로컬 AI 방식 전체를 복원하려면 백업의 관련 Python 파일과 `requirements.txt`를 함께 복원해야 합니다. 복원 전에 현재 수정본도 별도로 보관하세요.

## 구성 및 동작 메모

- `main.py`: 음성 전사, 통합 API 계획, 텍스트 API 계획, 손 모양 인식의 진입점.
- `module/speech.py`: 16 kHz 모노 입력, 4초 분리 창·3초 이동·1초 중첩, 스트림별 발화 감지 및 한국어 전사.
- `module/vision_assistant.py`: 모든 화자의 명령과 최신 JPEG를 묶어 순서대로 처리.
- `module/action_planner.py`: OpenAI API 호출 및 계획 응답 검증.
- `module/gesture.py`, `module/gesture_rules.py`: 손 모양 인식과 숫자·OK 규칙.
- `sketch_oct17a/sketch_oct17a.ino`: 별도로 실행되는 PCA9685 서보 중앙 신호 설정.
- `FLOWCHART.md`: 변경된 구조의 순서도.

각 음성 스트림은 0.7초 침묵으로 발화 종료를 판단합니다. 20초 이상의 발화는 내부 전사 조각으로 나눠도 최종 콜백은 한 번 전달합니다. 화자 지문을 쓰지 않는 기본 모드의 채널 연결은 중첩 파형 비교로 수행합니다. CPU 분리 속도가 3초 이동 간격을 따라가지 못하면 경고합니다.

## 검증

외부 요청이나 실제 장치 구동 없이 회귀 테스트를 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

API 모델·입력 형식 참고: [모델 문서](https://developers.openai.com/api/docs/models/gpt-5.4-mini), [구조화된 응답](https://developers.openai.com/api/docs/guides/structured-outputs), [이미지 입력](https://developers.openai.com/api/docs/guides/images-vision).
