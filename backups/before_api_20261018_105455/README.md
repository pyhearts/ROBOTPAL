# ROBOTPAL 음성·영상 로봇 행동 계획

16 kHz 모노 마이크 입력을 최대 두 화자로 분리하고, 등록된 두 명 또는 `기타`로 식별해 발화가 끝날 때 한 번만 전달합니다.

## 설치와 실행

Python 3.12 가상환경에서 의존성을 설치합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

최초 실행 또는 화자 변경 시 두 명을 등록합니다. 각 사람은 안내에 따라 약 8초간 말합니다. 원본 음성은 저장하지 않습니다.

```powershell
.\.venv\Scripts\python.exe .\main.py --enroll
```

이후 일반 실행은 다음과 같습니다.

```powershell
.\.venv\Scripts\python.exe .\main.py
```

결과를 다른 기능에 연결하려면 `main`에서 `run_live_transcription`을 가져와 콜백을 전달하거나, 음성 기능만 사용할 때는 `module.speech`에서 직접 가져올 수 있습니다.

## 손 모양 인식

카메라 기능은 `main.py --gesture`로 실행합니다. 최초 실행에는 공식 MediaPipe 모델을 `models/gesture_recognizer.task`로 자동 다운로드하므로 인터넷 연결이 필요합니다.

```powershell
# 화면 없이 기본 카메라 사용
.\.venv\Scripts\python.exe .\main.py --gesture

# 손 관절과 현재 결과 미리보기
.\.venv\Scripts\python.exe .\main.py --gesture --preview

# 두 번째 카메라 사용
.\.venv\Scripts\python.exe .\main.py --gesture --camera 1 --preview

# 숫자 1~5와 OK 모양 우선 인식
.\.venv\Scripts\python.exe .\main.py --gesture --rule-set numbers --preview
```

지원하는 손 모양은 주먹(`Closed_Fist`), 손바닥(`Open_Palm`), 검지(`Pointing_Up`), 엄지 아래/위(`Thumb_Down`, `Thumb_Up`), 브이(`Victory`), 아이 러브 유(`ILoveYou`)입니다. 같은 모양이 약 0.4초 유지되면 한 번만 전달합니다. 로봇 기능과 연결하려면 `main` 또는 `module.gesture`에서 `run_gesture_recognition`을 가져와 콜백을 전달합니다.

`--rule-set numbers`에서는 21개 손 관절 좌표로 `Number_1`부터 `Number_5`, `OK`를 우선 판정합니다. 숫자 3은 검지·중지·약지 또는 엄지·검지·중지 조합을 모두 허용합니다. 규칙 결과의 confidence `1.0`은 학습 모델의 확률이 아니라 규칙에 명확히 일치했다는 뜻입니다. 숫자 모드에서 규칙이 불확실하면 `Pointing_Up`, `Victory`, `Open_Palm`은 출력하지 않아 숫자 라벨과의 흔들림을 방지합니다.

## AI 행동 계획

AI 계획기는 이 컴퓨터의 Ollama 모델로 8족 로봇의 상위 행동 단계를 한국어로 만듭니다. 결과는 계획만 출력하며 서보나 로봇을 직접 제어하지 않습니다. API 키는 필요하지 않습니다.

### 모델과 Ollama 준비

인터넷을 사용할 수 있을 때 [Windows용 Ollama](https://ollama.com/download/windows)를 설치하고, Ollama가 실행 중인 상태에서 모델을 내려받으세요. 기본 모델은 [qwen3.5:4b](https://ollama.com/library/qwen3.5%3A4b)이며 이미지와 텍스트 입력을 지원합니다. 모델 파일 외에도 실행용 RAM 또는 GPU 메모리가 필요합니다.

```powershell
ollama pull qwen3.5:4b
```

Ollama의 클라우드 기능을 끄려면 트레이에서 기존 Ollama를 종료한 뒤 아래 명령으로 서버를 시작하세요. 이 터미널은 열어 둡니다. 환경 변수는 Ollama 서버를 시작하는 프로세스에 적용해야 합니다. [Ollama 로컬 전용 설정](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features)

```powershell
$env:OLLAMA_NO_CLOUD = "1"
ollama serve
```

앱은 기본적으로 `http://127.0.0.1:11434`에 연결하며 HTTP 루프백 주소만 허용합니다. 프록시 환경 변수와 리다이렉트를 사용하지 않고, 로컬 가중치가 확인되지 않는 모델 및 클라우드 모델을 거절합니다. 모델을 자동 다운로드하거나 외부 AI 서비스로 전환하지 않습니다.

다른 로컬 모델을 사용하려면 앱을 실행하는 터미널에 `OLLAMA_MODEL`을 설정하세요. 음성·영상 모드에는 이미지 입력을 지원하는 모델이 필요합니다.

```powershell
$env:OLLAMA_MODEL = "qwen3.5:4b"
```

### 텍스트 명령

Ollama 서버가 실행 중인 상태에서 별도 터미널로 다음 명령을 실행합니다.

```powershell
# 대화형으로 명령 입력
.\.venv\Scripts\python.exe .\main.py --plan

# 명령을 인자로 전달
.\.venv\Scripts\python.exe .\main.py --plan --command "앞으로 30cm 이동해"
```

다른 Python 코드에서는 `from module.action_planner import create_action_plan`으로 계획 기능을 가져올 수 있습니다. `create_action_plan(command, image=jpeg_bytes)`처럼 이미지 바이트를 함께 전달하거나 기존처럼 텍스트만 전달할 수 있습니다. 결과는 `ActionPlan`이며 `status`, `summary`, `steps`, `error` 필드를 제공합니다. 예전 `ai_action_planner.py` 진입점과 `from main import create_action_plan`도 유지합니다.

### 등록 화자의 음성 + 최신 카메라 프레임

처음에는 인터넷을 사용할 수 있을 때 `main.py --enroll`로 두 화자를 등록한 뒤, 기본 `main.py`를 한 번 실행해 음성 분리 모델과 Whisper 모델까지 준비하세요. 마이크를 듣기 시작하면 Ctrl+C로 종료합니다. 두 명의 등록자 모두 명령 권한을 가지며 `기타`는 미등록 음성을 뜻하는 예약 이름입니다.

Ollama와 음성 모델 준비가 끝나면 다음 명령으로 통합 모드를 시작합니다.

```powershell
.\.venv\Scripts\python.exe .\main.py --assist

# 두 번째 카메라 사용
.\.venv\Scripts\python.exe .\main.py --assist --camera 1
```

카메라는 계속 최신 프레임 하나를 메모리에 유지합니다. 등록된 화자의 발화가 끝나 전사 콜백이 도착하면, 그 시점의 프레임을 JPEG로 복사해 전사된 명령과 함께 로컬 AI로 전달합니다. 발화 종료와 전사 완료 사이에는 음성 처리 지연이 있으므로 촬영 시점은 전사 완료 시점입니다. 미등록 화자의 발화는 AI에 보내지 않습니다.

AI 처리는 음성 인식과 별도 작업자에서 FIFO 순서로 진행됩니다. 처리 중 다음 명령이 들어와도 각 명령은 접수 당시의 프레임을 유지합니다. 반복 발화도 별도 명령으로 처리하며, 영상이나 JPEG 파일을 디스크에 저장하지 않습니다. 모델이 느리면 명령 대기가 늘어날 수 있고 요청별 제한 시간은 120초입니다.

카메라가 끊기거나 프레임이 없거나 1초 넘게 갱신되지 않았다면 해당 명령은 보내지 않습니다. Ollama 미실행·모델 부재·응답 형식 오류도 표시하고 로봇을 구동하지 않습니다. Ctrl+C로 음성 입력을 종료하면 접수된 명령을 순서대로 마무리하고 종료합니다.

통합 모드는 음성 모델도 로컬 캐시만 사용합니다. 준비된 파일이 부족하면 다운로드 대신 오류를 표시합니다. 기본 음성 전사·화자 등록·손 모양 모드는 기존처럼 최초 모델 준비에 인터넷을 사용할 수 있습니다.

## 코드 구조

- `main.py`: 짧은 명령행 진입점이며 기존 공개 함수도 다시 제공합니다.
- `module/`: 애플리케이션 기능 모듈을 묶는 Python 패키지입니다.
  - `action_planner.py`: AI 텍스트 명령을 검증된 행동 계획으로 변환합니다.
  - `vision_assistant.py`: 등록 화자의 발화와 최신 프레임을 묶어 로컬 AI에 순서대로 전달합니다.
  - `speech.py`: 음성 분리, 화자 등록·식별, 발화 종료와 전사를 담당합니다.
  - `gesture.py`: 카메라 입력, MediaPipe 결과 안정화와 미리보기를 담당합니다.
  - `gesture_rules.py`: 숫자와 OK 모양의 순수 관절 좌표 규칙을 담습니다.

기존 코드처럼 `from main import run_live_transcription` 또는 `from main import run_gesture_recognition`을 계속 사용할 수 있습니다. 기능별 코드에서는 `module.speech`와 `module.gesture`에서 직접 가져와도 됩니다.

## 동작 메모

- SepFormer는 4초 창, 3초 이동, 1초 중첩으로 실행됩니다.
- 각 분리 스트림은 독립적으로 0.7초 침묵을 감지합니다.
- 20초를 넘는 발화는 내부 전사 조각으로 나누되 최종 콜백은 한 번만 호출합니다.
- 시작 시 CPU 분리 속도를 측정하며, 3초 이동 간격을 따라가지 못하면 경고합니다.
- 모델은 `models/`, 화자 지문은 `data/speaker_profiles.json`에 저장되며 Git에서 제외됩니다.
