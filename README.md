# ROBOTPAL 음성·손 모양 입력

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

## 코드 구조

- `main.py`: 짧은 명령행 진입점이며 기존 공개 함수도 다시 제공합니다.
- `module/`: 애플리케이션 기능 모듈을 묶는 Python 패키지입니다.
  - `speech.py`: 음성 분리, 화자 등록·식별, 발화 종료와 전사를 담당합니다.
  - `gesture.py`: 카메라 입력, MediaPipe 결과 안정화와 미리보기를 담당합니다.
  - `gesture_rules.py`: 숫자와 OK 모양의 순수 관절 좌표 규칙을 담습니다.
- `test_modules.py`: 장치와 모델 없이 실행되는 통합 회귀 테스트입니다.

기존 코드처럼 `from main import run_live_transcription` 또는 `from main import run_gesture_recognition`을 계속 사용할 수 있습니다. 기능별 코드에서는 `module.speech`와 `module.gesture`에서 직접 가져와도 됩니다.

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_modules.py
```

## 동작 메모

- SepFormer는 4초 창, 3초 이동, 1초 중첩으로 실행됩니다.
- 각 분리 스트림은 독립적으로 0.7초 침묵을 감지합니다.
- 20초를 넘는 발화는 내부 전사 조각으로 나누되 최종 콜백은 한 번만 호출합니다.
- 시작 시 CPU 분리 속도를 측정하며, 3초 이동 간격을 따라가지 못하면 경고합니다.
- 모델은 `models/`, 화자 지문은 `data/speaker_profiles.json`에 저장되며 Git에서 제외됩니다.
