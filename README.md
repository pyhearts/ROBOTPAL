# ROBOTPAL

마이크의 한국어 발화를 전사하고, 선택적으로 카메라 이미지와 함께 발화를 분류해 대화 답변이나 이동·회전·정지의 상위 행동 계획을 출력합니다. **실제 로봇 제어에는 연결되어 있지 않습니다.** 손 모양 인식은 독립 모드입니다.

## 설치

Python 3.12 환경에서 기본 기능 전체를 설치합니다. 화자 분리·등록은 별도 선택 설치입니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

필요한 기능만 설치할 수도 있습니다.

| 설치 파일 | 용도 |
|---|---|
| `requirements-core.txt` | 텍스트 분류·대화·행동 계획 |
| `requirements-speech.txt` | 직접 마이크·VAD·전사, core 포함 |
| `requirements-vision.txt` | 카메라·JPEG 처리 |
| `requirements-gesture.txt` | 손 모양 인식, vision 포함 |
| `requirements-legacy-speech.txt` | 선택적 화자 등록·식별·SepFormer, speech 포함 |

통합 모드는 speech와 vision이 필요합니다. 기본 전사는 Faster Whisper에 포함된 Silero ONNX 모델을 사용하며 PyTorch·SpeechBrain을 직접 불러오지 않습니다. 로컬 Whisper의 첫 실행에는 모델 다운로드가 필요합니다.

## 주요 실행법

```powershell
# 음성 전사만 실행 — API 키 불필요
.\.venv\Scripts\python.exe .\main.py

# 마이크 번호 확인과 입력 선택
.\.venv\Scripts\python.exe .\main.py --list-mics
.\.venv\Scripts\python.exe .\main.py --input-device 1

# 음성·카메라로 답변 또는 행동 계획 생성
.\.venv\Scripts\python.exe .\main.py --assist --input-device 1 --camera 0

# 텍스트 입력 — 카메라를 열지 않음
.\.venv\Scripts\python.exe .\main.py --plan --command "앞으로 30cm 이동해"
.\.venv\Scripts\python.exe .\main.py --plan

# 손 모양 인식 — API 키 불필요
.\.venv\Scripts\python.exe .\main.py --gesture --preview
.\.venv\Scripts\python.exe .\main.py --gesture --rule-set numbers --preview

# 전체 옵션
.\.venv\Scripts\python.exe .\main.py --help
```

`ai_action_planner.py`도 같은 텍스트 처리 경로를 사용하는 대화형 진입점입니다. 기본 실행법과 `--assist`, `--plan`, `--gesture`는 유지합니다. TTS는 코드 보관 상태이며 [복원 안내](archive/tts/RESTORE.md)를 참고하세요.

## API 설정과 처리 정책

AI 모드를 실행할 PowerShell 창에서 키를 설정합니다. `.env`를 자동으로 읽지는 않습니다.

```powershell
$env:OPENAI_API_KEY = "본인의_API_키"
# 선택: 분류 이후 대화·계획에 사용할 모델
$env:OPENAI_MODEL = "gpt-6-luna"
```

기본 모델과 분류 모델은 현재 코드에서 `gpt-6-luna`로 설정되어 있습니다. 분류는 `/decisions`, 대화·행동 계획은 `/responses`를 사용합니다. 서비스의 모델 접근 권한이 필요합니다.

| 분류 | 처리 |
|---|---|
| `concrete_command` | 방향·종료 조건이 명확한 이동·회전의 계획 생성 |
| `stop_command` | 추가 Responses 호출 없이 고정 정지 계획 출력 |
| `conversation` | 이미지 없이 독립적인 짧은 답변 |
| `visual_question` | 접수 당시 이미지로 답변, 영상이 없으면 고정 안내 |
| `not_command`, `needs_details`, `unsupported`, `uncertain` | 추가 Responses 호출 없이 고정 안내 |

확신도가 0.8 미만이면 불확실 안내를 출력합니다. 이 값은 초기 기준이며 실제 한국어 정확도를 보장하지 않습니다. 분류 실패 시 계획 API로 우회하지 않습니다. 대화에는 이전 문맥이나 도구를 전달하지 않습니다.

분류 제한 시간은 20초, 계획은 120초, 대화는 30초이며 자동 재시도는 없습니다. Responses에는 `store=False`를 지정하지만 서비스의 별도 보관 정책까지 비활성화하는 의미는 아닙니다.

최종 발화마다 분류 API 비용이 발생하며 대화·장면 질문·구체적 명령은 추가 API 비용이 발생할 수 있습니다. 기본 전사는 로컬에서 수행합니다. `--stt-backend openai`를 명시한 경우에만 발화 오디오를 전사 API로 전송하며 별도 비용이 발생합니다.

## 음성과 영상의 동작

직접 마이크 모드는 16 kHz 모노 입력을 Silero로 감지하고 로컬 Whisper small로 전사합니다. 기본 발화 종료 침묵은 0.45초입니다. 임시 인식은 표시만 하고 최종 품질 검사를 통과한 문장만 명령으로 전달합니다. 긴 발화는 조각으로 전사해도 최종 콜백은 한 번 전달하며, 한 조각이라도 품질 검사에 실패하면 전체를 보류합니다.

화자를 식별하거나 겹친 목소리를 분리하지 않습니다. 반복 발화도 별도 명령으로 받아 FIFO로 처리합니다. 최종 전사 접수 시점의 JPEG를 명령과 함께 고정하므로 API 처리 시점에 이미지가 바뀌지 않습니다. 프레임이 없거나 1초보다 오래됐으면 영상 없이 분류합니다.

영상이 없어도 일반 대화와 정지 계획을 처리합니다. 통합 모드의 구체적인 행동 요청과 장면 질문은 영상이 없으면 안내만 출력합니다. 텍스트 계획 모드의 구체적인 행동 요청은 영상 없이 처리할 수 있습니다.

Ctrl+C에서는 입력을 중단하고 마지막 발화의 전사를 마친 뒤 카메라를 닫고 접수된 AI 명령을 순서대로 마무리합니다. API가 느리면 종료까지 시간이 걸릴 수 있습니다. 정지도 FIFO로 계획을 출력하며 하드웨어 긴급 정지 기능은 아닙니다.

음성 설정·진단·녹음·동일 음원 비교는 [SPEECH_GUIDE.md](SPEECH_GUIDE.md)에 설명합니다.

## 선택적 화자 등록·분리

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-legacy-speech.txt
```

```python
from module.speech import enroll_speakers, run_separated_transcription

enroll_speakers()  # 별도로 한 번 실행
run_separated_transcription(identify_speakers=True)
```

선택 구현은 `module/legacy_speech.py`에 있으며 기본 실행에서는 불러오지 않습니다. 기존 `load_speaker_profiles()`와 `run_live_transcription(identify_speakers=True)` 호출도 유지합니다. CPU 음성 분리 지연이 발생할 수 있으며 화자 식별이 AI 명령 권한 제한을 제공하지는 않습니다.

## 로그와 코드 사용

실행마다 `logs/`에 `.log`와 `.jsonl`을 만듭니다. 한국 시간으로 화면 출력·전사·분류·API 사용량·오류를 기록하고 명령 ID로 결과를 연결합니다. 통합 모드는 명령별 JPEG를 `<실행 이름>_images/`에 저장합니다. 저장된 JPEG와 API에 전달할 JPEG는 같습니다. 일반 대화에 이미지를 보내지 않아도 접수 시 선택된 이미지는 저장됩니다.

원본 마이크 오디오와 연속 영상은 저장하지 않습니다. 평가용 WAV는 명시적으로 녹음을 선택할 때만 저장합니다. 로그에는 전사 내용과 이미지가 남으며 자동 삭제하지 않습니다. API 키는 텍스트 로그에서 가립니다.

```python
from module.decision_router import process_command, print_command_result
from module.session_log import capture_output

with capture_output():
    result = process_command("앞으로 30cm 이동해")
    print_command_result(result)
```

`CommandResult`, `CommandDecision`, `ActionPlan`과 `process_command`의 호출 방식은 유지합니다. 저수준 `create_action_plan`은 분류를 생략하므로 일반 진입점은 `process_command`를 사용하세요.

## 검증과 구조

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

외부 API·실제 장치 없이 회귀 테스트를 수행합니다. 보관 코드는 기본 테스트 대상에서 제외합니다. 실제 인식 품질·카메라·장치 연결은 별도 확인이 필요합니다.

본체는 CLI, 직접 마이크, 공유 발화·전사 작업자, 선택적 과거 음성, 카메라, 명령 처리, API 공통 처리로 나뉩니다. [FLOWCHART.md](FLOWCHART.md)에 실행 흐름을 기록합니다. Arduino 스케치는 독립 실행되는 서보 중앙 신호 설정이며 Python과 연결되어 있지 않습니다. 기존 `backups/`와 모델·가상환경·녹음 자산은 그대로 유지합니다.
