# 로컬 TTS 사용

기본 엔진은 **GPT-SoVITS v2ProPlus**입니다. 기준 녹음과 일치하는 대본을 함께 참조해 목소리·억양·리듬을 생성합니다. 녹음을 이어 붙이거나 피치를 인위적으로 흔들지 않습니다. 기존 MeloTTS + OpenVoice V2는 `--tts-engine openvoice`로 사용할 수 있습니다.

TTS의 녹음·대본·합성 문장은 외부 서비스에 전송하지 않습니다. 작업자는 네트워크를 차단하고 API 키를 전달받지 않습니다. 기존 Decisions·대화·행동 계획 생성은 계속 OpenAI API를 사용합니다.

## 설치

최초 설치에만 인터넷과 수 GB의 공간이 필요합니다. 앱의 `.venv`와 분리한 Python 3.10 CPU 환경 `.gsv-venv`를 사용합니다.

```powershell
.\.venv\Scripts\python.exe .\setup_reference_tts.py
```

고정된 공식 소스 버전과 모델을 받고 오프라인 로딩까지 확인합니다. `models/reference_tts/manifest.json`에 모델 해시·소스 및 모델 리비전을, `installed-requirements.txt`에 실제 패키지를 기록합니다. 한국어 전용으로 사용하지 않는 중국어 BERT·자동 언어 감지 모델은 로딩하지 않습니다. 모델 누락 때 실행 중 다운로드하거나 클라우드로 전환하지 않습니다.

## 기준 녹음 준비

본인 또는 사용 동의를 받은 같은 화자의 녹음을 사용하세요. 조용한 곳에서 일정한 마이크 거리로 자연스럽게 읽은 문장이 좋습니다. 평서문·질문·안내 문장을 다양하게 녹음하면 참조 선택 폭이 넓어집니다. 파일 수보다 정확한 발음·대본과 잡음 없는 발화가 중요합니다.

```powershell
.\.venv\Scripts\python.exe .\main.py --list-mics
.\.venv\Scripts\python.exe .\main.py --record-sample .\data\tts\reference.wav --input-device 1 --sample-seconds 20
.\.venv\Scripts\python.exe .\main.py --prepare-tts --tts-reference ".\남성1.m4a" ".\남성2.m4a" --tts-voice .\data\tts\new_voice.json
```

기존 로컬 Whisper와 VAD로 각 녹음에서 깨끗한 3.2~9.5초 구간을 찾고 최대 2개씩 추립니다. 인식 확신도가 낮거나 무음·클리핑이 심한 구간은 제외합니다. 32kHz 모노 WAV 사본의 원래 속도·억양을 유지하고 양끝에 5ms 페이드만 적용합니다. 원본은 수정하지 않고 SHA-256을 기록합니다.

기본 설정은 `data/tts/reference_voice.json`, 발화 WAV와 자동 대본은 별도 `data/tts/references_*/dataset.json`에 저장합니다. 질문에는 질문 참조를 우선 선택하고, 없으면 품질 점수가 높은 구간을 사용합니다. 다른 원본 녹음의 좋은 구간도 보조 목소리 참조로 함께 전달합니다(최대 3개). 추가 학습 없이 참조하는 방식이며 모든 녹음의 리듬을 학습하는 것은 아닙니다.

대본은 자동 인식 결과이며 검수 완료로 표시하지 않습니다. 틀린 대본은 자연스러움과 발음에 영향을 줄 수 있습니다. 구간을 들어 보고 `dataset.json`의 `text`를 실제 발화에 맞게 수정한 뒤 새 설정으로 저장할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe .\tools\curate_reference_voice.py --finalize .\data\tts\references_실제폴더\dataset.json --voice .\data\tts\reference_voice_reviewed.json
```

실행에도 `--tts-voice .\data\tts\reference_voice_reviewed.json`을 지정하세요. 기존 설정은 덮어쓰지 않습니다. 새 녹음은 `--prepare-tts --tts-reference 경로 --tts-voice 새경로`로 준비합니다. 대본·설정·WAV 해시를 확인해 변경 후 예전 캐시 사용을 막습니다.

## 시험 재생과 실행

이 PC에는 두 녹음으로 만든 기본 참조 설정이 준비되어 있습니다. 아래 시험은 API를 호출하지 않습니다.

```powershell
.\.venv\Scripts\python.exe .\main.py --list-speakers
.\.venv\Scripts\python.exe .\main.py --test-tts "안녕하세요. 잘 들리시나요?"
.\.venv\Scripts\python.exe .\main.py --assist --input-device 1 --tts

# 출력 장치와 속도 조정
.\.venv\Scripts\python.exe .\main.py --assist --input-device 1 --tts --output-device 3 --tts-speed 1.0
```

`--tts`를 생략하면 화면에만 출력합니다. 기본 음성 전사·손 모양 모드는 기존 동작을 따릅니다. `--tts-python`, `--tts-models`, `--tts-voice`로 경로를 지정합니다. 한국어 전용이며 다른 언어 품질은 검증하지 않았습니다. GPT-SoVITS의 `--tts-style`은 합성 설정을 바꾸지 않습니다. 참조 발화로 리듬을 생성합니다.

기존 방식 비교:

```powershell
.\.venv\Scripts\python.exe .\main.py --test-tts "안녕하세요. 잘 들리시나요?" --tts-engine openvoice --tts-style natural
```

기존 `data/tts/voice.json`, 모델·캐시·환경은 보존했습니다. 새 PC에서 기존 엔진을 설치하려면 `setup_tts.py`를 사용하세요.

## 여성 목소리 선택

`여성1.m4a`만 사용한 별도 목소리는 `data/tts/female_voice.json`에 준비했습니다. 남성 목소리와 섞지 않고 원본 녹음과 기존 기본 설정은 보존합니다. 여성 목소리로 실행할 때만 아래 설정을 추가하세요.

```powershell
.\.venv\Scripts\python.exe .\main.py --test-tts "안녕하세요. 오늘은 무엇을 도와드릴까요?" --tts-voice .\data\tts\female_voice.json
.\.venv\Scripts\python.exe .\main.py --assist --input-device 1 --tts --tts-voice .\data\tts\female_voice.json
```

`--tts-voice`를 생략하면 기존 남성 기본 목소리를 사용합니다. 여성 목소리의 참조 발화와 자동 대본은 `data/tts/references_76a361abd1/dataset.json`에서 확인할 수 있습니다. 대본은 자동 전사이므로 청취 검수가 필요합니다.

## 출력과 마이크

- 일반 대화·장면 질문은 최종 답변만 읽으며, 300자를 넘으면 완결 문장까지만 읽습니다. 전체 답변은 화면과 로그에 남깁니다.
- 행동 계획은 계획임을 밝히고 요약을 읽습니다. 실제 동작 완료를 주장하지 않습니다.
- 오류·재질문·정지는 고정 안내를 읽습니다. 무의미한 입력·임시 전사·진행 로그는 읽지 않습니다.
- 재생 중과 종료 후 300ms는 마이크 입력을 제외합니다. 버퍼와 늦게 처리된 전사도 재생 구간과 대조합니다. 재생 중 끼어들기는 지원하지 않습니다.
- 합성 중에는 듣기를 유지합니다. CPU 스레드 2개를 사용하지만 전사와 자원을 공유합니다.
- 대기 음성은 최대 4개입니다. 가득 차면 새 음성만 생략하고 화면·로그는 유지합니다.
- 모델 시작 제한은 120초, GPT-SoVITS 문장 합성 제한은 240초입니다(OpenVoice 60초). 실패 시 자동 재시도하지 않고 다음 출력은 계속 시도합니다. 초기화 실패 시 음성 비서의 화면 출력은 계속합니다.
- Ctrl+C는 재생과 대기 TTS를 취소합니다.

## 저장·검증

`data/tts/cache/`에 문장·모델·목소리·속도·엔진 버전별 WAV를 저장합니다. 동일 문장은 재합성하지 않습니다. 기준 녹음·모델·실행 환경·캐시는 Git에서 제외합니다.

실행별 `logs/<실행 이름>_audio/`에 생성·재사용 WAV를 복사하고 `command_id`로 AI 답변·계획·이미지와 연결합니다. `tts_synthesis_profile`에 엔진·주 참조 ID·보조 참조 ID·합성 시간을 기록합니다. 기존 대기·재생·실패·건너뜀 로그도 유지합니다.

2026-10-18에 두 원본에서 4개 구간을 준비했습니다. 준비 당시 이름은 `녹음 (2).m4a`, `녹음 (4).m4a`이며 현재 `남성1.m4a`, `남성2.m4a`와 각각 해시가 같습니다. 자동 대본과 구간 정보는 `data/tts/reference_dataset_20261018_155451/dataset.json`에 있습니다. 새로 추가된 다른 화자의 녹음은 이 목소리에 섞지 않았습니다.

두 녹음을 함께 참조하는 최종 설정에서 시험 문장 3개의 합성 시간은 약 19~28초였습니다. `reports/tts_20261018_160140/reference_voice.wav`와 `previous_voice.wav`에 같은 문장의 새·기존 음성이 있습니다. CPU에서 실시간 응답을 보장하지 않으며 자연스러움·목소리 유사도는 실제 청취로 비교해야 합니다. 실제 스피커와 물리적 에코는 이 검증에 포함하지 않았습니다. 고정 안내 8개는 준비 단계에서 캐시하며, 일반 답변은 처음에만 합성합니다.

```powershell
# 소리를 재생하거나 API를 호출하지 않고 측정
.\.venv\Scripts\python.exe .\tools\benchmark_local_tts.py
# 전사 동시 실행 비교는 메모리를 더 사용
.\.venv\Scripts\python.exe .\tools\benchmark_local_tts.py --compare-stt
```

변경 전 소스는 `backups/before_reference_tts_20261018_150325/`에 보관했습니다.

출처: [GPT-SoVITS 공식 소스](https://github.com/RVC-Boss/GPT-SoVITS), [공식 모델 배포](https://huggingface.co/lj1995/GPT-SoVITS), [OpenVoice](https://github.com/myshell-ai/OpenVoice), [MeloTTS](https://github.com/myshell-ai/MeloTTS).
