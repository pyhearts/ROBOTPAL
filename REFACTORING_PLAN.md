# ROBOTPAL 리팩토링 구현 기록

2026-10-21. TTS를 코드로 보관하고 본체를 단순화하는 상세 계획을 적용했습니다. 이전 초안의 TTS 런타임 유지와 하위 명령 추가 방안은 채택하지 않았습니다.

## 적용한 경계

- `main.py`: 로그 세션과 실행 호출만 담당.
- `module/cli.py`: 기존 옵션 등록·조합 검사·음성 설정 생성·모드별 지연 import와 실행 연결. 텍스트 보조 진입점도 같은 경로 사용.
- `module/speech.py`: 직접 마이크 수집과 입력 마무리, 선택적 과거 기능의 공개 위임 함수.
- `module/utterance.py`: 직접·분리 음성이 공유하는 발화 상태와 작업 자료형·감지기.
- `module/transcription.py`: 공유 전사 모델 로딩과 최종·임시 전사 작업자.
- `module/legacy_speech.py`: 화자 프로필·등록·식별·SepFormer·분리 실행.
- `module/camera.py`: 최신 프레임 수집과 신선도 검사·JPEG 생성.
- `module/vision_assistant.py`: 최종 발화와 이미지 접수, FIFO 명령 처리, 자원 수명 관리.
- `module/api_common.py`: API 설정·모델 선택·공통 오류·응답 필드 접근·이미지 입력 구성.

## TTS 보관

`archive/tts/source/`에 전용 소스·설치 목록·도구·테스트를 원래 상대 경로로 보관했습니다. `integration_context/`에는 수정 전 본체 연결 지점 4개 파일이 있습니다. 기존 안내 문서도 보관했습니다. 총 22개 원본·사본의 SHA-256이 일치하며 `manifest.json`에 기록했습니다.

현재 본체에는 TTS 생성·재생·결과 제출·재생 큐·재생 중 마이크 차단·생성 음성 로그 저장 경로가 없습니다. 구형 TTS 옵션은 종료 코드 2와 복원 안내로 거절합니다. 보관본은 현재 위치에서 실행 가능한 도구가 아닙니다. 복원 절차는 [RESTORE.md](archive/tts/RESTORE.md)에 있습니다.

모델·가상환경·참조 녹음·프로필·캐시·기존 로그와 백업은 이동하거나 삭제하지 않았습니다.

## 유지하는 사용법과 정책

기본 실행, `--assist`, `--plan`, `--gesture`, 마이크 진단·녹음·STT 비교를 유지합니다. `CommandDecision`, `CommandResult`, `ActionPlan`, `process_command`도 유지합니다.

과거 음성 기능은 별도 설치로 실행 가능하며 `enroll_speakers`, `load_speaker_profiles`, `run_separated_transcription`, `run_live_transcription(identify_speakers=True)`가 선택 구현에 위임합니다.

임시 전사는 명령으로 전달하지 않고, 긴 발화는 품질 검사를 통과해야 최종 한 번 전달합니다. 접수 당시 JPEG·명령 ID·FIFO·분류 후 API 호출·이미지 전송 범위·제한 시간·자동 재시도 0회 정책을 유지합니다.

종료는 마이크 수집 중단 → 오디오 및 최종 전사 완료 → 카메라 종료 → AI 명령 마무리 → API 클라이언트 종료 → 로그 종료 순서입니다. 초기화 실패에서도 시작한 자원을 정리합니다.

## 설치와 문서

기본 전체 설치 목록에서 SpeechBrain·PyTorch를 제외하고 core, speech, vision, gesture 목록을 조합합니다. 과거 음성 기능은 legacy-speech 목록으로 추가 설치합니다. 기본 VAD는 Faster Whisper의 Silero ONNX 구현을 사용합니다.

README, 음성 안내, 순서도를 현재 구조에 맞췄습니다. 검사 중 확인한 기존 숫자 손 모양 미리보기의 누락된 상수 import도 수정하고 회귀 테스트를 추가했습니다.

## 검증 결과

- 이전 기준선: 119개 테스트 통과.
- 기존 본체 테스트: 88개 유지.
- TTS 테스트: 31개를 원본 그대로 보관.
- 새 경계·호환·종료·보관 무결성 테스트: 13개.
- 최종 자동 검증: `python -m unittest discover -s tests -q`, 총 101개 통과.
- 별도 프로세스에서 음성·카메라·손 모양 의존성 import를 차단하고 두 텍스트 진입점을 실행.
- 기본 음성 경로의 과거 모듈 미로딩, 선택 기능 위임, 화자 프로필 저장·읽기, 분리 작업자 종료 확인.
- 초기화·작업자 시작 실패와 종료 중 오류의 자원 정리 확인.

실제 마이크·카메라·외부 API 품질 확인은 수행하지 않았습니다. 실제 장치에서는 짧은·긴·연속 발화, 카메라 없는 시작, 명령 대기 중 종료를 추가 확인해야 합니다. TTS 재합성·재생은 이번 검증 범위가 아닙니다.
