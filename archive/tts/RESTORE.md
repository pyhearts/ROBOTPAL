# TTS 보관본과 복원 안내

2026-10-21에 본체에서 분리한 참고 구현입니다. `source/`는 원래 상대 경로를 유지한 TTS 전용 코드이고, `integration_context/`는 분리 직전 본체 연결 지점의 원본입니다. **현재 위치에서 실행 가능함을 보장하지 않으며 기본 실행·설치·테스트에는 포함하지 않습니다.**

## 보존 내용 검증

`manifest.json`의 `files`에는 원본 경로, 이 폴더 기준 보관 경로, SHA-256이 있습니다. 보관 직후 원본과 사본의 해시가 일치함을 확인했습니다. `baseline_tests`에는 분리 전 119개 테스트의 이름을 기록했습니다. 그중 TTS 전용 31개는 `source/tests/`에 보관했고 본체의 기존 88개는 활성 테스트로 유지합니다.

## 외부 자산 — 기존 프로젝트 루트에 그대로 유지

| 용도 | 원래 위치 |
|---|---|
| GPT-SoVITS Python·소스 | `.gsv-venv/`, `.gsv-sources/GPT-SoVITS/` |
| OpenVoice Python·소스 | `.tts-venv/`, `.tts-sources/` |
| 설치 런타임·도구 | `.tts-python/`, `.tts-tools/` |
| GPT-SoVITS 모델 | `models/reference_tts/` |
| OpenVoice 모델 | `models/tts/` |
| 음성 프로필 | `data/tts/reference_voice.json`, `data/tts/voice.json` 등 기존 사용자 지정 JSON |
| 변환된 참조 녹음·대본·캐시 | `data/tts/`와 `data/tts/cache/` |
| 원본 녹음 | 프로젝트 루트의 기존 `.m4a` 파일 및 사용자가 지정한 녹음 위치 |

자산·가상환경·기존 로그를 삭제하거나 이동하지 않았습니다. 프로필이 참조하는 파일명은 각 JSON에 남아 있습니다. 보관 manifest는 소스 파일의 무결성을 검증하며 대용량 외부 자산의 유효성까지 검증하지 않습니다.

## 복원 시 확인할 연결 지점

1. 엔진을 먼저 선택하고 필요한 전용 의존성을 별도 환경에 설치합니다. 현재 본체의 설치 목록에 보관된 두 엔진의 의존성을 합치지 않습니다.
2. 경로를 다시 설정합니다. 보관 소스의 `ROOT = Path(__file__).resolve().parent.parent`, 작업자 파일의 이웃 경로, 도구의 `sys.path`와 설치 스크립트는 이전 프로젝트 배치를 전제로 합니다. 단순히 이 보관 폴더에서 실행하면 경로가 맞지 않습니다.
3. `source/module/local_tts.py`의 `spoken_text`를 참고해 현재 `CommandResult`를 음성 문장으로 바꾸는 연결부를 작성합니다. 행동 계획을 실제 실행 완료로 표현하지 않습니다.
4. 작업자의 표준 입출력 JSON 통신, 초기화·합성 제한 시간, 캐시 키, 참조 프로필·모델 검증을 복구합니다. 작업자에 API 키를 전달하지 않는 처리와 로컬 합성의 네트워크 차단도 확인합니다.
5. `integration_context/main.py`와 `module/vision_assistant.py`에서 결과 제출 위치를 확인합니다. 현재 CLI 옵션과 본체 시그니처에는 TTS가 없습니다. 예전 파일로 본체 전체를 덮어쓰지 않습니다.
6. `integration_context/module/speech.py`와 `source/module/audio_gate.py`에서 재생 중 마이크 차단을 확인합니다. 수집된 오디오의 시각·재생 세대, 늦게 완료된 전사, 임시 인식, 발화 초기화까지 함께 검토합니다. 입력 차단 없이 스피커만 연결하지 않습니다.
7. `integration_context/module/session_log.py`의 생성 음성 보관과 명령 ID 연결을 현재 로그 경계에 맞춰 복구합니다.
8. 새 종료 정책을 정하고 이미 닫힌 TTS로 AI 결과가 전달되지 않도록 검증합니다. 과거 종료 순서를 그대로 복원하지 않습니다.

## 재검증 조건

보관 테스트의 import·mock 경로를 새 구조에 맞춰 옮긴 뒤 합성·캐시·프로필 변조·작업자 제한 시간·실패 후 복구 테스트를 실행합니다. 마이크 차단·재생 중 종료·대기 명령 처리 테스트도 복원해야 합니다. 마지막으로 실제 엔진으로 참조 음성 준비, 합성, 스피커 재생, 자기 음성 재입력 차단을 확인합니다.

기존 사용법은 [TTS_GUIDE.md](TTS_GUIDE.md)에 원문으로 보관했습니다. 그 문서의 실행 명령은 분리 전 사용법입니다.
