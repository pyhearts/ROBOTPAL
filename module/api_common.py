"""Shared API configuration, error translation and multimodal input."""
from __future__ import annotations
import base64
import os
from typing import Any

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_HOST = "https://api.openai.com/v1"
REQUEST_TIMEOUT_SECONDS = 120.0

class PlannerResponseError(RuntimeError):
    """Raised when the model does not return a usable structured plan."""


class PlannerConnectionError(RuntimeError):
    """Raised when API configuration or the OpenAI service is unavailable."""


def create_api_client() -> Any:
    """Create an OpenAI client without logging or persisting credentials."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise PlannerConnectionError("OPENAI_API_KEY 환경 변수에 OpenAI API 키를 설정해 주세요.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise PlannerConnectionError(
            "OpenAI Python 패키지가 없습니다. requirements.txt의 의존성을 설치해 주세요."
        ) from exc
    return OpenAI(
        api_key=key,
        base_url=DEFAULT_HOST,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )


def response_value(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def selected_model(model: str | None = None) -> str:
    return (model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def friendly_api_error(exc: Exception, *, model: str = DEFAULT_MODEL) -> str:
    """Report useful errors without echoing request data or credentials."""
    status = response_value(exc, "status_code")
    if status == 401:
        return "OpenAI API 인증에 실패했습니다. OPENAI_API_KEY를 확인해 주세요."
    if status == 403:
        return "OpenAI API 또는 선택한 모델에 대한 접근 권한이 없습니다."
    if status == 404:
        return f"OpenAI 모델 {model}을 사용할 수 없습니다. OPENAI_MODEL과 접근 권한을 확인해 주세요."
    if status == 429:
        return "OpenAI API 사용 한도 또는 요청 제한에 도달했습니다. 결제·사용량을 확인해 주세요."
    if status == 400:
        return "OpenAI가 요청을 거절했습니다. 모델의 이미지 입력·구조화된 응답 지원을 확인해 주세요."
    if "Timeout" in type(exc).__name__:
        return "OpenAI API 응답 시간이 초과되었습니다."
    if isinstance(exc, ConnectionError) or "Connect" in type(exc).__name__:
        return "OpenAI API에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요."
    return "OpenAI API 요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요."


def user_content(text: str, image: bytes | None = None) -> list[dict[str, Any]]:
    if image is not None and (not isinstance(image, bytes) or not image):
        raise ValueError("image에는 비어 있지 않은 JPEG 이미지 바이트를 전달해 주세요.")
    content = [{"type": "input_text", "text": text.strip()}]
    if image is not None:
        encoded = base64.b64encode(image).decode("ascii")
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}", "detail": "auto"})
    return content
