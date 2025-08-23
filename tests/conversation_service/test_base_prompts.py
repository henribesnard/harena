"""Tests for base system prompts and response patterns."""

from conversation_service.prompts.base_prompts import (
    ADVANCED_BEHAVIOUR,
    BASE_SYSTEM_MESSAGES,
    BaseResponse,
    ErrorResponse,
    build_system_prompt,
)


def test_build_system_prompt_contains_advanced_behaviour():
    prompt = build_system_prompt().lower()
    assert "financial advice" in prompt
    assert "human operator" in prompt


def test_base_response_optional_fields():
    data = {"message": "hello", "confidence": 0.9}
    res = BaseResponse(**data)
    assert res.message == "hello"
    assert res.language is None

    data_full = {
        "message": "bonjour",
        "confidence": 0.8,
        "language": "fr",
        "extra": {"note": "salut"},
    }
    res_full = BaseResponse(**data_full)
    assert res_full.extra == {"note": "salut"}


def test_error_response_requires_error():
    try:
        ErrorResponse(confidence=0.1)
    except Exception:  # broad as Pydantic raises ValidationError
        pass
    else:
        raise AssertionError("ErrorResponse should require an error field")
