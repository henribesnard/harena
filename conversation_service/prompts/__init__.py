"""Prompt templates and response schemas for conversation agents."""

from .base_prompts import (
    ADVANCED_BEHAVIOUR,
    BASE_SYSTEM_MESSAGES,
    BaseResponse,
    ErrorResponse,
    build_system_prompt,
)

__all__ = [
    "ADVANCED_BEHAVIOUR",
    "BASE_SYSTEM_MESSAGES",
    "BaseResponse",
    "ErrorResponse",
    "build_system_prompt",
]

