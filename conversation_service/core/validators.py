"""Pydantic validation helpers for cached objects."""

from __future__ import annotations

from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def validate_model(model: Type[T], data: Any) -> T:
    """Validate ``data`` against ``model``.

    If ``data`` is already an instance of ``model`` it is returned as-is.
    Otherwise ``model.model_validate`` is used to construct and validate an instance.
    """

    if isinstance(data, model):
        return data
    try:
        return model.model_validate(data)  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - support for Pydantic v1
        return model(**data)
