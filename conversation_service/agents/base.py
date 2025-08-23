from __future__ import annotations

"""Common utilities for OpenAI-based agents with Pydantic validation."""

import json
import time
from typing import Any, Dict, List, Type, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from conversation_service.models.agent_models import AgentConfig, AgentStep


class BaseAgent:
    """Base class for agents relying on the OpenAI client."""

    def __init__(self, client: Any, config: AgentConfig) -> None:
        self.client = client
        self.config = config
        self.last_step: AgentStep | None = None

    async def _run(
        self, messages: List[Dict[str, str]], response_model: Type[BaseModel]
    ) -> BaseModel:
        """Execute a prompt and parse the response as ``response_model``."""
        start_time = time.perf_counter()
        schema = (
            response_model.model_json_schema()
            if hasattr(response_model, "model_json_schema")
            else {}
        )
        response = await self.client.chat_completion(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                *messages,
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": response_model.__name__, "schema": schema},
            },
            agent_name=self.config.name,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        usage = getattr(response, "usage", None)
        metrics: Dict[str, Any] = {"duration_ms": duration_ms}
        if usage is not None:
            metrics.update(
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                    "cost": getattr(usage, "total_cost", 0.0),
                }
            )
        reasoning_trace = None
        try:
            reasoning_trace = response.choices[0].message.reasoning  # type: ignore[attr-defined]
        except Exception:
            reasoning_trace = None
        step = AgentStep(
            agent_name=self.config.name,
            success=True,
            error_message=None,
            metrics=metrics,
            from_cache=getattr(response, "from_cache", False),
            reasoning_trace=reasoning_trace,
        )
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            if hasattr(response_model, "model_validate"):
                result = response_model.model_validate(data)
            else:
                result = response_model(**data)  # type: ignore[arg-type]
                annotations = get_type_hints(response_model)
                for field, field_type in annotations.items():
                    value = getattr(result, field, None)
                    if value is None:
                        continue
                    origin = get_origin(field_type)
                    args = get_args(field_type)
                    if origin in (list, List) and args:
                        inner = args[0]
                        if isinstance(inner, type) and issubclass(inner, BaseModel):
                            setattr(
                                result,
                                field,
                                [
                                    inner(**v) if isinstance(v, dict) else v
                                    for v in value
                                ],
                            )
                    elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
                        if isinstance(value, dict):
                            setattr(result, field, field_type(**value))
            self.last_step = step
            return result
        except Exception as exc:
            step.success = False
            step.error_message = f"Invalid response: {content}"
            self.last_step = step
            raise ValueError(
                f"Invalid response from agent {self.config.name}: {content}"
            ) from exc
