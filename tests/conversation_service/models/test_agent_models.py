import json

import pytest
from pydantic import ValidationError

from conversation_service.models import (
    AgentConfig,
    AgentStep,
    AgentTrace,
    DynamicFinancialEntity,
    IntentResult,
    EntityType,
    IntentType,
)


def test_agent_models_validation_and_json():
    step = AgentStep(agent_name="retriever", success=True)
    trace = AgentTrace(steps=[step], total_time_ms=12.5)
    data = {"steps": [s.model_dump() for s in trace.steps], "total_time_ms": trace.total_time_ms}
    json_data = json.dumps(data)
    loaded = AgentTrace(**json.loads(json_data))
    loaded_step = loaded.steps[0]
    assert loaded_step.model_dump() == step.model_dump()
    assert loaded.total_time_ms == trace.total_time_ms

    # Basic instantiation should succeed with valid data
    AgentStep(agent_name="retriever", success=True)
    AgentTrace(steps=[step], total_time_ms=12.5)


def test_agent_config_and_enums():
    config = AgentConfig(
        name="classifier",
        system_prompt="You are a bot.",
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=100,
        timeout=10,
        few_shot_examples=[["hi", "hello there"]],
        cache_ttl=10,
        cache_strategy="memory",
    )
    assert config.name == "classifier"
    assert config.few_shot_examples[0] == ["hi", "hello there"]
    assert config.cache_strategy == "memory"

    with pytest.raises(ValidationError):
        AgentConfig(name="bad", system_prompt="x", model="bad-model")

    with pytest.raises(ValidationError):
        AgentConfig(
            name="bad", system_prompt="x", model="gpt-4o-mini", few_shot_examples=[["only one"]]
        )

    with pytest.raises(ValidationError):
        AgentConfig(
            name="bad", system_prompt="x", model="gpt-4o-mini", cache_ttl=0
        )

    with pytest.raises(ValidationError):
        AgentConfig(
            name="bad", system_prompt="x", model="gpt-4o-mini", cache_strategy="disk"
        )

    intent = IntentResult(intent_type=IntentType.GREETING, confidence_score=0.9)
    assert intent.intent_type is IntentType.GREETING

    entity = DynamicFinancialEntity(
        entity_type=EntityType.ACCOUNT,
        raw_value="123",
        normalized_value="123",
        context="account number is 123",
        metadata={"source": "test"},
        confidence_score=0.8,
    )
    assert entity.entity_type is EntityType.ACCOUNT

    assert entity.metadata == {"source": "test"}

    with pytest.raises(ValidationError):
        DynamicFinancialEntity(
            entity_type="BAD", raw_value="1", confidence_score=0.1
        )
