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
    step = AgentStep(agent="retriever", status="ok")
    trace = AgentTrace(steps=[step], total_time_ms=12.5)
    data = {"steps": [s.model_dump() for s in trace.steps], "total_time_ms": trace.total_time_ms}
    json_data = json.dumps(data)
    loaded = AgentTrace(**json.loads(json_data))
    assert loaded.steps[0].model_dump() == step.model_dump()
    assert loaded.total_time_ms == trace.total_time_ms

    with pytest.raises(ValidationError):
        AgentStep(agent="", status="ok")

    with pytest.raises(ValidationError):
        AgentTrace(steps=[step], total_time_ms=-1.0)

    with pytest.raises(ValidationError):
        AgentTrace(steps=[], total_time_ms=1.0)


def test_agent_config_and_enums():
    config = AgentConfig(
        name="classifier",
        system_prompt="You are a bot.",
        model="gpt-4",
        temperature=0.5,
        max_tokens=100,
        timeout=10,
    )
    assert config.name == "classifier"

    intent = IntentResult(intent_type=IntentType.GREETING, confidence_score=0.9)
    assert intent.intent_type is IntentType.GREETING

    with pytest.raises(ValidationError):
        IntentResult(intent_type="INVALID", confidence_score=0.5)

    entity = DynamicFinancialEntity(
        entity_type=EntityType.ACCOUNT,
        value="123",
        confidence_score=0.8,
    )
    assert entity.entity_type is EntityType.ACCOUNT

    with pytest.raises(ValidationError):
        DynamicFinancialEntity(entity_type="BAD", value="1", confidence_score=0.1)
