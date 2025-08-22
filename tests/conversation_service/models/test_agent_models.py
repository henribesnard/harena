import json

import pytest
from pydantic import ValidationError

from conversation_service.models import AgentStep, AgentTrace


def test_agent_models_validation_and_json():
    step = AgentStep(agent="retriever", status="ok")
    trace = AgentTrace(steps=[step], total_time_ms=12.5)
    data = {"steps": [s.dict() for s in trace.steps], "total_time_ms": trace.total_time_ms}
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
