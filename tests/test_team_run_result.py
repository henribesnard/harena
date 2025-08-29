import pytest
import importlib.util
from pathlib import Path
from typing import List, Optional

spec = importlib.util.spec_from_file_location(
    "team_run_result",
    Path(__file__).resolve().parents[1]
    / "conversation_service/models/responses/team_run_result.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore[attr-defined]
TeamRunResult = module.TeamRunResult
TeamRunResult.model_rebuild(
    _types_namespace={
        "IntentClassificationResult": object,
        "EntityExtractionResult": object,
        "Optional": Optional,
        "List": List,
    }
)


def test_valid_workflow_stage():
    result = TeamRunResult(
        workflow_stage="intent_classification",
        workflow_success=True,
        coherence_validation=True,
        agents_sequence=["IntentClassifier"],
        processing_time_ms=123.4,
    )
    assert result.workflow_stage == "intent_classification"


def test_invalid_workflow_stage():
    with pytest.raises(ValueError):
        TeamRunResult(
            workflow_stage="unknown",
            workflow_success=False,
            coherence_validation=False,
            agents_sequence=[],
            processing_time_ms=0.0,
        )
