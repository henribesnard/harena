import json
import asyncio
import pytest

from conversation_service.prompts.autogen import (
    AUTOGEN_ENTITY_EXTRACTION_SYSTEM_MESSAGE,
    get_entity_extraction_prompt_for_autogen,
    TEAM_ORCHESTRATION_PHASE2_MESSAGE,
)
from conversation_service.prompts.system_prompts import (
    ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT,
)
from conversation_service.agents.financial.intent_classifier import (
    IntentClassifierAgent,
)


def test_autogen_entity_prompt_reuses_phase1_sections():
    """AUTOGEN_ENTITY_EXTRACTION_SYSTEM_MESSAGE should embed Phase 1 base prompt."""
    assert ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT in AUTOGEN_ENTITY_EXTRACTION_SYSTEM_MESSAGE


def test_get_entity_extraction_prompt_strategy_switch():
    """Strategy section changes based on intent confidence."""
    proactive = get_entity_extraction_prompt_for_autogen({"confidence": 0.9})
    conservative = get_entity_extraction_prompt_for_autogen({"confidence": 0.2})

    assert "Confiance d'intention suffisante" in proactive
    assert "Confiance d'intention faible" in conservative


def test_phase1_phase2_intent_classifier_parity(monkeypatch):
    """Phase 1 and Phase 2 classifiers return same intent/confidence for same input."""
    phase1_agent = IntentClassifierAgent()
    phase1_agent.system_message = INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT

    phase2_agent = IntentClassifierAgent()

    async def fake_llm(_msg: str):
        return json.dumps({"intent": "SEARCH_BY_MERCHANT", "confidence": 0.87})

    monkeypatch.setattr(phase1_agent, "a_generate_reply", fake_llm, raising=False)
    monkeypatch.setattr(phase2_agent, "a_generate_reply", fake_llm, raising=False)

    msg = "mes achats Amazon"
    res1 = asyncio.run(phase1_agent.classify_for_team(msg, user_id=1))
    res2 = asyncio.run(phase2_agent.classify_for_team(msg, user_id=1))

    assert res1["intent"] == res2["intent"]
    assert res1["confidence"] == res2["confidence"]


def test_team_orchestration_message_includes_workflow_and_retry():
    """TEAM_ORCHESTRATION_PHASE2_MESSAGE should mention workflow and retry rules."""
    assert "Workflow strict" in TEAM_ORCHESTRATION_PHASE2_MESSAGE
    assert "R\u00e8gles de retry" in TEAM_ORCHESTRATION_PHASE2_MESSAGE
