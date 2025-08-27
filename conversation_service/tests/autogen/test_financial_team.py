import asyncio

import pytest


class DummyIntentAgent:
    """Stub intent classifier returning a fixed intent."""

    async def classify_intent(self, message: str):
        return {"intent": "BALANCE_INQUIRY"}


class DummyEntityAgent:
    """Stub entity extractor verifying that context is passed correctly."""

    def __init__(self):
        self.last_context = None

    async def extract_entities(self, message: str, context=None):
        self.last_context = context
        return {"entities": [{"merchant": "Amazon"}]}


class FailingEntityAgent(DummyEntityAgent):
    async def extract_entities(self, message: str, context=None):
        raise RuntimeError("entity extractor failed")


class FinancialTeam:
    """Minimal team orchestrating intent and entity agents."""

    def __init__(self, intent_agent, entity_agent):
        self.intent_agent = intent_agent
        self.entity_agent = entity_agent

    async def process(self, message: str):
        intent = await self.intent_agent.classify_intent(message)
        try:
            entities = await self.entity_agent.extract_entities(
                message, context={"intent": intent["intent"]}
            )
        except Exception as exc:  # pragma: no cover - failure path
            entities = {"error": str(exc)}
        return {"intent": intent, "entities": entities}


def test_workflow_intent_to_entities():
    intent_agent = DummyIntentAgent()
    entity_agent = DummyEntityAgent()
    team = FinancialTeam(intent_agent, entity_agent)

    result = asyncio.run(team.process("Solde Amazon"))

    assert result["intent"]["intent"] == "BALANCE_INQUIRY"
    assert result["entities"]["entities"][0]["merchant"] == "Amazon"
    assert entity_agent.last_context == {"intent": "BALANCE_INQUIRY"}


def test_agent_failure_handling():
    intent_agent = DummyIntentAgent()
    entity_agent = FailingEntityAgent()
    team = FinancialTeam(intent_agent, entity_agent)

    result = asyncio.run(team.process("Solde Amazon"))

    assert result["intent"]["intent"] == "BALANCE_INQUIRY"
    assert "error" in result["entities"]
