import json
import asyncio


class DummyLLM:
    """Simple LLM stub that records prompts and returns a fixed JSON payload."""

    def __init__(self, payload=None):
        self.payload = payload or {"entities": [{"type": "merchant", "value": "Amazon"}]}
        self.last_prompt = None

    async def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return json.dumps(self.payload)


class EntityExtractorAgent:
    """Minimal entity extractor using an LLM stub.

    The agent simply forwards the prompt (optionally enriched with a
    collaboration context) to the provided LLM client and parses the JSON
    response.
    """

    def __init__(self, llm):
        self.llm = llm

    async def extract_entities(self, message: str, context: dict | None = None):
        prompt = message
        if context:
            prompt = f"{context.get('collaboration', '')}\n{message}".strip()
        result = await self.llm.complete(prompt)
        return json.loads(result)


def test_extract_entities_json():
    """The agent should return parsed JSON from the LLM response."""

    llm = DummyLLM()
    agent = EntityExtractorAgent(llm)

    result = asyncio.run(agent.extract_entities("J'ai dépensé 20€ chez Amazon"))

    assert isinstance(result, dict)
    assert result["entities"][0]["value"] == "Amazon"


def test_extract_entities_collaboration_context():
    """The collaboration context should be included in the prompt sent to the LLM."""

    llm = DummyLLM()
    agent = EntityExtractorAgent(llm)

    context = {"collaboration": "intent: BALANCE_INQUIRY"}
    asyncio.run(agent.extract_entities("Mon solde chez Amazon", context=context))

    assert "BALANCE_INQUIRY" in llm.last_prompt
    assert llm.last_prompt.startswith("intent: BALANCE_INQUIRY")
