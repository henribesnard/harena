import asyncio
import importlib.util
from pathlib import Path

module_path = Path(__file__).resolve().parents[2] / "agents" / "response_generator_agent.py"
spec = importlib.util.spec_from_file_location("response_generator_agent", module_path)
agent_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(agent_module)  # type: ignore[arg-type]
ResponseGeneratorAgent = agent_module.ResponseGeneratorAgent


class DummyLLM:
    def __init__(self) -> None:
        self.calls = []

    async def generate(self, prompt: str) -> str:  # pragma: no cover - trivial
        self.calls.append(prompt)
        return f"LLM:{len(self.calls)}"


def test_generation_personalisation_and_cache():
    async def run_test():
        llm = DummyLLM()
        agent = ResponseGeneratorAgent(llm)

        search_results = [
            {"summary": "Balance is 100€", "title": "Compte", "url": "x"}
        ]
        context = {
            "user_profile": {
                "name": "Alice",
                "preferences": {"currency": "EUR"},
            },
            "intent": "BALANCE_INQUIRY",
            "entities": [{"type": "ACCOUNT", "value": "checking"}],
        }

        first = await agent.generate("user1", search_results, context)
        assert first.startswith("LLM:")
        assert len(llm.calls) == 1
        prompt = llm.calls[0]
        assert "Balance is 100€" in prompt
        assert "Alice" in prompt
        assert "BALANCE_INQUIRY" in prompt
        assert "ACCOUNT" in prompt
        assert "EUR" in prompt

        second = await agent.generate("user1", search_results, context)
        assert second == first
        assert len(llm.calls) == 1

    asyncio.run(run_test())


def test_cache_expiry():
    async def run_test():
        llm = DummyLLM()
        agent = ResponseGeneratorAgent(llm, ttl=1)

        search_results = [{"summary": "Balance is 100€"}]
        context = {}

        await agent.generate("user1", search_results, context)
        await agent.generate("user1", search_results, context)
        assert len(llm.calls) == 1

        await asyncio.sleep(1.1)
        await agent.generate("user1", search_results, context)
        assert len(llm.calls) == 2

    asyncio.run(run_test())


def test_cache_varies_with_context():
    async def run_test():
        llm = DummyLLM()
        agent = ResponseGeneratorAgent(llm)

        search_results = [{"summary": "Balance is 100€"}]
        ctx1 = {"user_profile": {"name": "Alice", "preferences": {"currency": "EUR"}}}
        ctx2 = {"user_profile": {"name": "Alice", "preferences": {"currency": "USD"}}}

        await agent.generate("user1", search_results, ctx1)
        await agent.generate("user1", search_results, ctx1)
        await agent.generate("user1", search_results, ctx2)
        assert len(llm.calls) == 2

    asyncio.run(run_test())


class FailingLLM:
    async def generate(self, prompt: str) -> str:  # pragma: no cover - trivial
        raise RuntimeError("boom")


def test_error_handling_returns_suggestion():
    async def run_test():
        llm = FailingLLM()
        agent = ResponseGeneratorAgent(llm)

        search_results = [{"title": "Alt"}]
        result = await agent.generate("u", search_results, {})
        assert "Suggestion" in result
        assert "Alt" in result

    asyncio.run(run_test())

