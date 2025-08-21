import asyncio
import pytest

agent_module = pytest.importorskip("conversation_service.agents.response_generator_agent")
ResponseGeneratorAgent = agent_module.ResponseGeneratorAgent


class DummyLLM:
    def __init__(self) -> None:
        self.calls = []

    async def generate(self, prompt: str) -> str:  # pragma: no cover - trivial
        self.calls.append(prompt)
        return f"LLM:{len(self.calls)}"


def test_generation_and_cache():
    async def run_test():
        llm = DummyLLM()
        agent = ResponseGeneratorAgent(llm)

        search_results = [{"summary": "Balance is 100€"}]
        context = {"user_profile": {"name": "Alice"}}

        first = await agent.generate("user1", search_results, context)
        assert first.startswith("LLM:")
        assert len(llm.calls) == 1
        assert "Balance is 100€" in llm.calls[0]
        assert "Alice" in llm.calls[0]

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

