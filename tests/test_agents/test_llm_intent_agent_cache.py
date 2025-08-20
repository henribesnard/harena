import asyncio

from conversation_service.agents.llm_intent_agent import LLMIntentAgent


def test_intent_agent_caches_results(openai_mock):
    async def run():
        agent = LLMIntentAgent(openai_client=openai_mock)
        first = await agent.detect_intent("bonjour", user_id=1)
        second = await agent.detect_intent("bonjour", user_id=1)
        assert "cache_hit" not in first["metadata"]
        assert second["metadata"]["cache_hit"] is True
        assert first["content"] == second["content"]

    asyncio.run(run())
