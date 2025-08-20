from conversation_service.agents.llm_intent_agent import LLMIntentAgent
class DummyOpenAIClient:
    def __init__(self):
        class _Completions:
            async def create(_self, *args, **kwargs):
                class Choice:
                    message = type("Msg", (), {"content": "{}"})
                return type("Resp", (), {"choices": [Choice()]})

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        self.chat = _Chat()


def test_llm_intent_agent_prefers_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    agent = LLMIntentAgent(openai_client=DummyOpenAIClient())
    assert agent.config.model_client_config["api_key"] == "openai-test-key"


def test_llm_intent_agent_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = LLMIntentAgent(openai_client=DummyOpenAIClient())
    assert agent.config.model_client_config["api_key"] == ""
