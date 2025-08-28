"""Tests for the Autogen-based IntentClassifierAgent."""

from conversation_service.agents.financial.intent_classifier import (
    IntentClassifierAgent,
)
from conversation_service.prompts.autogen.intent_classification_prompts import (
    AUTOGEN_INTENT_SYSTEM_MESSAGE,
)


def test_intent_classifier_agent_configuration():
    agent = IntentClassifierAgent()
    assert agent.name == "intent_classifier"
    assert agent.system_message == AUTOGEN_INTENT_SYSTEM_MESSAGE
    assert agent.max_consecutive_auto_reply == 1
    assert agent.llm_config == {
        "config_list": [
            {
                "model": "deepseek-chat",
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 800,
                "cache_seed": 42,
            }
        ]
    }
