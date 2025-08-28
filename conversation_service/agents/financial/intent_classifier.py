"""Autogen-based agent for financial intent classification."""

from autogen import AssistantAgent
from conversation_service.prompts.autogen.intent_classification_prompts import (
    AUTOGEN_INTENT_SYSTEM_MESSAGE,
)

try:
    from autogen.agentchat.contrib.capabilities.teachability import Teachability
except Exception:  # pragma: no cover - capability is optional
    Teachability = None


class IntentClassifierAgent(AssistantAgent):
    """Assistant agent configured for financial intent classification."""

    def __init__(self, *_, **__):
        super().__init__(
            name="intent_classifier",
            system_message=AUTOGEN_INTENT_SYSTEM_MESSAGE,
            llm_config=self._create_llm_config(),
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
        )
        if Teachability is not None:
            self.add_capability(Teachability(verbosity=0))

    @staticmethod
    def _create_llm_config():
        return {
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

    async def classify_intent(self, *args, **kwargs):  # pragma: no cover
        """Placeholder classification method."""
        raise NotImplementedError("Intent classification not implemented.")
