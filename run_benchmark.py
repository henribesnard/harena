import asyncio
from typing import Iterable

from conversation_service.agents.llm_intent_agent import (
    LLMIntentAgent,
    LLMOutputParsingError,
)


def run_benchmark(agent: LLMIntentAgent, samples: Iterable[str]) -> None:
    """Simple benchmark runner for the intent agent.

    Any :class:`LLMOutputParsingError` raised by the agent is caught and
    displayed to make debugging easier during benchmarking.
    """
    for text in samples:
        try:
            asyncio.run(agent.detect_intent(text, user_id=1))
        except LLMOutputParsingError as err:
            print(f"Parsing error: {err}")
        except Exception as err:  # pragma: no cover - benchmark helper
            print(f"Unexpected error: {err}")
