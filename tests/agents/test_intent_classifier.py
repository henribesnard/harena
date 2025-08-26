import pytest
from unittest.mock import AsyncMock, patch, ANY

from conversation_service.agents.financial.intent_classifier import IntentClassifierAgent
from conversation_service.prompts.harena_intents import HarenaIntentType
from conversation_service.models.responses.conversation_responses import IntentClassificationResult


@pytest.mark.asyncio
async def test_classify_intent_success() -> None:
    deepseek_client = AsyncMock()
    deepseek_client.chat_completion = AsyncMock(return_value={
        "choices": [
            {
                "message": {
                    "content": '{"intent": "BALANCE_INQUIRY", "confidence": 0.9, "reasoning": "Balance check"}'
                }
            }
        ]
    })

    cache_manager = AsyncMock()
    cache_manager.get_semantic_cache = AsyncMock(return_value=None)
    cache_manager.set_semantic_cache = AsyncMock()

    agent = IntentClassifierAgent(deepseek_client=deepseek_client, cache_manager=cache_manager)

    with patch(
        "conversation_service.utils.validation_utils.validate_intent_response",
        AsyncMock(return_value=True),
    ):
        result = await agent.classify_intent("What is my balance?")

    assert result.intent_type == HarenaIntentType.BALANCE_INQUIRY
    assert result.confidence == 0.9
    assert result.category == "ACCOUNT_BALANCE"
    assert result.is_supported is True

    deepseek_client.chat_completion.assert_called_once_with(
        messages=ANY,
        max_tokens=300,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    cache_manager.set_semantic_cache.assert_called_once()


@pytest.mark.asyncio
async def test_classify_intent_returns_cached_result() -> None:
    cached = IntentClassificationResult(
        intent_type=HarenaIntentType.BALANCE_INQUIRY,
        confidence=0.8,
        reasoning="cached",
        original_message="query",
        category="ACCOUNT_BALANCE",
        is_supported=True,
        alternatives=[],
        processing_time_ms=5,
    ).dict()

    deepseek_client = AsyncMock()
    deepseek_client.chat_completion = AsyncMock()

    cache_manager = AsyncMock()
    cache_manager.get_semantic_cache = AsyncMock(return_value=cached)
    cache_manager.set_semantic_cache = AsyncMock()

    agent = IntentClassifierAgent(deepseek_client=deepseek_client, cache_manager=cache_manager)

    result = await agent.classify_intent("balance")

    assert result.intent_type == HarenaIntentType.BALANCE_INQUIRY
    deepseek_client.chat_completion.assert_not_called()
    cache_manager.set_semantic_cache.assert_not_called()


@pytest.mark.asyncio
async def test_classify_intent_deepseek_error() -> None:
    deepseek_client = AsyncMock()
    deepseek_client.chat_completion = AsyncMock(return_value={
        "choices": [
            {"message": {"content": "invalid-json"}}
        ]
    })

    cache_manager = AsyncMock()
    cache_manager.get_semantic_cache = AsyncMock(return_value=None)
    cache_manager.set_semantic_cache = AsyncMock()

    agent = IntentClassifierAgent(deepseek_client=deepseek_client, cache_manager=cache_manager)

    with patch(
        "conversation_service.utils.validation_utils.validate_intent_response",
        AsyncMock(return_value=True),
    ):
        result = await agent.classify_intent("balance")

    assert result.intent_type == HarenaIntentType.ERROR
    deepseek_client.chat_completion.assert_called_once_with(
        messages=ANY,
        max_tokens=300,
        temperature=0.1,
        response_format={"type": "json_object"},
    )


@pytest.mark.asyncio
async def test_classify_intent_malformed_json_error() -> None:
    deepseek_client = AsyncMock()
    deepseek_client.chat_completion = AsyncMock(return_value={
        "choices": [
            {"message": {"content": '{"intent": "BALANCE_INQUIRY",}'}}
        ]
    })

    cache_manager = AsyncMock()
    cache_manager.get_semantic_cache = AsyncMock(return_value=None)
    cache_manager.set_semantic_cache = AsyncMock()

    agent = IntentClassifierAgent(deepseek_client=deepseek_client, cache_manager=cache_manager)

    with patch(
        "conversation_service.utils.validation_utils.validate_intent_response",
        AsyncMock(return_value=True),
    ):
        result = await agent.classify_intent("balance")

    assert result.intent_type == HarenaIntentType.ERROR
    deepseek_client.chat_completion.assert_called_once_with(
        messages=ANY,
        max_tokens=300,
        temperature=0.1,
        response_format={"type": "json_object"},
    )


@pytest.mark.asyncio
async def test_classify_intent_empty_message() -> None:
    deepseek_client = AsyncMock()
    deepseek_client.chat_completion = AsyncMock()

    cache_manager = AsyncMock()
    cache_manager.get_semantic_cache = AsyncMock()
    cache_manager.set_semantic_cache = AsyncMock()

    agent = IntentClassifierAgent(deepseek_client=deepseek_client, cache_manager=cache_manager)

    result = await agent.classify_intent("")

    assert result.intent_type == HarenaIntentType.UNKNOWN
    deepseek_client.chat_completion.assert_not_called()
    cache_manager.get_semantic_cache.assert_not_called()
