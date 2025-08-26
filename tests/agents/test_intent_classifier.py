"""
Tests complets pour IntentClassifierAgent avec JSON Output forcé
"""
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from conversation_service.agents.financial.intent_classifier import IntentClassifierAgent
from conversation_service.prompts.harena_intents import HarenaIntentType
from conversation_service.models.responses.conversation_responses import (
    IntentClassificationResult, IntentAlternative
)
from conversation_service.clients.deepseek_client import DeepSeekClient, DeepSeekError
from conversation_service.core.cache_manager import CacheManager


class TestIntentClassifierAgent:
    """Suite de tests complète pour IntentClassifierAgent"""

    @pytest.fixture
    def mock_deepseek_client(self):
        """Mock DeepSeek client avec réponses JSON valides"""
        client = AsyncMock(spec=DeepSeekClient)
        client.chat_completion = AsyncMock(return_value={
            "choices": [
                {
                    "message": {
                        "content": '{"intent": "BALANCE_INQUIRY", "confidence": 0.95, "reasoning": "Utilisateur demande son solde"}'
                    }
                }
            ],
            "usage": {"total_tokens": 150}
        })
        return client

    @pytest.fixture 
    def mock_cache_manager(self):
        """Mock Cache manager"""
        cache = AsyncMock(spec=CacheManager)
        cache.get_semantic_cache = AsyncMock(return_value=None)
        cache.set_semantic_cache = AsyncMock(return_value=True)
        return cache

    @pytest.fixture
    def agent(self, mock_deepseek_client, mock_cache_manager):
        """Agent avec mocks configurés"""
        return IntentClassifierAgent(
            deepseek_client=mock_deepseek_client,
            cache_manager=mock_cache_manager
        )

    @pytest.mark.asyncio
    async def test_classify_intent_success_balance_inquiry(
        self, agent, mock_deepseek_client, mock_cache_manager
    ):
        """Test classification réussie pour demande de solde"""
        
        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent.classify_intent("Mon solde")

        # Vérifications résultat
        assert isinstance(result, IntentClassificationResult)
        assert result.intent_type == HarenaIntentType.BALANCE_INQUIRY
        assert result.confidence == 0.95
        assert result.reasoning == "Utilisateur demande son solde"
        assert result.category == "ACCOUNT_BALANCE"
        assert result.is_supported is True
        assert result.processing_time_ms is not None

        # Vérifications appels
        mock_deepseek_client.chat_completion.assert_called_once()
        call_kwargs = mock_deepseek_client.chat_completion.call_args.kwargs
        
        # Vérification JSON Output forcé
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["temperature"] == 0.1
        assert "max_tokens" in call_kwargs

        # Vérification cache
        mock_cache_manager.set_semantic_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_intent_merchant_search(self, agent, mock_deepseek_client):
        """Test classification pour recherche marchand"""
        
        mock_deepseek_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": '{"intent": "SEARCH_BY_MERCHANT", "confidence": 0.94, "reasoning": "Recherche transactions Amazon"}'
                }
            }]
        }

        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent.classify_intent("Mes achats Amazon")

        assert result.intent_type == HarenaIntentType.SEARCH_BY_MERCHANT
        assert result.confidence == 0.94
        assert result.category == "FINANCIAL_QUERY"
        assert result.is_supported is True

    @pytest.mark.asyncio
    async def test_classify_intent_unsupported_transfer(self, agent, mock_deepseek_client):
        """Test classification pour intention non supportée"""
        
        mock_deepseek_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": '{"intent": "TRANSFER_REQUEST", "confidence": 0.97, "reasoning": "Demande de virement"}'
                }
            }]
        }

        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent.classify_intent("Faire un virement de 100€")

        assert result.intent_type == HarenaIntentType.TRANSFER_REQUEST
        assert result.confidence == 0.97
        assert result.category == "UNSUPPORTED"
        assert result.is_supported is False

    @pytest.mark.asyncio
    async def test_classify_intent_with_alternatives(self, agent, mock_deepseek_client):
        """Test classification avec alternatives"""
        
        mock_deepseek_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "intent": "SPENDING_ANALYSIS",
                        "confidence": 0.75,
                        "reasoning": "Analyse des dépenses demandée",
                        "alternatives": [
                            {"intent": "SEARCH_BY_CATEGORY", "confidence": 0.65},
                            {"intent": "BALANCE_INQUIRY", "confidence": 0.45}
                        ]
                    })
                }
            }]
        }

        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent.classify_intent("Combien j'ai dépensé en restaurants ?")

        assert result.intent_type == HarenaIntentType.SPENDING_ANALYSIS
        assert len(result.alternatives) == 2
        assert result.alternatives[0].intent_type == HarenaIntentType.SEARCH_BY_CATEGORY
        assert result.alternatives[0].confidence == 0.65

    @pytest.mark.asyncio
    async def test_classify_intent_cache_hit(self, agent, mock_deepseek_client, mock_cache_manager):
        """Test récupération depuis cache"""
        
        cached_result = {
            "intent_type": "GREETING",
            "confidence": 0.99,
            "reasoning": "Salutation standard",
            "original_message": "Bonjour",
            "category": "CONVERSATIONAL",
            "is_supported": True,
            "alternatives": [],
            "processing_time_ms": 5
        }
        
        mock_cache_manager.get_semantic_cache.return_value = cached_result

        result = await agent.classify_intent("Bonjour")

        assert result.intent_type == HarenaIntentType.GREETING
        assert result.confidence == 0.99
        
        # Vérification pas d'appel DeepSeek
        mock_deepseek_client.chat_completion.assert_not_called()
        
        # Vérification pas de nouvelle mise en cache
        mock_cache_manager.set_semantic_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_classify_intent_empty_message(self, agent):
        """Test avec message vide"""
        
        result = await agent.classify_intent("")
        
        assert result.intent_type == HarenaIntentType.UNKNOWN
        assert result.reasoning == "Message vide après nettoyage"
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_classify_intent_message_too_long(self, agent):
        """Test avec message trop long"""
        
        long_message = "A" * 1000  # Message très long
        
        result = await agent.classify_intent(long_message)
        
        assert result.intent_type == HarenaIntentType.UNCLEAR_INTENT
        assert "trop long" in result.reasoning

    @pytest.mark.asyncio
    async def test_classify_intent_json_parse_error(self, agent, mock_deepseek_client):
        """Test avec réponse JSON invalide malgré JSON Output"""
        
        mock_deepseek_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": '{"intent": "BALANCE_INQUIRY", "confidence": 0.9'  # JSON invalide
                }
            }]
        }

        result = await agent.classify_intent("Mon solde")

        assert result.intent_type == HarenaIntentType.ERROR
        assert "JSON parsing error" in result.reasoning

    @pytest.mark.asyncio
    async def test_classify_intent_invalid_intent_type(self, agent, mock_deepseek_client):
        """Test avec type d'intention invalide"""
        
        mock_deepseek_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": '{"intent": "INVALID_INTENT", "confidence": 0.9, "reasoning": "Test"}'
                }
            }]
        }

        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent.classify_intent("Test")

        assert result.intent_type == HarenaIntentType.UNCLEAR_INTENT
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_classify_intent_confidence_out_of_range(self, agent, mock_deepseek_client):
        """Test avec confidence hors limites"""
        
        mock_deepseek_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": '{"intent": "BALANCE_INQUIRY", "confidence": 1.5, "reasoning": "Test"}'
                }
            }]
        }

        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent.classify_intent("Mon solde")

        assert result.confidence == 0.5  # Correction automatique

    @pytest.mark.asyncio
    async def test_classify_intent_deepseek_error(self, agent, mock_deepseek_client):
        """Test avec erreur DeepSeek"""
        
        mock_deepseek_client.chat_completion.side_effect = DeepSeekError("API Error")

        result = await agent.classify_intent("Test")

        assert result.intent_type == HarenaIntentType.ERROR
        assert "API Error" in result.reasoning

    @pytest.mark.asyncio
    async def test_classify_intent_validation_failure(self, agent, mock_deepseek_client):
        """Test avec échec de validation"""
        
        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=False)
        ):
            result = await agent.classify_intent("Test")

        assert result.intent_type == HarenaIntentType.UNCLEAR_INTENT
        assert "Validation qualité échouée" in result.reasoning

    @pytest.mark.asyncio
    async def test_classify_intent_with_user_context(self, agent, mock_deepseek_client):
        """Test avec contexte utilisateur"""
        
        user_context = {
            "recent_intents": ["BALANCE_INQUIRY", "SEARCH_BY_MERCHANT"],
            "user_id": 123
        }

        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent.classify_intent("Mes dépenses", user_context)

        # Vérification que le contexte est utilisé dans le prompt
        call_args = mock_deepseek_client.chat_completion.call_args
        prompt_content = call_args[1]["messages"][1]["content"]
        assert "CONTEXTE" in prompt_content

    @pytest.mark.asyncio
    async def test_classify_intent_prompt_construction(self, agent, mock_deepseek_client):
        """Test construction du prompt avec few-shots"""
        
        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            await agent.classify_intent("Test message")

        call_args = mock_deepseek_client.chat_completion.call_args
        messages = call_args[1]["messages"]
        
        # Vérification structure messages
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        
        # Vérification contenu prompt utilisateur
        user_prompt = messages[1]["content"]
        assert "EXEMPLES DE RÉFÉRENCE" in user_prompt
        assert "FORMAT JSON OBLIGATOIRE" in user_prompt
        assert "Test message" in user_prompt

    @pytest.mark.asyncio
    async def test_classify_intent_processing_time_tracking(self, agent, mock_deepseek_client):
        """Test suivi du temps de traitement"""
        
        # Simulation délai
        async def delayed_response(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            return {
                "choices": [{
                    "message": {
                        "content": '{"intent": "BALANCE_INQUIRY", "confidence": 0.9, "reasoning": "Test"}'
                    }
                }]
            }
        
        mock_deepseek_client.chat_completion = delayed_response

        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent.classify_intent("Mon solde")

        assert result.processing_time_ms >= 100  # Au moins le délai simulé

    def test_agent_initialization(self, mock_deepseek_client, mock_cache_manager):
        """Test initialisation de l'agent"""
        
        agent = IntentClassifierAgent(
            deepseek_client=mock_deepseek_client,
            cache_manager=mock_cache_manager
        )

        assert agent.name == "intent_classifier"
        assert agent.deepseek_client == mock_deepseek_client
        assert agent.cache_manager == mock_cache_manager
        assert len(agent.supported_intents) > 0
        assert len(agent.few_shot_examples) > 0

    @pytest.mark.asyncio
    async def test_execute_method_compatibility(self, agent):
        """Test méthode execute pour compatibilité BaseAgent"""
        
        with patch.object(agent, 'classify_intent') as mock_classify:
            mock_classify.return_value = MagicMock()
            
            await agent.execute("test input", {"context": "test"})
            
            mock_classify.assert_called_once_with("test input", {"context": "test"})

    @pytest.mark.asyncio
    async def test_contextual_example_selection(self, agent, mock_deepseek_client):
        """Test sélection d'exemples contextuels"""
        
        # Test avec message contenant "Amazon"
        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            await agent.classify_intent("Mes achats Amazon")

        call_args = mock_deepseek_client.chat_completion.call_args
        prompt = call_args[1]["messages"][1]["content"]
        
        # Le prompt devrait contenir des exemples pertinents pour "Amazon"
        assert "Amazon" in prompt or "amazon" in prompt

    @pytest.mark.asyncio 
    async def test_classify_intent_greeting_high_confidence(self, agent, mock_deepseek_client):
        """Test classification salutation avec haute confiance"""
        
        mock_deepseek_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": '{"intent": "GREETING", "confidence": 0.99, "reasoning": "Salutation claire"}'
                }
            }]
        }

        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent.classify_intent("Bonjour")

        assert result.intent_type == HarenaIntentType.GREETING
        assert result.confidence == 0.99
        assert result.is_supported is True

    @pytest.mark.asyncio
    async def test_classify_intent_unclear_low_confidence(self, agent, mock_deepseek_client):
        """Test classification ambiguë avec faible confiance"""
        
        mock_deepseek_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": '{"intent": "UNCLEAR_INTENT", "confidence": 0.3, "reasoning": "Message ambigu"}'
                }
            }]
        }

        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent.classify_intent("Euh... aide")

        assert result.intent_type == HarenaIntentType.UNCLEAR_INTENT
        assert result.confidence == 0.3
        assert result.is_supported is False


# Tests de stress et edge cases
class TestIntentClassifierEdgeCases:
    """Tests des cas limites et situations exceptionnelles"""

    @pytest.fixture
    def agent_no_cache(self):
        """Agent sans cache"""
        mock_client = AsyncMock()
        mock_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": '{"intent": "ERROR", "confidence": 0.5, "reasoning": "Test"}'}}]
        })
        return IntentClassifierAgent(mock_client, None)

    @pytest.mark.asyncio
    async def test_classify_intent_no_cache_manager(self, agent_no_cache):
        """Test sans gestionnaire de cache"""
        
        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent_no_cache.classify_intent("Test")

        assert isinstance(result, IntentClassificationResult)

    @pytest.mark.asyncio
    async def test_classify_intent_special_characters(self, agent_no_cache):
        """Test avec caractères spéciaux"""
        
        special_message = "Mes dépenses € $ £ ¥ 中文 🚀"
        
        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent_no_cache.classify_intent(special_message)

        assert isinstance(result, IntentClassificationResult)

    @pytest.mark.asyncio
    async def test_classify_intent_numbers_and_symbols(self, agent_no_cache):
        """Test avec chiffres et symboles"""
        
        message = "Dépenses > 100€ && < 500€ #important @urgent"
        
        with patch(
            "conversation_service.utils.validation_utils.validate_intent_response",
            AsyncMock(return_value=True)
        ):
            result = await agent_no_cache.classify_intent(message)

        assert isinstance(result, IntentClassificationResult)