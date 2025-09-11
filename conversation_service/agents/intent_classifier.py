"""
Intent Classifier Agent - LLM Component
Version 2.0 - Hybrid IA + Pure Logic Architecture

Responsabilité : Classification autonome des intentions utilisateur avec extraction d'entités.
- Few-shot learning sans règles rigides
- Fallback multi-providers (DeepSeek → OpenAI → Ollama)
- Cache intelligent des classifications
- JSON Output forcé pour fiabilité
"""

import logging
import json
import hashlib
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass

from conversation_service.agents.base_agent import BaseAgent
from conversation_service.clients.deepseek_client import DeepSeekClient, DeepSeekError
from conversation_service.core.cache_manager import CacheManager
from conversation_service.utils.metrics_collector import metrics_collector
# Remove unused imports - using dataclass ClassificationResult instead

logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """Résultat de classification d'intention"""
    intent: str
    sub_intent: Optional[str]
    confidence: float
    entities: List[Dict[str, Any]]
    reasoning: Optional[str]
    provider_used: str
    processing_time_ms: int
    cached: bool = False

class IntentClassifierAgent(BaseAgent):
    """
    Agent LLM pour classification d'intentions avec few-shot learning
    
    Architecture v2.0 - Fonctionnement autonome sans règles rigides
    """
    
    def __init__(
        self, 
        deepseek_client: DeepSeekClient,
        cache_manager: Optional[CacheManager] = None,
        config: Optional[Dict] = None
    ):
        super().__init__()
        self.deepseek_client = deepseek_client
        self.cache_manager = cache_manager
        
        # Configuration LLM providers
        self.config = config or {}
        self.providers_config = self.config.get("llm_providers", {
            "primary": "deepseek",
            "fallback": "openai", 
            "local": "ollama_llama3"
        })
        
        self.llm_configs = self.config.get("llm_configurations", {
            "deepseek": {
                "model": "deepseek-chat",
                "temperature": 0.1,
                "max_tokens": 500,
                "timeout": 3000
            },
            "openai": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 500,
                "timeout": 5000
            }
        })
        
        # Cache configuration
        self.cache_ttl = {
            "classification": 3600,  # 1h pour classifications
            "patterns": 86400        # 24h pour patterns fréquents
        }
        
        logger.info("IntentClassifierAgent v2.0 initialisé")
    
    async def classify_intent(
        self, 
        user_message: str, 
        conversation_context: Optional[List[Dict]] = None,
        user_id: Optional[int] = None
    ) -> ClassificationResult:
        """
        Classification d'intention avec fallback multi-providers
        
        Args:
            user_message: Message utilisateur à classifier
            conversation_context: Contexte conversationnel (5 derniers échanges)
            user_id: ID utilisateur pour personnalisation
            
        Returns:
            ClassificationResult avec intention, entités et métadonnées
        """
        start_time = datetime.now()
        
        try:
            # Génération clé cache
            cache_key = self._generate_cache_key(user_message, conversation_context)
            
            # Vérification cache
            if self.cache_manager:
                cached_result = await self._get_cached_classification(cache_key)
                if cached_result:
                    metrics_collector.increment_counter("intent_classifier.cache.hits")
                    return cached_result
                    
                metrics_collector.increment_counter("intent_classifier.cache.misses")
            
            # Classification via LLM avec fallback
            classification_result = await self._classify_with_fallback(
                user_message, conversation_context, user_id
            )
            
            # Cache du résultat
            if self.cache_manager and classification_result.confidence >= 0.7:
                await self._cache_classification_result(cache_key, classification_result)
            
            # Métriques
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            classification_result.processing_time_ms = int(processing_time)
            
            metrics_collector.record_histogram(
                "intent_classifier.processing_time_ms", 
                processing_time
            )
            metrics_collector.increment_counter(
                f"intent_classifier.provider.{classification_result.provider_used}"
            )
            
            return classification_result
            
        except Exception as e:
            logger.error(f"Erreur classification intention: {str(e)}")
            metrics_collector.increment_counter("intent_classifier.errors")
            
            # Fallback d'urgence
            return ClassificationResult(
                intent="UNCLEAR_INTENT",
                sub_intent=None,
                confidence=0.1,
                entities=[],
                reasoning="Erreur technique classification",
                provider_used="fallback_emergency",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
    
    async def _classify_with_fallback(
        self,
        user_message: str,
        conversation_context: Optional[List[Dict]],
        user_id: Optional[int]
    ) -> ClassificationResult:
        """Classification avec stratégie de fallback"""
        
        providers = ["primary", "fallback", "local"]
        last_error = None
        
        for provider_type in providers:
            try:
                provider_name = self.providers_config.get(provider_type)
                if not provider_name:
                    continue
                
                logger.debug(f"Tentative classification avec {provider_name}")
                
                result = await self._classify_with_provider(
                    user_message, conversation_context, user_id, provider_name
                )
                
                if result:
                    result.provider_used = provider_name
                    return result
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name} échoué: {str(e)}")
                continue
        
        # Tous les providers ont échoué
        logger.error(f"Tous les providers LLM ont échoué. Dernière erreur: {last_error}")
        raise Exception(f"Classification impossible - tous providers échoués: {last_error}")
    
    async def _classify_with_provider(
        self,
        user_message: str,
        conversation_context: Optional[List[Dict]],
        user_id: Optional[int],
        provider_name: str
    ) -> Optional[ClassificationResult]:
        """Classification avec un provider spécifique"""
        
        if provider_name != "deepseek":
            # Pour l'instant, seul DeepSeek est implémenté
            logger.warning(f"Provider {provider_name} non implémenté, utilisation DeepSeek")
            provider_name = "deepseek"
        
        # Construction du prompt
        classification_prompt = self._build_classification_prompt(
            user_message, conversation_context
        )
        
        # Configuration LLM
        llm_config = self.llm_configs.get(provider_name, self.llm_configs["deepseek"])
        
        try:
            # Appel LLM avec JSON Output forcé
            response = await self.deepseek_client.chat_completion(
                messages=classification_prompt,
                max_tokens=llm_config["max_tokens"],
                temperature=llm_config["temperature"],
                response_format={"type": "json_object"}
            )
            
            # Parsing réponse
            if not response or "choices" not in response:
                logger.error("Réponse LLM invalide")
                return None
            
            content = response["choices"][0]["message"]["content"]
            classification_data = json.loads(content)
            
            # Validation et structuration résultat
            return self._parse_classification_response(classification_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON LLM: {str(e)} - Contenu: {content}")
            return None
        except Exception as e:
            logger.error(f"Erreur appel LLM {provider_name}: {str(e)}")
            return None
    
    def _build_classification_prompt(
        self, 
        user_message: str, 
        conversation_context: Optional[List[Dict]]
    ) -> List[Dict[str, str]]:
        """Construction du prompt de classification avec few-shot examples"""
        
        system_prompt = """Tu es un expert en classification d'intentions financières. 
        
Analyse le message utilisateur et le contexte pour identifier :
- L'intention principale (BALANCE_INQUIRY, TRANSACTION_SEARCH, SPENDING_ANALYSIS, etc.)
- Le sous-type d'intention si applicable
- Les entités financières (montants, dates, marchands, catégories)
- Le niveau de confiance de ta classification

IMPORTANT: Réponds UNIQUEMENT en JSON valide avec cette structure exacte:
{
    "intent": "INTENT_NAME",
    "sub_intent": "optional_sub_intent",
    "confidence": 0.95,
    "entities": [
        {
            "type": "amount",
            "value": "50",
            "currency": "EUR",
            "raw_text": "50 euros"
        },
        {
            "type": "date_range", 
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
            "raw_text": "ce mois"
        }
    ],
    "reasoning": "courte_explication_classification"
}

Exemples Few-Shot:

User: "Quel est mon solde actuel ?"
{
    "intent": "BALANCE_INQUIRY",
    "sub_intent": "CURRENT_BALANCE",
    "confidence": 0.98,
    "entities": [],
    "reasoning": "Demande directe de solde actuel"
}

User: "Mes dépenses chez Carrefour le mois dernier"
{
    "intent": "TRANSACTION_SEARCH",
    "sub_intent": "BY_MERCHANT_AND_DATE",
    "confidence": 0.95,
    "entities": [
        {
            "type": "merchant",
            "value": "Carrefour",
            "raw_text": "Carrefour"
        },
        {
            "type": "date_range",
            "relative_period": "last_month",
            "raw_text": "le mois dernier"
        }
    ],
    "reasoning": "Recherche transactions par marchand et période"
}

User: "Analyse mes dépenses de plus de 100€ en restaurants"
{
    "intent": "SPENDING_ANALYSIS",
    "sub_intent": "BY_AMOUNT_AND_CATEGORY",
    "confidence": 0.92,
    "entities": [
        {
            "type": "amount_threshold",
            "operator": "greater_than",
            "value": "100",
            "currency": "EUR",
            "raw_text": "plus de 100€"
        },
        {
            "type": "category",
            "value": "restaurants",
            "raw_text": "restaurants"
        }
    ],
    "reasoning": "Analyse dépenses avec filtres montant et catégorie"
}"""

        # Contexte conversationnel
        context_str = ""
        if conversation_context:
            context_str = "\n\nContexte conversationnel récent:\n"
            for i, exchange in enumerate(conversation_context[-3:]):  # 3 derniers échanges
                context_str += f"User: {exchange.get('user_message', '')}\n"
                context_str += f"Assistant: {exchange.get('assistant_response', '')}\n"
        
        user_prompt = f"""Message à classifier: "{user_message}"{context_str}

Classifie ce message selon les exemples donnés. Retourne UNIQUEMENT le JSON de classification."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _parse_classification_response(self, classification_data: Dict) -> ClassificationResult:
        """Parse et valide la réponse de classification"""
        
        # Validation champs obligatoires
        required_fields = ["intent", "confidence"]
        for field in required_fields:
            if field not in classification_data:
                raise ValueError(f"Champ obligatoire manquant: {field}")
        
        # Normalisation confidence
        confidence = float(classification_data["confidence"])
        if confidence < 0.0 or confidence > 1.0:
            logger.warning(f"Confidence invalide: {confidence}, normalisation à 0.5")
            confidence = 0.5
        
        return ClassificationResult(
            intent=classification_data["intent"].upper(),
            sub_intent=classification_data.get("sub_intent"),
            confidence=confidence,
            entities=classification_data.get("entities", []),
            reasoning=classification_data.get("reasoning"),
            provider_used="",  # Sera défini par le caller
            processing_time_ms=0  # Sera défini par le caller
        )
    
    def _generate_cache_key(
        self, 
        user_message: str, 
        conversation_context: Optional[List[Dict]]
    ) -> str:
        """Génération clé cache pour classification"""
        
        # Hash du message + contexte pour clé unique
        context_hash = ""
        if conversation_context:
            context_str = json.dumps(conversation_context, sort_keys=True)
            context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]
        
        message_hash = hashlib.md5(user_message.lower().encode()).hexdigest()[:12]
        
        return f"intent_classification:{message_hash}:{context_hash}"
    
    async def _get_cached_classification(self, cache_key: str) -> Optional[ClassificationResult]:
        """Récupération classification en cache"""
        try:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                # Reconstruction de l'objet depuis le cache
                cached_data["cached"] = True
                return ClassificationResult(**cached_data)
        except Exception as e:
            logger.debug(f"Erreur lecture cache: {str(e)}")
        return None
    
    async def _cache_classification_result(
        self, 
        cache_key: str, 
        result: ClassificationResult
    ) -> None:
        """Mise en cache du résultat de classification"""
        try:
            # Sérialisation pour cache
            cache_data = {
                "intent": result.intent,
                "sub_intent": result.sub_intent,
                "confidence": result.confidence,
                "entities": result.entities,
                "reasoning": result.reasoning,
                "provider_used": result.provider_used,
                "processing_time_ms": result.processing_time_ms
            }
            
            await self.cache_manager.set(
                cache_key, 
                cache_data, 
                ttl=self.cache_ttl["classification"]
            )
            
        except Exception as e:
            logger.debug(f"Erreur mise en cache: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check de l'agent classificateur"""
        try:
            # Test simple de classification
            test_result = await self.classify_intent(
                "Quel est mon solde ?",
                conversation_context=[],
                user_id=None
            )
            
            return {
                "status": "healthy",
                "component": "intent_classifier_agent",
                "last_test": datetime.now(timezone.utc).isoformat(),
                "test_intent": test_result.intent,
                "test_confidence": test_result.confidence,
                "test_provider": test_result.provider_used,
                "test_processing_time_ms": test_result.processing_time_ms,
                "cache_available": self.cache_manager is not None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "intent_classifier_agent",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }