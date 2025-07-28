"""
🚀 Client DeepSeek - Fallback LLM Optimisé

Client optimisé pour DeepSeek avec gestion cache, retry et monitoring coûts.
Reprend la logique éprouvée du fichier original avec améliorations.
"""

import json
import re
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
from conversation_service.models.enums import IntentType
from conversation_service.models.exceptions import LLMFallbackError, DeepSeekAPIError, TimeoutError
from conversation_service.config import get_deepseek_config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Réponse structurée du LLM"""
    intent: str
    confidence: float
    entities: Dict[str, Any]
    raw_response: str
    tokens_used: int
    cost_estimate: float
    processing_time_ms: float


class DeepSeekClient:
    """
    Client DeepSeek optimisé pour détection d'intention
    
    Fonctionnalités:
    - Retry intelligent avec backoff exponentiel
    - Cache réponses pour économiser coûts
    - Monitoring usage et coûts temps réel
    - Prompts optimisés classification intention
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = get_deepseek_config()
        
        # Client OpenAI async pour DeepSeek
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
        
        # Cache des réponses LLM (économie coûts)
        self._response_cache: Dict[str, LLMResponse] = {}
        self._cache_max_size = 100
        
        # Métriques usage
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_response_time": 0.0,
            "intent_distribution": {}
        }
        
        # Prompt système optimisé (du fichier original)
        self.system_prompt = """Tu es un classificateur d'intentions financières expert.

Analyse la requête utilisateur et retourne UNIQUEMENT un JSON avec cette structure exacte:
{
  "intent": "ACCOUNT_BALANCE|SEARCH_BY_CATEGORY|BUDGET_ANALYSIS|TRANSFER|SEARCH_BY_DATE|CARD_MANAGEMENT|GREETING|HELP|GOODBYE|UNKNOWN",
  "confidence": 0.0-1.0,
  "entities": {}
}

Intentions supportées:
- ACCOUNT_BALANCE: Demande de solde compte
- SEARCH_BY_CATEGORY: Recherche dépenses par catégorie 
- BUDGET_ANALYSIS: Analyse budget et dépenses
- TRANSFER: Virement d'argent
- SEARCH_BY_DATE: Recherche par date/période
- CARD_MANAGEMENT: Gestion carte bancaire
- GREETING: Salutation
- HELP: Demande d'aide
- GOODBYE: Au revoir
- UNKNOWN: Intention non claire

Extrait les entités pertinentes dans le champ "entities".
Sois précis et confiant dans tes classifications."""

    async def classify_intent(
        self, 
        query: str, 
        use_cache: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Classification d'intention principale via DeepSeek
        
        Args:
            query: Requête utilisateur
            use_cache: Utiliser cache si disponible
            context: Contexte conversationnel optionnel
            
        Returns:
            LLMResponse avec intention et entités
        """
        if not query or not query.strip():
            raise LLMFallbackError("Query vide pour classification LLM")
        
        self._metrics["total_requests"] += 1
        start_time = time.time()
        
        # Vérification cache
        cache_key = self._generate_cache_key(query, context)
        if use_cache and cache_key in self._response_cache:
            self._metrics["cache_hits"] += 1
            cached_response = self._response_cache[cache_key]
            self.logger.debug(f"Cache hit pour: {query[:50]}...")
            return cached_response
        
        try:
            # Construction messages
            messages = self._build_messages(query, context)
            
            # Appel API DeepSeek avec retry
            api_response = await self._make_api_request(messages)
            
            # Parsing réponse
            llm_response = self._parse_api_response(api_response, start_time)
            
            # Mise en cache si confiance élevée
            if use_cache and llm_response.confidence >= 0.6:
                self._cache_response(cache_key, llm_response)
            
            # Mise à jour métriques
            self._update_metrics(llm_response)
            
            self._metrics["successful_requests"] += 1
            return llm_response
            
        except Exception as e:
            self._metrics["failed_requests"] += 1
            self.logger.error(f"Erreur classification DeepSeek: {e}")
            
            if "timeout" in str(e).lower():
                raise TimeoutError(
                    f"Timeout DeepSeek après {self.config.timeout}s",
                    operation="intent_classification",
                    timeout_seconds=self.config.timeout
                )
            else:
                raise LLMFallbackError(
                    f"Erreur DeepSeek: {str(e)}",
                    llm_provider="deepseek",
                    api_error=str(e)
                )
    
    def _generate_cache_key(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Génère clé cache unique pour query + context"""
        # Normalisation query pour cache
        normalized_query = query.lower().strip()
        
        # Hash simple du contexte
        context_hash = ""
        if context:
            context_str = json.dumps(context, sort_keys=True, default=str)
            context_hash = str(hash(context_str))
        
        return f"{normalized_query}:{context_hash}"
    
    def _build_messages(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Construit messages pour API DeepSeek"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Ajout contexte si fourni
        if context:
            context_info = f"Contexte: {json.dumps(context, ensure_ascii=False)}\n\n"
            user_content = context_info + query
        else:
            user_content = query
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    async def _make_api_request(self, messages: List[Dict[str, str]]) -> Any:
        """Appel API DeepSeek avec retry intelligent"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"}  # Force JSON
                )
                return response
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    # Backoff exponentiel
                    wait_time = 2 ** attempt
                    self.logger.warning(
                        f"Tentative {attempt + 1} échouée, retry dans {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Tous les retries échoués: {e}")
        
        # Si toutes les tentatives ont échoué
        raise DeepSeekAPIError(
            str(last_exception), 
            status_code=getattr(last_exception, 'status_code', 500)
        )
    
    def _parse_api_response(self, api_response: Any, start_time: float) -> LLMResponse:
        """Parse et valide la réponse de l'API DeepSeek"""
        processing_time = (time.time() - start_time) * 1000
        
        try:
            # Extraction contenu réponse
            response_content = api_response.choices[0].message.content.strip()
            
            # Nettoyage si markdown JSON
            if "```json" in response_content:
                response_content = response_content.split("```json")[1].split("```")[0]
            elif "```" in response_content:
                response_content = response_content.split("```")[1].split("```")[0]
            
            # Parsing JSON
            try:
                parsed_data = json.loads(response_content)
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON invalide: {response_content}")
                # Fallback parsing
                parsed_data = self._fallback_parse_response(response_content)
            
            # Validation structure
            intent = parsed_data.get("intent", "UNKNOWN")
            confidence = float(parsed_data.get("confidence", 0.0))
            entities = parsed_data.get("entities", {})
            
            # Validation intention supportée
            if intent not in [e.value for e in IntentType]:
                self.logger.warning(f"Intention non supportée: {intent}")
                intent = "UNKNOWN"
                confidence = 0.0
            
            # Calcul coût
            tokens_used = api_response.usage.total_tokens if api_response.usage else 100
            cost = tokens_used * self.config.cost_per_1k_tokens / 1000
            
            return LLMResponse(
                intent=intent,
                confidence=min(max(confidence, 0.0), 1.0),  # Clamp [0,1]
                entities=entities,
                raw_response=response_content,
                tokens_used=tokens_used,
                cost_estimate=cost,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Erreur parsing réponse DeepSeek: {e}")
            raise LLMFallbackError(
                f"Erreur parsing réponse: {str(e)}",
                llm_provider="deepseek"
            )
    
    def _fallback_parse_response(self, response_content: str) -> Dict[str, Any]:
        """Parsing fallback si JSON malformé"""
        # Tentative extraction patterns basiques
        intent = "UNKNOWN"
        confidence = 0.0
        entities = {}
        
        # Extraction intention via regex
        intent_match = re.search(r'"intent":\s*"([^"]+)"', response_content)
        if intent_match:
            intent = intent_match.group(1)
        
        # Extraction confiance via regex
        conf_match = re.search(r'"confidence":\s*([0-9.]+)', response_content)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                confidence = 0.0
        
        self.logger.warning(f"Fallback parsing utilisé: intent={intent}, conf={confidence}")
        
        return {
            "intent": intent,
            "confidence": confidence,
            "entities": entities
        }
    
    def _cache_response(self, cache_key: str, response: LLMResponse):
        """Met en cache une réponse LLM"""
        if len(self._response_cache) >= self._cache_max_size:
            # Éviction LRU simple
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[cache_key] = response
        self.logger.debug(f"Réponse mise en cache pour: {cache_key[:50]}...")
    
    def _update_metrics(self, response: LLMResponse):
        """Met à jour métriques d'usage"""
        self._metrics["total_tokens"] += response.tokens_used
        self._metrics["total_cost"] += response.cost_estimate
        
        # Distribution par intention
        intent = response.intent
        if intent not in self._metrics["intent_distribution"]:
            self._metrics["intent_distribution"][intent] = 0
        self._metrics["intent_distribution"][intent] += 1
        
        # Temps de réponse moyen
        total_requests = self._metrics["total_requests"]
        if total_requests > 1:
            self._metrics["avg_response_time"] = (
                (self._metrics["avg_response_time"] * (total_requests - 1) + response.processing_time_ms) / 
                total_requests
            )
        else:
            self._metrics["avg_response_time"] = response.processing_time_ms
    
    async def batch_classify(
        self, 
        queries: List[str], 
        use_cache: bool = True
    ) -> List[LLMResponse]:
        """Classification batch de plusieurs requêtes"""
        if not queries:
            return []
        
        self.logger.info(f"Début classification batch de {len(queries)} requêtes")
        
        # Traitement parallèle des requêtes
        tasks = [
            self.classify_intent(query, use_cache=use_cache) 
            for query in queries
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrage erreurs et logging
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Erreur requête {i}: {result}")
                    # Créer réponse par défaut pour erreur
                    error_response = LLMResponse(
                        intent="UNKNOWN",
                        confidence=0.0,
                        entities={},
                        raw_response=f"ERROR: {str(result)}",
                        tokens_used=0,
                        cost_estimate=0.0,
                        processing_time_ms=0.0
                    )
                    valid_results.append(error_response)
                else:
                    valid_results.append(result)
            
            return valid_results
            
        except Exception as e:
            self.logger.error(f"Erreur batch classification: {e}")
            raise LLMFallbackError(f"Erreur batch: {str(e)}")
    
    def get_client_metrics(self) -> Dict[str, Any]:
        """Retourne métriques détaillées du client"""
        total_requests = self._metrics["total_requests"]
        
        if total_requests == 0:
            return {"total_requests": 0, "client_ready": True}
        
        success_rate = self._metrics["successful_requests"] / total_requests
        cache_hit_rate = self._metrics["cache_hits"] / total_requests
        
        return {
            "usage_stats": {
                "total_requests": total_requests,
                "successful_requests": self._metrics["successful_requests"],
                "failed_requests": self._metrics["failed_requests"],
                "success_rate": round(success_rate, 3),
                "cache_hits": self._metrics["cache_hits"],
                "cache_hit_rate": round(cache_hit_rate, 3)
            },
            "cost_analytics": {
                "total_tokens": self._metrics["total_tokens"],
                "total_cost_usd": round(self._metrics["total_cost"], 4),
                "avg_tokens_per_request": round(
                    self._metrics["total_tokens"] / total_requests, 1
                ),
                "avg_cost_per_request": round(
                    self._metrics["total_cost"] / total_requests, 6
                )
            },
            "performance_metrics": {
                "avg_response_time_ms": round(self._metrics["avg_response_time"], 2),
                "cache_size": len(self._response_cache),
                "cache_max_size": self._cache_max_size
            },
            "intent_distribution": self._metrics["intent_distribution"],
            "configuration": {
                "model": self.config.model,
                "base_url": self.config.base_url,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                "temperature": self.config.temperature
            }
        }
    
    def clear_cache(self):
        """Vide le cache des réponses"""
        cache_size = len(self._response_cache)
        self._response_cache.clear()
        self.logger.info(f"Cache vidé: {cache_size} entrées supprimées")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Statistiques du cache"""
        return {
            "cache_size": len(self._response_cache),
            "cache_max_size": self._cache_max_size,
            "cache_usage_percent": round(
                len(self._response_cache) / self._cache_max_size * 100, 1
            ),
            "total_cache_hits": self._metrics["cache_hits"]
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test de connectivité avec DeepSeek"""
        test_query = "bonjour"
        start_time = time.time()
        
        try:
            response = await self.classify_intent(test_query, use_cache=False)
            connection_time = (time.time() - start_time) * 1000
            
            return {
                "status": "connected",
                "response_time_ms": round(connection_time, 2),
                "model": self.config.model,
                "test_intent": response.intent,
                "test_confidence": response.confidence
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            }
    
    def reset_metrics(self):
        """Remet à zéro les métriques (utile pour tests)"""
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_response_time": 0.0,
            "intent_distribution": {}
        }


# Instance singleton du client DeepSeek
_deepseek_client_instance = None

async def get_deepseek_client() -> DeepSeekClient:
    """Factory function async pour récupérer instance DeepSeekClient singleton"""
    global _deepseek_client_instance
    if _deepseek_client_instance is None:
        _deepseek_client_instance = DeepSeekClient()
    return _deepseek_client_instance


def get_deepseek_client_sync() -> DeepSeekClient:
    """Factory function synchrone pour récupérer instance DeepSeekClient"""
    global _deepseek_client_instance
    if _deepseek_client_instance is None:
        _deepseek_client_instance = DeepSeekClient()
    return _deepseek_client_instance


# Fonction utilitaire de classification rapide
async def quick_intent_classification(query: str) -> Tuple[str, float]:
    """Classification rapide d'intention via DeepSeek"""
    client = await get_deepseek_client()
    response = await client.classify_intent(query)
    return response.intent, response.confidence


# Exports publics
__all__ = [
    "DeepSeekClient",
    "LLMResponse",
    "get_deepseek_client",
    "get_deepseek_client_sync",
    "quick_intent_classification"
]