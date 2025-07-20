"""
Fallback LLM DeepSeek pour requêtes complexes niveau L2
Intégration optimisée avec cache intelligent
"""

import asyncio
import json
from typing import Dict, Any, Optional
import httpx

from .models import IntentResult, IntentLevel, IntentConfidence, IntentType
from conversation_service.config import settings
from conversation_service.utils.logging import get_logger

logger = get_logger(__name__)


class LLMFallback:
    """
    Fallback DeepSeek pour analyse intentions complexes
    Utilisé uniquement pour 3% des requêtes les plus ambiguës
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=settings.DEEPSEEK_RESPONSE_TIMEOUT,
            headers={
                "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
        )
        
        self.system_prompt = self._build_financial_system_prompt()
    
    def _build_financial_system_prompt(self) -> str:
        """Prompt système optimisé pour intentions financières"""
        return """Tu es un expert en analyse d'intentions pour une application bancaire française.

Ton rôle: analyser des requêtes utilisateur complexes et extraire:
1. L'intention principale (balance_check, expense_analysis, transfer, transaction_search, budget_inquiry, general_query)
2. Les entités importantes (montants, dates, catégories, contacts)
3. Un score de confiance

Réponds UNIQUEMENT en JSON valide avec cette structure:
{
    "intent_type": "expense_analysis",
    "entities": {
        "category": "restaurant",
        "time_period": "janvier",
        "amount_range": "50-100"
    },
    "confidence_score": 0.89,
    "reasoning": "Requête claire sur analyse dépenses restaurant"
}

Intentions disponibles:
- balance_check: vérification soldes comptes
- expense_analysis: analyse dépenses par catégorie/période  
- transfer: virements et transferts
- transaction_search: recherche transactions spécifiques
- budget_inquiry: questions sur budgets et limites
- general_query: questions générales banking

Sois précis et extrait toutes les entités pertinentes."""
    
    async def analyze_complex_intent(
        self,
        original_query: str,
        normalized_query: str, 
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> IntentResult:
        """
        Analyse LLM pour intentions complexes/ambiguës
        Performance: 200-500ms selon charge API
        """
        try:
            # Construction prompt avec contexte
            user_prompt = self._build_user_prompt(
                original_query, normalized_query, conversation_context
            )
            
            # Appel DeepSeek API
            response = await self._call_deepseek_api(user_prompt)
            
            if response:
                return self._parse_llm_response(response, original_query)
            else:
                return self._create_fallback_result(original_query)
                
        except Exception as e:
            logger.error(f"Erreur LLM fallback: {e}")
            return self._create_fallback_result(original_query, error=str(e))
    
    def _build_user_prompt(
        self,
        original_query: str,
        normalized_query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Construction prompt utilisateur avec contexte"""
        
        prompt_parts = [
            f"Requête utilisateur: '{original_query}'",
            f"Requête normalisée: '{normalized_query}'"
        ]
        
        # Ajout contexte conversationnel si disponible
        if context:
            if "previous_intent" in context:
                prompt_parts.append(f"Intention précédente: {context['previous_intent']}")
            
            if "conversation_history" in context:
                recent_history = context["conversation_history"][-2:]  # 2 derniers échanges
                prompt_parts.append(f"Historique récent: {recent_history}")
        
        prompt_parts.append("Analyse cette requête et réponds en JSON.")
        
        return "\n".join(prompt_parts)
    
    async def _call_deepseek_api(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Appel API DeepSeek avec gestion erreurs"""
        
        payload = {
            "model": settings.DEEPSEEK_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": settings.DEEPSEEK_MAX_TOKENS,
            "temperature": 0.1,  # Peu de créativité pour intentions
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = await self.client.post(
                settings.DEEPSEEK_API_URL,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                logger.warning(f"DeepSeek API error: {response.status_code}")
                return None
                
        except asyncio.TimeoutError:
            logger.warning("DeepSeek API timeout")
            return None
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            return None
    
    def _parse_llm_response(self, llm_response: Dict[str, Any], original_query: str) -> IntentResult:
        """Parse et validation réponse LLM"""
        
        try:
            intent_type = llm_response.get("intent_type", "general_query")
            entities = llm_response.get("entities", {})
            confidence_score = llm_response.get("confidence_score", 0.7)
            reasoning = llm_response.get("reasoning", "")
            
            # Validation intention
            valid_intents = [intent.value for intent in IntentType]
            if intent_type not in valid_intents:
                intent_type = "general_query"
                confidence_score = max(0.6, confidence_score - 0.2)
            
            # Validation score confiance
            confidence_score = max(0.0, min(1.0, float(confidence_score)))
            
            # Enrichissement entités avec requête originale
            entities["original_query"] = original_query
            entities["llm_reasoning"] = reasoning
            
            return IntentResult(
                intent_type=intent_type,
                entities=entities,
                confidence=IntentConfidence(
                    score=confidence_score,
                    level=IntentLevel.L2_LLM
                ),
                level=IntentLevel.L2_LLM,
                latency_ms=0,  # Sera mis à jour par le moteur principal
                metadata={
                    "llm_provider": "deepseek",
                    "llm_model": settings.DEEPSEEK_CHAT_MODEL,
                    "llm_reasoning": reasoning,
                    "raw_response": llm_response
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur parsing réponse LLM: {e}")
            return self._create_fallback_result(original_query, error=str(e))
    
    def _create_fallback_result(self, original_query: str, error: str = None) -> IntentResult:
        """Création résultat fallback garanti"""
        
        return IntentResult(
            intent_type="general_query",
            entities={
                "original_query": original_query,
                "fallback_reason": error or "llm_unavailable"
            },
            confidence=IntentConfidence(
                score=0.6,
                level=IntentLevel.L2_LLM
            ),
            level=IntentLevel.L2_LLM,
            latency_ms=0,
            metadata={
                "fallback": True,
                "error": error,
                "llm_provider": "deepseek"
            }
        )
    
    async def close(self):
        """Fermeture propre client HTTP"""
        await self.client.aclose()