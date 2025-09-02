"""
Exemple d'intégration de l'équipe AutoGen dans l'infrastructure existante
Démontre l'utilisation avec FastAPI, cache et métriques
"""

import logging
import asyncio
from typing import Dict, Any
from fastapi import HTTPException

from .multi_agent_financial_team import MultiAgentFinancialTeam
from ..clients.deepseek_client import DeepSeekClient

logger = logging.getLogger("conversation_service.integration")


class TeamIntegrationService:
    """
    Service d'intégration équipe AutoGen avec API existante
    """
    
    def __init__(self):
        """Initialise service avec client DeepSeek existant"""
        
        # Client DeepSeek existant
        self.deepseek_client = DeepSeekClient()
        
        # Équipe AutoGen
        self.financial_team = None
        self._initialize_team()
    
    def _initialize_team(self):
        """Initialisation lazy de l'équipe"""
        try:
            self.financial_team = MultiAgentFinancialTeam(
                deepseek_client=self.deepseek_client
            )
            logger.info("Équipe AutoGen initialisée avec succès")
        except Exception as e:
            logger.error(f"Échec initialisation équipe: {e}")
            self.financial_team = None
    
    async def process_financial_query(
        self, 
        user_message: str, 
        user_id: int
    ) -> Dict[str, Any]:
        """
        Point d'entrée principal pour requêtes financières
        Utilise équipe AutoGen si disponible, fallback sur agents individuels
        """
        
        if self.financial_team:
            try:
                # Traitement équipe AutoGen (infrastructure intégrée)
                result = await self.financial_team.process_user_message(
                    user_message, 
                    user_id
                )
                
                # Enrichissement résultat pour API
                return self._format_team_response(result)
                
            except Exception as e:
                logger.warning(f"Équipe AutoGen échouée, fallback: {e}")
                return await self._fallback_to_individual_agents(user_message, user_id)
        else:
            # Fallback direct si équipe non disponible
            return await self._fallback_to_individual_agents(user_message, user_id)
    
    def _format_team_response(self, team_result: Dict) -> Dict[str, Any]:
        """Formate réponse équipe pour API existante"""
        
        intent_result = team_result.get("intent_result", {})
        entities_result = team_result.get("entities_result", {})
        
        return {
            "processing_method": "multi_agent_team",
            "success": team_result.get("workflow_success", False),
            "intent": {
                "classification": intent_result.get("intent", "UNKNOWN"),
                "confidence": intent_result.get("confidence", 0.0),
                "reasoning": intent_result.get("reasoning", "")
            },
            "entities": entities_result.get("entities", {}),
            "entities_confidence": entities_result.get("confidence", 0.0),
            "team_metrics": {
                "coherence_score": team_result.get("coherence_validation", {}).get("score", 0.0),
                "agents_used": team_result.get("agents_sequence", []),
                "processing_quality": team_result.get("team_metadata", {}).get("processing_quality", "unknown")
            },
            "from_cache": team_result.get("from_cache", False),
            "metadata": {
                "workflow_completed": team_result.get("workflow_completed"),
                "timestamp": team_result.get("cache_timestamp") or team_result.get("team_metadata", {}).get("processing_timestamp")
            }
        }
    
    async def _fallback_to_individual_agents(
        self, 
        user_message: str, 
        user_id: int
    ) -> Dict[str, Any]:
        """Fallback sur agents individuels si équipe non disponible"""
        
        # Import local pour éviter dépendances circulaires
        from ..agents.financial.intent_classifier import IntentClassifierAgent
        from ..agents.financial.entity_extractor import EntityExtractorAgent
        
        try:
            # Agents individuels (mode standard, pas AutoGen)
            intent_agent = IntentClassifierAgent(autogen_mode=False)
            entity_agent = EntityExtractorAgent(autogen_mode=False)
            
            # Classification intention
            intent_result = await intent_agent.process_user_message(user_message, user_id)
            
            # Extraction entités (pas de contexte équipe)
            entities_result = await entity_agent.process_user_message(user_message, user_id)
            
            return {
                "processing_method": "individual_agents_fallback",
                "success": True,
                "intent": intent_result,
                "entities": entities_result.get("entities", {}),
                "entities_confidence": entities_result.get("confidence", 0.0),
                "team_metrics": {
                    "coherence_score": 0.5,  # Score neutre sans validation équipe
                    "agents_used": ["intent_classifier", "entity_extractor"],
                    "processing_quality": "fallback"
                },
                "from_cache": False,
                "metadata": {
                    "workflow_completed": "individual_agents_fallback",
                    "fallback_reason": "team_unavailable"
                }
            }
            
        except Exception as e:
            logger.error(f"Échec fallback agents individuels: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Traitement impossible: équipe et agents individuels échoués"
            )
    
    async def get_team_health(self) -> Dict[str, Any]:
        """Health check équipe pour monitoring"""
        
        if not self.financial_team:
            return {
                "status": "unavailable",
                "reason": "team_not_initialized",
                "fallback_available": True
            }
        
        try:
            health_result = await self.financial_team.health_check()
            return {
                "status": "available",
                "team_health": health_result,
                "fallback_available": True
            }
        except Exception as e:
            return {
                "status": "degraded",
                "error": str(e),
                "fallback_available": True
            }
    
    async def get_team_metrics(self) -> Dict[str, Any]:
        """Métriques détaillées équipe pour dashboard"""
        
        if not self.financial_team:
            return {
                "available": False,
                "reason": "team_not_initialized"
            }
        
        return {
            "available": True,
            "statistics": self.financial_team.get_team_statistics(),
            "health": await self.financial_team.health_check()
        }


# Exemple d'intégration FastAPI
"""
# Dans api/routes/conversation.py

from teams.integration_example import TeamIntegrationService

# Service global
team_service = TeamIntegrationService()

@router.post("/conversation/financial")
async def process_financial_conversation(
    request: ConversationRequest,
    user_id: int = Depends(get_current_user_id)
):
    try:
        # Traitement avec équipe AutoGen intégrée
        result = await team_service.process_financial_query(
            user_message=request.message,
            user_id=user_id
        )
        
        return ConversationResponse(
            success=result["success"],
            intent=result["intent"],
            entities=result["entities"],
            processing_method=result["processing_method"],
            metrics=result["team_metrics"],
            from_cache=result["from_cache"]
        )
        
    except Exception as e:
        logger.error(f"Erreur traitement conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/team/health")
async def get_team_health():
    return await team_service.get_team_health()

@router.get("/team/metrics") 
async def get_team_metrics():
    return await team_service.get_team_metrics()
"""


# Exemple d'utilisation standalone
async def example_usage():
    """Exemple d'utilisation directe de l'équipe"""
    
    # Initialisation service
    service = TeamIntegrationService()
    
    # Messages test
    test_messages = [
        "Combien j'ai dépensé chez Carrefour ce mois ?",
        "Mon solde actuel",
        "Mes achats de plus de 50€ la semaine dernière",
        "Historique des virements en janvier"
    ]
    
    # Traitement avec métriques
    for i, message in enumerate(test_messages):
        print(f"\n=== Test {i+1}: {message} ===")
        
        try:
            result = await service.process_financial_query(message, user_id=123)
            
            print(f"Méthode: {result['processing_method']}")
            print(f"Succès: {result['success']}")
            print(f"Intent: {result['intent']['classification']} ({result['intent']['confidence']:.2f})")
            print(f"Entités: {len(result['entities'])} types extraites")
            print(f"Score cohérence: {result['team_metrics']['coherence_score']:.2f}")
            print(f"Cache: {result['from_cache']}")
            
        except Exception as e:
            print(f"Erreur: {e}")
    
    # Health check final
    print(f"\n=== Health Check ===")
    health = await service.get_team_health()
    print(f"Status équipe: {health['status']}")
    
    # Métriques finales
    print(f"\n=== Métriques Équipe ===")
    metrics = await service.get_team_metrics()
    if metrics["available"]:
        stats = metrics["statistics"]["team_metrics"]
        print(f"Conversations traitées: {stats['conversations_processed']}")
        print(f"Taux succès: {stats['success_rate']:.1%}")
        print(f"Temps moyen: {stats['avg_processing_time_ms']:.0f}ms")


if __name__ == "__main__":
    asyncio.run(example_usage())