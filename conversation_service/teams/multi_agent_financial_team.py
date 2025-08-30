"""
Équipe AutoGen multi-agents pour traitement financier
Intégration complète avec l'infrastructure existante (cache, métriques, logging)
"""

import logging
import time
import json
import asyncio
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

# AutoGen imports avec fallback
try:
    from autogen import GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    GroupChat = None
    GroupChatManager = None

# Agents existants adaptés
from ..agents.financial.intent_classifier import IntentClassifierAgent
from ..agents.financial.entity_extractor import EntityExtractorAgent

# Infrastructure existante réutilisée
from ..core.cache_manager import CacheManager
from ..utils.metrics_collector import MetricsCollector, MetricType
from ..prompts.autogen.team_orchestration import (
    MULTI_AGENT_TEAM_ORCHESTRATION_PROMPT,
    get_orchestration_prompt_for_context,
    get_workflow_completion_message,
    get_workflow_error_message
)

# Configuration du logger
logger = logging.getLogger("conversation_service.teams")


class MultiAgentFinancialTeam:
    """
    Équipe AutoGen intégrée avec infrastructure conversation service
    Orchestre intent_classifier + entity_extractor avec cache et métriques existants
    """
    
    def __init__(self, deepseek_client=None):
        """
        Initialise équipe avec agents et infrastructure existante
        
        Args:
            deepseek_client: Client DeepSeek existant (optionnel, utilise env si absent)
        """
        
        if not AUTOGEN_AVAILABLE:
            raise ImportError("AutoGen non disponible. Installez: pip install pyautogen")
        
        # Infrastructure existante partagée
        self.cache_manager = CacheManager()
        self.metrics_collector = MetricsCollector()
        
        # Configuration AutoGen avec infrastructure existante
        self.deepseek_client = deepseek_client
        self._setup_agents()
        self._setup_autogen_groupchat()
        
        # Métriques équipe (extension existant)
        self.team_metrics = {
            "conversations_processed": 0,
            "success_rate": 0.0,
            "avg_processing_time_ms": 0.0,
            "total_processing_time_ms": 0.0,
            "cache_hit_rate": 0.0,
            "coherence_avg_score": 0.0
        }
        
        logger.info("Équipe multi-agents initialisée avec infrastructure existante")
    
    def _setup_agents(self):
        """Configuration agents avec prompts AutoGen étendus"""
        
        # Agent Intent (mode AutoGen équipe)
        self.intent_classifier = IntentClassifierAgent(
            autogen_mode=True,  # Utilise prompts collaboration étendus
            deepseek_client=self.deepseek_client
        )
        
        # Agent Entity (mode AutoGen équipe)  
        self.entity_extractor = EntityExtractorAgent(
            autogen_mode=True,  # Utilise prompts collaboration étendus
            deepseek_client=self.deepseek_client
        )
        
        logger.info("Agents configurés en mode AutoGen collaboration")
    
    def _setup_autogen_groupchat(self):
        """Configuration GroupChat AutoGen avec orchestration spécialisée"""
        
        # Configuration LLM DeepSeek pour AutoGen
        llm_config = {
            "config_list": [{
                "model": "deepseek-chat",
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": "https://api.deepseek.com/v1"
            }],
            "temperature": 0.0,
            "cache_seed": 42,
            "timeout": 30
        }
        
        # Récupération agents AutoGen sous-jacents
        self.intent_autogen_agent = self.intent_classifier.get_autogen_agent()
        self.entity_autogen_agent = self.entity_extractor.get_autogen_agent()
        
        # Vérification disponibilité agents
        if not self.intent_autogen_agent or not self.entity_autogen_agent:
            raise RuntimeError("Agents AutoGen sous-jacents non disponibles")
        
        # GroupChat séquentiel strict: Intent → Entity
        self.group_chat = GroupChat(
            agents=[self.intent_autogen_agent, self.entity_autogen_agent],
            messages=[],
            max_round=3,  # Intent + Entity + validation si nécessaire
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False,
            enable_clear_history=True
        )
        
        # Manager orchestration (utilise prompt spécialisé)
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=llm_config,
            system_message=MULTI_AGENT_TEAM_ORCHESTRATION_PROMPT,
            name="financial_team_manager"
        )
        
        logger.info("GroupChat AutoGen configuré avec orchestration séquentielle")
    
    async def process_user_message(self, user_message: str, user_id: int) -> Dict[str, Any]:
        """
        Traitement message utilisateur avec équipe AutoGen
        Intégré cache et métriques existants
        
        Args:
            user_message: Message utilisateur à traiter
            user_id: ID utilisateur pour cache et metrics
            
        Returns:
            Dict contenant résultats intent + entities + métriques équipe
        """
        
        start_time = time.time()
        
        # Cache équipe (réutilise CacheManager existant)
        cache_key = f"multi_agent_team_{user_id}_{hash(user_message)}"
        cached_result = await self.cache_manager.get(cache_key)
        
        if cached_result:
            self.metrics_collector.record_metric(
                "cache_hits", 
                1, 
                MetricType.COUNTER, 
                labels={"team": "multi_agent_financial"}
            )
            return self._enrich_cached_result(cached_result)
        
        try:
            # Message initiation équipe AutoGen
            team_initiation_message = f"""TRAITEMENT ÉQUIPE MULTI-AGENTS

Message utilisateur: "{user_message}"
User ID: {user_id}
Timestamp: {datetime.now(timezone.utc).isoformat()}

WORKFLOW SÉQUENTIEL:
1. intent_classifier: Analyser intention + préparer contexte entity
2. entity_extractor: Extraire entités selon contexte intent

Commençons avec l'analyse d'intention."""
            
            # Workflow AutoGen avec clear history
            self.group_chat.reset()
            
            # Initier conversation équipe
            await self._execute_team_workflow(team_initiation_message)
            
            # Extraction résultats
            team_results = await self._extract_team_results()
            
            # Cache résultat + métriques (infrastructure existante)
            processing_time_ms = int((time.time() - start_time) * 1000)
            await self._cache_and_update_metrics(
                cache_key, 
                team_results, 
                processing_time_ms,
                user_id
            )
            
            return team_results
            
        except Exception as e:
            # Gestion erreur cohérente existante
            return await self._handle_team_failure(e, user_message, user_id, start_time)
    
    async def _execute_team_workflow(self, initiation_message: str):
        """Exécute workflow équipe AutoGen de manière asynchrone"""
        
        def run_autogen_workflow():
            """Fonction synchrone pour AutoGen"""
            # Utilisation de l'agent AutoGen sous-jacent stocké
            return self.intent_autogen_agent.initiate_chat(
                self.manager,
                message=initiation_message,
                max_turns=3,
                clear_history=True
            )
        
        # Exécution asynchrone du workflow synchrone AutoGen
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_autogen_workflow)
        
        return result
    
    async def _extract_team_results(self) -> Dict[str, Any]:
        """Extraction résultats depuis conversation AutoGen"""
        
        conversation_history = self.group_chat.messages
        
        # Parsing résultats agents
        intent_result = None
        entities_result = None
        
        for message in conversation_history:
            agent_name = message.get("name", "")
            content = message.get("content", "")
            
            try:
                # Tentative parsing JSON
                parsed_content = json.loads(content)
                
                if agent_name == "intent_classifier" and not intent_result:
                    if "intent" in parsed_content:
                        intent_result = parsed_content
                        logger.debug(f"Intent extrait: {parsed_content.get('intent')}")
                        
                elif agent_name == "entity_extractor" and not entities_result:
                    if "entities" in parsed_content:
                        entities_result = parsed_content
                        logger.debug(f"Entities extraites: {len(parsed_content.get('entities', {}))}")
                        
            except json.JSONDecodeError:
                # Log erreur mais continue (résilience existante)
                logger.warning(f"JSON invalide agent {agent_name}: {content[:100]}")
                continue
        
        # Validation cohérence (logique métier existante)
        coherence_score = self._validate_intent_entity_coherence(
            intent_result, entities_result
        )
        
        # Structure résultat équipe
        workflow_success = bool(intent_result and entities_result and coherence_score > 0.5)
        
        return {
            "workflow_completed": "intent_entity_extraction_complete",
            "intent_result": intent_result or self._get_fallback_intent(),
            "entities_result": entities_result or self._get_fallback_entities(),
            "workflow_success": workflow_success,
            "coherence_validation": {
                "score": coherence_score,
                "threshold_met": coherence_score > 0.7,
                "validation_rules_applied": True
            },
            "agents_sequence": ["intent_classifier", "entity_extractor"],
            "conversation_history": conversation_history,
            "team_metadata": {
                "messages_exchanged": len(conversation_history),
                "processing_quality": "high" if coherence_score > 0.8 else "medium" if coherence_score > 0.5 else "low"
            }
        }
    
    def _validate_intent_entity_coherence(self, intent_result: Dict, entities_result: Dict) -> float:
        """Validation cohérence intention-entités (logique métier existante)"""
        
        if not intent_result or not entities_result:
            return 0.0
            
        intent_type = intent_result.get("intent", "")
        entities = entities_result.get("entities", {})
        
        # Règles cohérence métier (extension existante)
        coherence_checks = {
            "SEARCH_BY_MERCHANT": lambda e: len(e.get("merchants", [])) > 0,
            "SEARCH_BY_AMOUNT": lambda e: len(e.get("amounts", [])) > 0,
            "SPENDING_ANALYSIS": lambda e: len(e.get("amounts", [])) > 0 or len(e.get("categories", [])) > 0,
            "BALANCE_INQUIRY": lambda e: True,  # Pas d'entités spécifiques requises
            "SEARCH_BY_DATE": lambda e: len(e.get("dates", [])) > 0,
            "SEARCH_BY_OPERATION_TYPE": lambda e: len(e.get("operation_types", [])) > 0,
            "TRANSACTION_HISTORY": lambda e: True,  # Accepte entités vides
            "ACCOUNT_BALANCE": lambda e: True,
            "GENERAL_INQUIRY": lambda e: True,
            "GREETING": lambda e: True,
            "UNCLEAR_INTENT": lambda e: True
        }
        
        check_function = coherence_checks.get(intent_type, lambda e: True)
        intent_confidence = intent_result.get("confidence", 0.5)
        entity_confidence = entities_result.get("confidence", 0.5)
        
        if check_function(entities):
            # Bonus cohérence: moyenne pondérée des confiances
            coherence_score = min((intent_confidence + entity_confidence) / 2 + 0.2, 1.0)
        else:
            # Malus incohérence
            coherence_score = max((intent_confidence + entity_confidence) / 2 - 0.3, 0.0)
        
        logger.debug(f"Cohérence calculée: {coherence_score:.3f} pour {intent_type}")
        return coherence_score
    
    async def _cache_and_update_metrics(
        self, 
        cache_key: str, 
        results: Dict, 
        processing_time_ms: int,
        user_id: int
    ):
        """Cache et métriques intégrés infrastructure existante"""
        
        # Cache (réutilise CacheManager existant)
        await self.cache_manager.set(
            cache_key, 
            results, 
            ttl=3600  # 1h cache équipe
        )
        
        # Métriques Prometheus existantes
        self.metrics_collector.record_metric(
            "team_execution_time_ms",
            processing_time_ms,
            MetricType.HISTOGRAM,
            labels={
                "team_name": "multi_agent_financial",
                "success": str(results["workflow_success"]),
                "user_id": str(user_id)
            }
        )
        
        self.metrics_collector.record_metric(
            "team_conversations_total",
            1,
            MetricType.COUNTER,
            labels={"team_name": "multi_agent_financial"}
        )
        
        # Métriques équipe internes
        self._update_team_metrics(processing_time_ms, results)
        
        logger.info(f"Métriques enregistrées: {processing_time_ms}ms, succès: {results['workflow_success']}")
    
    def _update_team_metrics(self, processing_time_ms: int, results: Dict):
        """Mise à jour métriques équipe (pattern existant)"""
        
        self.team_metrics["conversations_processed"] += 1
        self.team_metrics["total_processing_time_ms"] += processing_time_ms
        
        # Moyenne mobile temps traitement
        self.team_metrics["avg_processing_time_ms"] = (
            self.team_metrics["total_processing_time_ms"] / 
            self.team_metrics["conversations_processed"]
        )
        
        # Taux de succès
        success = results["workflow_success"]
        total_processed = self.team_metrics["conversations_processed"]
        
        if success:
            current_successes = self.team_metrics["success_rate"] * (total_processed - 1)
            self.team_metrics["success_rate"] = (current_successes + 1.0) / total_processed
        else:
            current_successes = self.team_metrics["success_rate"] * (total_processed - 1)
            self.team_metrics["success_rate"] = current_successes / total_processed
        
        # Score cohérence moyen
        coherence_score = results.get("coherence_validation", {}).get("score", 0.0)
        current_coherence_sum = self.team_metrics["coherence_avg_score"] * (total_processed - 1)
        self.team_metrics["coherence_avg_score"] = (current_coherence_sum + coherence_score) / total_processed
    
    async def _handle_team_failure(
        self, 
        error: Exception, 
        user_message: str, 
        user_id: int, 
        start_time: float
    ) -> Dict[str, Any]:
        """Gestion erreur équipe cohérente avec patterns existants"""
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Log erreur (format existant)
        logger.error(
            f"Échec équipe multi-agents: {str(error)}", 
            extra={
                "user_id": user_id,
                "message": user_message[:100],
                "processing_time_ms": processing_time_ms,
                "error_type": type(error).__name__
            }
        )
        
        # Métriques échec
        self.metrics_collector.record_metric(
            "team_failures_total",
            1,
            MetricType.COUNTER,
            labels={
                "team_name": "multi_agent_financial",
                "error_type": type(error).__name__,
                "user_id": str(user_id)
            }
        )
        
        # Fallback structuré (cohérent existant)
        return {
            "workflow_completed": "team_failed_with_fallback",
            "intent_result": self._get_fallback_intent(),
            "entities_result": self._get_fallback_entities(),
            "workflow_success": False,
            "coherence_validation": {
                "score": 0.0,
                "threshold_met": False,
                "fallback_applied": True
            },
            "error_context": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "processing_time_ms": processing_time_ms,
                "fallback_strategy": "minimal_results"
            },
            "team_metadata": {
                "agents_attempted": ["intent_classifier", "entity_extractor"],
                "recovery_applied": True
            }
        }
    
    def _get_fallback_intent(self) -> Dict[str, Any]:
        """Intent de fallback en cas d'échec"""
        return {
            "intent": "GENERAL_INQUIRY",
            "confidence": 0.3,
            "reasoning": "Équipe AutoGen échouée, fallback appliqué"
        }
    
    def _get_fallback_entities(self) -> Dict[str, Any]:
        """Entités de fallback en cas d'échec"""
        return {
            "entities": {
                "amounts": [],
                "merchants": [],
                "dates": [],
                "categories": [],
                "operation_types": [],
                "text_search": []
            },
            "confidence": 0.0,
            "reasoning": "Extraction impossible après échec équipe"
        }
    
    def _enrich_cached_result(self, cached_result: Dict) -> Dict[str, Any]:
        """Enrichit résultat cache avec métadonnées actuelles"""
        
        cached_result["from_cache"] = True
        cached_result["cache_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return cached_result
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check équipe intégré monitoring existant"""
        
        # Test connectivité agents AutoGen
        intent_healthy = hasattr(self.intent_autogen_agent, 'llm_config') if self.intent_autogen_agent else False
        entity_healthy = hasattr(self.entity_autogen_agent, 'llm_config') if self.entity_autogen_agent else False
        
        # Test infrastructure
        cache_healthy = await self._test_cache_connectivity()
        metrics_healthy = self.metrics_collector is not None
        
        # Debug des statuts
        logger.info(f"Health check détails:")
        logger.info(f"  - Intent agent healthy: {intent_healthy}")
        logger.info(f"  - Entity agent healthy: {entity_healthy}")  
        logger.info(f"  - Cache healthy: {cache_healthy}")
        logger.info(f"  - Metrics healthy: {metrics_healthy}")
        
        # Status global
        overall_healthy = intent_healthy and entity_healthy and cache_healthy and metrics_healthy
        
        return {
            "team_name": "multi_agent_financial_team",
            "overall_status": "healthy" if overall_healthy else "degraded",
            "agents_status": {
                "intent_classifier": "healthy" if intent_healthy else "unhealthy",
                "entity_extractor": "healthy" if entity_healthy else "unhealthy"
            },
            "infrastructure_status": {
                "cache_manager": "healthy" if cache_healthy else "unhealthy",
                "metrics_collector": "healthy" if metrics_healthy else "unhealthy"
            },
            "autogen_groupchat": {
                "configured": self.group_chat is not None,
                "max_rounds": getattr(self.group_chat, 'max_round', 0),
                "agents_count": len(getattr(self.group_chat, 'agents', []))
            },
            "performance_metrics": self.team_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _test_cache_connectivity(self) -> bool:
        """Test rapide connectivité cache"""
        try:
            test_key = f"health_check_{int(time.time())}"
            await self.cache_manager.set(test_key, {"test": True}, ttl=10)
            result = await self.cache_manager.get(test_key)
            return result is not None
        except Exception:
            return False
    
    def get_team_statistics(self) -> Dict[str, Any]:
        """Statistiques détaillées équipe pour monitoring"""
        
        return {
            "team_metrics": self.team_metrics.copy(),
            "agents_info": {
                "intent_classifier": {
                    "name": getattr(self.intent_classifier, 'name', 'intent_classifier'),
                    "configured": True
                },
                "entity_extractor": {
                    "name": getattr(self.entity_extractor, 'name', 'entity_extractor'),
                    "configured": True
                }
            },
            "autogen_config": {
                "max_rounds": getattr(self.group_chat, 'max_round', 0),
                "speaker_selection": getattr(self.group_chat, 'speaker_selection_method', 'round_robin'),
                "allow_repeat": getattr(self.group_chat, 'allow_repeat_speaker', False)
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }