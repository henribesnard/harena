"""
Conversation Orchestrator - Service Principal Phase 5
Architecture v2.0 - Pipeline Lineaire

Responsabilite : Orchestration complete du pipeline conversation
- Pipeline lineaire : Context e Classification e Query e Execution e Response
- Coordination entre agents logiques et agents LLM
- Gestion etat conversation et cache cross-agents
- Metriques temps reel et health monitoring
- Support WebSocket streaming
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Agents logiques (Phase 3)
from ..core.context_manager import ContextManager, ContextCompressionRequest
from ..core.query_builder import QueryBuilder, QueryBuildRequest
from ..core.query_executor import QueryExecutor, QueryExecutionRequest

# Agents LLM (Phase 4)
from ..agents.llm import (
    LLMProviderManager,
    IntentClassifier, ClassificationRequest,
    ResponseGenerator, ResponseGenerationRequest
)

# Advanced Agents (Phase 3.5)
from ..agents.reasoning_agent import ReasoningAgent

# Configuration
from ..config.settings import ConfigManager

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """etapes du pipeline de conversation"""
    CONTEXT_ANALYSIS = "context_analysis"
    INTENT_CLASSIFICATION = "intent_classification" 
    QUERY_BUILDING = "query_building"
    QUERY_EXECUTION = "query_execution"
    RESPONSE_GENERATION = "response_generation"
    COMPLETED = "completed"
    ERROR = "error"

class ConversationMode(Enum):
    """Modes de conversation supportes"""
    SYNCHRONOUS = "sync"      # Reponse complete
    STREAMING = "stream"      # Streaming temps reel
    BATCH = "batch"          # Traitement differe

@dataclass
class ConversationRequest:
    """Requete de conversation complete"""
    user_message: str
    user_id: int
    conversation_id: Optional[str] = None
    mode: ConversationMode = ConversationMode.SYNCHRONOUS
    stream_response: bool = False
    include_insights: bool = True
    max_context_turns: int = 10
    timeout_seconds: int = 30
    jwt_token: Optional[str] = None

@dataclass
class PipelineMetrics:
    """Metriques de performance du pipeline"""
    stage_timings: Dict[str, int] = field(default_factory=dict)
    total_processing_time_ms: int = 0
    context_compression_ratio: float = 0.0
    cache_hit_rate: float = 0.0
    tokens_used: int = 0
    model_used: str = ""
    
@dataclass
class ConversationResult:
    """Resultat complet de conversation"""
    success: bool
    response_text: str
    conversation_id: str
    pipeline_stage: PipelineStage
    insights: List[Dict[str, Any]]
    data_visualizations: List[Dict[str, Any]]
    metrics: PipelineMetrics
    error_message: Optional[str] = None

    # Donnees intermediaires pour debug/monitoring
    classified_intent: Optional[Dict[str, Any]] = None
    built_query: Optional[Dict[str, Any]] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    search_total_hits: int = 0
    context_snapshot: Optional[Dict[str, Any]] = None

class ConversationOrchestrator:
    """
    Service principal orchestrant le pipeline complet de conversation
    
    Pipeline lineaire :
    User Input e Context e Classification e Query Build e Query Execute e Response e User
    """
    
    def __init__(
        self,
        context_manager: ContextManager,
        intent_classifier: IntentClassifier,
        query_builder: QueryBuilder,
        query_executor: QueryExecutor,
        response_generator: ResponseGenerator,
        config_manager: ConfigManager,
        reasoning_agent: Optional[ReasoningAgent] = None
    ):
        # Agents du pipeline
        self.context_manager = context_manager
        self.intent_classifier = intent_classifier
        self.query_builder = query_builder
        self.query_executor = query_executor
        self.response_generator = response_generator
        self.config_manager = config_manager
        self.reasoning_agent = reasoning_agent  # Optional for complex queries
        
        # Cache cross-pipeline pour optimisation
        self._conversation_cache: Dict[str, Any] = {}
        
        # Statistiques globales
        self.stats = {
            "conversations_processed": 0,
            "successful_conversations": 0,
            "streaming_conversations": 0,
            "pipeline_errors": {stage.value: 0 for stage in PipelineStage},
            "avg_processing_time_ms": 0,
            "total_tokens_consumed": 0
        }
        
        # Configuration pipeline
        self.pipeline_config = {
            "max_retries": 2,
            "stage_timeout_seconds": 10,
            "enable_parallel_insights": True,
            "context_compression_threshold": 8000  # tokens
        }
        
        logger.info("ConversationOrchestrator initialise")
    
    async def initialize(self) -> bool:
        """Initialise tous les composants du pipeline"""
        
        initialization_tasks = [
            ("intent_classifier", self.intent_classifier.initialize()),
            ("response_generator", self.response_generator.initialize())
        ]
        
        success_count = 0
        
        for component_name, init_task in initialization_tasks:
            try:
                if await init_task:
                    success_count += 1
                    logger.info(f"Composant {component_name} initialise")
                else:
                    logger.error(f"echec initialisation {component_name}")
            except Exception as e:
                logger.error(f"Erreur initialisation {component_name}: {str(e)}")
        
        # Query builder et executor n'ont pas de initialize async
        success_count += 2
        
        pipeline_ready = success_count >= 4  # Au moins 4/5 composants OK
        
        if pipeline_ready:
            logger.info("ConversationOrchestrator pret - Pipeline initialise")
        else:
            logger.error("ConversationOrchestrator en mode degrade")
        
        return pipeline_ready
    
    async def process_conversation(self, request: ConversationRequest) -> ConversationResult:
        """
        Traite une conversation complete via le pipeline lineaire
        
        Args:
            request: Requete de conversation avec message utilisateur
            
        Returns:
            ConversationResult avec reponse et metriques
        """
        start_time = datetime.now()
        metrics = PipelineMetrics()
        current_stage = PipelineStage.CONTEXT_ANALYSIS
        
        try:
            # Generation conversation_id si necessaire
            conversation_id = request.conversation_id or f"conv_{request.user_id}_{int(start_time.timestamp())}"
            
            logger.info(f"Demarrage pipeline conversation {conversation_id}")
            
            # === STAGE 1: CONTEXT ANALYSIS ===
            stage_start = datetime.now()
            current_stage = PipelineStage.CONTEXT_ANALYSIS
            
            context_result = await self._execute_context_analysis(request, conversation_id)
            
            metrics.stage_timings["context_analysis"] = self._get_stage_time(stage_start)
            
            if not context_result["success"]:
                return self._build_error_result(
                    conversation_id, current_stage, metrics, start_time,
                    f"Context analysis failed: {context_result['error']}"
                )
            
            # === STAGE 2: INTENT CLASSIFICATION ===
            stage_start = datetime.now() 
            current_stage = PipelineStage.INTENT_CLASSIFICATION
            
            classification_result = await self._execute_intent_classification(
                request, context_result["context_snapshot"]
            )
            
            metrics.stage_timings["intent_classification"] = self._get_stage_time(stage_start)
            
            if not classification_result.success:
                return self._build_error_result(
                    conversation_id, current_stage, metrics, start_time,
                    f"Intent classification failed: {classification_result.error_message}"
                )
            
            # Log des entités extraites pour debugging
            entities_count = len(classification_result.entities) if classification_result.entities else 0
            if entities_count > 0:
                entities_list = [f"{entity.name}={entity.value}" for entity in classification_result.entities]
                logger.info(f"Entités extraites ({entities_count}): {', '.join(entities_list)}")
            else:
                logger.info("Aucune entité extraite")

            # === ROUTING: CHECK IF COMPLEX QUERY NEEDS REASONING AGENT ===
            if self.reasoning_agent and self._is_complex_query(classification_result, request.user_message):
                logger.info(f"Complex query detected - routing to Reasoning Agent")

                # Use Reasoning Agent for complex multi-step queries
                stage_start = datetime.now()
                current_stage = PipelineStage.RESPONSE_GENERATION

                try:
                    reasoning_result = await self.reasoning_agent.reason_and_execute(
                        user_question=request.user_message,
                        user_id=request.user_id,
                        context={
                            "classified_intent": {
                                "intent_group": classification_result.intent_group,
                                "intent_subtype": classification_result.intent_subtype,
                                "confidence": classification_result.confidence
                            },
                            "conversation_context": context_result["context_snapshot"]
                        }
                    )

                    metrics.stage_timings["reasoning_agent"] = self._get_stage_time(stage_start)
                    metrics.total_processing_time_ms = self._get_total_time(start_time)

                    # Convert reasoning result to standard format
                    if reasoning_result.success:
                        await self._update_conversation_stats(metrics, True)
                        await self._save_conversation_turn(conversation_id, request, reasoning_result.final_answer)

                        logger.info(f"Reasoning Agent completed successfully in {reasoning_result.total_execution_time_ms}ms")

                        return ConversationResult(
                            success=True,
                            response_text=reasoning_result.final_answer,
                            conversation_id=conversation_id,
                            pipeline_stage=PipelineStage.COMPLETED,
                            insights=[],  # Reasoning Agent doesn't generate insights (yet)
                            data_visualizations=[],
                            metrics=metrics,
                            classified_intent={
                                "intent_group": classification_result.intent_group,
                                "intent_subtype": classification_result.intent_subtype,
                                "confidence": classification_result.confidence,
                                "entities": [entity.__dict__ for entity in classification_result.entities]
                            },
                            built_query=None,  # Reasoning Agent manages queries internally
                            search_results=[],
                            search_total_hits=0,
                            context_snapshot=context_result["context_snapshot"]
                        )
                    else:
                        logger.warning(f"Reasoning Agent failed, falling back to standard pipeline")
                        # Fall through to standard pipeline

                except Exception as e:
                    logger.error(f"Reasoning Agent error: {str(e)}, falling back to standard pipeline")
                    # Fall through to standard pipeline

            # === ROUTAGE SELON INTENTION ===
            # Import des helpers d'intention
            from ..prompts.harena_intents import can_direct_response, requires_search, HarenaIntentType
            
            # Conversion de l'intention en enum si c'est une string
            intent_enum = None
            if hasattr(classification_result, 'intent_group'):
                intent_str = classification_result.intent_group
            else:
                intent_str = getattr(classification_result, 'intent_type', 'UNKNOWN')
            
            try:
                # Essayer d'abord avec la string exacte
                intent_enum = HarenaIntentType(intent_str)
            except ValueError:
                try:
                    # Essayer en majuscules si échec
                    intent_enum = HarenaIntentType(intent_str.upper())
                except ValueError:
                    logger.warning(f"Intention inconnue: {intent_str}, fallback vers UNKNOWN")
                    intent_enum = HarenaIntentType.UNKNOWN
            
            # Si l'intention peut aller directement à la génération de réponse
            if can_direct_response(intent_enum):
                logger.info(f"Intention {intent_enum.value} - court-circuit vers génération de réponse")
                
                # Passer directement à la génération de réponse (stage 5)
                stage_start = datetime.now()
                current_stage = PipelineStage.RESPONSE_GENERATION
                
                response_result = await self._execute_response_generation(
                    request, classification_result, [],  # Pas de résultats de recherche
                    context_result["context_snapshot"]["recent_turns"],
                    search_aggregations=None
                )
                
                metrics.stage_timings["response_generation"] = self._get_stage_time(stage_start)
                
                if not response_result.success:
                    return self._build_error_result(
                        conversation_id, current_stage, metrics, start_time,
                        f"Response generation failed: {response_result.error_message}"
                    )
                
                # Pipeline terminé (court-circuité)
                metrics.total_processing_time_ms = self._get_total_time(start_time)
                metrics.tokens_used = response_result.tokens_used
                metrics.model_used = response_result.model_used
                
                await self._update_conversation_stats(metrics, True)
                await self._save_conversation_turn(conversation_id, request, response_result.response_text)
                
                logger.info(f"Pipeline {conversation_id} court-circuité en {metrics.total_processing_time_ms}ms")
                
                return ConversationResult(
                    success=True,
                    response_text=response_result.response_text,
                    conversation_id=conversation_id,
                    pipeline_stage=PipelineStage.COMPLETED,
                    insights=[insight.__dict__ for insight in response_result.insights],
                    data_visualizations=response_result.data_visualizations,
                    metrics=metrics,
                    classified_intent={
                        "intent_group": classification_result.intent_group,
                        "intent_subtype": classification_result.intent_subtype,
                        "confidence": classification_result.confidence,
                        "entities": [entity.__dict__ for entity in classification_result.entities]
                    },
                    built_query=None,  # Pas de recherche effectuée
                    search_results=[],  # Pas de recherche effectuée
                    search_total_hits=0,  # Pas de recherche effectuée
                    context_snapshot=context_result["context_snapshot"]
                )
            
            # === STAGE 3: QUERY BUILDING (seulement si recherche nécessaire) ===
            stage_start = datetime.now()
            current_stage = PipelineStage.QUERY_BUILDING
            
            query_build_result = await self._execute_query_building(
                classification_result, request, context_result["user_context"]
            )
            
            metrics.stage_timings["query_building"] = self._get_stage_time(stage_start)
            
            if not query_build_result.success:
                return self._build_error_result(
                    conversation_id, current_stage, metrics, start_time,
                    f"Query building failed: {query_build_result.validation_errors}"
                )
            
            # === STAGE 4: QUERY EXECUTION ===
            stage_start = datetime.now()
            current_stage = PipelineStage.QUERY_EXECUTION
            
            query_execution_result = await self._execute_query_execution(
                query_build_result.query, request
            )
            
            metrics.stage_timings["query_execution"] = self._get_stage_time(stage_start)
            
            if not query_execution_result.success:
                # On continue meme si pas de resultats (reponse generale possible)
                logger.warning(f"Query execution warning: {query_execution_result.error_message}")
            
            # === STAGE 5: RESPONSE GENERATION ===
            stage_start = datetime.now()
            current_stage = PipelineStage.RESPONSE_GENERATION
            
            response_result = await self._execute_response_generation(
                request, classification_result, query_execution_result.results if query_execution_result.success else [],
                context_result["context_snapshot"]["recent_turns"],
                search_aggregations=query_execution_result.aggregations if query_execution_result and query_execution_result.success else None
            )
            
            metrics.stage_timings["response_generation"] = self._get_stage_time(stage_start)
            
            if not response_result.success:
                return self._build_error_result(
                    conversation_id, current_stage, metrics, start_time,
                    f"Response generation failed: {response_result.error_message}"
                )
            
            # === PIPELINE COMPLETED ===
            metrics.total_processing_time_ms = self._get_total_time(start_time)
            metrics.tokens_used = response_result.tokens_used
            metrics.model_used = response_result.model_used
            
            # Mise e jour statistiques
            await self._update_conversation_stats(metrics, True)
            
            # Sauvegarde tour de conversation
            await self._save_conversation_turn(
                conversation_id, request, response_result.response_text
            )
            
            logger.info(f"Pipeline {conversation_id} termine en {metrics.total_processing_time_ms}ms")
            
            return ConversationResult(
                success=True,
                response_text=response_result.response_text,
                conversation_id=conversation_id,
                pipeline_stage=PipelineStage.COMPLETED,
                insights=[insight.__dict__ for insight in response_result.insights],
                data_visualizations=response_result.data_visualizations,
                metrics=metrics,
                classified_intent={
                    "intent_group": classification_result.intent_group,
                    "intent_subtype": classification_result.intent_subtype,
                    "confidence": classification_result.confidence,
                    "entities": [entity.__dict__ for entity in classification_result.entities]
                },
                built_query=query_build_result.query if query_build_result.success else None,
                search_results=query_execution_result.results if query_execution_result.success else [],
                search_total_hits=query_execution_result.total_hits if query_execution_result.success else 0,
                context_snapshot=context_result["context_snapshot"]
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Pipeline timeout pour conversation {conversation_id}")
            return self._build_error_result(
                conversation_id, current_stage, metrics, start_time,
                f"Pipeline timeout at stage {current_stage.value}"
            )
            
        except Exception as e:
            logger.error(f"Erreur inattendue pipeline {conversation_id}: {str(e)}")
            return self._build_error_result(
                conversation_id, current_stage, metrics, start_time,
                f"Unexpected pipeline error: {str(e)}"
            )
    
    async def process_conversation_stream(
        self,
        request: ConversationRequest
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Traite une conversation en mode streaming

        Yields des dictionnaires (pas des strings SSE) pour que la route puisse les formatter
        """
        try:
            # etapes pre-streaming (rapides)
            conversation_id = request.conversation_id or f"stream_{request.user_id}_{int(datetime.now().timestamp())}"

            yield {'type': 'status', 'message': 'Analyse du contexte...', 'conversation_id': conversation_id}
            
            # Context + Classification (parallel pour speed)
            context_task = self._execute_context_analysis(request, conversation_id)
            classification_task = self._execute_intent_classification_simple(request)
            
            context_result, classification_result = await asyncio.gather(
                context_task, classification_task, return_exceptions=True
            )

            # Debug logging
            logger.error(f"DEBUG - context_result type: {type(context_result)}, value: {str(context_result)[:200]}")
            logger.error(f"DEBUG - classification_result type: {type(classification_result)}")

            if isinstance(classification_result, Exception):
                logger.error(f"Classification exception: {classification_result}")
                yield {'type': 'error', 'message': f'Classification failed: {str(classification_result)}'}
                return

            if not hasattr(classification_result, 'success') or not classification_result.success:
                logger.error(f"Classification failed or missing success attribute")
                yield {'type': 'error', 'message': 'Classification failed'}
                return

            yield {'type': 'status', 'message': 'Construction de la requete...', 'intent': classification_result.intent_group}
            
            # Query building + execution
            user_context = {}
            if not isinstance(context_result, Exception) and isinstance(context_result, dict):
                user_context = context_result.get("user_context", {})

            query_build_result = await self._execute_query_building(
                classification_result, request, user_context
            )
            
            search_results = []
            query_execution_result = None
            if query_build_result.success:
                query_execution_result = await self._execute_query_execution(query_build_result.query, request)
                if query_execution_result.success:
                    search_results = query_execution_result.results
            
            yield {'type': 'status', 'message': f'Generation de la reponse... ({len(search_results)} resultats trouves)'}

            # Streaming response generation
            conversation_context = []
            if not isinstance(context_result, Exception) and isinstance(context_result, dict):
                context_snapshot = context_result.get("context_snapshot", {})
                if isinstance(context_snapshot, dict):
                    conversation_context = context_snapshot.get("recent_turns", [])

            response_request = ResponseGenerationRequest(
                intent_group=classification_result.intent_group,
                intent_subtype=classification_result.intent_subtype,
                user_message=request.user_message,
                search_results=search_results,
                conversation_context=conversation_context,
                user_profile={"user_id": request.user_id},
                user_id=request.user_id,
                conversation_id=conversation_id,
                generate_insights=request.include_insights,
                stream_response=True,
                search_aggregations=query_execution_result.aggregations if query_execution_result and query_execution_result.success else None
            )

            # Stream de la reponse finale
            yield {'type': 'response_start'}

            async for chunk in self.response_generator.generate_streaming_response(response_request):
                yield {'type': 'response_chunk', 'content': chunk}

            yield {'type': 'response_end', 'conversation_id': conversation_id}
            
            # Statistiques
            self.stats["streaming_conversations"] += 1
            
        except Exception as e:
            logger.error(f"Erreur streaming conversation: {str(e)}")
            yield {'type': 'error', 'message': f'Streaming error: {str(e)}'}
    
    async def _execute_context_analysis(
        self, 
        request: ConversationRequest, 
        conversation_id: str
    ) -> Dict[str, Any]:
        """Execute l'analyse de contexte (Stage 1)"""
        
        try:
            # Recuperation contexte conversation
            context_snapshot = await self.context_manager.get_conversation_context(
                conversation_id, max_turns=request.max_context_turns
            )
            
            # Compression si necessaire
            if (context_snapshot and 
                context_snapshot.total_tokens > self.pipeline_config["context_compression_threshold"]):
                
                compression_request = ContextCompressionRequest(
                    conversation_history=context_snapshot.conversation_history,
                    target_token_count=self.pipeline_config["context_compression_threshold"] // 2,
                    user_id=request.user_id
                )
                
                compression_result = await self.context_manager.compress_conversation_context(compression_request)
                
                if compression_result.success:
                    context_snapshot.conversation_history = compression_result.compressed_history
                    context_snapshot.total_tokens = compression_result.final_token_count
            
            # Extraction contexte utilisateur pour query building
            user_context = {
                "recent_intents": [],
                "conversation_length": len(context_snapshot.conversation_history) if context_snapshot else 0,
                "user_preferences": {}
            }
            
            if context_snapshot:
                # Extraction intentions recentes
                for turn in context_snapshot.conversation_history[-3:]:
                    if turn.metadata and "classified_intent" in turn.metadata:
                        user_context["recent_intents"].append(turn.metadata["classified_intent"])
            
            return {
                "success": True,
                "context_snapshot": {
                    "conversation_history": context_snapshot.conversation_history if context_snapshot else [],
                    "recent_turns": [
                        {"role": turn.role, "content": turn.content} 
                        for turn in (context_snapshot.conversation_history[-3:] if context_snapshot else [])
                    ],
                    "total_tokens": context_snapshot.total_tokens if context_snapshot else 0
                },
                "user_context": user_context
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_intent_classification(
        self, 
        request: ConversationRequest, 
        context_snapshot: Dict[str, Any]
    ) -> Any:  # ClassificationResult
        """Execute la classification d'intention (Stage 2)"""
        
        classification_request = ClassificationRequest(
            user_message=request.user_message,
            conversation_context=context_snapshot.get("recent_turns", []),
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            use_context=len(context_snapshot.get("recent_turns", [])) > 0
        )
        
        return await self.intent_classifier.classify_intent(classification_request)
    
    async def _execute_intent_classification_simple(self, request: ConversationRequest) -> Any:
        """Classification simplifiee pour streaming"""
        
        classification_request = ClassificationRequest(
            user_message=request.user_message,
            conversation_context=[],
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            use_context=False
        )
        
        return await self.intent_classifier.classify_intent(classification_request)
    
    async def _execute_query_building(
        self, 
        classification_result: Any,  # ClassificationResult 
        request: ConversationRequest,
        user_context: Dict[str, Any]
    ) -> Any:  # QueryBuildResult
        """Execute la construction de requete (Stage 3)"""
        
        # Extraction entites pour injection dans template avec normalisation
        entities = {}
        has_amount_entity = False
        amount_value = None
        operator_value = None
        amount_min_value = None
        amount_max_value = None

        for entity in classification_result.entities:
            # Collecter amount et operator séparément pour les fusionner
            if entity.name == "amount":
                amount_value = entity.value
                has_amount_entity = True
                continue
            elif entity.name == "operator":
                operator_value = entity.value
                continue
            elif entity.name == "amount_min":
                amount_min_value = entity.value
                has_amount_entity = True
                continue
            elif entity.name == "amount_max":
                amount_max_value = entity.value
                has_amount_entity = True
                continue

            # Normalisation des noms d'entités anciens -> nouveaux
            normalized_name = self._normalize_entity_name(entity.name)
            normalized_value = self._normalize_entity_value(entity.name, entity.value)
            entities[normalized_name] = normalized_value

            # Tracker si on a une entité de montant (autres formes)
            if entity.name in ["amount_threshold", "montant"]:
                has_amount_entity = True

        # FUSION amount_min + amount_max en objet montant avec range
        if amount_min_value is not None and amount_max_value is not None:
            montant_obj = {
                "operator": "range",
                "min": float(amount_min_value) if isinstance(amount_min_value, str) else amount_min_value,
                "max": float(amount_max_value) if isinstance(amount_max_value, str) else amount_max_value,
                "currency": "EUR"
            }
            entities["montant"] = montant_obj
            logger.info(f"Fusion amount_min + amount_max (range): {montant_obj}")
        # FUSION amount + operator en objet montant
        elif amount_value is not None:
            montant_obj = {
                "amount": float(amount_value) if isinstance(amount_value, str) else amount_value,
                "operator": operator_value if operator_value else "eq",  # eq par défaut
                "currency": "EUR"
            }
            entities["montant"] = montant_obj
            logger.info(f"Fusion amount + operator: {montant_obj}")

        # Si on a un montant mais pas de transaction_type explicite, inférer debit par défaut
        if has_amount_entity and "transaction_type" not in entities:
            # Analyser le message utilisateur pour détecter le contexte
            user_message = (request.user_message or "").lower()
            if user_message and any(word in user_message for word in ["revenus", "gains", "versements", "salaires", "crédits", "credits"]):
                entities["transaction_type"] = "credit"
            else:
                # Par défaut pour les montants : dépenses (debit)
                entities["transaction_type"] = "debit"

            logger.info(f"Ajout automatique transaction_type: {entities['transaction_type']} (inféré du contexte)")
        
        # Log pour debugging
        logger.info(f"Query building - Intent: {classification_result.intent_group}, Subtype: {classification_result.intent_subtype}, Entities: {json.dumps(entities, ensure_ascii=False, default=str)}")

        # Sélection intelligente du template basé sur les entités
        intent_group_upper = classification_result.intent_group.upper()
        intent_subtype = classification_result.intent_subtype

        # Mapping financial_query -> transaction_search avec subtype "filter"
        if intent_group_upper == "FINANCIAL_QUERY":
            intent_subtype = "filter"  # Utiliser le template filter de transaction_search
            logger.info(f"Mapping FINANCIAL_QUERY -> TRANSACTION_SEARCH with subtype 'filter'")

        # Pour transaction_search sans subtype mais avec des entités de filtrage, utiliser "filter"
        elif intent_group_upper == "TRANSACTION_SEARCH" and not intent_subtype:
            # Vérifier si on a des entités qui nécessitent le template filter
            filter_entities = ["merchant", "merchants", "montant", "categories", "date_range", "operation_type"]
            has_filter_entities = any(entity_key in entities for entity_key in filter_entities)

            if has_filter_entities:
                intent_subtype = "filter"
                logger.info(f"Auto-selecting 'filter' template for TRANSACTION_SEARCH with entities: {list(entities.keys())}")

        query_build_request = QueryBuildRequest(
            intent_group=intent_group_upper,
            intent_subtype=intent_subtype,
            entities=entities,
            user_context=user_context,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        return await self.query_builder.build_query(query_build_request)
    
    def _normalize_entity_name(self, entity_name: str) -> str:
        """Normalise les noms d'entités pour compatibilité avec les templates"""
        
        # Mapping des anciens noms vers les nouveaux
        entity_name_mappings = {
            "amount_threshold": "montant",
            "amount": "montant", 
            "amount_min": "montant",
            "amount_max": "montant",
            "merchant": "merchant_name",
            "date_range": "date_range",
            "category": "category_name",
            "transaction_type": "transaction_type"  # Pas de changement, mais explicite
        }
        
        return entity_name_mappings.get(entity_name, entity_name)
    
    def _normalize_entity_value(self, original_name: str, value: Any) -> Any:
        """Normalise les valeurs d'entités pour compatibilité avec les templates"""
        
        # Cas spécial : amount_threshold (valeur simple) -> montant (objet avec opérateur)
        if original_name == "amount_threshold":
            try:
                amount = float(value) if isinstance(value, str) else value
                return {
                    "operator": "gte",  # Par défaut pour amount_threshold (fallback cases)
                    "amount": amount,
                    "currency": "EUR"
                }
            except (ValueError, TypeError):
                logger.warning(f"Impossible de convertir amount_threshold: {value}")
                return value
        
        # Cas spécial : amount simple -> objet avec opérateur equal
        if original_name == "amount" and not isinstance(value, dict):
            try:
                amount = float(value) if isinstance(value, str) else value
                return {
                    "operator": "eq",
                    "amount": amount,
                    "currency": "EUR"
                }
            except (ValueError, TypeError):
                logger.warning(f"Impossible de convertir amount: {value}")
                return value
        
        return value
    
    async def _execute_query_execution(
        self, 
        query: Dict[str, Any], 
        request: ConversationRequest
    ) -> Any:  # QueryExecutionResult
        """Execute la requete search_service (Stage 4)"""
        
        query_execution_request = QueryExecutionRequest(
            query=query,
            user_id=request.user_id,
            timeout_ms=request.timeout_seconds * 1000,
            cache_ttl_seconds=300,
            retry_count=self.pipeline_config["max_retries"],
            jwt_token=request.jwt_token
        )
        
        return await self.query_executor.execute_query(query_execution_request)
    
    async def _execute_response_generation(
        self, 
        request: ConversationRequest,
        classification_result: Any,  # ClassificationResult
        search_results: List[Dict[str, Any]],
        conversation_context: List[Dict[str, str]],
        search_aggregations: Optional[Dict[str, Any]] = None
    ) -> Any:  # ResponseGenerationResult
        """Execute la generation de reponse (Stage 5)"""
        
        response_request = ResponseGenerationRequest(
            intent_group=classification_result.intent_group,
            intent_subtype=classification_result.intent_subtype,
            user_message=request.user_message,
            search_results=search_results,
            conversation_context=conversation_context,
            user_profile={"user_id": request.user_id},
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            generate_insights=request.include_insights,
            stream_response=request.stream_response,
            search_aggregations=search_aggregations
        )
        
        return await self.response_generator.generate_response(response_request)
    
    async def _save_conversation_turn(
        self, 
        conversation_id: str, 
        request: ConversationRequest, 
        response_text: str
    ) -> None:
        """Sauvegarde le tour de conversation dans le contexte"""
        
        try:
            await self.context_manager.add_conversation_turn(
                conversation_id=conversation_id,
                user_message=request.user_message,
                assistant_response=response_text,
                user_id=request.user_id
            )
        except Exception as e:
            logger.warning(f"Erreur sauvegarde turn conversation {conversation_id}: {str(e)}")
    
    def _build_error_result(
        self, 
        conversation_id: str, 
        failed_stage: PipelineStage, 
        metrics: PipelineMetrics,
        start_time: datetime,
        error_message: str
    ) -> ConversationResult:
        """Construit un resultat d'erreur"""
        
        metrics.total_processing_time_ms = self._get_total_time(start_time)
        
        # Statistiques erreur
        self.stats["pipeline_errors"][failed_stage.value] += 1
        asyncio.create_task(self._update_conversation_stats(metrics, False))
        
        # Reponse de fallback selon le stage d'echec
        fallback_responses = {
            PipelineStage.CONTEXT_ANALYSIS: "Je rencontre des difficultes e analyser le contexte de notre conversation.",
            PipelineStage.INTENT_CLASSIFICATION: "Je n'arrive pas e bien comprendre votre demande. Pouvez-vous reformuler ?",
            PipelineStage.QUERY_BUILDING: "Je ne parviens pas e construire une requete appropriee pour vos donnees.",
            PipelineStage.QUERY_EXECUTION: "Je n'arrive pas e acceder e vos donnees pour le moment.",
            PipelineStage.RESPONSE_GENERATION: "J'ai des difficultes e formuler une reponse. Veuillez reessayer."
        }
        
        return ConversationResult(
            success=False,
            response_text=fallback_responses.get(failed_stage, "Une erreur technique est survenue."),
            conversation_id=conversation_id,
            pipeline_stage=failed_stage,
            insights=[],
            data_visualizations=[],
            metrics=metrics,
            error_message=error_message,
            built_query=None
        )
    
    async def _update_conversation_stats(self, metrics: PipelineMetrics, success: bool):
        """Met e jour les statistiques globales"""
        
        self.stats["conversations_processed"] += 1
        
        if success:
            self.stats["successful_conversations"] += 1
        
        self.stats["total_tokens_consumed"] += metrics.tokens_used
        
        # Moyenne mobile temps de traitement
        current_avg = self.stats["avg_processing_time_ms"]
        total_conversations = self.stats["conversations_processed"]
        
        self.stats["avg_processing_time_ms"] = (
            (current_avg * (total_conversations - 1) + metrics.total_processing_time_ms) / total_conversations
        )
    
    def _get_stage_time(self, stage_start: datetime) -> int:
        """Calcule temps d'une etape en ms"""
        return int((datetime.now() - stage_start).total_seconds() * 1000)
    
    def _get_total_time(self, start_time: datetime) -> int:
        """Calcule temps total en ms"""
        return int((datetime.now() - start_time).total_seconds() * 1000)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Recupere les statistiques completes du pipeline"""
        
        # Calcul taux de succes
        success_rate = 0.0
        if self.stats["conversations_processed"] > 0:
            success_rate = self.stats["successful_conversations"] / self.stats["conversations_processed"]
        
        # Agregation stats des composants
        component_stats = {
            "context_manager": self.context_manager.get_stats(),
            "intent_classifier": self.intent_classifier.get_stats(),
            "query_builder": self.query_builder.get_stats(),
            "query_executor": self.query_executor.get_stats(),
            "response_generator": self.response_generator.get_stats()
        }
        
        return {
            "pipeline_stats": {
                **self.stats,
                "success_rate": success_rate,
                "cache_entries": len(self._conversation_cache)
            },
            "component_stats": component_stats,
            "config": self.pipeline_config
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check complet du pipeline"""
        
        try:
            # Test rapide du pipeline complet
            test_request = ConversationRequest(
                user_message="Test sante du pipeline",
                user_id=0,
                conversation_id="health_check",
                timeout_seconds=5,
                include_insights=False
            )
            
            # Test avec timeout court
            try:
                result = await asyncio.wait_for(
                    self.process_conversation(test_request), 
                    timeout=5.0
                )
                pipeline_healthy = result.success
            except asyncio.TimeoutError:
                pipeline_healthy = False
            
            # Health check des composants
            components_health = {}
            healthy_components = 0
            
            health_checks = [
                ("intent_classifier", self.intent_classifier.health_check()),
                ("response_generator", self.response_generator.health_check()),
                ("query_executor", self.query_executor.health_check())
            ]
            
            for component_name, health_check_task in health_checks:
                try:
                    component_health = await health_check_task
                    components_health[component_name] = component_health
                    if component_health.get("status") == "healthy":
                        healthy_components += 1
                except Exception as e:
                    components_health[component_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            
            # Logique de status global : priorité aux composants individuels
            if healthy_components == len(health_checks):  # Tous les composants healthy
                overall_status = "healthy"  # Même si pipeline test échoue
            elif healthy_components >= 2:  # Au moins 2/3 composants OK
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
            return {
                "status": overall_status,
                "component": "conversation_orchestrator",
                "pipeline_test_success": pipeline_healthy,
                "healthy_components": f"{healthy_components}/{len(health_checks)}",
                "components": components_health,
                "stats": self.get_pipeline_stats(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "conversation_orchestrator",
                "error": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _is_complex_query(self, classification_result: Any, user_message: str) -> bool:
        """
        Détecte si une question est complexe et nécessite le Reasoning Agent

        Critères de complexité:
        - Mots-clés de comparaison: "compare", "vs", "différence", "variation"
        - Mots-clés temporels multiples: "ce mois" + "mois dernier"
        - Questions multi-parties avec "et"
        - Calculs explicites: "total", "moyenne", "combien"
        - Analyse de tendance: "évolution", "progression", "trend"

        Returns:
            True si la question est complexe, False sinon
        """

        message_lower = user_message.lower()

        # Keywords indicating complex comparison queries
        comparison_keywords = [
            "compare", "comparer", "comparaison",
            "vs", "versus",
            "différence", "difference",
            "variation",
            "évolution", "evolution",
            "progression",
            "tendance", "trend"
        ]

        # Temporal keywords indicating multi-period analysis
        temporal_keywords = [
            "ce mois", "mois dernier", "mois précédent",
            "cette année", "année dernière", "année précédente",
            "ce trimestre", "trimestre dernier",
            "cette semaine", "semaine dernière"
        ]

        # Multi-part questions
        multi_part_indicators = [
            " et ", " puis ", " ensuite ",
            " ainsi que ", " également "
        ]

        # Check for comparison keywords
        has_comparison = any(keyword in message_lower for keyword in comparison_keywords)

        # Check for multiple temporal references (indicates period comparison)
        temporal_count = sum(1 for keyword in temporal_keywords if keyword in message_lower)
        has_multi_temporal = temporal_count >= 2

        # Check for multi-part questions
        has_multi_part = any(indicator in message_lower for indicator in multi_part_indicators)

        # Complex if:
        # 1. Has comparison keywords
        # 2. Has multiple temporal references (e.g., "ce mois vs mois dernier")
        # 3. Has multi-part structure
        is_complex = has_comparison or has_multi_temporal or has_multi_part

        if is_complex:
            logger.info(f"Complex query detected: comparison={has_comparison}, "
                       f"multi_temporal={has_multi_temporal}, multi_part={has_multi_part}")

        return is_complex

    async def close(self):
        """Ferme proprement tous les composants"""

        try:
            # Fermeture des composants avec connexions
            await self.query_executor.close()

            logger.info("ConversationOrchestrator ferme proprement")

        except Exception as e:
            logger.error(f"Erreur fermeture ConversationOrchestrator: {str(e)}")