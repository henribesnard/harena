"""
Agent Orchestrator - Coordonne les agents LangChain avec auto-correction
Pipeline: Analyze → Build Query → Execute → Correct (if needed) → Generate Response
"""
import logging
import asyncio
import httpx
import json
import dataclasses
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..models import (
    UserQuery, QueryAnalysis, ElasticsearchQuery, SearchResults,
    ConversationResponse, AgentResponse
)
from ..agents.intent_router_agent import IntentRouterAgent
from ..agents.query_analyzer_agent import QueryAnalyzerAgent
from ..agents.elasticsearch_builder_agent import ElasticsearchBuilderAgent
from ..agents.response_generator_agent import ResponseGeneratorAgent
from ..agents.analytics_agent import AnalyticsAgent
from ..models.intent import IntentCategory
from ..core.aggregation_enricher import AggregationEnricher
from ..core.category_validator import category_validator
from ..services.redis_conversation_cache import RedisConversationCache
from ..services.analytics_service import AnalyticsService
from ..services.user_profile_service import UserProfileService
from ..utils.token_counter import TokenCounter
from ..config.settings import settings

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrateur d'agents autonomes avec auto-correction

    Pipeline:
    1. QueryAnalyzerAgent: Analyse la requête utilisateur
    2. ElasticsearchBuilderAgent: Construit la query Elasticsearch
    3. Execute query sur search_service
    4. Si échec → ElasticsearchBuilderAgent.correct_query()
    5. ResponseGeneratorAgent: Génère la réponse finale

    Auto-correction:
    - Maximum 2 tentatives de correction
    - Utilise les erreurs Elasticsearch pour guider la correction
    """

    def __init__(
        self,
        search_service_url: str = "http://localhost:3002",
        max_correction_attempts: int = 2,
        llm_model: str = "gpt-4o-mini"
    ):
        # Initialiser les agents
        self.intent_router = IntentRouterAgent(llm_model=llm_model)
        self.query_analyzer = QueryAnalyzerAgent(llm_model=llm_model)
        self.query_builder = ElasticsearchBuilderAgent(llm_model=llm_model)
        self.response_generator = ResponseGeneratorAgent(llm_model="gpt-4o")
        self.analytics_agent = AnalyticsAgent(llm_model=llm_model)

        # Initialiser l'enrichisseur d'agrégations
        self.aggregation_enricher = AggregationEnricher()

        # Initialiser le service d'analytics
        self.analytics_service = AnalyticsService()

        self.search_service_url = search_service_url
        self.max_correction_attempts = max_correction_attempts

        # HTTP client pour search_service
        # Timeout augmenté pour laisser le temps aux agrégations complexes de s'exécuter
        self.http_client = httpx.AsyncClient(timeout=60.0)

        # Initialize Redis conversation cache (if enabled)
        self.conversation_cache: Optional[RedisConversationCache] = None
        if settings.REDIS_CONVERSATION_CACHE_ENABLED:
            try:
                token_counter = TokenCounter(model=llm_model)
                self.conversation_cache = RedisConversationCache(
                    redis_url=settings.REDIS_URL,
                    max_messages_per_conversation=settings.MAX_CONVERSATION_MESSAGES,
                    max_context_tokens=settings.MAX_CONVERSATION_CONTEXT_TOKENS,
                    cache_ttl_seconds=settings.CONVERSATION_CACHE_TTL_SECONDS,
                    token_counter=token_counter
                )
                logger.info("✅ Redis conversation cache enabled")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Redis cache: {e}")
                self.conversation_cache = None
        else:
            logger.info("ℹ️ Redis conversation cache disabled")

        # Initialize user profile service (if enabled)
        self.user_profile_service: Optional[UserProfileService] = None
        if settings.BUDGET_PROFILE_ENABLED:
            try:
                self.user_profile_service = UserProfileService()
                logger.info("✅ UserProfileService enabled")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize UserProfileService: {e}")
                self.user_profile_service = None
        else:
            logger.info("ℹ️ UserProfileService disabled")

        # Statistiques
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "corrections_needed": 0,
            "avg_pipeline_time_ms": 0,
            "conversational_responses": 0,
            "searches_avoided": 0
        }

        logger.info(f"AgentOrchestrator initialized with search_service: {search_service_url}")

    async def process_query(
        self,
        user_query: UserQuery,
        jwt_token: Optional[str] = None
    ) -> ConversationResponse:
        """
        Traite une requête utilisateur de bout en bout

        Args:
            user_query: Requête utilisateur
            jwt_token: Token JWT pour l'authentification

        Returns:
            ConversationResponse avec la réponse finale
        """
        start_time = datetime.now()
        self.stats["total_queries"] += 1

        try:
            logger.info(f"Processing query for user {user_query.user_id}: {user_query.message[:100]}")

            # === ÉTAPE 0: Routage d'intention (NOUVEAU) ===
            logger.info("Step 0: Intent classification")
            intent_response = await self.intent_router.classify_intent(user_query)

            if not intent_response.success:
                return self._create_error_response(
                    f"Failed to classify intent: {intent_response.error}"
                )

            intent_classification = intent_response.data
            logger.info(f"Intent classified: {intent_classification.category.value}, requires_search={intent_classification.requires_search}")

            # Calculer la date actuelle une fois pour tout le pipeline
            current_date = datetime.now().strftime("%Y-%m-%d")
            logger.debug(f"Using current_date: {current_date}")

            # === CAS 1: Réponse conversationnelle (pas de recherche) ===
            if not intent_classification.requires_search:
                self.stats["conversational_responses"] += 1
                self.stats["searches_avoided"] += 1
                self.stats["successful_queries"] += 1

                # Utiliser la réponse suggérée ou générer une réponse persona
                if intent_classification.suggested_response:
                    response_text = intent_classification.suggested_response
                else:
                    response_text = self.intent_router.get_persona_response(
                        intent_classification.category
                    )

                pipeline_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                self._update_avg_time(pipeline_time_ms)

                logger.info(f"Conversational response generated in {pipeline_time_ms}ms")

                # === Sauvegarder dans l'historique Redis (pour conversations aussi) ===
                if self.conversation_cache:
                    try:
                        conversation_id_int = int(user_query.conversation_id) if user_query.conversation_id else None

                        # Sauvegarder le message utilisateur
                        self.conversation_cache.add_message(
                            user_id=user_query.user_id,
                            role="user",
                            content=user_query.message,
                            conversation_id=conversation_id_int
                        )

                        # Sauvegarder la réponse
                        self.conversation_cache.add_message(
                            user_id=user_query.user_id,
                            role="assistant",
                            content=response_text,
                            conversation_id=conversation_id_int,
                            metadata={
                                "intent": intent_classification.category.value,
                                "conversational": True
                            }
                        )
                        logger.info("✅ Conversational exchange saved to Redis cache")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to save conversational exchange: {e}")

                return ConversationResponse(
                    success=True,
                    message=response_text,
                    search_results=None,
                    metadata={
                        "pipeline_time_ms": pipeline_time_ms,
                        "intent": intent_classification.category.value,
                        "requires_search": False,
                        "confidence": intent_classification.confidence
                    }
                )

            # === CAS 2: Pipeline analytique (comparaisons, tendances, prévisions) ===
            if intent_classification.category in [
                IntentCategory.COMPARATIVE_ANALYSIS,
                IntentCategory.TREND_ANALYSIS,
                IntentCategory.PREDICTIVE_ANALYSIS,
                IntentCategory.OPTIMIZATION_RECOMMENDATION,
                IntentCategory.BUDGET_ANALYSIS
            ]:
                logger.info(f"Analytical intent detected: {intent_classification.category.value}")
                return await self._handle_analytical_query(
                    user_query=user_query,
                    intent_classification=intent_classification,
                    current_date=current_date,
                    jwt_token=jwt_token,
                    start_time=start_time
                )

            # === CAS 3: Pipeline financier standard (recherche simple) ===
            logger.info("Financial intent detected, proceeding with standard search pipeline")

            # === ÉTAPE 1: Analyse de la requête ===
            logger.info("Step 1: Analyzing user query")
            analysis_response = await self.query_analyzer.analyze(
                user_query=user_query,
                current_date=current_date
            )

            if not analysis_response.success:
                return self._create_error_response(
                    f"Failed to analyze query: {analysis_response.error}"
                )

            query_analysis: QueryAnalysis = analysis_response.data
            logger.info(f"Query analyzed: intent={query_analysis.intent}, confidence={query_analysis.confidence:.2f}")

            # === ÉTAPE 2: Construction de la query Elasticsearch ===
            logger.info("Step 2: Building Elasticsearch query")
            build_response = await self.query_builder.build_query(
                query_analysis=query_analysis,
                user_id=user_query.user_id,
                current_date=current_date,
                user_query=user_query.message  # Passer la question originale
            )

            if not build_response.success:
                return self._create_error_response(
                    f"Failed to build query: {build_response.error}"
                )

            es_query: ElasticsearchQuery = build_response.data
            logger.info(f"Query built: size={es_query.size}, has_aggs={es_query.aggregations is not None}")
            logger.info(f"Generated ES query: {json.dumps(es_query.query, ensure_ascii=False, indent=2)}")

            # === ÉTAPE 2.3: Validation et correction automatique des catégories et operation_types ===
            logger.info("Step 2.3: Validating and correcting filters (categories + operation_types)")
            if 'filters' in es_query.query:
                original_filters = es_query.query['filters'].copy()

                # Passer le message utilisateur pour détection automatique si category manquante
                corrected_filters = category_validator.validate_and_correct_filters(
                    es_query.query['filters'],
                    user_message=user_query.message
                )

                if corrected_filters != original_filters:
                    es_query.query['filters'] = corrected_filters
                    logger.info(f"✅ Filters corrected: {json.dumps(corrected_filters, ensure_ascii=False, indent=2)}")
                    # Mettre à jour aussi query_analysis pour le fallback
                    query_analysis.filters = corrected_filters
                else:
                    logger.debug("✓ Filters already valid, no correction needed")

            # === ÉTAPE 2.5: Enrichissement des agrégations avec templates ===
            # Enrichir avec templates si agrégations complexes demandées
            if query_analysis.aggregations_needed:
                logger.info(f"Enriching query with aggregation templates: {query_analysis.aggregations_needed}")

                es_query.query = self.aggregation_enricher.enrich(
                    query=es_query.query,
                    aggregations_requested=query_analysis.aggregations_needed
                )

                logger.info(f"Query after enrichment: {json.dumps(es_query.query.get('aggregations', {}), ensure_ascii=False, indent=2)}")
            else:
                # Fallback: Détecter depuis le message si pas détecté par QueryAnalyzer
                detected_templates = self.aggregation_enricher.detect_from_query_text(
                    user_query.message
                )

                if detected_templates:
                    logger.info(f"Detected aggregation templates from message: {detected_templates}")
                    es_query.query = self.aggregation_enricher.enrich(
                        query=es_query.query,
                        aggregations_requested=detected_templates
                    )

                    logger.info(f"Query after fallback enrichment: {json.dumps(es_query.query.get('aggregations', {}), ensure_ascii=False, indent=2)}")

            # === ÉTAPE 3: Exécution de la query (avec auto-correction) ===
            logger.info("Step 3: Executing query with auto-correction")
            search_results = await self._execute_with_correction(
                es_query=es_query,
                query_analysis=query_analysis,
                user_id=user_query.user_id,
                jwt_token=jwt_token
            )

            if not search_results:
                return self._create_error_response(
                    "Failed to execute query after multiple correction attempts"
                )

            # === ÉTAPE 3.5: Fallback textuel si 0 résultats avec filtre catégorie ===
            # Si 0 résultats ET un filtre de catégorie était présent, tenter recherche textuelle
            if (search_results.total == 0 and
                query_analysis.filters.get('category_name')):

                logger.warning("0 results with category filter, attempting text search fallback")

                # Extraire la catégorie recherchée pour l'utiliser comme query texte
                category_filter = query_analysis.filters.get('category_name', {})
                if isinstance(category_filter, dict):
                    search_text = category_filter.get('match') or category_filter.get('term', '')
                else:
                    search_text = str(category_filter)

                if search_text:
                    logger.info(f"Fallback: searching text for '{search_text}'")

                    # Créer une nouvelle analyse sans filtre catégorie
                    # Note: QueryAnalysis est un dataclass, utiliser dataclasses.replace()
                    fallback_filters = query_analysis.filters.copy()
                    fallback_filters.pop('category_name', None)
                    fallback_analysis = dataclasses.replace(query_analysis, filters=fallback_filters)

                    # Construire une nouvelle query avec recherche textuelle
                    fallback_build = await self.query_builder.build_query(
                        query_analysis=fallback_analysis,
                        user_id=user_query.user_id,
                        current_date=current_date,
                        user_query=user_query.message
                    )

                    if fallback_build.success:
                        fallback_query = fallback_build.data

                        # Ajouter le paramètre query pour recherche textuelle
                        if 'query' not in fallback_query.query:
                            fallback_query.query['query'] = search_text

                        # Exécuter la query de fallback
                        fallback_results = await self._execute_with_correction(
                            es_query=fallback_query,
                            query_analysis=fallback_analysis,
                            user_id=user_query.user_id,
                            jwt_token=jwt_token
                        )

                        if fallback_results and fallback_results.total > 0:
                            logger.info(f"Fallback successful: {fallback_results.total} results found with text search")
                            search_results = fallback_results
                            # Mettre à jour query_analysis pour la génération de réponse
                            query_analysis = fallback_analysis
                        else:
                            logger.info("Fallback text search also returned 0 results")
                    else:
                        logger.warning(f"Failed to build fallback query: {fallback_build.error}")

            logger.info(f"Query executed successfully: {search_results.total} results found")

            # === ÉTAPE 3.8: Charger le contexte utilisateur (profil async + historique sync) ===
            logger.info("Step 3.8: Loading user context (profile + conversation history)")
            context_loading_start = datetime.now()

            # Charger le profil utilisateur (async)
            user_profile = None
            if self.user_profile_service:
                try:
                    user_profile = await self.user_profile_service.get_user_profile(
                        user_id=user_query.user_id,
                        jwt_token=jwt_token
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load user profile: {e}")
                    user_profile = None

            # Charger l'historique de conversation (sync)
            conversation_history = []
            conversation_id_int = None
            if user_query.conversation_id:
                try:
                    conversation_id_int = int(user_query.conversation_id)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid conversation_id: {user_query.conversation_id}")

            if self.conversation_cache and conversation_id_int:
                try:
                    conversation_history = self.conversation_cache.get_conversation_history(
                        user_id=user_query.user_id,
                        conversation_id=conversation_id_int,
                        include_system_message=False
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load conversation history: {e}")
                    conversation_history = []

            context_loading_ms = int((datetime.now() - context_loading_start).total_seconds() * 1000)
            logger.info(
                f"✅ Context loaded in {context_loading_ms}ms: "
                f"profile={bool(user_profile)}, "
                f"history={len(conversation_history) if conversation_history else 0} messages"
            )

            # === ÉTAPE 4: Génération de la réponse ===
            logger.info("Step 4: Generating response")
            response_result = await self.response_generator.generate_response(
                user_message=user_query.message,
                search_results=search_results,
                original_query_analysis=query_analysis.__dict__,
                conversation_history=conversation_history if conversation_history else [],
                user_profile=user_profile
            )

            if not response_result.success:
                return self._create_error_response(
                    f"Failed to generate response: {response_result.error}"
                )

            conversation_response: ConversationResponse = response_result.data

            # === ÉTAPE 4.5: Sauvegarder dans l'historique Redis ===
            if self.conversation_cache:
                try:
                    conversation_id_int = int(user_query.conversation_id) if user_query.conversation_id else None

                    # Sauvegarder le message utilisateur
                    self.conversation_cache.add_message(
                        user_id=user_query.user_id,
                        role="user",
                        content=user_query.message,
                        conversation_id=conversation_id_int
                    )

                    # Sauvegarder la réponse de l'assistant
                    self.conversation_cache.add_message(
                        user_id=user_query.user_id,
                        role="assistant",
                        content=conversation_response.message,
                        conversation_id=conversation_id_int,
                        metadata={
                            "total_results": search_results.total,
                            "intent": intent_classification.category.value
                        }
                    )
                    logger.info("✅ Conversation saved to Redis cache")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save to conversation cache: {e}")

            # === Mise à jour des statistiques ===
            pipeline_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.stats["successful_queries"] += 1
            self._update_avg_time(pipeline_time_ms)

            logger.info(f"Query processed successfully in {pipeline_time_ms}ms")

            # Ajouter les métadonnées de pipeline
            conversation_response.metadata["pipeline_time_ms"] = pipeline_time_ms
            conversation_response.metadata["intent"] = intent_classification.category.value
            conversation_response.metadata["requires_search"] = True
            conversation_response.metadata["query_analysis"] = {
                "intent": query_analysis.intent,
                "confidence": query_analysis.confidence,
                "aggregations_requested": query_analysis.aggregations_needed
            }
            # Ajouter la requête Elasticsearch générée
            conversation_response.metadata["elasticsearch_query"] = es_query.query

            return conversation_response

        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
            self.stats["failed_queries"] += 1
            return self._create_error_response(f"Unexpected error: {str(e)}")

    async def _execute_with_correction(
        self,
        es_query: ElasticsearchQuery,
        query_analysis: QueryAnalysis,
        user_id: int,
        jwt_token: Optional[str] = None
    ) -> Optional[SearchResults]:
        """
        Exécute la query avec auto-correction en cas d'échec

        Args:
            es_query: Query Elasticsearch initiale
            query_analysis: Analyse originale de la requête
            user_id: ID utilisateur
            jwt_token: Token JWT

        Returns:
            SearchResults si succès, None si échec après corrections
        """
        current_query = es_query
        attempts = 0

        while attempts <= self.max_correction_attempts:
            try:
                # Tentative d'exécution
                search_results = await self._execute_query(current_query, user_id, jwt_token)

                if search_results:
                    logger.info(f"Query executed successfully on attempt {attempts + 1}")
                    return search_results

            except Exception as e:
                error_message = str(e)
                logger.warning(f"Query execution failed (attempt {attempts + 1}): {error_message}")

                # Si c'est la dernière tentative, on abandonne
                if attempts >= self.max_correction_attempts:
                    logger.error(f"Max correction attempts reached ({self.max_correction_attempts})")
                    return None

                # Sinon, tenter une correction
                logger.info(f"Attempting query correction (attempt {attempts + 1})")
                self.stats["corrections_needed"] += 1

                correction_response = await self.query_builder.correct_query(
                    failed_query=current_query,
                    error_message=error_message,
                    original_analysis=query_analysis,
                    user_id=user_id
                )

                if not correction_response.success:
                    logger.error(f"Query correction failed: {correction_response.error}")
                    return None

                # Utiliser la query corrigée pour la prochaine tentative
                current_query = correction_response.data
                logger.info("Query corrected, retrying execution")

            attempts += 1

        return None

    async def _execute_query(
        self,
        es_query: ElasticsearchQuery,
        user_id: int,
        jwt_token: Optional[str] = None
    ) -> Optional[SearchResults]:
        """
        Exécute une query Elasticsearch via search_service

        Args:
            es_query: Query Elasticsearch
            user_id: ID utilisateur
            jwt_token: Token JWT

        Returns:
            SearchResults si succès, None si échec

        Raises:
            Exception: Si l'exécution échoue (pour permettre la correction)
        """
        try:
            # NOTE: es_query.query contient maintenant directement le format search_service
            # Format: {user_id, filters, sort, page_size, aggregations}
            # On l'envoie DIRECTEMENT à search_service sans transformation!

            search_payload = es_query.query  # Le format search_service complet généré par l'agent

            # S'assurer que user_id est présent (au cas où)
            if "user_id" not in search_payload:
                search_payload["user_id"] = user_id

            # CRITIQUE: S'assurer que sort est TOUJOURS présent (obligatoire pour search_service)
            if "sort" not in search_payload or not search_payload.get("sort"):
                search_payload["sort"] = [{"date": {"order": "desc"}}]
                logger.debug("Added default sort criteria")

            # Headers
            headers = {
                "Content-Type": "application/json"
            }
            if jwt_token:
                headers["Authorization"] = f"Bearer {jwt_token}"

            # Appel à search_service
            url = f"{self.search_service_url}/api/v1/search/search"
            logger.info(f"Search payload: {json.dumps(search_payload, ensure_ascii=False, indent=2)}")
            logger.debug(f"Calling search_service: {url}")

            response = await self.http_client.post(
                url,
                json=search_payload,
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()

                # Parser la réponse de search_service (format custom, pas Elasticsearch brut)
                # search_service retourne: {results: [...], total_hits: X, aggregations: {...}}
                transactions = data.get("results", [])
                total = data.get("total_hits", 0)
                aggregations = data.get("aggregations")
                took = data.get("response_metadata", {}).get("processing_time_ms", 0)

                search_results = SearchResults(
                    hits=transactions,  # search_service appelle ça "results"
                    total=total,  # search_service appelle ça "total_hits"
                    aggregations=aggregations,
                    took_ms=took
                )

                logger.info(f"✅ Parsed {len(transactions)} transactions from {total} total results")

                logger.debug(f"Search successful: {total} results, took {took}ms")
                return search_results

            else:
                # Erreur HTTP - lever une exception pour permettre la correction
                error_text = response.text
                logger.warning(f"Search service returned {response.status_code}: {error_text}")
                raise Exception(f"Search service error ({response.status_code}): {error_text}")

        except httpx.TimeoutException:
            logger.error("Search service timeout")
            raise Exception("Search service timeout")

        except httpx.RequestError as e:
            logger.error(f"Search service request error: {str(e)}")
            raise Exception(f"Search service unavailable: {str(e)}")

        except Exception as e:
            # Re-lever l'exception pour permettre la correction
            raise

    async def _handle_analytical_query(
        self,
        user_query: UserQuery,
        intent_classification,
        current_date: str,
        jwt_token: Optional[str],
        start_time: datetime
    ) -> ConversationResponse:
        """
        Pipeline pour questions analytiques complexes

        Étapes:
        1. AnalyticsAgent.plan_analysis() → génère plan d'exécution
        2. Exécuter N requêtes ES selon le plan
        3. AnalyticsService.compute_insights() → calculs post-agrégation
        4. ResponseGenerator avec insights enrichis

        Args:
            user_query: Requête utilisateur
            intent_classification: Classification d'intention
            current_date: Date actuelle
            jwt_token: Token JWT
            start_time: Timestamp de début

        Returns:
            ConversationResponse avec analyse complète
        """
        try:
            logger.info("=== ANALYTICAL PIPELINE START ===")

            # Étape 1: Planifier l'analyse
            logger.info("Step 1: Planning analytical query")
            plan_response = await self.analytics_agent.plan_analysis(
                user_message=user_query.message,
                intent=intent_classification.category,
                current_date=current_date
            )

            if not plan_response.success:
                return self._create_error_response(f"Failed to plan analysis: {plan_response.error}")

            analysis_plan = plan_response.data
            logger.info(f"Analysis plan: {len(analysis_plan.get('queries', []))} queries, operations: {analysis_plan.get('analytics_operations')}")

            # Étape 2: Exécuter les requêtes ES selon le plan
            logger.info("Step 2: Executing queries")
            all_results = []

            for idx, query_spec in enumerate(analysis_plan.get("queries", []), 1):
                logger.info(f"Executing query {idx}/{len(analysis_plan['queries'])}")

                # Construire la query ES à partir du spec
                es_query = await self._build_query_from_spec(
                    query_spec=query_spec,
                    user_id=user_query.user_id,
                    current_date=current_date
                )

                if not es_query:
                    logger.warning(f"Failed to build query {idx}, skipping")
                    continue

                # Exécuter la query
                search_results = await self._execute_with_correction(
                    es_query=es_query,
                    query_analysis=None,  # Pas besoin pour queries analytiques
                    user_id=user_query.user_id,
                    jwt_token=jwt_token
                )

                if search_results:
                    all_results.append({
                        "query_index": idx,
                        "spec": query_spec,
                        "results": search_results
                    })
                    logger.info(f"Query {idx} executed: {search_results.total} results")
                else:
                    logger.warning(f"Query {idx} returned no results")

            if not all_results:
                return self._create_error_response("No results from analytical queries")

            # Étape 3: Calculer insights avec AnalyticsService
            logger.info("Step 3: Computing insights")
            insights = self._compute_analytical_insights(
                analysis_plan=analysis_plan,
                query_results=all_results
            )

            # Étape 4: Charger contexte utilisateur (profil + historique)
            logger.info("Step 3.5: Loading user context")
            context_loading_start = datetime.now()

            # Charger le profil utilisateur
            user_profile = None
            if self.user_profile_service:
                try:
                    user_profile = await self.user_profile_service.get_user_profile(
                        user_id=user_query.user_id,
                        jwt_token=jwt_token
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load user profile: {e}")
                    user_profile = None

            # Charger historique conversation
            conversation_history = []
            if self.conversation_cache:
                try:
                    conversation_id_int = int(user_query.conversation_id) if user_query.conversation_id else None
                    conversation_history = self.conversation_cache.get_conversation_history(
                        user_id=user_query.user_id,
                        conversation_id=conversation_id_int,
                        include_system_message=False
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load conversation history: {e}")

            context_loading_ms = int((datetime.now() - context_loading_start).total_seconds() * 1000)
            logger.info(
                f"✅ Context loaded in {context_loading_ms}ms: "
                f"profile={bool(user_profile)}, "
                f"history={len(conversation_history)} messages"
            )

            # Étape 5: Générer réponse analytique
            logger.info("Step 4: Generating analytical response")
            response_text = await self._generate_analytical_response(
                user_message=user_query.message,
                analysis_plan=analysis_plan,
                insights=insights,
                conversation_history=conversation_history,
                user_profile=user_profile
            )

            # Calculer temps pipeline
            pipeline_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.stats["successful_queries"] += 1
            self._update_avg_time(pipeline_time_ms)

            logger.info(f"=== ANALYTICAL PIPELINE COMPLETE ({pipeline_time_ms}ms) ===")

            # Sauvegarder dans Redis
            if self.conversation_cache:
                try:
                    conversation_id_int = int(user_query.conversation_id) if user_query.conversation_id else None

                    self.conversation_cache.add_message(
                        user_id=user_query.user_id,
                        role="user",
                        content=user_query.message,
                        conversation_id=conversation_id_int
                    )

                    self.conversation_cache.add_message(
                        user_id=user_query.user_id,
                        role="assistant",
                        content=response_text,
                        conversation_id=conversation_id_int,
                        metadata={
                            "intent": intent_classification.category.value,
                            "analytical": True,
                            "operations": analysis_plan.get("analytics_operations")
                        }
                    )
                    logger.info("✅ Analytical exchange saved to Redis")
                except Exception as e:
                    logger.warning(f"Failed to save analytical exchange: {e}")

            return ConversationResponse(
                success=True,
                message=response_text,
                search_results=all_results[0]["results"] if all_results else None,
                metadata={
                    "pipeline_time_ms": pipeline_time_ms,
                    "intent": intent_classification.category.value,
                    "analytical": True,
                    "num_queries_executed": len(all_results),
                    "operations": analysis_plan.get("analytics_operations"),
                    "insights": insights
                }
            )

        except Exception as e:
            logger.error(f"Error in analytical pipeline: {str(e)}", exc_info=True)
            return self._create_error_response(f"Analytical pipeline error: {str(e)}")

    async def _build_query_from_spec(
        self,
        query_spec: Dict[str, Any],
        user_id: int,
        current_date: str
    ) -> Optional[ElasticsearchQuery]:
        """
        Construit une query Elasticsearch à partir d'un spec du plan analytique

        Args:
            query_spec: Spécification de query du plan
            user_id: ID utilisateur
            current_date: Date actuelle

        Returns:
            ElasticsearchQuery construite
        """
        try:
            # Créer une QueryAnalysis à partir du spec
            filters = query_spec.get("filters", {})
            aggregations = query_spec.get("aggregations", [])
            time_range = query_spec.get("time_range")

            query_analysis = QueryAnalysis(
                intent="analytical",
                entities=filters,
                filters=filters,
                aggregations_needed=aggregations,
                time_range=time_range,
                confidence=1.0
            )

            # Utiliser query_builder pour construire la query
            build_response = await self.query_builder.build_query(
                query_analysis=query_analysis,
                user_id=user_id,
                current_date=current_date
            )

            if not build_response.success:
                logger.error(f"Failed to build query: {build_response.error}")
                return None

            es_query = build_response.data

            # Enrichir avec templates d'agrégations si nécessaire
            if aggregations:
                es_query.query = self.aggregation_enricher.enrich(
                    query=es_query.query,
                    aggregations_requested=aggregations
                )

            # OPTIMISATION: Limiter la taille des résultats pour les requêtes analytiques
            # Quand on ne s'intéresse qu'aux agrégations, pas besoin de récupérer beaucoup de documents
            if aggregations and not es_query.query.get("size"):
                # Limiter à 1 document au lieu de 10 par défaut (on ne s'intéresse qu'aux agrégations)
                es_query.query["size"] = 1
                logger.debug(f"Analytical query optimized: size limited to 1 (aggregations only)")

            return es_query

        except Exception as e:
            logger.error(f"Error building query from spec: {str(e)}", exc_info=True)
            return None

    def _compute_analytical_insights(
        self,
        analysis_plan: Dict[str, Any],
        query_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calcule les insights analytiques à partir des résultats de requêtes

        Args:
            analysis_plan: Plan d'analyse
            query_results: Résultats des requêtes

        Returns:
            Dict d'insights calculés
        """
        insights = {
            "operations_performed": [],
            "results": {}
        }

        operations = analysis_plan.get("analytics_operations", [])
        analysis_type = analysis_plan.get("analysis_type", "")

        logger.info(f"Processing {len(operations)} operations: {operations}, query_results count: {len(query_results)}")

        for operation in operations:
            try:
                if operation == "compare_periods" and len(query_results) >= 2:
                    # Comparaison de 2 périodes
                    comparison = self.analytics_service.compare_periods(
                        period1_results=query_results[0]["results"],
                        period2_results=query_results[1]["results"],
                        metric="total_amount"
                    )
                    insights["results"]["comparison"] = comparison
                    insights["operations_performed"].append("compare_periods")

                elif operation == "detect_trend" and len(query_results) >= 1:
                    # Détection de tendance
                    trend = self.analytics_service.detect_trend(
                        time_series_results=query_results[0]["results"],
                        aggregation_name="monthly_trend"
                    )
                    insights["results"]["trend"] = trend
                    insights["operations_performed"].append("detect_trend")

                elif operation == "forecast_next_period" and len(query_results) >= 1:
                    # Prévision
                    forecast = self.analytics_service.forecast_next_period(
                        time_series_results=query_results[0]["results"],
                        aggregation_name="monthly_trend",
                        method="moving_average"
                    )
                    insights["results"]["forecast"] = forecast
                    insights["operations_performed"].append("forecast_next_period")

                elif operation == "calculate_savings_rate" and len(query_results) >= 2:
                    # Taux d'épargne (revenus vs dépenses)
                    savings_rate = self.analytics_service.calculate_savings_rate(
                        income_results=query_results[0]["results"],
                        expense_results=query_results[1]["results"]
                    )
                    insights["results"]["savings_rate"] = savings_rate
                    insights["operations_performed"].append("calculate_savings_rate")

                elif operation == "recommend_savings_opportunities" and len(query_results) >= 1:
                    # Recommandations d'économies
                    recommendations = self.analytics_service.recommend_savings_opportunities(
                        by_category_results=query_results[0]["results"],
                        target_reduction_pct=10.0
                    )
                    insights["results"]["recommendations"] = recommendations
                    insights["operations_performed"].append("recommend_savings_opportunities")

                elif operation == "classify_fixed_vs_variable" and len(query_results) >= 1:
                    # Classification charges fixes/variables
                    classification = self.analytics_service.classify_fixed_vs_variable(
                        by_category_results=query_results[0]["results"]
                    )
                    insights["results"]["fixed_vs_variable"] = classification
                    insights["operations_performed"].append("classify_fixed_vs_variable")

                elif operation == "analyze_multi_period_budget" and len(query_results) >= 3:
                    # Analyse budgétaire multi-périodes
                    period_labels = [qr.get("spec", {}).get("period_label", f"Period {i+1}")
                                    for i, qr in enumerate(query_results)]

                    multi_period_analysis = self.analytics_service.analyze_multi_period_budget(
                        period_results=query_results,
                        period_labels=period_labels
                    )
                    insights["results"]["multi_period_analysis"] = multi_period_analysis
                    insights["operations_performed"].append("analyze_multi_period_budget")

                elif operation == "overview_multi_period":
                    # Fallback: overview_multi_period est mappé vers analyze_multi_period_budget
                    # Car le LLM génère parfois cette opération qui n'existe pas dans le code
                    logger.info(f"🔄 Mapping 'overview_multi_period' to 'analyze_multi_period_budget' (query_results: {len(query_results)})")

                    if len(query_results) < 2:
                        logger.warning(f"Not enough query results for multi-period analysis: {len(query_results)} < 2")
                        continue
                    period_labels = [qr.get("spec", {}).get("period_label", f"Period {i+1}")
                                    for i, qr in enumerate(query_results)]

                    multi_period_analysis = self.analytics_service.analyze_multi_period_budget(
                        period_results=query_results,
                        period_labels=period_labels
                    )
                    insights["results"]["multi_period_analysis"] = multi_period_analysis
                    insights["operations_performed"].append("analyze_multi_period_budget")

                elif operation == "identify_spending_patterns" and len(query_results) >= 1:
                    # Identification de patterns de dépenses
                    patterns = self.analytics_service.identify_spending_patterns(
                        by_category_results=query_results[0]["results"],
                        time_series_results=query_results[0]["results"] if len(query_results) > 0 else None
                    )
                    insights["results"]["spending_patterns"] = patterns
                    insights["operations_performed"].append("identify_spending_patterns")

                elif operation == "calculate_budget_health_score" and len(query_results) >= 1:
                    # Score de santé budgétaire
                    # Extraire les métriques nécessaires
                    result_obj = query_results[0]["results"]

                    income = 0
                    expenses = 0
                    fixed_expenses = 0
                    discretionary_expenses = 0

                    if result_obj and result_obj.aggregations:
                        aggs = result_obj.aggregations

                        # Extraire income
                        if "income" in aggs:
                            income_agg = aggs["income"]
                            if "total_income" in income_agg:
                                income = income_agg["total_income"].get("value", 0) or 0

                        # Extraire expenses
                        if "expenses" in aggs:
                            exp_agg = aggs["expenses"]
                            if "total_expenses" in exp_agg:
                                expenses = exp_agg["total_expenses"].get("value", 0) or 0

                        # Extraire fixed/discretionary
                        if "fixed_expenses" in aggs:
                            fixed_agg = aggs["fixed_expenses"]
                            if "total_fixed" in fixed_agg:
                                fixed_expenses = fixed_agg["total_fixed"].get("value", 0) or 0

                        if "variable_expenses" in aggs:
                            var_agg = aggs["variable_expenses"]
                            if "total_variable" in var_agg:
                                discretionary_expenses = var_agg["total_variable"].get("value", 0) or 0

                    health_score = self.analytics_service.calculate_budget_health_score(
                        income=income,
                        expenses=expenses,
                        fixed_expenses=fixed_expenses,
                        discretionary_expenses=discretionary_expenses
                    )
                    insights["results"]["budget_health_score"] = health_score
                    insights["operations_performed"].append("calculate_budget_health_score")

            except Exception as e:
                logger.error(f"Error computing insight for operation {operation}: {str(e)}", exc_info=True)
                insights["results"][operation] = {"error": str(e)}

        logger.info(f"Computed {len(insights['operations_performed'])} insights")
        return insights

    async def _generate_analytical_response(
        self,
        user_message: str,
        analysis_plan: Dict[str, Any],
        insights: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Génère une réponse analytique enrichie et personnalisée

        Args:
            user_message: Message utilisateur
            analysis_plan: Plan d'analyse
            insights: Insights calculés
            conversation_history: Historique conversation
            user_profile: Profil budgétaire de l'utilisateur (optionnel)

        Returns:
            Réponse textuelle personnalisée
        """
        try:
            # Créer un prompt spécial pour réponses analytiques avec personnalisation
            analytical_prompt = ChatPromptTemplate.from_messages([
                ("system", """Tu es un conseiller financier expert en analyse de données.

Tu dois créer une réponse analytique basée sur des INSIGHTS calculés (pas des transactions brutes).

🎯 **PERSONNALISATION SELON LE PROFIL UTILISATEUR**

Tu as accès au profil budgétaire de l'utilisateur. Utilise-le pour PERSONNALISER ta réponse :

1. **Adapter le ton selon le segment budgétaire :**
   - Budget serré (épargne < 10%) → Ton ENCOURAGEANT, conseils d'OPTIMISATION
   - Équilibré (épargne 10-30%) → Ton NEUTRE, conseils de MAINTIEN
   - Confortable (épargne > 30%) → Ton POSITIF, conseils d'INVESTISSEMENT

2. **Contextualiser les montants :**
   - Comparer les dépenses au "reste à vivre" de l'utilisateur
   - Calculer les pourcentages par rapport au budget disponible
   - Mentionner l'impact sur le taux d'épargne actuel

3. **Tenir compte du pattern comportemental :**
   - Acheteur impulsif / erratic_spender → Suggérer planification et groupement d'achats
   - Planificateur → Valoriser la constance, optimisations fines
   - Dépensier haute fréquence → Adapter à ce rythme

4. **Graceful degradation :**
   - Si profil non disponible → Rester NEUTRE, ne pas faire de suppositions
   - Mentionner que l'analyse serait plus précise avec un profil complet

Format de réponse:
1. 📊 **Réponse directe avec chiffres clés**
   - Montants principaux (dépenses, revenus, épargne)
   - Taux d'épargne, tendances

2. 📈 **Analyse détaillée et interprétation**
   - Contextualiser selon le profil utilisateur
   - Expliquer les tendances et variations
   - Comparer aux périodes précédentes

3. 💡 **Insights et observations importantes**
   - Points d'attention spécifiques au profil
   - Meilleurs/pires périodes
   - Opportunités d'optimisation

4. ✅ **Recommandations actionnables**
   - Conseils adaptés au segment budgétaire
   - Actions concrètes et personnalisées
   - Objectifs réalistes selon le profil

Sois clair, précis, actionnable et TOUJOURS personnalisé selon le profil."""),
                ("user", """**PROFIL UTILISATEUR:**
{user_profile}

**Question:** {user_message}

**Type d'analyse:** {analysis_type}

**Insights calculés:**
{insights}

**Historique conversation:**
{conversation_history}

Génère une réponse analytique complète, personnalisée selon le profil utilisateur.""")
            ])

            chain = analytical_prompt | ChatOpenAI(model="gpt-4o", temperature=0.3)

            # Formater historique
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-3:]
            ]) if conversation_history else "Aucun historique"

            # Formater profil utilisateur
            from ..services.user_profile_service import UserProfileService
            profile_service = UserProfileService()
            profile_text = profile_service.format_profile_for_prompt(user_profile)

            # Invoquer LLM
            result = await chain.ainvoke({
                "user_message": user_message,
                "analysis_type": analysis_plan.get("analysis_type", ""),
                "insights": json.dumps(insights, ensure_ascii=False, indent=2),
                "conversation_history": history_text,
                "user_profile": profile_text
            })

            return result.content

        except Exception as e:
            logger.error(f"Error generating analytical response: {str(e)}", exc_info=True)
            return f"J'ai analysé vos données mais j'ai rencontré une difficulté pour formuler la réponse. Insights disponibles: {json.dumps(insights.get('operations_performed', []))}"

    def _create_error_response(self, error_message: str) -> ConversationResponse:
        """
        Crée une réponse d'erreur conviviale

        Args:
            error_message: Message d'erreur technique

        Returns:
            ConversationResponse avec message d'erreur
        """
        logger.error(f"Creating error response: {error_message}")

        user_friendly_message = (
            "Désolé, je rencontre des difficultés pour traiter votre demande. "
            "Pouvez-vous reformuler votre question ou essayer une autre recherche ?"
        )

        return ConversationResponse(
            success=False,
            message=user_friendly_message,
            metadata={
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        )

    def _extract_filters_from_es_query(self, es_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrait les filtres d'une query Elasticsearch et les convertit au format search_service

        Args:
            es_query: Query Elasticsearch avec structure {"bool": {"must": [...], "filter": [...]}}

        Returns:
            Dict de filtres au format search_service {"field": {"operator": value}}
        """
        filters = {}

        try:
            # Structure typique: {"bool": {"must": [...], "filter": [...]}}
            if isinstance(es_query, dict) and "bool" in es_query:
                bool_query = es_query["bool"]

                # Extraire les conditions de "must"
                must_conditions = bool_query.get("must", [])
                for condition in must_conditions:
                    if isinstance(condition, dict):
                        # Range query: {"range": {"amount": {"gte": 100}}}
                        if "range" in condition:
                            for field, range_value in condition["range"].items():
                                filters[field] = range_value

                        # Term query: {"term": {"transaction_type": "debit"}}
                        elif "term" in condition:
                            for field, term_value in condition["term"].items():
                                filters[field] = term_value

                        # Terms query: {"terms": {"category_name": ["food", "restaurant"]}}
                        elif "terms" in condition:
                            for field, terms_values in condition["terms"].items():
                                filters[field] = terms_values

                        # Match query: {"match": {"merchant_name": "carrefour"}}
                        elif "match" in condition:
                            for field, match_value in condition["match"].items():
                                filters[field] = match_value

                # Extraire les conditions de "filter" (sauf user_id qui est déjà dans le payload)
                filter_conditions = bool_query.get("filter", [])
                for condition in filter_conditions:
                    if isinstance(condition, dict):
                        # Range query
                        if "range" in condition:
                            for field, range_value in condition["range"].items():
                                filters[field] = range_value

                        # Term query
                        elif "term" in condition:
                            for field, term_value in condition["term"].items():
                                # Ne pas inclure user_id dans filters car déjà dans payload
                                if field != "user_id":
                                    filters[field] = term_value

                        # Terms query
                        elif "terms" in condition:
                            for field, terms_values in condition["terms"].items():
                                if field != "user_id":
                                    filters[field] = terms_values

                        # Match query
                        elif "match" in condition:
                            for field, match_value in condition["match"].items():
                                if field != "user_id":
                                    filters[field] = match_value

        except Exception as e:
            logger.warning(f"Error extracting filters from ES query: {str(e)}")
            # En cas d'erreur, retourner un dict vide
            return {}

        return filters

    def _update_avg_time(self, pipeline_time_ms: int):
        """Met à jour le temps moyen de pipeline"""
        current_avg = self.stats["avg_pipeline_time_ms"]
        total_queries = self.stats["total_queries"]

        if total_queries > 0:
            self.stats["avg_pipeline_time_ms"] = (
                (current_avg * (total_queries - 1) + pipeline_time_ms) / total_queries
            )

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'orchestrateur et des agents"""

        success_rate = 0.0
        if self.stats["total_queries"] > 0:
            success_rate = self.stats["successful_queries"] / self.stats["total_queries"]

        correction_rate = 0.0
        if self.stats["successful_queries"] > 0:
            correction_rate = self.stats["corrections_needed"] / self.stats["successful_queries"]

        conversational_rate = 0.0
        if self.stats["total_queries"] > 0:
            conversational_rate = self.stats["conversational_responses"] / self.stats["total_queries"]

        return {
            "orchestrator": {
                **self.stats,
                "success_rate": success_rate,
                "correction_rate": correction_rate,
                "conversational_rate": conversational_rate
            },
            "agents": {
                "intent_router": self.intent_router.get_stats(),
                "query_analyzer": self.query_analyzer.get_stats(),
                "query_builder": self.query_builder.get_stats(),
                "response_generator": self.response_generator.get_stats()
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check de l'orchestrateur et des agents"""
        try:
            # Vérifier search_service
            search_service_healthy = False
            try:
                response = await self.http_client.get(
                    f"{self.search_service_url}/health",
                    timeout=5.0
                )
                search_service_healthy = response.status_code == 200
            except:
                pass

            return {
                "status": "healthy" if search_service_healthy else "degraded",
                "orchestrator": "healthy",
                "search_service": "healthy" if search_service_healthy else "unhealthy",
                "agents": {
                    "intent_router": "initialized",
                    "query_analyzer": "initialized",
                    "query_builder": "initialized",
                    "response_generator": "initialized"
                },
                "stats": self.get_stats()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def process_query_stream(
        self,
        user_query: UserQuery,
        jwt_token: Optional[str] = None
    ):
        """
        Version streaming de process_query - Yield des événements de progression et chunks de réponse

        Cette méthode centralise TOUTE la logique du pipeline pour éviter la duplication
        entre l'endpoint stream et non-stream.

        Args:
            user_query: Requête utilisateur
            jwt_token: Token JWT pour l'authentification

        Yields:
            Dict avec type: 'status' | 'response_chunk' | 'response_data' | 'error'
        """
        start_time = datetime.now()
        self.stats["total_queries"] += 1

        try:
            logger.info(f"Processing STREAM query for user {user_query.user_id}: {user_query.message[:100]}")

            # === ÉTAPE 0: Routage d'intention ===
            yield {'type': 'status', 'message': '• Analyse de votre question...'}

            logger.info("Step 0: Intent classification")
            intent_response = await self.intent_router.classify_intent(user_query)

            if not intent_response.success:
                yield {'type': 'error', 'error': f"Failed to classify intent: {intent_response.error}"}
                return

            intent_classification = intent_response.data
            logger.info(f"Intent classified: {intent_classification.category.value}, requires_search={intent_classification.requires_search}")

            current_date = datetime.now().strftime("%Y-%m-%d")

            # === CAS 1: Réponse conversationnelle (pas de recherche) ===
            if not intent_classification.requires_search:
                self.stats["conversational_responses"] += 1
                self.stats["searches_avoided"] += 1
                self.stats["successful_queries"] += 1

                yield {'type': 'status', 'message': '• Préparation de la réponse...'}

                if intent_classification.suggested_response:
                    response_text = intent_classification.suggested_response
                else:
                    response_text = self.intent_router.get_persona_response(
                        intent_classification.category
                    )

                # Stream la réponse mot par mot
                words = response_text.split(' ')
                for i, word in enumerate(words):
                    chunk = word + (' ' if i < len(words) - 1 else '')
                    yield {'type': 'response_chunk', 'content': chunk}
                    await asyncio.sleep(0.05)

                pipeline_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                self._update_avg_time(pipeline_time_ms)

                yield {
                    'type': 'response_data',
                    'data': {
                        'success': True,
                        'message': response_text,
                        'search_results': None,
                        'metadata': {
                            'pipeline_time_ms': pipeline_time_ms,
                            'intent': intent_classification.category.value,
                            'requires_search': False
                        }
                    }
                }
                return

            # === CAS 2: Pipeline financier standard ===
            yield {'type': 'status', 'message': '• Recherche de vos transactions...'}

            logger.info("Step 1: Analyzing user query")
            analysis_response = await self.query_analyzer.analyze(
                user_query=user_query,
                current_date=current_date
            )

            if not analysis_response.success:
                yield {'type': 'error', 'error': f"Failed to analyze query: {analysis_response.error}"}
                return

            query_analysis: QueryAnalysis = analysis_response.data

            logger.info("Step 2: Building Elasticsearch query")
            build_response = await self.query_builder.build_query(
                query_analysis=query_analysis,
                user_id=user_query.user_id,
                current_date=current_date,
                user_query=user_query.message
            )

            if not build_response.success:
                yield {'type': 'error', 'error': f"Failed to build query: {build_response.error}"}
                return

            es_query: ElasticsearchQuery = build_response.data

            # Validation et correction des filtres
            if 'filters' in es_query.query:
                corrected_filters = category_validator.validate_and_correct_filters(
                    es_query.query['filters'],
                    user_message=user_query.message
                )
                if corrected_filters != es_query.query['filters']:
                    es_query.query['filters'] = corrected_filters
                    query_analysis.filters = corrected_filters

            # Enrichissement des agrégations
            if query_analysis.aggregations_needed:
                es_query.query = self.aggregation_enricher.enrich(
                    query=es_query.query,
                    aggregations_requested=query_analysis.aggregations_needed
                )

            yield {'type': 'status', 'message': '• Analyse de vos données...'}

            logger.info("Step 3: Executing query")
            search_results = await self._execute_with_correction(
                es_query=es_query,
                query_analysis=query_analysis,
                user_id=user_query.user_id,
                jwt_token=jwt_token
            )

            if not search_results:
                yield {'type': 'error', 'error': 'Failed to execute query after corrections'}
                return

            # Message de progression basé sur les résultats
            if search_results.total > 0:
                yield {'type': 'status', 'message': f'• {search_results.total} transaction(s) trouvée(s), génération de la réponse...'}
            else:
                yield {'type': 'status', 'message': '• Préparation de la réponse...'}

            logger.info(f"Query executed: {search_results.total} results")

            # === ÉTAPE 3.8: Charger le contexte utilisateur ===
            logger.info("Step 3.8: Loading user context")
            context_loading_start = datetime.now()

            user_profile = None
            if self.user_profile_service:
                try:
                    user_profile = await self.user_profile_service.get_user_profile(
                        user_id=user_query.user_id,
                        jwt_token=jwt_token
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load user profile: {e}")

            conversation_history = []
            conversation_id_int = None
            if user_query.conversation_id:
                try:
                    conversation_id_int = int(user_query.conversation_id)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid conversation_id: {user_query.conversation_id}")

            if self.conversation_cache and conversation_id_int:
                try:
                    conversation_history = self.conversation_cache.get_conversation_history(
                        user_id=user_query.user_id,
                        conversation_id=conversation_id_int,
                        include_system_message=False
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load conversation history: {e}")

            context_loading_ms = int((datetime.now() - context_loading_start).total_seconds() * 1000)
            logger.info(
                f"✅ Context loaded in {context_loading_ms}ms: "
                f"profile={bool(user_profile)}, "
                f"history={len(conversation_history)} messages"
            )

            # === ÉTAPE 4: Stream de la réponse ===
            logger.info("Step 4: Streaming response generation")
            accumulated_response = ""

            async for chunk in self.response_generator.generate_response_stream(
                user_message=user_query.message,
                search_results=search_results,
                original_query_analysis=query_analysis.__dict__,
                conversation_history=conversation_history,
                user_profile=user_profile
            ):
                accumulated_response += chunk
                yield {'type': 'response_chunk', 'content': chunk}

            # === Finalisation ===
            pipeline_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.stats["successful_queries"] += 1
            self._update_avg_time(pipeline_time_ms)

            logger.info(f"Stream query processed successfully in {pipeline_time_ms}ms")

            # Retourner les données finales
            yield {
                'type': 'response_data',
                'data': {
                    'success': True,
                    'message': accumulated_response,
                    'search_results': search_results,
                    'query_analysis': query_analysis,
                    'es_query': es_query,
                    'metadata': {
                        'pipeline_time_ms': pipeline_time_ms,
                        'intent': intent_classification.category.value,
                        'requires_search': True,
                        'total_results': search_results.total
                    }
                }
            }

        except Exception as e:
            logger.error(f"Unexpected error in stream pipeline: {str(e)}", exc_info=True)
            self.stats["failed_queries"] += 1
            yield {'type': 'error', 'error': str(e)}

    async def close(self):
        """Ferme proprement les connexions"""
        await self.http_client.aclose()
        logger.info("AgentOrchestrator closed")
