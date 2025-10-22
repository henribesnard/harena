"""
Agent Orchestrator - Coordonne les agents LangChain avec auto-correction
Pipeline: Analyze → Build Query → Execute → Correct (if needed) → Generate Response
"""
import logging
import asyncio
import httpx
import json
from typing import Dict, Any, Optional
from datetime import datetime

from ..models import (
    UserQuery, QueryAnalysis, ElasticsearchQuery, SearchResults,
    ConversationResponse, AgentResponse
)
from ..agents.intent_router_agent import IntentRouterAgent
from ..agents.query_analyzer_agent import QueryAnalyzerAgent
from ..agents.elasticsearch_builder_agent import ElasticsearchBuilderAgent
from ..agents.response_generator_agent import ResponseGeneratorAgent
from ..models.intent import IntentCategory
from ..core.aggregation_enricher import AggregationEnricher
from ..core.category_validator import category_validator

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

        # Initialiser l'enrichisseur d'agrégations
        self.aggregation_enricher = AggregationEnricher()

        self.search_service_url = search_service_url
        self.max_correction_attempts = max_correction_attempts

        # HTTP client pour search_service
        # Timeout augmenté pour laisser le temps aux agrégations complexes de s'exécuter
        self.http_client = httpx.AsyncClient(timeout=60.0)

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

            # === CAS 2: Pipeline financier complet (recherche requise) ===
            logger.info("Financial intent detected, proceeding with search pipeline")

            # Calculer la date actuelle une fois pour tout le pipeline
            current_date = datetime.now().strftime("%Y-%m-%d")
            logger.debug(f"Using current_date: {current_date}")

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
                current_date=current_date
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
                    fallback_analysis = query_analysis.model_copy(deep=True)
                    fallback_analysis.filters.pop('category_name', None)

                    # Construire une nouvelle query avec recherche textuelle
                    fallback_build = await self.query_builder.build_query(
                        query_analysis=fallback_analysis,
                        user_id=user_query.user_id,
                        current_date=current_date
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

            # === ÉTAPE 4: Génération de la réponse ===
            logger.info("Step 4: Generating response")
            response_result = await self.response_generator.generate_response(
                user_message=user_query.message,
                search_results=search_results,
                original_query_analysis=query_analysis.__dict__
            )

            if not response_result.success:
                return self._create_error_response(
                    f"Failed to generate response: {response_result.error}"
                )

            conversation_response: ConversationResponse = response_result.data

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

    async def close(self):
        """Ferme proprement les connexions"""
        await self.http_client.aclose()
        logger.info("AgentOrchestrator closed")
