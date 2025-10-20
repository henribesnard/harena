"""
Agent Orchestrator - Coordonne les agents LangChain avec auto-correction
Pipeline: Analyze → Build Query → Execute → Correct (if needed) → Generate Response
"""
import logging
import asyncio
import httpx
from typing import Dict, Any, Optional
from datetime import datetime

from ..models import (
    UserQuery, QueryAnalysis, ElasticsearchQuery, SearchResults,
    ConversationResponse, AgentResponse
)
from ..agents.query_analyzer_agent import QueryAnalyzerAgent
from ..agents.elasticsearch_builder_agent import ElasticsearchBuilderAgent
from ..agents.response_generator_agent import ResponseGeneratorAgent

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
        self.query_analyzer = QueryAnalyzerAgent(llm_model=llm_model)
        self.query_builder = ElasticsearchBuilderAgent(llm_model=llm_model)
        self.response_generator = ResponseGeneratorAgent(llm_model="gpt-4o")

        self.search_service_url = search_service_url
        self.max_correction_attempts = max_correction_attempts

        # HTTP client pour search_service
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Statistiques
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "corrections_needed": 0,
            "avg_pipeline_time_ms": 0
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

            # === ÉTAPE 1: Analyse de la requête ===
            logger.info("Step 1: Analyzing user query")
            analysis_response = await self.query_analyzer.analyze(user_query)

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
                current_date=datetime.now().strftime("%Y-%m-%d")
            )

            if not build_response.success:
                return self._create_error_response(
                    f"Failed to build query: {build_response.error}"
                )

            es_query: ElasticsearchQuery = build_response.data
            logger.info(f"Query built: size={es_query.size}, has_aggs={es_query.aggregations is not None}")

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
            conversation_response.metadata["query_analysis"] = {
                "intent": query_analysis.intent,
                "confidence": query_analysis.confidence,
                "aggregations_requested": query_analysis.aggregations_needed
            }

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
            # Construire la query complète
            full_query = {
                "query": es_query.query,
                "size": es_query.size
            }

            if es_query.aggregations:
                full_query["aggs"] = es_query.aggregations

            if es_query.sort:
                full_query["sort"] = es_query.sort

            # Headers
            headers = {
                "Content-Type": "application/json"
            }
            if jwt_token:
                headers["Authorization"] = f"Bearer {jwt_token}"

            # Appel à search_service
            url = f"{self.search_service_url}/api/search/query"
            logger.debug(f"Calling search_service: {url}")

            response = await self.http_client.post(
                url,
                json=full_query,
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()

                # Parser la réponse Elasticsearch
                hits = data.get("hits", {}).get("hits", [])
                total = data.get("hits", {}).get("total", {}).get("value", 0)
                aggregations = data.get("aggregations")
                took = data.get("took", 0)

                search_results = SearchResults(
                    hits=hits,
                    total=total,
                    aggregations=aggregations,
                    took_ms=took
                )

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

        return {
            "orchestrator": {
                **self.stats,
                "success_rate": success_rate,
                "correction_rate": correction_rate
            },
            "agents": {
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
