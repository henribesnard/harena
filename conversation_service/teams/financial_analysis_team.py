"""Equipe AutoGen pour l'analyse financière.

Cette implémentation orchestre deux agents (classification d'intention
et extraction d'entités) au sein d'un `GroupChat` AutoGen. Elle gère
également le cache et les métriques de performance de l'équipe.
"""

from __future__ import annotations

from typing import Any, Dict

import json
import logging
import time
from datetime import datetime

from autogen import GroupChat, GroupChatManager

from conversation_service.agents.financial import (
    EntityExtractorAgent,
    IntentClassifierAgent,
)
from conversation_service.core.cache_manager import CacheManager
from conversation_service.prompts.autogen import TEAM_ORCHESTRATION_PHASE2_MESSAGE as TEAM_ORCHESTRATION_MESSAGE
from conversation_service.utils.metrics_collector import (
    AdvancedMetricsCollector as MetricsCollector,
)


logger = logging.getLogger("conversation_service.teams")


class FinancialAnalysisTeam:
    """Equipe d'agents dédiée à l'analyse financière."""

    def __init__(self) -> None:
        # Agents principaux
        self.intent_classifier = IntentClassifierAgent()
        self.entity_extractor = EntityExtractorAgent()

        # Services transverses
        self.cache_manager = CacheManager()
        self.metrics_collector = MetricsCollector()

        # Orchestration AutoGen
        self._setup_autogen_groupchat()

        # Métriques propres à l'équipe
        self.team_metrics: Dict[str, Any] = {
            "total_requests": 0,
            "failures": 0,
            "avg_processing_time_ms": 0.0,
        }

    # ------------------------------------------------------------------
    def _setup_autogen_groupchat(self) -> None:
        """Configure le GroupChat et son manager."""

        self.group_chat = GroupChat(
            agents=[self.intent_classifier, self.entity_extractor],
            messages=[{"role": "system", "content": TEAM_ORCHESTRATION_MESSAGE}],
            max_round=3,
            speaker_selection_method="round_robin",
        )

        llm_config = {
            "config_list": [
                {
                    "model": "deepseek-chat",
                    "temperature": 0.0,
                    "cache_seed": 42,
                }
            ]
        }
        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat, llm_config=llm_config
        )

    # ------------------------------------------------------------------
    async def process_user_message(self, user_message: str, user_id: int) -> Dict[str, Any]:
        """Traite un message utilisateur via l'équipe."""

        cache_key = f"financial_team:{user_id}:{user_message}"
        start_time = time.perf_counter()

        try:
            cached = await self.cache_manager.get_semantic_cache(
                cache_key, cache_type="response"
            )
            if cached is not None:
                processing_time_ms = (time.perf_counter() - start_time) * 1000
                await self._cache_and_update_metrics(
                    cache_key, cached, processing_time_ms
                )
                return cached

            await self.intent_classifier.a_initiate_chat(
                self.group_chat_manager, message=user_message
            )

            results = await self._extract_team_results()
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            await self._cache_and_update_metrics(
                cache_key, results, processing_time_ms
            )
            return results

        except Exception as exc:  # pragma: no cover - sécurité runtime
            return await self._handle_team_failure(
                exc, user_message, user_id, start_time
            )

    # ------------------------------------------------------------------
    async def _extract_team_results(self) -> Dict[str, Any]:
        """Analyse les messages du group chat pour extraire les résultats."""

        intent_result: Dict[str, Any] | None = None
        entities_result: Dict[str, Any] | None = None
        errors: list[str] = []

        for msg in getattr(self.group_chat, "messages", []):
            name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", "")
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")

            if name == self.intent_classifier.name and content:
                try:
                    intent_result = json.loads(content)
                except Exception as exc:  # pragma: no cover - protection JSON
                    errors.append(f"intent_parse_error: {exc}")
            elif name == self.entity_extractor.name and content:
                try:
                    entities_result = json.loads(content)
                except Exception as exc:  # pragma: no cover - protection JSON
                    errors.append(f"entities_parse_error: {exc}")

        self._validate_intent_entity_coherence(intent_result, entities_result)

        return {
            "intent": intent_result or {},
            "entities": (entities_result or {}).get("entities", []),
            "errors": errors + (entities_result or {}).get("errors", []),
        }

    # ------------------------------------------------------------------
    def _validate_intent_entity_coherence(
        self, intent_result: Dict[str, Any] | None, entities_result: Dict[str, Any] | None
    ) -> bool:
        """Valide la cohérence entre l'intention et les entités."""

        if not intent_result or not entities_result:
            return False

        intent = intent_result.get("intent")
        success = entities_result.get("extraction_success", False)
        if intent == "GENERAL_INQUIRY" and success:
            return False
        return True

    # ------------------------------------------------------------------
    async def _cache_and_update_metrics(
        self, cache_key: str, results: Dict[str, Any], processing_time_ms: float
    ) -> None:
        """Met en cache les résultats et met à jour les métriques."""

        await self.cache_manager.set_semantic_cache(
            cache_key, results, cache_type="response"
        )

        self.metrics_collector.record_histogram(
            "financial_team.processing_time", processing_time_ms
        )

        success = not results.get("errors")
        counter_name = (
            "financial_team.success" if success else "financial_team.failure"
        )
        self.metrics_collector.increment_counter(counter_name)

        self._update_team_metrics(processing_time_ms, success)

    # ------------------------------------------------------------------
    def _update_team_metrics(self, processing_time_ms: float, success: bool) -> None:
        """Met à jour les métriques agrégées de l'équipe."""

        metrics = self.team_metrics
        metrics["total_requests"] += 1
        if not success:
            metrics["failures"] += 1

        count = metrics["total_requests"]
        metrics["avg_processing_time_ms"] = (
            (metrics["avg_processing_time_ms"] * (count - 1) + processing_time_ms)
            / count
        )

    # ------------------------------------------------------------------
    async def _handle_team_failure(
        self,
        error: Exception,
        user_message: str,
        user_id: int,
        start_time: float,
    ) -> Dict[str, Any]:
        """Gère les échecs globaux de l'équipe."""

        logger.exception("Echec équipe FinancialAnalysisTeam: %s", error)
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        self.metrics_collector.increment_counter("financial_team.failure")
        self.metrics_collector.record_histogram(
            "financial_team.processing_time", processing_time_ms
        )
        self._update_team_metrics(processing_time_ms, False)

        fallback = {
            "intent": {
                "intent": "GENERAL_INQUIRY",
                "confidence": 0.0,
                "team_context": {
                    "original_message": user_message,
                    "user_id": user_id,
                },
            },
            "entities": [],
            "errors": [str(error)],
        }
        return fallback

    # ------------------------------------------------------------------
    async def health_check(self) -> Dict[str, Any]:
        """Retourne l'état de santé de l'équipe."""

        cache_health = await self.cache_manager.health_check()
        return {
            "agents": {
                "intent_classifier": {
                    "success": getattr(self.intent_classifier, "success_count", 0),
                    "errors": getattr(self.intent_classifier, "error_count", 0),
                },
                "entity_extractor": {
                    "success": getattr(self.entity_extractor, "success_count", 0),
                    "errors": getattr(self.entity_extractor, "error_count", 0),
                },
            },
            "team_metrics": self.team_metrics,
            "cache": cache_health,
            "config": {
                "max_round": getattr(self.group_chat, "max_round", 0),
                "llm_model": "deepseek-chat",
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

