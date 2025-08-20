"""
Search Query Agent for interfacing with Search Service.

This agent generates optimized search queries for the Search Service based on
detected intents and user messages. It includes entity extraction, query
optimization, and standardized search service communication.

Classes:
    - SearchQueryAgent: Main search query generation agent
    - QueryOptimizer: Helper class for query optimization

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Search Service Integration
"""

import time
import logging
import unicodedata
import re
import json
import httpx
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .base_financial_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..models.service_contracts import (
    SearchServiceQuery,
    SearchServiceResponse,
    QueryMetadata,
    SearchParameters,
    SearchFilters,
)
from ..models.financial_models import IntentResult, FinancialEntity, EntityType
from ..core.deepseek_client import DeepSeekClient
from ..utils.validators import ContractValidator

logger = logging.getLogger(__name__)
# Dedicated logger for security/audit events
audit_logger = logging.getLogger("audit")


FRENCH_MONTHS = {
    "janvier": "01",
    "fevrier": "02",
    "mars": "03",
    "avril": "04",
    "mai": "05",
    "juin": "06",
    "juillet": "07",
    "aout": "08",
    "septembre": "09",
    "octobre": "10",
    "novembre": "11",
    "decembre": "12",
}


def resolve_relative_date(normalized_value: str) -> Dict[str, Dict[str, str]]:
    """Resolve a relative date keyword into an absolute date range."""
    now = datetime.utcnow()
    value = normalized_value.lower()

    if value == "current_month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        end = next_month - timedelta(days=1)
    elif value == "current_week":
        start = now - timedelta(days=now.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=6)
    else:
        return {}

    return {"date": {"gte": start.strftime("%Y-%m-%d"), "lte": end.strftime("%Y-%m-%d")}}


class QueryOptimizer:
    """Helper class for optimizing search queries."""

    @staticmethod
    def optimize_search_text(
        user_message: str, intent_result: IntentResult
    ) -> Optional[str]:
        """
        Optimize search text based on intent and entities.

        Args:
            user_message: Original user message
            intent_result: Detected intent with entities

        Returns:
            Optimized search text or ``None`` if no useful terms remain
        """
        # Start with clean user message
        search_text = user_message.lower().strip()

        # Remove accents for consistent search behavior
        search_text = unicodedata.normalize("NFD", search_text)
        search_text = "".join(
            ch for ch in search_text if unicodedata.category(ch) != "Mn"
        )

        # Remove punctuation
        search_text = re.sub(r"[^\w\s]", " ", search_text)

        # Remove common stop words, question words, and generic terms
        stop_words = {
            "le",
            "la",
            "les",
            "de",
            "du",
            "des",
            "un",
            "une",
            "mon",
            "ma",
            "mes",
            "recherche",
            "rechercher",
            "depense",
            "depenses",
            "transaction",
            "transactions",
            "combien",
            "pourquoi",
            "pour",
            "quel",
            "quelle",
            "quels",
            "quelles",
            "qui",
            "que",
            "quoi",
            "ou",
            "quand",
            "comment",
            "ce",
            "cet",
            "cette",
            "ces",
            "mois",
            "je",
            "j",
            "ai",
            "jai",
            "a",
            "euro",
            "euros",
            # Comparatives
            "superieur",
            "superieure",
            "superieurs",
            "superieures",
            "inferieur",
            "inferieure",
            "inferieurs",
            "inferieures",
            "plus",
            "moins",
            "egal",
            "egale",
            "egaux",
            "egales",
            # Monetary symbols and currency tokens
            "€",
            "$",
            "£",
            "¥",
            "dollar",
            "dollars",
            "yen",
            "yens",
            "usd",
            "eur",
            "gbp",
            "chf",
            # Numeric tokens
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        }
        words = [
            word
            for word in search_text.split()
            if word not in stop_words and not re.fullmatch(r"\d+(?:[.,]\d+)?", word)
        ]

        # Remove basic verb forms (infinitives) for more aggressive normalization
        words = [word for word in words if not word.endswith(("er", "ir", "re"))]

        # Add entity-based keywords without duplicating existing terms
        seen_words = set(words)
        has_amount = False
        if intent_result.entities:
            for entity in intent_result.entities:
                if entity.entity_type in {
                    EntityType.MERCHANT,
                    "MERCHANT",
                    EntityType.CATEGORY,
                    "CATEGORY",
                }:
                    if entity.normalized_value:
                        value = str(entity.normalized_value).lower()
                        if value not in seen_words:
                            words.append(value)
                            seen_words.add(value)
                if entity.entity_type in {EntityType.AMOUNT, "AMOUNT"}:
                    has_amount = True

        if has_amount:
            words = [w for w in words if not w.replace('.', '', 1).isdigit()]

        # Remove any remaining duplicates while preserving order and limit length
        unique_words: List[str] = []
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
        if not unique_words:
            return None

        optimized_text = " ".join(unique_words)[:200]
        logger.debug("Normalized search text: %s", optimized_text)
        return optimized_text  # Limit search text length

    @staticmethod
    def extract_date_filters(intent_result: IntentResult) -> Dict[str, Any]:
        """Extract date range filters from intent entities."""
        date_filters: Dict[str, Any] = {}

        if not intent_result.entities:
            return date_filters

        for entity in intent_result.entities:
            if entity.entity_type == EntityType.DATE_RANGE and entity.normalized_value:
                normalized_date = entity.normalized_value
                date: Dict[str, Any] = {}

                if isinstance(normalized_date, dict):
                    start_date = normalized_date.get("start_date")
                    end_date = normalized_date.get("end_date")
                    if start_date:
                        date["gte"] = start_date
                    if end_date:
                        date["lte"] = end_date
                elif isinstance(normalized_date, str):
                    try:
                        if len(normalized_date) == 7:  # YYYY-MM format
                            start = f"{normalized_date}-01"
                            date["gte"] = start
                            date["lte"] = start
                        elif len(normalized_date) == 10:  # YYYY-MM-DD format
                            date["gte"] = normalized_date
                            date["lte"] = normalized_date
                    except Exception:
                        pass

                if date:
                    date_filters["date"] = date
                    break
            if (
                entity.entity_type == EntityType.DATE
                and isinstance(entity.normalized_value, str)
            ):
                value = entity.normalized_value.lower().strip()
                value = unicodedata.normalize("NFD", value)
                value = "".join(
                    ch for ch in value if unicodedata.category(ch) != "Mn"
                )
                parts = value.split()
                year = None
                month = None
                for part in parts:
                    if part.isdigit() and len(part) == 4:
                        year = part
                    elif part in FRENCH_MONTHS:
                        month = FRENCH_MONTHS[part]
                if month:
                    if not year:
                        year = datetime.utcnow().strftime("%Y")
                    date_filters["date"] = {
                        "gte": f"{year}-{month}-01",
                        "lte": f"{year}-{month}-31",
                    }
                    break
            if entity.entity_type == EntityType.RELATIVE_DATE and isinstance(
                entity.normalized_value, str
            ):
                relative = resolve_relative_date(str(entity.normalized_value))
                if relative:
                    date_filters.update(relative)
                    break

        return date_filters

    @staticmethod
    def extract_amount_filters(intent_result: IntentResult) -> Dict[str, Any]:
        """Extract amount filters from intent entities.

        The mock intent agent may suggest whether the comparison should be
        greater-than or less-than through ``suggested_actions``.  When such a
        hint is present we build the corresponding range.  Otherwise we fall
        back to a loose ±10%% tolerance around the detected amount to keep the
        original behaviour.
        """
        amount_filters: Dict[str, Any] = {}

        if not intent_result.entities:
            return amount_filters

        for entity in intent_result.entities:
            if entity.entity_type == EntityType.AMOUNT:
                actions = intent_result.suggested_actions or []
                value = entity.normalized_value

                def _to_float(val: Any) -> Optional[float]:
                    if isinstance(val, str):
                        cleaned = (
                            val.replace("€", "")
                            .replace("euros", "")
                            .replace("euro", "")
                            .replace(",", ".")
                            .strip()
                        )
                        cleaned = re.sub(r"[^0-9.\-]", "", cleaned.lower())
                        try:
                            return float(cleaned)
                        except ValueError:
                            return None
                    if isinstance(val, (int, float)):
                        return float(val)
                    return None

                if isinstance(value, dict):
                    amount: Dict[str, float] = {}
                    for op in ("gte", "lte"):
                        if op in value:
                            num = _to_float(value[op])
                            if num is not None:
                                amount[op] = num
                    if amount:
                        amount_filters["amount"] = amount
                        break

                else:
                    num = _to_float(value)
                    if num is not None:
                        normalized_amount = num
                        abs_amount = abs(normalized_amount)

                        if "filter_by_amount_greater" in actions:
                            # Use absolute amount comparison for "greater than" queries
                            amount_filters["amount_abs"] = {"gte": abs_amount}
                        elif "filter_by_amount_less" in actions:
                            # Use absolute amount comparison for "less than" queries
                            amount_filters["amount_abs"] = {"lte": abs_amount}
                        else:
                            tolerance = abs(normalized_amount) * 0.1  # 10% tolerance
                            amount_filters["amount"] = {
                                "gte": normalized_amount - tolerance,
                                "lte": normalized_amount + tolerance,
                            }
                        break

        return amount_filters


class SearchQueryAgent(BaseFinancialAgent):
    """
    Agent for generating and executing search queries.

    This agent takes intent detection results and user messages, then:
    1. Extracts additional entities using AI
    2. Generates optimized SearchServiceQuery contracts
    3. Executes queries against the Search Service
    4. Returns structured search results

    Attributes:
        search_service_url: Base URL for the Search Service
        http_client: HTTP client for service communication
        query_optimizer: Helper for query optimization
    """

    def __init__(
        self,
        deepseek_client: DeepSeekClient,
        search_service_url: str,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the search query agent.

        Args:
            deepseek_client: Configured DeepSeek client
            search_service_url: Base URL for Search Service
            config: Optional agent configuration
        """
        if config is None:
            config = AgentConfig(
                name="search_query_agent",
                model_client_config={
                    "model": "deepseek-chat",
                    "api_key": deepseek_client.api_key,
                    "base_url": deepseek_client.base_url,
                },
                system_message=self._get_system_message(),
                max_consecutive_auto_reply=1,
                description="Search query generation and execution agent",
                temperature=0.2,  # Low temperature for consistent entity extraction
                max_tokens=250,
                timeout_seconds=12,
            )

        super().__init__(
            name=config.name, config=config, deepseek_client=deepseek_client
        )

        # Ensure a local name attribute exists without overriding base class behavior
        self._name = config.name

        self.search_service_url = search_service_url.rstrip("/")
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.query_optimizer = QueryOptimizer()
        self.default_limit = min(
            int(os.getenv("SEARCH_QUERY_DEFAULT_LIMIT", "100")), 100
        )

        logger.info(
            f"Initialized SearchQueryAgent with service URL: {search_service_url}"
        )

    def extract_amount_filters(self, intent_result: IntentResult) -> Dict[str, Any]:
        """Expose amount filter extraction for testing and reuse."""
        return self.query_optimizer.extract_amount_filters(intent_result)

    async def _execute_operation(
        self, input_data: Dict[str, Any], user_id: int
    ) -> Dict[str, Any]:
        """
        Execute search query operation.

        Args:
            input_data: Dict containing 'intent_result' and 'user_message'
            user_id: ID of the requesting user

        Returns:
            Dict with search results and metadata
        """
        intent_result = input_data.get("intent_result")
        user_message = input_data.get("user_message", "")

        if not intent_result:
            raise ValueError("intent_result is required for search query generation")

        return await self.process_search_request(intent_result, user_message, user_id)

    async def process_search_request(
        self,
        intent_result: IntentResult,
        user_message: str,
        user_id: int,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Process a search request end-to-end.

        Args:
            intent_result: Detected intent with entities
            user_message: Original user message
            user_id: ID of the requesting user

        Returns:
            Dictionary containing search results and metadata
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Extract additional entities using AI
            enhanced_entities = await self._extract_additional_entities(
                user_message, intent_result, user_id
            )

            # Step 2: Generate search service query contract
            limit = limit or self.default_limit
            search_query = await self._generate_search_contract(
                intent_result,
                user_message,
                user_id,
                enhanced_entities,
                limit=limit,
                offset=offset,
            )

            logger.info(
                "Search query generated: text=%s filters=%s",
                search_query.search_parameters.search_text,
                search_query.filters,
            )

            # Step 3: Execute search query (single call)
            search_response = await self._execute_search_query(search_query)

            results = list(getattr(search_response, "results", []) or [])

            execution_time = (time.perf_counter() - start_time) * 1000

            return {
                "content": f"Search completed: {len(results)} results",
                "metadata": {
                    "search_query": search_query.dict(),
                    "search_response": search_response.dict(),
                    "enhanced_entities": (
                        [e.dict() for e in enhanced_entities]
                        if enhanced_entities
                        else []
                    ),
                    "execution_time_ms": execution_time,
                    "search_results_count": len(results),
                },
                "confidence_score": min(
                    intent_result.confidence + 0.1, 1.0
                ),  # Boost confidence slightly
                "token_usage": {
                    "prompt_tokens": 50,  # Estimated
                    "completion_tokens": 20,
                    "total_tokens": 70,
                },
            }

        except Exception as e:
            logger.error(f"Search request processing failed: {e}")
            raise

    async def _generate_search_contract(
        self,
        intent_result: IntentResult,
        user_message: str,
        user_id: int,
        enhanced_entities: Optional[List[FinancialEntity]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> SearchServiceQuery:
        """
        Generate a SearchServiceQuery contract from intent and message.

        Args:
            intent_result: Detected intent with entities
            user_message: Original user message
            user_id: ID of the requesting user
            enhanced_entities: Additional entities from AI extraction

        Returns:
            SearchServiceQuery contract for the Search Service
        """
        # Optimize search text
        search_text = self.query_optimizer.optimize_search_text(
            user_message, intent_result
        )
        logger.debug("Optimized search text: %s", search_text)

        # Extract filters from entities
        date_filters = self.query_optimizer.extract_date_filters(intent_result)
        amount_filters = self.query_optimizer.extract_amount_filters(intent_result)

        # Combine all entities
        all_entities: List[FinancialEntity] = (
            intent_result.entities.copy() if intent_result.entities else []
        )
        if enhanced_entities:
            all_entities.extend(enhanced_entities)

        # Build search filters
        search_filters: Dict[str, Any] = {}
        search_filters.update(date_filters)
        search_filters.update(amount_filters)

        # Group entities by type for filter creation
        categories = [
            str(e.normalized_value)
            for e in all_entities
            if e.entity_type in {EntityType.CATEGORY, "CATEGORY"} and e.normalized_value
        ]
        if categories:
            search_filters["category_name"] = categories

        operation_types = [
            str(e.normalized_value)
            for e in all_entities
            if e.entity_type in {EntityType.OPERATION_TYPE, "OPERATION_TYPE"} and e.normalized_value
        ]
        if operation_types:
            search_filters["operation_type"] = operation_types[0]

        # Merchant name filtering is intentionally avoided. Some databases may
        # store transactions with an empty ``merchant_name`` field. Adding a
        # ``merchant_name`` filter would exclude those records, even though the
        # merchant can still be matched via full-text search on
        # ``primary_description``/``searchable_text``. Merchant entities are
        # still extracted earlier to enrich ``search_text``, so relying on the
        # text search is sufficient here.

        # Always filter by user_id for security and multi-tenant isolation
        search_filters["user_id"] = user_id
        logger.debug("Calculated search filters: %s", search_filters)

        # Create query metadata
        query_metadata = QueryMetadata(
            conversation_id=f"conv_{int(time.time())}",  # Placeholder - should come from context
            user_id=user_id,
            intent_type=intent_result.intent_type,
            language="fr",
            priority="normal",
            source_agent=self.name,
        )

        # Create search parameters based on intent
        default_limit = (
            100 if intent_result.intent_type == "TRANSACTION_SEARCH" else 10
        )
        limit = min(limit or default_limit, 100)
        search_params = SearchParameters(
            search_text=search_text,
            max_results=limit,
            offset=offset,
            include_highlights=True,
            boost_recent=intent_result.intent_type
            in [
                "BALANCE_CHECK",
                "SPENDING_ANALYSIS",
            ],
            fuzzy_matching=True if search_text and len(search_text.split()) > 1 else False,
        )

        # Create complete search query
        filters_obj = (
            SearchFilters(**search_filters) if search_filters else SearchFilters()
        )

        search_query = SearchServiceQuery(
            query_metadata=query_metadata,
            search_parameters=search_params,
            filters=filters_obj,
        )

        # Validate the query
        validator = ContractValidator()
        validation_errors = validator.validate_search_query(search_query.dict())
        if validation_errors:
            logger.warning(f"Search query validation warnings: {validation_errors}")

        return search_query

    async def _execute_search_query(
        self, query: SearchServiceQuery
    ) -> SearchServiceResponse:
        """Execute a single search query against the Search Service."""
        try:
            url = f"{self.search_service_url}/search"
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"ConversationService/{self.name}",
            }

            metadata = query.query_metadata
            audit_logger.info(
                "search query execution: user_id=%s conversation_id=%s query_id=%s intent_type=%s",
                metadata.user_id,
                metadata.conversation_id,
                metadata.query_id,
                metadata.intent_type,
            )
            logger.info(
                (
                    "Sending search request to Search Service: user_id=%s, "
                    "conversation_id=%s, query_id=%s"
                ),
                metadata.user_id,
                metadata.conversation_id,
                metadata.query_id,
            )
            logger.info(
                "Search parameters before request: text=%s filters=%s",
                query.search_parameters.search_text,
                query.filters.model_dump(exclude_none=True),
            )

            request_payload = (
                query.to_search_request()
                if hasattr(query, "to_search_request")
                else query.dict()
            )
            logger.debug("Search request payload: %s", request_payload)

            response = await self.http_client.post(
                url=url, json=request_payload, headers=headers
            )
            response.raise_for_status()
            response_data = response.json()

            for result in response_data.get("results", []):
                if "currency_code" in result and "currency" not in result:
                    result["currency"] = result.pop("currency_code")
                if "primary_description" in result and "description" not in result:
                    result["description"] = result.pop("primary_description")
                if "merchant_name" in result and "merchant" not in result:
                    result["merchant"] = result.pop("merchant_name")
                if "category_name" in result and "category" not in result:
                    result["category"] = result.pop("category_name")
                if "account_id" in result and isinstance(result["account_id"], int):
                    result["account_id"] = str(result["account_id"])

            search_response = SearchServiceResponse(**response_data)
            results_len = len(search_response.results or [])
            logger.info("Search service returned %s results", results_len)
            if search_response.results:
                logger.info("First search result: %s", search_response.results[0])
            meta = search_response.response_metadata
            total_res = (
                meta.get("total_results", 0)
                if isinstance(meta, dict)
                else getattr(meta, "total_results", 0)
            )
            returned_res = (
                meta.get("returned_results", 0)
                if isinstance(meta, dict)
                else getattr(meta, "returned_results", 0)
            )
            logger.info("Search service returned %s total results", total_res)
            logger.info(
                "Search query executed successfully: %s results", returned_res
            )

            return search_response

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Search service HTTP error: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(f"Search service error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Search service request error: {e}")
            raise Exception(f"Search service unavailable: {e}")
        except Exception as e:
            logger.error(f"Search query execution failed: {e}")
            raise

    async def _extract_additional_entities(
        self, message: str, intent_result: IntentResult, user_id: int
    ) -> List[FinancialEntity]:
        """
        Extract additional entities using AI that weren't detected previously.

        Args:
            message: User message
            intent_result: Intent with existing entities

        Returns:
            List of additional entities found
        """
        try:
            # Prepare context for entity extraction
            existing_entities = intent_result.entities if intent_result.entities else []
            context = self._prepare_entity_extraction_context(
                message, existing_entities
            )

            # Call DeepSeek for entity extraction
            logger.debug("Extracting additional entities for user_id=%s", user_id)
            response = await self.deepseek_client.generate_response(
                messages=[
                    {"role": "system", "content": self._get_entity_extraction_prompt()},
                    {"role": "user", "content": context},
                ],
                temperature=0.1,
                max_tokens=120,
                user=str(user_id),
                use_cache=True,
            )

            # Parse AI response
            additional_entities = self._parse_entity_response(response.content)

            return additional_entities

        except Exception as e:
            logger.warning(f"Additional entity extraction failed: {e}")
            return []

    def _prepare_entity_extraction_context(
        self, message: str, existing_entities: List[FinancialEntity]
    ) -> str:
        """Prepare context for AI entity extraction."""
        context = f'Message: "{message}"\n\n'

        if existing_entities:
            context += "Entités déjà détectées:\n"
            entities_by_type: Dict[str, List[str]] = {}
            for entity in existing_entities:
                entities_by_type.setdefault(
                    getattr(entity.entity_type, "value", entity.entity_type), []
                ).append(str(entity.normalized_value))
            for entity_type, values in entities_by_type.items():
                context += f"- {entity_type}: {values}\n"
            context += "\n"

        context += (
            "Extrais toutes les entités financières supplémentaires non détectées."
        )
        return context

    def _parse_entity_response(self, ai_content: str) -> List[FinancialEntity]:
        """Parse AI entity extraction response."""
        entities: List[FinancialEntity] = []

        try:
            entities_idx = ai_content.lower().find("entities:")
            if entities_idx != -1:
                json_block = ai_content[entities_idx + len("entities:") :].strip()
                logger.debug(f"Raw entities JSON block: {json_block}")
                try:
                    parsed = json.loads(json_block)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict):
                                for key, value in item.items():
                                    if not value or (
                                        isinstance(value, str)
                                        and value.lower() == "aucune"
                                    ):
                                        continue
                                    try:
                                        entity_type = EntityType(key.upper())
                                    except ValueError:
                                        continue
                                    entities.append(
                                        FinancialEntity(
                                            entity_type=entity_type,
                                            raw_value=str(value),
                                            normalized_value=str(value),
                                            confidence=0.8,
                                        )
                                    )
                        logger.debug(
                            "Recognized entities: %s",
                            [
                                f"{getattr(e.entity_type, 'value', e.entity_type)}: {e.normalized_value}"
                                for e in entities
                            ],
                        )
                        return entities
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parsing failed: {e}")

            # Fallback to simple line-based parsing
            lines = ai_content.strip().split("\n")
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        entity_type_str = parts[0].strip().upper()
                        entity_value = parts[1].strip()

                        if entity_value and entity_value.lower() != "aucune":
                            try:
                                entity_type = EntityType(entity_type_str)
                            except ValueError:
                                continue

                            entities.append(
                                FinancialEntity(
                                    entity_type=entity_type,
                                    raw_value=entity_value,
                                    normalized_value=entity_value,
                                    confidence=0.8,
                                )
                            )

            logger.debug(
                "Recognized entities: %s",
                [
                    f"{getattr(e.entity_type, 'value', e.entity_type)}: {e.normalized_value}"
                    for e in entities
                ],
            )

        except Exception as e:
            logger.warning(f"Failed to parse entity response: {e}")

        return entities

    def _get_system_message(self) -> str:
        """Get system message for the agent."""
        return """Tu es un agent spécialisé dans la génération de requêtes de recherche pour les données financières.

Ton rôle est de:
1. Analyser les intentions détectées et les messages utilisateur
2. Extraire des entités financières supplémentaires 
3. Générer des requêtes optimisées pour le service de recherche
4. Exécuter les requêtes et retourner les résultats structurés

Types d'entités à extraire:
- MERCHANT: Noms de commerçants
- CATEGORY: Catégories de transactions
- AMOUNT: Montants et plages de montants
- DATE_RANGE: Dates et périodes
- TRANSACTION_TYPE: Types de transactions (débit/crédit)

Optimise les requêtes pour la pertinence et la performance."""

    def _get_entity_extraction_prompt(self) -> str:
        """Get entity extraction prompt for DeepSeek."""
        return """Extrais toutes les entités financières du message utilisateur.

Types d'entités à rechercher:
- MERCHANT: Noms de commerçants, magasins, services
- CATEGORY: Catégories de dépenses (alimentation, transport, etc.)
- AMOUNT: Montants, prix, valeurs monétaires
- DATE_RANGE: Dates, périodes, mois, années
- TRANSACTION_TYPE: Type de transaction (achat, virement, etc.)

Format de réponse:
TYPE_ENTITE: valeur_trouvée
TYPE_ENTITE: autre_valeur

Si aucune entité supplémentaire n'est trouvée, réponds: "aucune"."""

    async def close(self) -> None:
        """Close HTTP client resources."""
        await self.http_client.aclose()
        logger.info("SearchQueryAgent HTTP client closed")
