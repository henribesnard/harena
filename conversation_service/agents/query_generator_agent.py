"""
Query Generator Agent for Harena Conversation Service.

This module implements the query generation agent using AutoGen v0.4 framework.
It transforms user intentions and entities into optimized Elasticsearch queries
for the search_service with standardized contracts.

Key Features:
- AutoGen AssistantAgent with specialized Elasticsearch query generation
- Intent-specific query templates with few-shot learning
- Standardized SearchServiceQuery contract generation
- Multi-query type support: filtered_search, aggregated_search, text_search
- Intelligent query optimization based on user patterns
- Comprehensive validation and error handling

Author: Harena Conversation Team
Created: 2025-01-31
Version: 1.0.0 - AutoGen v0.4 + Search Service Integration
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid

# AutoGen imports
from autogen import AssistantAgent
from openai import AsyncOpenAI

# Local imports
from ..models.core_models import (
    IntentType, AgentResponse, FinancialEntity, 
    SearchServiceQuery, QueryType, HarenaValidators
)
from ..models.agent_models import AgentConfig
from ..base_agent import BaseFinancialAgent
from ..prompts.query_prompts import (
    QUERY_GENERATION_SYSTEM_PROMPT, QUERY_FEW_SHOT_EXAMPLES,
    INTENT_QUERY_TEMPLATES, ELASTICSEARCH_FIELD_MAPPING
)
from ..core.cache_manager import CacheManager
from ..core.metrics_collector import MetricsCollector
from ..utils.logging import get_structured_logger

__all__ = ["QueryGeneratorAgent", "QueryOptimizer", "SearchQueryValidator"]

# Configure logging
logger = get_structured_logger(__name__)

# ================================
# QUERY OPTIMIZATION SYSTEM
# ================================

@dataclass
class QueryOptimizationRule:
    """Rule for query optimization based on patterns."""
    pattern: str
    optimization: Dict[str, Any]
    priority: int = 1
    conditions: Optional[Dict[str, Any]] = None

class QueryOptimizer:
    """
    Intelligent query optimizer for Elasticsearch queries.
    
    Applies performance optimizations, field selection, and
    query structure improvements based on intent patterns.
    """
    
    # Performance optimization rules
    OPTIMIZATION_RULES = [
        QueryOptimizationRule(
            pattern="high_volume_user",
            optimization={
                "limit": 10,  # Reduce from default 20
                "fields": ["user_id", "amount", "merchant_name", "date", "category_name"],
                "timeout_ms": 3000
            },
            priority=1
        ),
        QueryOptimizationRule(
            pattern="temporal_analysis",
            optimization={
                "aggregations": {"enabled": True, "bucket_size": "month"},
                "fields": ["date", "month_year", "amount", "category_name"],
                "sort": [{"date": "desc"}]
            },
            priority=2
        ),
        QueryOptimizationRule(
            pattern="category_analysis", 
            optimization={
                "aggregations": {"enabled": True, "group_by": ["category_name"]},
                "fields": ["category_name", "amount", "merchant_name", "date"],
                "sort": [{"amount": "desc"}]
            },
            priority=2
        ),
        QueryOptimizationRule(
            pattern="merchant_analysis",
            optimization={
                "fields": ["merchant_name", "amount", "date", "category_name"],
                "sort": [{"date": "desc"}],
                "limit": 15
            },
            priority=2
        )
    ]
    
    @classmethod
    def optimize_query(
        cls, 
        base_query: Dict[str, Any], 
        intent: IntentType,
        user_patterns: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply optimization rules to base query."""
        optimized_query = base_query.copy()
        
        # Apply intent-specific optimizations
        intent_pattern = cls._get_intent_pattern(intent)
        
        for rule in cls.OPTIMIZATION_RULES:
            if rule.pattern == intent_pattern or rule.pattern == "high_volume_user":
                # Apply optimization
                for key, value in rule.optimization.items():
                    if key == "fields":
                        optimized_query.setdefault("search_parameters", {})["fields"] = value
                    elif key == "limit":
                        optimized_query.setdefault("search_parameters", {})["limit"] = value
                    elif key == "timeout_ms":
                        optimized_query.setdefault("search_parameters", {})["timeout_ms"] = value
                    elif key == "aggregations":
                        optimized_query.setdefault("aggregations", {}).update(value)
                    elif key == "sort":
                        optimized_query.setdefault("search_parameters", {})["sort"] = value
        
        return optimized_query
    
    @classmethod
    def _get_intent_pattern(cls, intent: IntentType) -> str:
        """Map intent to optimization pattern."""
        pattern_mapping = {
            IntentType.TEMPORAL_ANALYSIS: "temporal_analysis",
            IntentType.CATEGORY_ANALYSIS: "category_analysis", 
            IntentType.MERCHANT_ANALYSIS: "merchant_analysis",
            IntentType.SPENDING_ANALYSIS: "category_analysis",
            IntentType.BUDGET_ANALYSIS: "temporal_analysis"
        }
        return pattern_mapping.get(intent, "default")

# ================================
# QUERY VALIDATION SYSTEM
# ================================

class SearchQueryValidator:
    """
    Comprehensive validator for SearchServiceQuery contracts.
    
    Ensures generated queries are syntactically valid, semantically correct,
    and optimized for the target Elasticsearch cluster.
    """
    
    REQUIRED_FIELDS = ["query_metadata", "search_parameters", "filters"]
    VALID_OPERATORS = ["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "between", "match"]
    VALID_QUERY_TYPES = ["simple_search", "filtered_search", "aggregated_search", "text_search"]
    
    @classmethod
    def validate_query(cls, query: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate SearchServiceQuery structure and content.
        
        Returns (is_valid, error_messages)
        """
        errors = []
        
        # Check required top-level fields
        for field in cls.REQUIRED_FIELDS:
            if field not in query:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate query metadata
        metadata_errors = cls._validate_metadata(query.get("query_metadata", {}))
        errors.extend(metadata_errors)
        
        # Validate search parameters
        search_errors = cls._validate_search_parameters(query.get("search_parameters", {}))
        errors.extend(search_errors)
        
        # Validate filters
        filter_errors = cls._validate_filters(query.get("filters", {}))
        errors.extend(filter_errors)
        
        # Validate aggregations if present
        if "aggregations" in query:
            agg_errors = cls._validate_aggregations(query["aggregations"])
            errors.extend(agg_errors)
        
        return len(errors) == 0, errors
    
    @classmethod
    def _validate_metadata(cls, metadata: Dict[str, Any]) -> List[str]:
        """Validate query metadata structure."""
        errors = []
        required_meta_fields = ["intent_type", "agent_name", "timestamp"]
        
        for field in required_meta_fields:
            if field not in metadata:
                errors.append(f"Missing metadata field: {field}")
        
        # Validate intent type
        if "intent_type" in metadata:
            try:
                IntentType(metadata["intent_type"])
            except ValueError:
                errors.append(f"Invalid intent_type: {metadata['intent_type']}")
        
        return errors
    
    @classmethod
    def _validate_search_parameters(cls, params: Dict[str, Any]) -> List[str]:
        """Validate search parameters."""
        errors = []
        
        # Validate query_type
        if "query_type" in params:
            if params["query_type"] not in cls.VALID_QUERY_TYPES:
                errors.append(f"Invalid query_type: {params['query_type']}")
        
        # Validate numeric parameters
        if "limit" in params:
            if not isinstance(params["limit"], int) or params["limit"] <= 0:
                errors.append("limit must be positive integer")
        
        if "timeout_ms" in params:
            if not isinstance(params["timeout_ms"], int) or params["timeout_ms"] <= 0:
                errors.append("timeout_ms must be positive integer")
        
        return errors
    
    @classmethod
    def _validate_filters(cls, filters: Dict[str, Any]) -> List[str]:
        """Validate filter structure and operators."""
        errors = []
        
        # Validate each filter type
        for filter_type in ["required", "optional", "ranges"]:
            if filter_type in filters:
                if not isinstance(filters[filter_type], list):
                    errors.append(f"{filter_type} must be a list")
                    continue
                
                for i, filter_item in enumerate(filters[filter_type]):
                    if not isinstance(filter_item, dict):
                        errors.append(f"{filter_type}[{i}] must be a dict")
                        continue
                    
                    # Check required filter fields
                    required = ["field", "operator", "value"]
                    for req_field in required:
                        if req_field not in filter_item:
                            errors.append(f"{filter_type}[{i}] missing {req_field}")
                    
                    # Validate operator
                    if "operator" in filter_item:
                        if filter_item["operator"] not in cls.VALID_OPERATORS:
                            errors.append(f"Invalid operator: {filter_item['operator']}")
        
        return errors
    
    @classmethod
    def _validate_aggregations(cls, aggregations: Dict[str, Any]) -> List[str]:
        """Validate aggregation structure."""
        errors = []
        
        if "enabled" in aggregations and not isinstance(aggregations["enabled"], bool):
            errors.append("aggregations.enabled must be boolean")
        
        if "types" in aggregations and not isinstance(aggregations["types"], list):
            errors.append("aggregations.types must be list")
        
        return errors

# ================================
# MAIN QUERY GENERATOR AGENT
# ================================

class QueryGeneratorAgent(BaseFinancialAgent):
    """
    Query Generation Agent using AutoGen v0.4 framework.
    
    Transforms user intentions and extracted entities into optimized
    Elasticsearch queries with standardized SearchServiceQuery contracts.
    """
    
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        search_client,
        cache_manager: Optional[CacheManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        enable_optimization: bool = True
    ):
        """Initialize Query Generator Agent."""
        
        config = AgentConfig(
            name="query_generator",
            model_name="gpt-4",  # Use GPT-4 for complex query generation
            temperature=0.1,  # Low temperature for consistent query structure
            max_tokens=800,
            timeout_seconds=15,
            system_message=QUERY_GENERATION_SYSTEM_PROMPT,
            few_shot_examples=QUERY_FEW_SHOT_EXAMPLES
        )
        
        super().__init__(config, openai_client, cache_manager, metrics_collector)
        
        self.search_client = search_client
        self.enable_optimization = enable_optimization
        self.query_optimizer = QueryOptimizer()
        self.validator = SearchQueryValidator()
        
        # Query generation statistics
        self.total_queries = 0
        self.successful_queries = 0
        self.optimization_hits = 0
        self.validation_failures = 0
        
        logger.info(
            "Query Generator Agent initialized",
            model=config.model_name,
            optimization_enabled=enable_optimization,
            temperature=config.temperature
        )
    
    async def _process_implementation(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate optimized search query from intent and entities."""
        
        user_message = input_data.get("user_message", "")
        user_id = input_data.get("user_id")
        intent_result = input_data.get("intent", {})
        entities = input_data.get("entities", [])
        context = input_data.get("context", {})
        
        if not user_message or not intent_result:
            raise ValueError("user_message and intent are required")
        
        intent_type = intent_result.get("intent")
        intent_confidence = intent_result.get("confidence", 0.0)
        
        try:
            # Step 1: Generate base query using LLM
            base_query = await self._generate_base_query(
                user_message, intent_type, entities, user_id, context
            )
            
            # Step 2: Apply optimizations
            if self.enable_optimization:
                optimized_query = self.query_optimizer.optimize_query(
                    base_query, IntentType(intent_type), context.get("user_patterns")
                )
                self.optimization_hits += 1
            else:
                optimized_query = base_query
            
            # Step 3: Validate query structure
            is_valid, validation_errors = self.validator.validate_query(optimized_query)
            
            if not is_valid:
                self.validation_failures += 1
                logger.warning(
                    "Query validation failed",
                    intent=intent_type,
                    errors=validation_errors,
                    query_preview=str(optimized_query)[:200]
                )
                
                # Fallback to simple query
                optimized_query = self._create_fallback_query(user_id, intent_type)
            
            # Step 4: Execute search if enabled
            search_results = None
            if hasattr(self.search_client, 'search') and self.search_client:
                try:
                    search_results = await self.search_client.search(optimized_query)
                    logger.debug(
                        "Search executed successfully",
                        results_count=len(search_results.get("results", [])),
                        query_id=optimized_query.get("query_metadata", {}).get("query_id")
                    )
                except Exception as search_error:
                    logger.warning(f"Search execution failed: {search_error}")
                    search_results = {"results": [], "error": str(search_error)}
            
            self.total_queries += 1
            self.successful_queries += 1
            
            return {
                "query": optimized_query,
                "search_results": search_results,
                "generation_metadata": {
                    "intent": intent_type,
                    "intent_confidence": intent_confidence,
                    "entities_count": len(entities),
                    "optimized": self.enable_optimization,
                    "validation_passed": is_valid,
                    "validation_errors": validation_errors if not is_valid else [],
                    "query_complexity": self._assess_query_complexity(optimized_query)
                }
            }
            
        except Exception as e:
            logger.error(
                "Query generation failed",
                intent=intent_type,
                user_message=user_message[:100],
                error=str(e),
                exc_info=True
            )
            
            # Return fallback query on error
            fallback_query = self._create_fallback_query(user_id, intent_type)
            
            return {
                "query": fallback_query,
                "search_results": None,
                "generation_metadata": {
                    "fallback_used": True,
                    "error": str(e),
                    "intent": intent_type
                }
            }
    
    async def _generate_base_query(
        self,
        user_message: str,
        intent_type: str,
        entities: List[Dict[str, Any]],
        user_id: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate base Elasticsearch query using LLM."""
        
        # Build comprehensive prompt
        prompt = self._build_query_generation_prompt(
            user_message, intent_type, entities, user_id, context
        )
        
        # Call OpenAI with few-shot examples
        response = await self._call_openai(prompt)
        
        # Parse and validate JSON response
        try:
            query_json = json.loads(response["content"].strip())
            
            # Ensure required metadata
            query_json.setdefault("query_metadata", {})
            query_json["query_metadata"].update({
                "query_id": str(uuid.uuid4()),
                "user_id": user_id,
                "intent_type": intent_type,
                "original_query": user_message,
                "agent_name": self.config.name,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.85
            })
            
            return query_json
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM query response: {e}")
            # Fallback to template-based generation
            return self._generate_template_query(intent_type, entities, user_id)
    
    def _build_query_generation_prompt(
        self,
        user_message: str,
        intent_type: str,
        entities: List[Dict[str, Any]],
        user_id: int,
        context: Dict[str, Any]
    ) -> str:
        """Build comprehensive prompt for query generation."""
        
        prompt_parts = [
            f"Message utilisateur : {user_message}",
            f"Intention détectée : {intent_type}",
            f"User ID : {user_id}"
        ]
        
        # Add entities information
        if entities:
            entities_text = "Entités extraites :\n"
            for entity in entities:
                entities_text += f"- {entity.get('entity_type', 'UNKNOWN')}: {entity.get('normalized_value', entity.get('raw_value', ''))}\n"
            prompt_parts.append(entities_text)
        
        # Add context hints
        if context:
            if context.get("previous_searches"):
                prompt_parts.append("Recherches récentes : " + str(context["previous_searches"][:2]))
            
            if context.get("user_patterns"):
                patterns = context["user_patterns"]
                if patterns.get("frequent_categories"):
                    prompt_parts.append(f"Catégories fréquentes : {patterns['frequent_categories'][:3]}")
        
        # Add intent-specific guidance
        intent_guidance = {
            "BALANCE_INQUIRY": "Requête simple sur user_id pour récupérer les dernières transactions",
            "CATEGORY_ANALYSIS": "Filtrage par catégorie + agrégation des montants",
            "MERCHANT_ANALYSIS": "Recherche textuelle sur merchant_name + filtrage",
            "SPENDING_ANALYSIS": "Agrégation des montants avec groupement temporel",
            "TRANSACTION_SEARCH": "Recherche multi-critères avec filtres combinés",
            "TEMPORAL_ANALYSIS": "Filtrage temporel + agrégation par période"
        }
        
        guidance = intent_guidance.get(intent_type, "Génère une requête adaptée à l'intention")
        prompt_parts.append(f"Guidance : {guidance}")
        
        prompt_parts.extend([
            "",
            "Génère une requête Elasticsearch optimisée au format SearchServiceQuery.",
            "RÉPONDS UNIQUEMENT en JSON valide avec la structure exacte :",
            """{
  "query_metadata": {
    "intent_type": "INTENT_TYPE",
    "agent_name": "query_generator",
    "timestamp": "2024-01-31T10:30:00Z"
  },
  "search_parameters": {
    "query_type": "filtered_search|aggregated_search|text_search",
    "fields": ["user_id", "amount", "merchant_name", "category_name", "date"],
    "limit": 20,
    "timeout_ms": 5000
  },
  "filters": {
    "required": [
      {"field": "user_id", "operator": "eq", "value": USER_ID}
    ],
    "optional": [],
    "ranges": []
  }
}"""
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_template_query(
        self,
        intent_type: str,
        entities: List[Dict[str, Any]],
        user_id: int
    ) -> Dict[str, Any]:
        """Generate query using predefined templates as fallback."""
        
        # Base template
        template = {
            "query_metadata": {
                "query_id": str(uuid.uuid4()),
                "user_id": user_id,
                "intent_type": intent_type,
                "agent_name": self.config.name,
                "timestamp": datetime.now().isoformat(),
                "template_used": True
            },
            "search_parameters": {
                "query_type": "filtered_search",
                "fields": ["user_id", "amount", "merchant_name", "category_name", "date"],
                "limit": 20,
                "timeout_ms": 5000
            },
            "filters": {
                "required": [
                    {"field": "user_id", "operator": "eq", "value": user_id}
                ],
                "optional": [],
                "ranges": []
            }
        }
        
        # Apply intent-specific template modifications
        if intent_type == "CATEGORY_ANALYSIS":
            template["search_parameters"]["query_type"] = "aggregated_search"
            template["aggregations"] = {
                "enabled": True,
                "types": ["sum", "count"],
                "group_by": ["category_name"]
            }
        
        elif intent_type == "TEMPORAL_ANALYSIS":
            template["search_parameters"]["query_type"] = "aggregated_search"
            template["aggregations"] = {
                "enabled": True,
                "types": ["sum"],
                "group_by": ["month_year"]
            }
        
        elif intent_type in ["MERCHANT_ANALYSIS", "TRANSACTION_SEARCH"]:
            template["search_parameters"]["query_type"] = "text_search"
        
        # Add entity-based filters
        for entity in entities:
            entity_type = entity.get("entity_type")
            normalized_value = entity.get("normalized_value")
            
            if entity_type == "CATEGORY" and normalized_value:
                template["filters"]["required"].append({
                    "field": "category_name.keyword",
                    "operator": "eq", 
                    "value": normalized_value
                })
            elif entity_type == "AMOUNT" and normalized_value:
                try:
                    amount = float(normalized_value)
                    template["filters"]["ranges"].append({
                        "field": "amount_abs",
                        "operator": "gte",
                        "value": amount
                    })
                except ValueError:
                    pass
            elif entity_type == "DATE_RANGE" and normalized_value:
                # Add date range filter based on normalized temporal value
                date_filter = self._parse_temporal_entity(normalized_value)
                if date_filter:
                    template["filters"]["ranges"].append(date_filter)
        
        return template
    
    def _parse_temporal_entity(self, temporal_value: str) -> Optional[Dict[str, Any]]:
        """Parse temporal entity into date range filter."""
        
        today = datetime.now()
        
        temporal_mapping = {
            "today": {
                "field": "date",
                "operator": "eq",
                "value": today.strftime("%Y-%m-%d")
            },
            "this_week": {
                "field": "date", 
                "operator": "between",
                "value": [
                    (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d"),
                    today.strftime("%Y-%m-%d")
                ]
            },
            "this_month": {
                "field": "month_year",
                "operator": "eq",
                "value": today.strftime("%Y-%m")
            },
            "last_month": {
                "field": "month_year",
                "operator": "eq", 
                "value": (today.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
            }
        }
        
        return temporal_mapping.get(temporal_value)
    
    def _create_fallback_query(self, user_id: int, intent_type: str) -> Dict[str, Any]:
        """Create minimal fallback query for error cases."""
        
        return {
            "query_metadata": {
                "query_id": str(uuid.uuid4()),
                "user_id": user_id,
                "intent_type": intent_type,
                "agent_name": self.config.name,
                "timestamp": datetime.now().isoformat(),
                "fallback": True
            },
            "search_parameters": {
                "query_type": "simple_search",
                "fields": ["user_id", "amount", "date", "merchant_name"],
                "limit": 10,
                "timeout_ms": 3000
            },
            "filters": {
                "required": [
                    {"field": "user_id", "operator": "eq", "value": user_id}
                ]
            }
        }
    
    def _assess_query_complexity(self, query: Dict[str, Any]) -> str:
        """Assess complexity level of generated query."""
        
        complexity_score = 0
        
        # Count filters
        filters = query.get("filters", {})
        complexity_score += len(filters.get("required", []))
        complexity_score += len(filters.get("optional", []))
        complexity_score += len(filters.get("ranges", []))
        
        # Check for aggregations
        if query.get("aggregations", {}).get("enabled"):
            complexity_score += 2
        
        # Check for text search
        if "text_search" in query:
            complexity_score += 1
        
        if complexity_score <= 2:
            return "simple"
        elif complexity_score <= 5:
            return "medium"
        else:
            return "complex"
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get query generation statistics."""
        
        success_rate = self.successful_queries / max(self.total_queries, 1)
        optimization_rate = self.optimization_hits / max(self.total_queries, 1)
        validation_failure_rate = self.validation_failures / max(self.total_queries, 1)
        
        return {
            "agent_name": self.config.name,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "success_rate": success_rate,
            "optimization_hits": self.optimization_hits,
            "optimization_rate": optimization_rate,
            "validation_failures": self.validation_failures,
            "validation_failure_rate": validation_failure_rate,
            "performance_metrics": {
                "avg_success_rate": success_rate,
                "optimization_effectiveness": optimization_rate,
                "query_reliability": 1.0 - validation_failure_rate
            }
        }