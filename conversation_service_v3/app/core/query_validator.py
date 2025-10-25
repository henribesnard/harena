"""
Query Validator - Validates Elasticsearch queries to prevent injection attacks

This validator ensures that LLM-generated Elasticsearch queries only contain
whitelisted fields and safe operations, preventing potential security issues
from malicious or compromised LLM outputs.

Author: Claude Code
Date: 2025-10-25
"""

import logging
from typing import Dict, Any, List, Set

logger = logging.getLogger(__name__)


class ElasticsearchQueryValidator:
    """
    Validates Elasticsearch queries against security rules

    Prevents injection attacks by:
    - Whitelisting allowed fields
    - Validating aggregation types
    - Checking for suspicious operations
    """

    # Allowed fields for filters and aggregations
    ALLOWED_FIELDS: Set[str] = {
        "transaction_type",
        "category_name",
        "category_name.keyword",
        "merchant_name",
        "merchant_name.keyword",
        "amount_abs",
        "amount",
        "date",
        "transaction_id",
        "user_id",
        "description",
        "description.keyword",
        "account_id",
        "account_name",
        "account_name.keyword",
        "bank_name",
        "balance"
    }

    # Allowed aggregation types
    ALLOWED_AGGREGATION_TYPES: Set[str] = {
        "sum",
        "avg",
        "min",
        "max",
        "count",
        "value_count",
        "cardinality",
        "terms",
        "date_histogram",
        "histogram",
        "stats",
        "extended_stats"
    }

    # Allowed query types (for filters)
    ALLOWED_QUERY_TYPES: Set[str] = {
        "term",
        "terms",
        "match",
        "match_phrase",
        "range",
        "bool",
        "exists"
    }

    @classmethod
    def validate_query(cls, query: Dict[str, Any], raise_on_error: bool = True) -> tuple[bool, List[str]]:
        """
        Validate an Elasticsearch query

        Args:
            query: The Elasticsearch query dict to validate
            raise_on_error: If True, raise ValueError on validation errors

        Returns:
            Tuple of (is_valid, list_of_errors)

        Raises:
            ValueError: If raise_on_error=True and validation fails
        """
        errors = []

        # Validate filters
        if "filters" in query:
            filter_errors = cls._validate_filters(query["filters"])
            errors.extend(filter_errors)

        # Validate aggregations
        if "aggregations" in query:
            agg_errors = cls._validate_aggregations(query["aggregations"])
            errors.extend(agg_errors)

        # Validate time_range if present
        if "time_range" in query:
            range_errors = cls._validate_time_range(query["time_range"])
            errors.extend(range_errors)

        is_valid = len(errors) == 0

        if not is_valid and raise_on_error:
            error_msg = f"Query validation failed: {'; '.join(errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not is_valid:
            logger.warning(f"Query validation warnings: {errors}")

        return is_valid, errors

    @classmethod
    def _validate_filters(cls, filters: Dict[str, Any]) -> List[str]:
        """Validate filter fields"""
        errors = []

        for field_name, field_value in filters.items():
            # Check if field is allowed
            if field_name not in cls.ALLOWED_FIELDS:
                errors.append(f"Field '{field_name}' is not in the whitelist")

            # Check for suspicious patterns
            if isinstance(field_value, str):
                if any(char in field_value for char in [';', '--', '/*', '*/']):
                    errors.append(f"Suspicious characters in filter value for '{field_name}'")

        return errors

    @classmethod
    def _validate_aggregations(cls, aggregations: Dict[str, Any]) -> List[str]:
        """Validate aggregation structure"""
        errors = []

        for agg_name, agg_config in aggregations.items():
            if isinstance(agg_config, dict):
                # Check aggregation type
                for agg_type, agg_body in agg_config.items():
                    if agg_type == "aggs":
                        # Nested aggregations - recursive check
                        nested_errors = cls._validate_aggregations(agg_body)
                        errors.extend(nested_errors)
                    elif agg_type not in cls.ALLOWED_AGGREGATION_TYPES:
                        errors.append(f"Aggregation type '{agg_type}' is not allowed")

                    # Validate field if present
                    if isinstance(agg_body, dict) and "field" in agg_body:
                        field = agg_body["field"]
                        if field not in cls.ALLOWED_FIELDS:
                            errors.append(f"Aggregation field '{field}' is not in the whitelist")

        return errors

    @classmethod
    def _validate_time_range(cls, time_range: Dict[str, Any]) -> List[str]:
        """Validate time range structure"""
        errors = []

        # Check that time_range has expected keys
        expected_keys = {"start", "end"}
        actual_keys = set(time_range.keys())

        if not expected_keys.issubset(actual_keys):
            errors.append(f"Time range missing required keys. Expected {expected_keys}, got {actual_keys}")

        return errors

    @classmethod
    def sanitize_query(cls, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a query by removing invalid fields

        This is a fallback for when strict validation is too restrictive.
        It removes any fields/aggregations that fail validation.

        Args:
            query: Query to sanitize

        Returns:
            Sanitized query with only valid fields
        """
        sanitized = query.copy()

        # Sanitize filters
        if "filters" in sanitized:
            sanitized["filters"] = {
                k: v for k, v in sanitized["filters"].items()
                if k in cls.ALLOWED_FIELDS
            }

        # Sanitize aggregations (simplified - just check top level)
        if "aggregations" in sanitized:
            sanitized["aggregations"] = {
                k: v for k, v in sanitized["aggregations"].items()
                if cls._is_valid_aggregation(k, v)
            }

        logger.info(f"Query sanitized. Original keys: {set(query.keys())}, Sanitized keys: {set(sanitized.keys())}")
        return sanitized

    @classmethod
    def _is_valid_aggregation(cls, agg_name: str, agg_config: Any) -> bool:
        """Check if a single aggregation is valid"""
        if not isinstance(agg_config, dict):
            return False

        for agg_type in agg_config.keys():
            if agg_type not in cls.ALLOWED_AGGREGATION_TYPES and agg_type != "aggs":
                return False

        return True


# Global validator instance
query_validator = ElasticsearchQueryValidator()
