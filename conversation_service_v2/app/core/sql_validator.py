"""Module 3: SQL Validation for security and performance."""

import sqlparse
import re
from typing import Dict, Any, List


class ValidationError(Exception):
    """Exception raised when SQL validation fails."""
    pass


class SQLValidator:
    """Validator for generated SQL queries."""

    # Dangerous keywords that should never appear
    FORBIDDEN_KEYWORDS = [
        'DELETE', 'DROP', 'TRUNCATE', 'UPDATE', 'INSERT',
        'ALTER', 'CREATE', 'GRANT', 'REVOKE', 'EXEC',
        'EXECUTE', 'CALL', 'DECLARE', 'CURSOR'
    ]

    # Required security patterns
    REQUIRED_PATTERNS = [
        r'user_id\s*=\s*:user_id',  # Must filter by user_id parameter
    ]

    def __init__(self):
        """Initialize the SQL validator."""
        pass

    def validate(self, sql_query: str, user_id: int) -> Dict[str, Any]:
        """
        Validate SQL query for security and syntax.

        Args:
            sql_query: SQL query to validate
            user_id: User ID for context

        Returns:
            dict: Validation result with details

        Raises:
            ValidationError: If validation fails
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Check for empty query
        if not sql_query or not sql_query.strip():
            raise ValidationError("SQL query is empty")

        # 2. Parse SQL syntax
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                raise ValidationError("Failed to parse SQL query")
        except Exception as e:
            raise ValidationError(f"SQL syntax error: {str(e)}")

        # 3. Convert to uppercase for keyword checking
        sql_upper = sql_query.upper()

        # 4. Check for forbidden keywords
        for keyword in self.FORBIDDEN_KEYWORDS:
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                errors.append(f"Forbidden keyword detected: {keyword}")

        # 5. Check for required security patterns
        for pattern in self.REQUIRED_PATTERNS:
            if not re.search(pattern, sql_query, re.IGNORECASE):
                errors.append(f"Missing required security pattern: {pattern}")

        # 6. Check that only SELECT is used
        statements = [stmt for stmt in parsed if stmt.get_type() != 'UNKNOWN']
        for stmt in statements:
            stmt_type = stmt.get_type()
            if stmt_type and stmt_type.upper() != 'SELECT':
                errors.append(f"Only SELECT queries are allowed, found: {stmt_type}")

        # 7. Check for potential SQL injection patterns
        dangerous_patterns = [
            r';\s*DROP',
            r';\s*DELETE',
            r'--\s*\n',  # SQL comments
            r'/\*.*\*/',  # Block comments (might hide malicious code)
            r'UNION\s+ALL\s+SELECT',  # Union-based injection
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, sql_upper):
                warnings.append(f"Potentially dangerous pattern detected: {pattern}")

        # 8. Raise error if validation failed
        if errors:
            raise ValidationError(f"SQL validation failed: {'; '.join(errors)}")

        return {
            "valid": True,
            "warnings": warnings,
            "parsed_statements": len(statements),
            "query_length": len(sql_query)
        }

    def sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize query parameters.

        Args:
            params: Query parameters to sanitize

        Returns:
            dict: Sanitized parameters
        """
        sanitized = {}

        for key, value in params.items():
            # Convert to appropriate type
            if isinstance(value, str):
                # Remove any potential SQL injection
                value = value.replace("'", "''")  # Escape single quotes
                value = value.replace(";", "")    # Remove semicolons
                value = value.replace("--", "")   # Remove SQL comments

            sanitized[key] = value

        return sanitized
