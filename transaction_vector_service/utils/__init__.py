"""
Utility functions and helpers for the Transaction Vector Service.

This package provides various utility functions for string processing,
date handling, currency operations, and other common tasks.
"""

from .text_processors import (
    clean_transaction_description, 
    remove_accents,
    extract_merchant_name,
    extract_location,
    is_likely_recurring,
    extract_payment_method
)

from .currency_utils import (
    format_amount,
    parse_amount,
    convert_currency,
    sum_amounts,
    is_positive_flow,
    get_currency_precision
)

from .date_utils import (
    parse_date,
    format_date,
    get_date_range,
    get_month_name,
    is_date_between,
    calculate_date_diff,
    get_next_occurrence
)

from .string_utils import (
    slugify,
    hash_text,
    truncate,
    format_name,
    normalize_text,
    extract_numbers,
    extract_emails,
    extract_urls,
    calculate_similarity,
    find_longest_common_substring,
    extract_keywords,
    format_phone_number
)

__all__ = [
    # Text processors
    "clean_transaction_description", 
    "remove_accents",
    "extract_merchant_name",
    "extract_location",
    "is_likely_recurring",
    "extract_payment_method",
    
    # Currency utils
    "format_amount",
    "parse_amount",
    "convert_currency",
    "sum_amounts",
    "is_positive_flow",
    "get_currency_precision",
    
    # Date utils
    "parse_date",
    "format_date",
    "get_date_range",
    "get_month_name",
    "is_date_between",
    "calculate_date_diff",
    "get_next_occurrence",
    
    # String utils
    "slugify",
    "hash_text",
    "truncate",
    "format_name",
    "normalize_text",
    "extract_numbers",
    "extract_emails",
    "extract_urls",
    "calculate_similarity",
    "find_longest_common_substring",
    "extract_keywords",
    "format_phone_number"
]