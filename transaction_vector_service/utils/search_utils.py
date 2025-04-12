"""
Search utility functions.

This module provides utilities for searching and indexing transactions,
including natural language query processing.
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, date, timedelta

from .date_utils import parse_date, get_date_range
from .currency_utils import parse_amount


def parse_natural_language_query(query: str) -> Dict[str, Any]:
    """
    Parse a natural language query into structured search parameters.
    
    Args:
        query: Natural language query string
        
    Returns:
        Dictionary of search parameters
    """
    search_params = {
        "text": [],
        "date_range": {},
        "amount_range": {},
        "merchants": [],
        "categories": []
    }
    
    # Extract date ranges
    date_patterns = [
        # "Last X days/weeks/months"
        (r'last\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)', _parse_relative_date),
        # "This week/month/year"
        (r'this\s+(week|month|year)', _parse_this_period),
        # "Last week/month/year"
        (r'last\s+(week|month|year)', _parse_last_period),
        # Date ranges: "from X to Y", "between X and Y"
        (r'(?:from|between)\s+(.+?)\s+(?:to|and)\s+(.+?)(?:\s|$)', _parse_explicit_range),
        # "Since X"
        (r'since\s+(.+?)(?:\s|$)', _parse_since_date),
        # "Before X"
        (r'before\s+(.+?)(?:\s|$)', _parse_before_date),
        # "On X"
        (r'on\s+(.+?)(?:\s|$)', _parse_on_date),
        # "Yesterday"
        (r'yesterday', lambda _: (date.today() - timedelta(days=1), date.today() - timedelta(days=1))),
        # "Today"
        (r'today', lambda _: (date.today(), date.today()))
    ]
    
    # Extract amount ranges
    amount_patterns = [
        # "More than X"
        (r'more than\s+([\d\.,]+)(?:\s*(?:€|EUR|USD|\$|£|GBP))?', _parse_min_amount),
        # "Less than X"
        (r'less than\s+([\d\.,]+)(?:\s*(?:€|EUR|USD|\$|£|GBP))?', _parse_max_amount),
        # "Between X and Y"
        (r'between\s+([\d\.,]+)\s+and\s+([\d\.,]+)(?:\s*(?:€|EUR|USD|\$|£|GBP))?', _parse_amount_range)
    ]
    
    # Extract merchants
    merchant_patterns = [
        # "at X", "from X"
        (r'(?:at|from)\s+([A-Za-z0-9\s]+?)(?:\s|$)', lambda match: match.group(1).strip()),
        # "to X"
        (r'to\s+([A-Za-z0-9\s]+?)(?:\s|$)', lambda match: match.group(1).strip())
    ]
    
    # Extract categories (would need a list of known categories in a real implementation)
    category_patterns = [
        # "in category X"
        (r'in\s+(?:category|categorie)\s+([A-Za-z\s]+?)(?:\s|$)', lambda match: match.group(1).strip()),
        # "for X" where X is a category
        (r'for\s+([A-Za-z\s]+?)(?:\s|$)', lambda match: match.group(1).strip())
    ]
    
    # Process date patterns
    for pattern, handler in date_patterns:
        matches = list(re.finditer(pattern, query, re.IGNORECASE))
        for match in matches:
            result = handler(match)
            if result:
                start_date, end_date = result
                if start_date:
                    search_params["date_range"]["start_date"] = start_date
                if end_date:
                    search_params["date_range"]["end_date"] = end_date
                # Remove matched text from query
                query = query.replace(match.group(0), " ")
    
    # Process amount patterns
    for pattern, handler in amount_patterns:
        matches = list(re.finditer(pattern, query, re.IGNORECASE))
        for match in matches:
            result = handler(match)
            if result:
                if "min_amount" in result:
                    search_params["amount_range"]["min_amount"] = result["min_amount"]
                if "max_amount" in result:
                    search_params["amount_range"]["max_amount"] = result["max_amount"]
                # Remove matched text from query
                query = query.replace(match.group(0), " ")
    
    # Process merchant patterns
    for pattern, handler in merchant_patterns:
        matches = list(re.finditer(pattern, query, re.IGNORECASE))
        for match in matches:
            merchant = handler(match)
            if merchant and merchant.lower() not in [m.lower() for m in search_params["merchants"]]:
                search_params["merchants"].append(merchant)
                # Remove matched text from query
                query = query.replace(match.group(0), " ")
    
    # Process category patterns
    for pattern, handler in category_patterns:
        matches = list(re.finditer(pattern, query, re.IGNORECASE))
        for match in matches:
            category = handler(match)
            if category and category.lower() not in [c.lower() for c in search_params["categories"]]:
                search_params["categories"].append(category)
                # Remove matched text from query
                query = query.replace(match.group(0), " ")
    
    # The remaining text is for full-text search
    query = re.sub(r'\s+', ' ', query).strip()
    if query:
        search_params["text"] = query
    
    return search_params


def _parse_relative_date(match) -> Tuple[Optional[date], Optional[date]]:
    """Parse relative date expressions like 'last 30 days'."""
    try:
        num = int(match.group(1))
        unit = match.group(2).lower()
        
        if unit in ["day", "days"]:
            return (date.today() - timedelta(days=num), date.today())
        elif unit in ["week", "weeks"]:
            return (date.today() - timedelta(days=num * 7), date.today())
        elif unit in ["month", "months"]:
            # Approximate months as 30 days
            return (date.today() - timedelta(days=num * 30), date.today())
        elif unit in ["year", "years"]:
            # Approximate years as 365 days
            return (date.today() - timedelta(days=num * 365), date.today())
    except (ValueError, IndexError):
        pass
    
    return (None, None)


def _parse_this_period(match) -> Tuple[Optional[date], Optional[date]]:
    """Parse period expressions like 'this week'."""
    try:
        unit = match.group(1).lower()
        today = date.today()
        
        if unit == "week":
            # Start of week (Monday)
            start = today - timedelta(days=today.weekday())
            return (start, today)
        elif unit == "month":
            # Start of month
            start = date(today.year, today.month, 1)
            return (start, today)
        elif unit == "year":
            # Start of year
            start = date(today.year, 1, 1)
            return (start, today)
    except (ValueError, IndexError):
        pass
    
    return (None, None)


def _parse_last_period(match) -> Tuple[Optional[date], Optional[date]]:
    """Parse period expressions like 'last week'."""
    try:
        unit = match.group(1).lower()
        today = date.today()
        
        if unit == "week":
            # Last week (Monday to Sunday)
            end = today - timedelta(days=today.weekday() + 1)
            start = end - timedelta(days=6)
            return (start, end)
        elif unit == "month":
            # Last month
            if today.month == 1:
                # January - go back to December of previous year
                start = date(today.year - 1, 12, 1)
                end = date(today.year - 1, 12, 31)
            else:
                # Previous month of same year
                start = date(today.year, today.month - 1, 1)
                # Last day of previous month
                if today.month == 3 and today.year % 4 == 0:
                    # February in leap year
                    end = date(today.year, 2, 29)
                else:
                    # Last day of month calculation
                    last_day = {
                        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
                    }
                    end = date(today.year, today.month - 1, last_day[today.month - 1])
            return (start, end)
        elif unit == "year":
            # Last year
            start = date(today.year - 1, 1, 1)
            end = date(today.year - 1, 12, 31)
            return (start, end)
    except (ValueError, IndexError):
        pass
    
    return (None, None)


def _parse_explicit_range(match) -> Tuple[Optional[date], Optional[date]]:
    """Parse explicit date ranges like 'from 2023-01-01 to 2023-01-31'."""
    try:
        start_str = match.group(1).strip()
        end_str = match.group(2).strip()
        
        start_date = parse_date(start_str)
        end_date = parse_date(end_str)
        
        return (start_date, end_date)
    except (ValueError, IndexError):
        pass
    
    return (None, None)


def _parse_since_date(match) -> Tuple[Optional[date], Optional[date]]:
    """Parse 'since' expressions like 'since 2023-01-01'."""
    try:
        date_str = match.group(1).strip()
        parsed_date = parse_date(date_str)
        
        return (parsed_date, date.today())
    except (ValueError, IndexError):
        pass
    
    return (None, None)


def _parse_before_date(match) -> Tuple[Optional[date], Optional[date]]:
    """Parse 'before' expressions like 'before 2023-01-01'."""
    try:
        date_str = match.group(1).strip()
        parsed_date = parse_date(date_str)
        
        return (None, parsed_date)
    except (ValueError, IndexError):
        pass
    
    return (None, None)


def _parse_on_date(match) -> Tuple[Optional[date], Optional[date]]:
    """Parse 'on' expressions like 'on 2023-01-01'."""
    try:
        date_str = match.group(1).strip()
        parsed_date = parse_date(date_str)
        
        return (parsed_date, parsed_date)
    except (ValueError, IndexError):
        pass
    
    return (None, None)


def _parse_min_amount(match) -> Dict[str, float]:
    """Parse 'more than' expressions like 'more than 100'."""
    try:
        amount_str = match.group(1).strip()
        amount = parse_amount(amount_str)
        
        if amount is not None:
            return {"min_amount": amount}
    except (ValueError, IndexError):
        pass
    
    return {}


def _parse_max_amount(match) -> Dict[str, float]:
    """Parse 'less than' expressions like 'less than 100'."""
    try:
        amount_str = match.group(1).strip()
        amount = parse_amount(amount_str)
        
        if amount is not None:
            return {"max_amount": amount}
    except (ValueError, IndexError):
        pass
    
    return {}


def _parse_amount_range(match) -> Dict[str, float]:
    """Parse amount ranges like 'between 100 and 200'."""
    try:
        min_str = match.group(1).strip()
        max_str = match.group(2).strip()
        
        min_amount = parse_amount(min_str)
        max_amount = parse_amount(max_str)
        
        result = {}
        if min_amount is not None:
            result["min_amount"] = min_amount
        if max_amount is not None:
            result["max_amount"] = max_amount
        
        return result
    except (ValueError, IndexError):
        pass
    
    return {}


def convert_nl_params_to_search_query(nl_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert natural language parameters to a structured search query.
    
    Args:
        nl_params: Natural language parameters
        
    Returns:
        Structured search query
    """
    search_query = {}
    
    # Add text search
    if "text" in nl_params and nl_params["text"]:
        search_query["query"] = nl_params["text"]
    
    # Add date range
    if "date_range" in nl_params and nl_params["date_range"]:
        if "start_date" in nl_params["date_range"]:
            search_query["start_date"] = nl_params["date_range"]["start_date"]
        if "end_date" in nl_params["date_range"]:
            search_query["end_date"] = nl_params["date_range"]["end_date"]
    
    # Add amount range
    if "amount_range" in nl_params and nl_params["amount_range"]:
        if "min_amount" in nl_params["amount_range"]:
            search_query["min_amount"] = nl_params["amount_range"]["min_amount"]
        if "max_amount" in nl_params["amount_range"]:
            search_query["max_amount"] = nl_params["amount_range"]["max_amount"]
    
    # Add merchants
    if "merchants" in nl_params and nl_params["merchants"]:
        search_query["merchant_names"] = nl_params["merchants"]
    
    # Add categories (would need to map category names to IDs in a real implementation)
    if "categories" in nl_params and nl_params["categories"]:
        # This is a placeholder - real implementation would look up category IDs
        search_query["categories"] = []
    
    # Set default parameters
    search_query["limit"] = 50
    search_query["offset"] = 0
    search_query["sort_by"] = "transaction_date"
    search_query["sort_order"] = "desc"
    
    return search_query


def generate_search_explanation(search_params: Dict[str, Any]) -> str:
    """
    Generate a human-readable explanation of a search query.
    
    Args:
        search_params: Search parameters
        
    Returns:
        Explanation string
    """
    parts = ["Searching for transactions"]
    
    # Explain date range
    if "start_date" in search_params and "end_date" in search_params:
        parts.append(f"between {search_params['start_date']} and {search_params['end_date']}")
    elif "start_date" in search_params:
        parts.append(f"from {search_params['start_date']} to now")
    elif "end_date" in search_params:
        parts.append(f"before {search_params['end_date']}")
    
    # Explain amount range
    if "min_amount" in search_params and "max_amount" in search_params:
        parts.append(f"with amount between {search_params['min_amount']} and {search_params['max_amount']}")
    elif "min_amount" in search_params:
        parts.append(f"with amount greater than {search_params['min_amount']}")
    elif "max_amount" in search_params:
        parts.append(f"with amount less than {search_params['max_amount']}")
    
    # Explain merchant filter
    if "merchant_names" in search_params and search_params["merchant_names"]:
        if len(search_params["merchant_names"]) == 1:
            parts.append(f"from merchant '{search_params['merchant_names'][0]}'")
        else:
            merchants = ", ".join(f"'{m}'" for m in search_params["merchant_names"])
            parts.append(f"from merchants {merchants}")
    
    # Explain category filter
    if "categories" in search_params and search_params["categories"]:
        if len(search_params["categories"]) == 1:
            parts.append(f"in category {search_params['categories'][0]}")
        else:
            categories = ", ".join(str(c) for c in search_params["categories"])
            parts.append(f"in categories {categories}")
    
    # Explain text search
    if "query" in search_params and search_params["query"]:
        parts.append(f"matching text '{search_params['query']}'")
    
    return " ".join(parts)


def extract_search_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract entities like dates, amounts, merchants from search text.
    
    Args:
        text: Search text
        
    Returns:
        Dictionary of extracted entities
    """
    entities = {
        "dates": [],
        "amounts": [],
        "merchants": [],
        "categories": []
    }
    
    # Extract dates
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # ISO format
        r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
        r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
        r'\d{2}/\d{2}/\d{2}',  # DD/MM/YY
        r'(?:yesterday|today|tomorrow)',  # Relative dates
        r'(?:last|this|next)\s+(?:week|month|year)'  # Relative periods
    ]
    
    for pattern in date_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities["dates"].append(match.group(0))
    
    # Extract amounts
    amount_patterns = [
        r'\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:€|EUR|USD|\$|£|GBP))?',  # 1,234.56 €
        r'\d+(?:\.\d{3})*(?:,\d{2})?(?:\s*(?:€|EUR|USD|\$|£|GBP))?',  # 1.234,56 €
        r'(?:€|EUR|USD|\$|£|GBP)\s*\d+(?:[,.]\d+)?'  # € 1234.56
    ]
    
    for pattern in amount_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities["amounts"].append(match.group(0))
    
    # Extract merchants (simplified - would need a merchant database in real implementation)
    merchant_indicators = [
        r'(?:at|from|to)\s+([A-Za-z0-9\s]+?)(?:\s|$)'
    ]
    
    for pattern in merchant_indicators:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) > 0:
                merchant = match.group(1).strip()
                if merchant and merchant not in entities["merchants"]:
                    entities["merchants"].append(merchant)
    
    # Extract categories (simplified - would need a category database in real implementation)
    category_indicators = [
        r'(?:in|for)\s+(?:category\s+)?([A-Za-z\s]+?)(?:\s|$)'
    ]
    
    for pattern in category_indicators:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) > 0:
                category = match.group(1).strip()
                if category and category not in entities["categories"]:
                    entities["categories"].append(category)
    
    return entities


def highlight_search_matches(
    transaction_text: str,
    search_terms: Union[str, List[str]],
    max_length: int = 100,
    highlight_prefix: str = "<em>",
    highlight_suffix: str = "</em>"
) -> str:
    """
    Highlight search terms in transaction text and truncate to a reasonable length.
    
    Args:
        transaction_text: Text to highlight in
        search_terms: Terms to highlight
        max_length: Maximum length of result
        highlight_prefix: Prefix for highlighted terms
        highlight_suffix: Suffix for highlighted terms
        
    Returns:
        Highlighted text snippet
    """
    if not transaction_text:
        return ""
    
    # Convert single search term to list
    if isinstance(search_terms, str):
        search_terms = [search_terms]
    
    # Normalize text for case-insensitive matching
    text_lower = transaction_text.lower()
    
    # Find all matches
    matches = []
    for term in search_terms:
        term_lower = term.lower()
        start = 0
        while start < len(text_lower):
            pos = text_lower.find(term_lower, start)
            if pos == -1:
                break
            matches.append((pos, pos + len(term)))
            start = pos + 1
    
    # Sort matches by position
    matches.sort()
    
    # Merge overlapping matches
    if matches:
        merged = [matches[0]]
        for current in matches[1:]:
            prev = merged[-1]
            if current[0] <= prev[1]:
                # Overlapping matches, merge them
                merged[-1] = (prev[0], max(prev[1], current[1]))
            else:
                # Non-overlapping match, add it
                merged.append(current)
        
        matches = merged
    
    # If no matches, return a truncated version of the text
    if not matches:
        if len(transaction_text) <= max_length:
            return transaction_text
        return transaction_text[:max_length - 3] + "..."
    
    # Create highlight markers at match positions
    markers = []
    for start, end in matches:
        markers.append((start, "START"))
        markers.append((end, "END"))
    
    # Sort markers by position
    markers.sort()
    
    # Apply highlighting
    result = []
    last_pos = 0
    highlighting = False
    
    for pos, marker_type in markers:
        if marker_type == "START":
            result.append(transaction_text[last_pos:pos])
            result.append(highlight_prefix)
            highlighting = True
        else:  # END
            result.append(transaction_text[last_pos:pos])
            result.append(highlight_suffix)
            highlighting = False
        
        last_pos = pos
    
    # Add remaining text
    result.append(transaction_text[last_pos:])
    
    highlighted_text = "".join(result)
    
    # Truncate if too long, making sure to not break highlight tags
    if len(highlighted_text) <= max_length:
        return highlighted_text
    
    # Find a good position to truncate
    # Try to include at least the first match
    first_match_end = matches[0][1] + len(highlight_suffix)
    
    if first_match_end + 3 <= max_length:
        # We can fit the first match and an ellipsis
        return highlighted_text[:max_length - 3] + "..."
    
    # If we can't fit the first match, just truncate at max_length
    # This could break highlight tags, so check for that
    trunc = highlighted_text[:max_length - 3] + "..."
    
    # Ensure we don't have mismatched tags
    tag_open_count = trunc.count(highlight_prefix)
    tag_close_count = trunc.count(highlight_suffix)
    
    if tag_open_count > tag_close_count:
        # We have an unclosed tag, add the closing tag
        trunc += highlight_suffix
    
    return trunc


def create_search_index(transactions: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Create a simple inverted index for text search.
    
    Args:
        transactions: List of transactions to index
        
    Returns:
        Inverted index mapping terms to transaction IDs
    """
    index = {}
    
    for tx in transactions:
        tx_id = tx.get("id")
        if not tx_id:
            continue
        
        # Get text to index
        texts = []
        if "description" in tx:
            texts.append(tx["description"])
        if "clean_description" in tx:
            texts.append(tx["clean_description"])
        if "normalized_merchant" in tx:
            texts.append(tx["normalized_merchant"])
        
        # Normalize and tokenize text
        tokens = set()
        for text in texts:
            if text:
                # Convert to lowercase
                text = text.lower()
                # Split into words
                words = re.findall(r'\b\w+\b', text)
                tokens.update(words)
        
        # Add to index
        for token in tokens:
            if token not in index:
                index[token] = set()
            index[token].add(tx_id)
    
    return index


def search_index(
    index: Dict[str, Set[str]],
    query: str
) -> List[str]:
    """
    Search the index for matching transaction IDs.
    
    Args:
        index: Inverted index
        query: Search query
        
    Returns:
        List of matching transaction IDs
    """
    if not query:
        return []
    
    # Normalize and tokenize query
    query_lower = query.lower()
    query_tokens = set(re.findall(r'\b\w+\b', query_lower))
    
    # If no tokens, return empty list
    if not query_tokens:
        return []
    
    # Find matches
    match_sets = []
    for token in query_tokens:
        if token in index:
            match_sets.append(index[token])
    
    # Handle no matches
    if not match_sets:
        return []
    
    # Start with the first set
    matches = match_sets[0]
    
    # Intersect with other sets for AND semantics
    for match_set in match_sets[1:]:
        matches = matches.intersection(match_set)
    
    return list(matches)