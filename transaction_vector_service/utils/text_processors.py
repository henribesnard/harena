"""
Text processing utilities for transaction descriptions.

This module provides functions for cleaning, normalizing, and extracting
information from transaction descriptions.
"""

import re
import unicodedata
from typing import Optional, List, Dict, Any, Tuple


def clean_transaction_description(description: str) -> str:
    """
    Clean and normalize a transaction description.
    
    Args:
        description: Raw transaction description
        
    Returns:
        Cleaned transaction description
    """
    if not description:
        return ""
    
    # Convert to lowercase
    text = description.lower()
    
    # Remove accents and special characters
    text = remove_accents(text)
    
    # Remove common transaction prefixes
    prefixes = [
        "paiement par carte",
        "achat par carte",
        "carte bancaire",
        "cb ",
        "achat du ",
        "achat le ",
        "facture ",
        "prlv ",
        "virement ",
        "vir ",
        "prelevement ",
    ]
    
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Remove date patterns
    text = re.sub(r'\d{1,2}/\d{1,2}(/\d{2,4})?', '', text)
    
    # Remove reference numbers
    text = re.sub(r'ref:?\s+[a-z0-9]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'id:?\s+[a-z0-9]+', '', text, flags=re.IGNORECASE)
    
    # Remove transaction IDs and order numbers
    text = re.sub(r'transaction\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'commande\s+n[o°]?\s*\d+', '', text, flags=re.IGNORECASE)
    
    # Remove amounts and currency codes
    text = re.sub(r'\d+[.,]\d+\s*(?:eur|gbp|usd)?', '', text, flags=re.IGNORECASE)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_accents(text: str) -> str:
    """
    Remove accents and normalize Unicode characters.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    return ''.join(c for c in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(c))


def extract_merchant_name(description: str) -> str:
    """
    Extract merchant name from transaction description.
    
    Args:
        description: Cleaned transaction description
        
    Returns:
        Extracted merchant name
    """
    # Very simple implementation - first word or first two words
    words = description.split()
    
    if len(words) >= 2:
        # Return first two words as merchant name
        return ' '.join(words[:2])
    elif len(words) == 1:
        return words[0]
    else:
        return ""


def extract_location(description: str) -> Optional[Dict[str, str]]:
    """
    Extract location information from transaction description.
    
    Args:
        description: Cleaned transaction description
        
    Returns:
        Dictionary with location information or None
    """
    # Look for common location patterns
    
    # Country codes (e.g., "FR", "UK", "USA")
    country_match = re.search(r'\b([A-Z]{2,3})\b', description.upper())
    
    # City names followed by country code (e.g., "PARIS FR")
    city_country_match = re.search(r'\b([A-Z][A-Za-z]+)\s+([A-Z]{2})\b', description.upper())
    
    if city_country_match:
        return {
            "city": city_country_match.group(1).title(),
            "country_code": city_country_match.group(2).upper()
        }
    elif country_match:
        return {
            "country_code": country_match.group(1).upper()
        }
    
    return None


def is_likely_recurring(descriptions: List[str]) -> bool:
    """
    Check if a list of transaction descriptions likely represent recurring payments.
    
    Args:
        descriptions: List of transaction descriptions
        
    Returns:
        True if likely recurring, False otherwise
    """
    if len(descriptions) < 3:
        return False
    
    # Clean descriptions
    cleaned = [clean_transaction_description(d) for d in descriptions]
    
    # Check for high similarity between descriptions
    base = cleaned[0]
    similarity_scores = [compare_strings(base, d) for d in cleaned[1:]]
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    
    # If average similarity is high, likely recurring
    return avg_similarity > 0.8


def compare_strings(str1: str, str2: str) -> float:
    """
    Compare two strings and return a similarity score (0-1).
    Simple implementation using character overlap.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Similarity score (0-1)
    """
    if not str1 or not str2:
        return 0.0
    
    # Convert strings to sets of characters
    set1 = set(str1)
    set2 = set(str2)
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def extract_payment_method(description: str) -> Optional[str]:
    """
    Extract payment method from transaction description.
    
    Args:
        description: Transaction description
        
    Returns:
        Payment method or None
    """
    description = description.lower()
    
    if re.search(r'\bcarte\b|\bcb\b|card', description):
        return "card"
    elif re.search(r'\bvirement\b|\bvir\b|transfer', description):
        return "transfer"
    elif re.search(r'\bprelevement\b|\bprlv\b|direct debit', description):
        return "direct_debit"
    elif re.search(r'\bcheque\b|\bchq\b|check', description):
        return "check"
    elif re.search(r'\bespece\b|\bcash\b', description):
        return "cash"
    
    return None