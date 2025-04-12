"""
String utility functions.

This module provides utilities for string processing, formatting,
and text manipulation.
"""

import re
import hashlib
import unicodedata
from typing import List, Optional, Dict, Set, Any, Tuple


def slugify(text: str) -> str:
    """
    Convert text to a URL-friendly slug.
    
    Args:
        text: Text to slugify
        
    Returns:
        Slugified text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove accents
    text = ''.join(c for c in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(c))
    
    # Replace non-alphanumeric characters with hyphens
    text = re.sub(r'[^a-z0-9]+', '-', text)
    
    # Remove leading/trailing hyphens
    text = text.strip('-')
    
    return text


def hash_text(text: str, algorithm: str = 'md5') -> str:
    """
    Generate a hash of the given text.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hexadecimal hash string
    """
    if algorithm == 'md5':
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(text.encode('utf-8')).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def truncate(text: str, max_length: int, suffix: str = '...') -> str:
    """
    Truncate text to a maximum length, adding a suffix if truncated.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_name(first_name: str, last_name: str, format_type: str = 'full') -> str:
    """
    Format a name according to the specified format.
    
    Args:
        first_name: First name
        last_name: Last name
        format_type: Format type (full, initial, last_first)
        
    Returns:
        Formatted name
    """
    if not first_name and not last_name:
        return ""
    
    if format_type == 'full':
        return f"{first_name} {last_name}".strip()
    
    elif format_type == 'initial':
        if first_name:
            return f"{first_name[0]}. {last_name}".strip()
        return last_name
    
    elif format_type == 'last_first':
        return f"{last_name}, {first_name}".strip()
    
    return f"{first_name} {last_name}".strip()


def normalize_text(text: str) -> str:
    """
    Normalize text by removing accents, trimming whitespace, and converting to lowercase.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove accents
    text = ''.join(c for c in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(c))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_numbers(text: str) -> List[str]:
    """
    Extract all numbers from a text string.
    
    Args:
        text: Text to extract numbers from
        
    Returns:
        List of extracted numbers
    """
    return re.findall(r'\d+(?:\.\d+)?', text)


def extract_emails(text: str) -> List[str]:
    """
    Extract all email addresses from a text string.
    
    Args:
        text: Text to extract emails from
        
    Returns:
        List of extracted emails
    """
    return re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)


def extract_urls(text: str) -> List[str]:
    """
    Extract all URLs from a text string.
    
    Args:
        text: Text to extract URLs from
        
    Returns:
        List of extracted URLs
    """
    return re.findall(r'https?://[^\s]+', text)


def calculate_similarity(str1: str, str2: str, method: str = 'jaccard') -> float:
    """
    Calculate the similarity between two strings.
    
    Args:
        str1: First string
        str2: Second string
        method: Similarity method (jaccard, levenshtein)
        
    Returns:
        Similarity score (0-1)
    """
    if not str1 or not str2:
        return 0.0
    
    if method == 'jaccard':
        # Jaccard similarity: size of intersection / size of union
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    elif method == 'levenshtein':
        # Levenshtein distance-based similarity
        distance = levenshtein_distance(str1.lower(), str2.lower())
        max_len = max(len(str1), len(str2))
        
        return 1 - (distance / max_len) if max_len > 0 else 1.0
    
    else:
        raise ValueError(f"Unsupported similarity method: {method}")


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def find_longest_common_substring(str1: str, str2: str) -> str:
    """
    Find the longest common substring between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Longest common substring
    """
    if not str1 or not str2:
        return ""
    
    # Convert to lowercase for case-insensitive comparison
    str1 = str1.lower()
    str2 = str2.lower()
    
    # Create a table to store lengths of longest common suffixes
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Variables to keep track of maximum length and ending position
    max_length = 0
    end_position = 0
    
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_position = i
    
    # Extract the longest common substring
    return str1[end_position - max_length:end_position]


def extract_keywords(text: str, min_length: int = 3, max_count: int = 10) -> List[str]:
    """
    Extract potential keywords from text.
    
    Args:
        text: Text to extract keywords from
        min_length: Minimum keyword length
        max_count: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    # Normalize text
    text = normalize_text(text)
    
    # Remove common stop words (simplified version)
    stop_words = {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'a', 'à',
        'au', 'aux', 'en', 'par', 'pour', 'sur', 'avec', 'ce', 'ces', 'cette',
        'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre',
        'nos', 'votre', 'vos', 'leur', 'leurs', 'the', 'a', 'an', 'and', 'or',
        'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'in', 'on',
        'at', 'to', 'for', 'with', 'by', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up',
        'down', 'of', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
    }
    
    # Split into words
    words = re.findall(r'\b\w+\b', text)
    
    # Filter words
    filtered_words = [
        word for word in words
        if word not in stop_words and len(word) >= min_length
    ]
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Return top words
    return [word for word, count in sorted_words[:max_count]]


def format_phone_number(phone: str, country: str = 'FR') -> str:
    """
    Format a phone number according to country conventions.
    
    Args:
        phone: Phone number to format
        country: Country code
        
    Returns:
        Formatted phone number
    """
    # Remove non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    if country == 'FR':
        # Format French phone numbers (0X XX XX XX XX)
        if len(digits) == 10:
            if digits.startswith('0'):
                return f"{digits[0:2]} {digits[2:4]} {digits[4:6]} {digits[6:8]} {digits[8:10]}"
            else:
                return f"0{digits[0:1]} {digits[1:3]} {digits[3:5]} {digits[5:7]} {digits[7:9]}"
        # Format international French numbers (+33 X XX XX XX XX)
        elif len(digits) == 11 and digits.startswith('33'):
            return f"+33 {digits[2:3]} {digits[3:5]} {digits[5:7]} {digits[7:9]} {digits[9:11]}"
    
    # For other countries or unrecognized formats, return as is with some basic formatting
    if len(digits) > 10:
        return f"+{digits[:2]} {digits[2:]}"
    else:
        return digits