"""
Validation utility functions.

This module provides utilities for validating different types of data,
such as emails, phone numbers, IBAN, etc.
"""

import re
from typing import Optional, Union, Dict, Any, List, Tuple
from datetime import date


def is_valid_email(email: str) -> bool:
    """
    Validate an email address.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def is_valid_phone(phone: str, country: str = "FR") -> bool:
    """
    Validate a phone number for a specific country.
    
    Args:
        phone: Phone number to validate
        country: Country code
        
    Returns:
        True if valid, False otherwise
    """
    # Remove non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    if country == "FR":
        # French phone number: 10 digits starting with 0, or 11 digits starting with 33
        return (len(digits) == 10 and digits.startswith('0')) or \
               (len(digits) == 11 and digits.startswith('33'))
    
    elif country == "US":
        # US phone number: 10 digits, or 11 digits starting with 1
        return len(digits) == 10 or (len(digits) == 11 and digits.startswith('1'))
    
    # Default validation: at least 8 digits
    return len(digits) >= 8


def is_valid_iban(iban: str) -> bool:
    """
    Validate an International Bank Account Number (IBAN).
    
    Args:
        iban: IBAN to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Remove spaces and convert to uppercase
    iban = re.sub(r'\s+', '', iban.upper())
    
    # Check format: country code (2 letters) + check digits (2 digits) + BBAN
    if not re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$', iban):
        return False
    
    # Move first 4 characters to the end
    rearranged = iban[4:] + iban[:4]
    
    # Convert letters to numbers (A=10, B=11, ..., Z=35)
    numeric = ''
    for char in rearranged:
        if char.isalpha():
            numeric += str(ord(char) - 55)  # A is ASCII 65, so A -> 10 requires -55
        else:
            numeric += char
    
    # Check if the number mod 97 equals 1
    return int(numeric) % 97 == 1


def is_valid_credit_card(card_number: str) -> bool:
    """
    Validate a credit card number using the Luhn algorithm.
    
    Args:
        card_number: Credit card number to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Remove non-digit characters
    digits = re.sub(r'\D', '', card_number)
    
    if not digits or len(digits) < 13 or len(digits) > 19:
        return False
    
    # Luhn algorithm
    total = 0
    odd_even = len(digits) % 2
    
    for i, digit in enumerate(digits):
        if ((i + odd_even) % 2) == 0:
            # Double every second digit
            doubled = int(digit) * 2
            # If doubled value is greater than 9, subtract 9
            total += doubled if doubled < 10 else doubled - 9
        else:
            total += int(digit)
    
    # If the total mod 10 is 0, the number is valid
    return total % 10 == 0


def is_valid_postal_code(postal_code: str, country: str = "FR") -> bool:
    """
    Validate a postal code for a specific country.
    
    Args:
        postal_code: Postal code to validate
        country: Country code
        
    Returns:
        True if valid, False otherwise
    """
    # Remove spaces
    postal_code = re.sub(r'\s+', '', postal_code)
    
    if country == "FR":
        # French postal code: 5 digits
        return bool(re.match(r'^\d{5}$', postal_code))
    
    elif country == "US":
        # US postal code: 5 digits or 5+4 digits
        return bool(re.match(r'^\d{5}(-\d{4})?$', postal_code))
    
    elif country == "UK":
        # UK postal code: complex pattern
        return bool(re.match(r'^[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}$', postal_code, re.IGNORECASE))
    
    # Default validation: at least 3 characters
    return len(postal_code) >= 3


def is_valid_amount(amount: Union[str, float, int], min_value: float = None, max_value: float = None) -> bool:
    """
    Validate a monetary amount.
    
    Args:
        amount: Amount to validate
        min_value: Minimum valid value
        max_value: Maximum valid value
        
    Returns:
        True if valid, False otherwise
    """
    # Convert string to float if needed
    if isinstance(amount, str):
        try:
            # Replace comma with dot for French number format
            amount = float(amount.replace(',', '.'))
        except ValueError:
            return False
    
    # Check min value
    if min_value is not None and amount < min_value:
        return False
    
    # Check max value
    if max_value is not None and amount > max_value:
        return False
    
    return True


def is_valid_bic(bic: str) -> bool:
    """
    Validate a Bank Identifier Code (BIC).
    
    Args:
        bic: BIC to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Remove spaces and convert to uppercase
    bic = re.sub(r'\s+', '', bic.upper())
    
    # BIC format: 8 or 11 characters
    # AAAABBCC or AAAABBCCDDD
    # AAAA - bank code
    # BB - country code
    # CC - location code
    # DDD - branch code (optional)
    return bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', bic))


def validate_transaction_data(transaction: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate transaction data structure.
    
    Args:
        transaction: Transaction data to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    required_fields = ["account_id", "amount", "description", "transaction_date"]
    for field in required_fields:
        if field not in transaction:
            errors.append(f"Missing required field: {field}")
    
    # Validate amount
    if "amount" in transaction:
        try:
            float(transaction["amount"])
        except (ValueError, TypeError):
            errors.append("Invalid amount: must be a number")
    
    # Validate dates if present
    date_fields = ["transaction_date", "value_date", "booking_date"]
    for field in date_fields:
        if field in transaction and transaction[field]:
            if not isinstance(transaction[field], (str, date)):
                errors.append(f"Invalid {field}: must be a date string or date object")
            elif isinstance(transaction[field], str):
                # Check date format (YYYY-MM-DD)
                if not re.match(r'^\d{4}-\d{2}-\d{2}$', transaction[field]):
                    errors.append(f"Invalid {field} format: must be YYYY-MM-DD")
    
    return len(errors) == 0, errors


def sanitize_input(text: str, allowed_tags: List[str] = None) -> str:
    """
    Sanitize input text to prevent XSS attacks.
    
    Args:
        text: Text to sanitize
        allowed_tags: List of allowed HTML tags
        
    Returns:
        Sanitized text
    """
    # By default, remove all HTML tags
    if allowed_tags is None:
        return re.sub(r'<[^>]*>', '', text)
    
    # If specific tags are allowed, remove only disallowed tags
    if not allowed_tags:
        return re.sub(r'<[^>]*>', '', text)
    
    # Create pattern to match disallowed tags
    allowed_pattern = '|'.join(allowed_tags)
    pattern = re.compile(r'<(?!/?(?:' + allowed_pattern + r'))[^>]*>', re.IGNORECASE)
    
    return pattern.sub('', text)