"""
Currency utility functions.

This module provides utilities for currency formatting, conversion,
and other currency-related operations.
"""

from typing import Optional, Dict, Any, Union
from datetime import date, datetime
import locale
import re

from ..config.constants import DEFAULT_CURRENCY, CURRENCY_SYMBOLS


def format_amount(amount: float, currency: str = DEFAULT_CURRENCY, include_symbol: bool = True) -> str:
    """
    Format a monetary amount according to locale and currency.
    
    Args:
        amount: Amount to format
        currency: Currency code (default: EUR)
        include_symbol: Whether to include the currency symbol
        
    Returns:
        Formatted amount string
    """
    # Set locale based on currency
    locale_map = {
        "EUR": "fr_FR",
        "USD": "en_US",
        "GBP": "en_GB"
    }
    
    selected_locale = locale_map.get(currency, "fr_FR")
    
    try:
        locale.setlocale(locale.LC_ALL, f"{selected_locale}.UTF-8")
    except locale.Error:
        # Fallback to basic formatting if locale not available
        symbol = CURRENCY_SYMBOLS.get(currency, "€") if include_symbol else ""
        formatted = f"{abs(amount):,.2f}".replace(",", " ").replace(".", ",")
        return f"{formatted} {symbol}".strip()
    
    # Format using locale
    if include_symbol:
        return locale.currency(abs(amount), symbol=True, grouping=True)
    else:
        return locale.format_string("%.2f", abs(amount), grouping=True)


def parse_amount(amount_str: str) -> Optional[float]:
    """
    Parse an amount string into a float.
    
    Args:
        amount_str: String representing an amount
        
    Returns:
        Parsed amount or None if parsing failed
    """
    # Remove currency symbols
    for symbol in CURRENCY_SYMBOLS.values():
        amount_str = amount_str.replace(symbol, "")
    
    # Clean up the string
    amount_str = amount_str.strip()
    
    # Handle different number formats
    try:
        # Try to parse with comma as decimal separator (French format)
        if "," in amount_str and "." not in amount_str:
            amount_str = amount_str.replace(" ", "").replace(",", ".")
        # Try to parse with dot as decimal separator (English format)
        elif "." in amount_str:
            amount_str = amount_str.replace(" ", "")
        
        # Remove any remaining non-numeric characters except minus sign and decimal point
        amount_str = re.sub(r'[^\d\-\.]', '', amount_str)
        
        return float(amount_str)
    except ValueError:
        return None


def convert_currency(
    amount: float, 
    from_currency: str, 
    to_currency: str,
    exchange_rate: Optional[float] = None,
    rate_date: Optional[Union[date, datetime]] = None
) -> float:
    """
    Convert an amount between currencies.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code
        to_currency: Target currency code
        exchange_rate: Optional explicit exchange rate
        rate_date: Optional date for historical exchange rate
        
    Returns:
        Converted amount
    """
    # If currencies are the same, no conversion needed
    if from_currency == to_currency:
        return amount
    
    # If exchange rate is provided, use it
    if exchange_rate is not None:
        return amount * exchange_rate
    
    # Otherwise, use hardcoded exchange rates for demonstration
    # In a real implementation, this would fetch rates from an API
    rates = {
        "EUR": {
            "USD": 1.09,
            "GBP": 0.86
        },
        "USD": {
            "EUR": 0.92,
            "GBP": 0.79
        },
        "GBP": {
            "EUR": 1.16,
            "USD": 1.27
        }
    }
    
    # Lookup the exchange rate
    if from_currency in rates and to_currency in rates[from_currency]:
        return amount * rates[from_currency][to_currency]
    
    # If conversion not possible, return original amount
    return amount


def sum_amounts(amounts: Dict[str, float], target_currency: str = DEFAULT_CURRENCY) -> float:
    """
    Sum amounts in different currencies, converting to a target currency.
    
    Args:
        amounts: Dictionary mapping currency codes to amounts
        target_currency: Currency to convert all amounts to
        
    Returns:
        Total amount in target currency
    """
    total = 0.0
    
    for currency, amount in amounts.items():
        converted = convert_currency(amount, currency, target_currency)
        total += converted
    
    return total


def is_positive_flow(operation_type: str) -> bool:
    """
    Determine if an operation type represents a positive cash flow (income).
    
    Args:
        operation_type: Type of operation
        
    Returns:
        True if positive flow, False otherwise
    """
    positive_types = {"income", "refund", "transfer_in", "deposit"}
    return operation_type.lower() in positive_types


def get_currency_precision(currency: str) -> int:
    """
    Get the number of decimal places for a currency.
    
    Args:
        currency: Currency code
        
    Returns:
        Number of decimal places
    """
    # Most currencies use 2 decimal places
    currency_precisions = {
        "JPY": 0,
        "KRW": 0,
        "HUF": 0,
        "BHD": 3,
        "OMR": 3,
        "JOD": 3
    }
    
    return currency_precisions.get(currency, 2)