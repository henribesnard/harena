"""
Date utility functions.

This module provides utilities for date parsing, formatting,
and date-related operations.
"""

import re
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple, Union
import calendar


def parse_date(date_str: str) -> Optional[date]:
    """
    Parse a date string into a date object, supporting various formats.
    
    Args:
        date_str: String representation of a date
        
    Returns:
        Date object or None if parsing failed
    """
    formats = [
        "%Y-%m-%d",      # ISO format (2023-01-15)
        "%d/%m/%Y",      # French format (15/01/2023)
        "%d-%m-%Y",      # Alternative format (15-01-2023)
        "%d/%m/%y",      # Short year French format (15/01/23)
        "%d %b %Y",      # Day month name year (15 Jan 2023)
        "%d %B %Y",      # Day full month name year (15 January 2023)
        "%Y%m%d"         # Compact format (20230115)
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    # Try to extract date using regex for more complex strings
    # Look for patterns like "DD/MM/YYYY" or "YYYY-MM-DD" in text
    iso_pattern = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
    if iso_pattern:
        year, month, day = map(int, iso_pattern.groups())
        try:
            return date(year, month, day)
        except ValueError:
            pass
    
    french_pattern = re.search(r'(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{2,4})', date_str)
    if french_pattern:
        day, month, year = map(int, french_pattern.groups())
        if year < 100:
            year += 2000 if year < 50 else 1900
        try:
            return date(year, month, day)
        except ValueError:
            pass
    
    return None


def format_date(
    d: Union[date, datetime],
    format_str: str = "%d/%m/%Y",
    locale: str = "fr_FR"
) -> str:
    """
    Format a date according to the specified format and locale.
    
    Args:
        d: Date or datetime to format
        format_str: Format string
        locale: Locale for date formatting
        
    Returns:
        Formatted date string
    """
    if isinstance(d, datetime):
        d = d.date()
    
    try:
        # Use locale-aware formatting
        import locale as loc
        current_locale = loc.getlocale(loc.LC_TIME)
        loc.setlocale(loc.LC_TIME, locale)
        formatted = d.strftime(format_str)
        loc.setlocale(loc.LC_TIME, current_locale)
        return formatted
    except (ImportError, loc.Error):
        # Fallback to standard formatting
        return d.strftime(format_str)


def get_date_range(
    period: str,
    reference_date: Optional[date] = None
) -> Tuple[date, date]:
    """
    Get start and end dates for a specified period.
    
    Args:
        period: Period name (today, yesterday, this_week, last_week, this_month, etc.)
        reference_date: Reference date, defaults to today
        
    Returns:
        Tuple of (start_date, end_date)
    """
    if reference_date is None:
        reference_date = date.today()
    
    if period == "today":
        return reference_date, reference_date
    
    elif period == "yesterday":
        yesterday = reference_date - timedelta(days=1)
        return yesterday, yesterday
    
    elif period == "this_week":
        # Start from Monday of current week
        start = reference_date - timedelta(days=reference_date.weekday())
        # End on Sunday
        end = start + timedelta(days=6)
        return start, end
    
    elif period == "last_week":
        # Start from Monday of previous week
        start = reference_date - timedelta(days=reference_date.weekday() + 7)
        # End on Sunday of previous week
        end = start + timedelta(days=6)
        return start, end
    
    elif period == "this_month":
        # Start from first day of current month
        start = date(reference_date.year, reference_date.month, 1)
        # End on last day of current month
        last_day = calendar.monthrange(reference_date.year, reference_date.month)[1]
        end = date(reference_date.year, reference_date.month, last_day)
        return start, end
    
    elif period == "last_month":
        # Get first day of current month
        first_day_current = date(reference_date.year, reference_date.month, 1)
        # Get last day of previous month
        last_day_previous = first_day_current - timedelta(days=1)
        # Get first day of previous month
        first_day_previous = date(last_day_previous.year, last_day_previous.month, 1)
        return first_day_previous, last_day_previous
    
    elif period == "this_quarter":
        # Determine current quarter
        quarter = (reference_date.month - 1) // 3 + 1
        # Start from first day of the quarter
        start_month = (quarter - 1) * 3 + 1
        start = date(reference_date.year, start_month, 1)
        # End on last day of the quarter
        if quarter < 4:
            end_month = quarter * 3
            end_year = reference_date.year
        else:
            end_month = 12
            end_year = reference_date.year
        last_day = calendar.monthrange(end_year, end_month)[1]
        end = date(end_year, end_month, last_day)
        return start, end
    
    elif period == "last_quarter":
        # Determine current quarter
        quarter = (reference_date.month - 1) // 3 + 1
        # Determine previous quarter
        if quarter > 1:
            prev_quarter = quarter - 1
            year = reference_date.year
        else:
            prev_quarter = 4
            year = reference_date.year - 1
        
        # Start from first day of the previous quarter
        start_month = (prev_quarter - 1) * 3 + 1
        start = date(year, start_month, 1)
        
        # End on last day of the previous quarter
        end_month = prev_quarter * 3
        last_day = calendar.monthrange(year, end_month)[1]
        end = date(year, end_month, last_day)
        return start, end
    
    elif period == "this_year":
        return date(reference_date.year, 1, 1), date(reference_date.year, 12, 31)
    
    elif period == "last_year":
        return date(reference_date.year - 1, 1, 1), date(reference_date.year - 1, 12, 31)
    
    elif period == "last_7_days":
        return reference_date - timedelta(days=6), reference_date
    
    elif period == "last_30_days":
        return reference_date - timedelta(days=29), reference_date
    
    elif period == "last_90_days":
        return reference_date - timedelta(days=89), reference_date
    
    elif period == "last_365_days":
        return reference_date - timedelta(days=364), reference_date
    
    # Default to today
    return reference_date, reference_date


def get_month_name(month: int, short: bool = False, locale: str = "fr_FR") -> str:
    """
    Get the name of a month.
    
    Args:
        month: Month number (1-12)
        short: Whether to return short name
        locale: Locale for month name
        
    Returns:
        Month name
    """
    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")
    
    # Create a date object for the month
    d = date(2023, month, 1)
    
    try:
        # Use locale-aware formatting
        import locale as loc
        current_locale = loc.getlocale(loc.LC_TIME)
        loc.setlocale(loc.LC_TIME, locale)
        if short:
            name = d.strftime("%b")
        else:
            name = d.strftime("%B")
        loc.setlocale(loc.LC_TIME, current_locale)
        return name
    except (ImportError, loc.Error):
        # Fallback to English month names
        english_months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        english_short = [m[:3] for m in english_months]
        
        if short:
            return english_short[month - 1]
        return english_months[month - 1]


def is_date_between(d: date, start: date, end: date) -> bool:
    """
    Check if a date is between two other dates (inclusive).
    
    Args:
        d: Date to check
        start: Start date
        end: End date
        
    Returns:
        True if date is between start and end (inclusive)
    """
    return start <= d <= end


def calculate_date_diff(d1: date, d2: date) -> Dict[str, Any]:
    """
    Calculate the difference between two dates in various units.
    
    Args:
        d1: First date
        d2: Second date
        
    Returns:
        Dictionary with differences in days, weeks, months, years
    """
    # Ensure d1 is earlier than d2
    if d1 > d2:
        d1, d2 = d2, d1
    
    delta = d2 - d1
    days = delta.days
    
    # Calculate approximate differences
    years = days // 365
    months = days // 30
    weeks = days // 7
    
    # More accurate month calculation
    month_diff = (d2.year - d1.year) * 12 + d2.month - d1.month
    if d2.day < d1.day:
        month_diff -= 1
    
    return {
        "days": days,
        "weeks": weeks,
        "months": month_diff,
        "years": years,
        "total_days": days
    }


def get_next_occurrence(
    base_date: date,
    frequency: str,
    interval: int = 1,
    day_of_month: Optional[int] = None,
    day_of_week: Optional[int] = None
) -> date:
    """
    Calculate the next occurrence of a recurring date.
    
    Args:
        base_date: Base date to calculate from
        frequency: Frequency (daily, weekly, monthly, yearly)
        interval: Number of periods between occurrences
        day_of_month: Specific day of month for monthly recurrence
        day_of_week: Specific day of week for weekly recurrence (0=Monday, 6=Sunday)
        
    Returns:
        Date of next occurrence
    """
    if frequency == "daily":
        return base_date + timedelta(days=interval)
    
    elif frequency == "weekly":
        next_date = base_date + timedelta(days=7 * interval)
        
        # Adjust to specific day of week if needed
        if day_of_week is not None:
            days_to_add = (day_of_week - next_date.weekday()) % 7
            next_date += timedelta(days=days_to_add)
        
        return next_date
    
    elif frequency == "biweekly":
        return base_date + timedelta(days=14 * interval)
    
    elif frequency == "monthly":
        # Calculate next month
        year = base_date.year
        month = base_date.month + interval
        
        while month > 12:
            month -= 12
            year += 1
        
        # Determine day of month
        if day_of_month is not None:
            day = min(day_of_month, calendar.monthrange(year, month)[1])
        else:
            day = min(base_date.day, calendar.monthrange(year, month)[1])
        
        return date(year, month, day)
    
    elif frequency == "quarterly":
        # Calculate 3 months ahead
        return get_next_occurrence(base_date, "monthly", interval * 3, day_of_month)
    
    elif frequency == "semiannual":
        # Calculate 6 months ahead
        return get_next_occurrence(base_date, "monthly", interval * 6, day_of_month)
    
    elif frequency == "yearly":
        # Calculate next year
        year = base_date.year + interval
        month = base_date.month
        day = min(base_date.day, calendar.monthrange(year, month)[1])
        
        return date(year, month, day)
    
    # Default fallback
    return base_date