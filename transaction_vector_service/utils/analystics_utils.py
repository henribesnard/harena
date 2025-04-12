"""
Analytics utility functions.

This module provides utilities for statistical analysis and data mining
of transaction data.
"""

import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

from .date_utils import get_date_range, get_next_occurrence
from .text_processors import clean_transaction_description


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of values
        
    Returns:
        Dictionary of statistics
    """
    if not values:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "variance": 0.0,
            "std_dev": 0.0,
            "range": 0.0
        }
    
    arr = np.array(values)
    return {
        "count": len(values),
        "sum": float(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "variance": float(np.var(arr)),
        "std_dev": float(np.std(arr)),
        "range": float(np.max(arr) - np.min(arr))
    }


def detect_outliers(values: List[float], method: str = 'zscore', threshold: float = 3.0) -> List[int]:
    """
    Detect outliers in a list of values.
    
    Args:
        values: List of values
        method: Detection method ('zscore', 'iqr', 'percentile')
        threshold: Threshold for outlier detection
        
    Returns:
        List of indices of outliers
    """
    if not values or len(values) < 3:
        return []
    
    arr = np.array(values)
    outlier_indices = []
    
    if method == 'zscore':
        # Z-score method: identify values beyond threshold standard deviations
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std == 0:
            return []
        
        z_scores = np.abs((arr - mean) / std)
        outlier_indices = np.where(z_scores > threshold)[0].tolist()
    
    elif method == 'iqr':
        # IQR method: identify values outside threshold * IQR range
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return []
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outlier_indices = np.where((arr < lower_bound) | (arr > upper_bound))[0].tolist()
    
    elif method == 'percentile':
        # Percentile method: identify values outside specified percentiles
        low_percentile = threshold
        high_percentile = 100 - threshold
        
        low_bound = np.percentile(arr, low_percentile)
        high_bound = np.percentile(arr, high_percentile)
        
        outlier_indices = np.where((arr < low_bound) | (arr > high_bound))[0].tolist()
    
    return outlier_indices


def group_transactions_by(
    transactions: List[Dict[str, Any]],
    group_by: str,
    date_format: str = '%Y-%m-%d'
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group transactions by a specified field.
    
    Args:
        transactions: List of transaction dictionaries
        group_by: Field to group by ('day', 'week', 'month', 'year', 'category', 'merchant')
        date_format: Format for date keys
        
    Returns:
        Dictionary of grouped transactions
    """
    grouped = defaultdict(list)
    
    for transaction in transactions:
        # Skip transactions without required fields
        if 'transaction_date' not in transaction:
            continue
        
        # Parse date if it's a string
        tx_date = transaction['transaction_date']
        if isinstance(tx_date, str):
            try:
                tx_date = datetime.strptime(tx_date, date_format).date()
            except ValueError:
                continue
        
        # Group by specified field
        if group_by == 'day':
            key = tx_date.strftime(date_format)
        elif group_by == 'week':
            # Get the Monday of the week
            monday = tx_date - timedelta(days=tx_date.weekday())
            key = monday.strftime(date_format)
        elif group_by == 'month':
            key = tx_date.strftime('%Y-%m')
        elif group_by == 'year':
            key = str(tx_date.year)
        elif group_by == 'category':
            key = str(transaction.get('category_id', 'unknown'))
        elif group_by == 'merchant':
            key = transaction.get('normalized_merchant', 'unknown')
        else:
            key = str(transaction.get(group_by, 'unknown'))
        
        grouped[key].append(transaction)
    
    return dict(grouped)


def calculate_frequency_distribution(values: List[Any]) -> Dict[Any, int]:
    """
    Calculate the frequency distribution of values.
    
    Args:
        values: List of values
        
    Returns:
        Dictionary mapping values to frequencies
    """
    return dict(Counter(values))


def analyze_spending_by_category(
    transactions: List[Dict[str, Any]],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze spending by category over a period.
    
    Args:
        transactions: List of transaction dictionaries
        start_date: Start date for analysis
        end_date: End date for analysis
        
    Returns:
        Dictionary of category statistics
    """
    # Filter transactions by date
    filtered = transactions
    if start_date or end_date:
        filtered = []
        for tx in transactions:
            tx_date = tx.get('transaction_date')
            if isinstance(tx_date, str):
                try:
                    tx_date = datetime.strptime(tx_date, '%Y-%m-%d').date()
                except ValueError:
                    continue
            
            if start_date and tx_date < start_date:
                continue
            if end_date and tx_date > end_date:
                continue
            
            filtered.append(tx)
    
    # Group by category
    by_category = defaultdict(list)
    for tx in filtered:
        category_id = tx.get('category_id')
        if category_id is not None:
            by_category[str(category_id)].append(tx)
    
    # Calculate statistics for each category
    result = {}
    for category_id, category_txs in by_category.items():
        # Get only negative amounts (expenses)
        amounts = [tx.get('amount', 0) for tx in category_txs if tx.get('amount', 0) < 0]
        
        if not amounts:
            continue
        
        total_amount = sum(amounts)
        avg_amount = total_amount / len(amounts)
        
        result[category_id] = {
            'transaction_count': len(category_txs),
            'expense_count': len(amounts),
            'total_amount': total_amount,
            'average_amount': avg_amount,
            'min_amount': min(amounts),
            'max_amount': max(amounts),
            'first_date': min(tx.get('transaction_date') for tx in category_txs),
            'last_date': max(tx.get('transaction_date') for tx in category_txs)
        }
    
    return result


def detect_recurring_patterns(
    transactions: List[Dict[str, Any]],
    similarity_threshold: float = 0.8,
    date_variance_days: int = 3,
    min_occurrences: int = 3
) -> List[Dict[str, Any]]:
    """
    Detect recurring transaction patterns.
    
    Args:
        transactions: List of transaction dictionaries
        similarity_threshold: Threshold for text similarity
        date_variance_days: Maximum variance in days between occurrences
        min_occurrences: Minimum number of occurrences
        
    Returns:
        List of detected patterns
    """
    # Group transactions by merchant
    by_merchant = defaultdict(list)
    for tx in transactions:
        merchant = tx.get('normalized_merchant')
        if merchant:
            by_merchant[merchant].append(tx)
    
    # Detect patterns within each merchant group
    patterns = []
    
    for merchant, merchant_txs in by_merchant.items():
        # Skip merchants with too few transactions
        if len(merchant_txs) < min_occurrences:
            continue
        
        # Group by approximate amount
        by_amount = defaultdict(list)
        for tx in merchant_txs:
            amount = tx.get('amount', 0)
            
            # Find a matching amount group
            matched = False
            for key_amount in list(by_amount.keys()):
                variance_pct = abs(amount - key_amount) / max(abs(key_amount), 0.01)
                if variance_pct <= 0.05:  # 5% tolerance
                    by_amount[key_amount].append(tx)
                    matched = True
                    break
            
            if not matched:
                by_amount[amount].append(tx)
        
        # For each amount group, check for recurring patterns
        for amount, amount_txs in by_amount.items():
            if len(amount_txs) < min_occurrences:
                continue
            
            # Sort by date
            sorted_txs = sorted(amount_txs, key=lambda x: x.get('transaction_date'))
            
            # Calculate intervals between consecutive dates
            dates = []
            for tx in sorted_txs:
                tx_date = tx.get('transaction_date')
                if isinstance(tx_date, str):
                    try:
                        tx_date = datetime.strptime(tx_date, '%Y-%m-%d').date()
                        dates.append(tx_date)
                    except ValueError:
                        continue
                elif isinstance(tx_date, date):
                    dates.append(tx_date)
            
            if len(dates) < min_occurrences:
                continue
            
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            
            # Skip if no intervals
            if not intervals:
                continue
            
            # Calculate average interval
            avg_interval = sum(intervals) / len(intervals)
            
            # Calculate variance in intervals
            variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
            std_dev = variance ** 0.5
            
            # Skip if variance is too high
            if std_dev > date_variance_days:
                continue
            
            # Determine frequency based on average interval
            frequency = "unknown"
            if 25 <= avg_interval <= 35:
                frequency = "monthly"
            elif 6 <= avg_interval <= 8:
                frequency = "weekly"
            elif 13 <= avg_interval <= 15:
                frequency = "biweekly"
            elif 85 <= avg_interval <= 95:
                frequency = "quarterly"
            elif 350 <= avg_interval <= 380:
                frequency = "annual"
            
            # Calculate next expected date
            next_date = get_next_occurrence(dates[-1], frequency)
            
            # Add the pattern
            patterns.append({
                'merchant': merchant,
                'amount': amount,
                'average_interval_days': avg_interval,
                'interval_std_dev': std_dev,
                'frequency': frequency,
                'transaction_count': len(amount_txs),
                'first_date': dates[0],
                'last_date': dates[-1],
                'next_expected_date': next_date,
                'confidence': max(0, min(1, 1 - (std_dev / date_variance_days)))
            })
    
    # Sort patterns by confidence
    patterns.sort(key=lambda x: x['confidence'], reverse=True)
    
    return patterns


def calculate_cash_flow_projection(
    transactions: List[Dict[str, Any]],
    num_days: int = 30,
    include_recurring: bool = True
) -> Dict[str, Any]:
    """
    Create a cash flow projection for future days.
    
    Args:
        transactions: List of transaction dictionaries
        num_days: Number of days to project
        include_recurring: Whether to include recurring transactions
        
    Returns:
        Projection data
    """
    # Get current date
    today = date.today()
    end_date = today + timedelta(days=num_days)
    
    # Split transactions into income and expenses
    incomes = [tx for tx in transactions if tx.get('amount', 0) > 0]
    expenses = [tx for tx in transactions if tx.get('amount', 0) < 0]
    
    # Detect recurring patterns if enabled
    recurring_patterns = []
    if include_recurring:
        recurring_patterns = detect_recurring_patterns(transactions)
    
    # Initialize daily projection
    projection = {
        "daily": {},
        "cumulative": {},
        "starting_balance": 0,  # This would be current balance in real implementation
        "ending_balance": 0,
        "total_projected_income": 0,
        "total_projected_expenses": 0,
        "recurring_transactions": []
    }
    
    # Calculate daily projection
    for day_offset in range(num_days + 1):
        current_date = today + timedelta(days=day_offset)
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Initialize daily data
        projection["daily"][date_str] = {
            "date": date_str,
            "income": 0,
            "expenses": 0,
            "net": 0,
            "transactions": []
        }
        
        # Add recurring transactions for this day
        for pattern in recurring_patterns:
            next_date = pattern.get("next_expected_date")
            if isinstance(next_date, str):
                try:
                    next_date = datetime.strptime(next_date, "%Y-%m-%d").date()
                except ValueError:
                    continue
            
            if next_date == current_date:
                amount = pattern.get("amount", 0)
                
                # Add to daily totals
                if amount > 0:
                    projection["daily"][date_str]["income"] += amount
                    projection["total_projected_income"] += amount
                else:
                    projection["daily"][date_str]["expenses"] += amount
                    projection["total_projected_expenses"] += amount
                
                # Add transaction detail
                projection["daily"][date_str]["transactions"].append({
                    "description": f"[Projected] {pattern.get('merchant', 'Recurring payment')}",
                    "amount": amount,
                    "confidence": pattern.get("confidence", 0),
                    "recurring": True
                })
                
                # Add to recurring transactions list
                projection["recurring_transactions"].append({
                    "date": date_str,
                    "merchant": pattern.get("merchant"),
                    "amount": amount,
                    "confidence": pattern.get("confidence", 0)
                })
        
        # Calculate daily net
        projection["daily"][date_str]["net"] = (
            projection["daily"][date_str]["income"] + 
            projection["daily"][date_str]["expenses"]
        )
    
    # Calculate cumulative values
    running_balance = projection["starting_balance"]
    for day_offset in range(num_days + 1):
        current_date = today + timedelta(days=day_offset)
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Update running balance
        running_balance += projection["daily"][date_str]["net"]
        
        # Store cumulative balance
        projection["cumulative"][date_str] = running_balance
    
    # Update ending balance
    projection["ending_balance"] = running_balance
    
    return projection


def find_unusual_transactions(
    transactions: List[Dict[str, Any]],
    method: str = 'zscore',
    threshold: float = 2.5
) -> List[Dict[str, Any]]:
    """
    Find unusual or anomalous transactions.
    
    Args:
        transactions: List of transaction dictionaries
        method: Detection method ('zscore', 'iqr', 'percentile')
        threshold: Threshold for anomaly detection
        
    Returns:
        List of unusual transactions with reason
    """
    # Group transactions by merchant
    by_merchant = defaultdict(list)
    for tx in transactions:
        merchant = tx.get('normalized_merchant', 'unknown')
        by_merchant[merchant].append(tx)
    
    unusual_transactions = []
    
    # Analyze each merchant group
    for merchant, merchant_txs in by_merchant.items():
        # Skip merchants with too few transactions
        if len(merchant_txs) < 3:
            continue
        
        # Extract amounts
        amounts = [abs(tx.get('amount', 0)) for tx in merchant_txs]
        
        # Detect amount outliers
        outlier_indices = detect_outliers(amounts, method, threshold)
        
        # Add outliers to unusual transactions
        for idx in outlier_indices:
            tx = merchant_txs[idx]
            avg_amount = sum(amounts) / len(amounts)
            
            unusual_transactions.append({
                **tx,
                'unusual_reason': 'amount',
                'typical_amount': avg_amount,
                'deviation_percent': (abs(tx.get('amount', 0)) - avg_amount) / avg_amount * 100
            })
    
    # Look for unusual transaction dates (e.g., weekends for normally weekday merchants)
    # This is a simplified implementation
    
    # Sort by deviation
    unusual_transactions.sort(key=lambda x: abs(x['deviation_percent']), reverse=True)
    
    return unusual_transactions