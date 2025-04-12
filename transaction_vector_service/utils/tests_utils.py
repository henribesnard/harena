"""
Testing utility functions.

This module provides utilities for testing the Transaction Vector Service,
including mock data generation and test helpers.
"""

import random
import uuid
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union


def generate_mock_transaction(
    user_id: int = 1,
    account_id: int = 1,
    date_offset: int = 0,
    amount: Optional[float] = None,
    merchant: Optional[str] = None,
    category_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a mock transaction with random or specified values.
    
    Args:
        user_id: User ID for the transaction
        account_id: Account ID for the transaction
        date_offset: Days from today (negative for past, positive for future)
        amount: Transaction amount (random if None)
        merchant: Merchant name (random if None)
        category_id: Category ID (random if None)
        
    Returns:
        Mock transaction dictionary
    """
    # Generate random transaction date
    tx_date = date.today() + timedelta(days=date_offset)
    
    # Generate random amount if not specified
    if amount is None:
        # 80% chance of negative amount (expense)
        if random.random() < 0.8:
            amount = round(random.uniform(-500, -1), 2)
        else:
            amount = round(random.uniform(10, 2000), 2)
    
    # Generate random merchant if not specified
    if merchant is None:
        merchants = [
            "Supermarket", "Restaurant", "Gas Station", "Online Store",
            "Coffee Shop", "Pharmacy", "Clothing Store", "Electronics",
            "Streaming Service", "Utility Company", "Internet Provider",
            "Mobile Carrier", "Insurance", "Gym", "Public Transport"
        ]
        merchant = random.choice(merchants)
    
    # Generate random category if not specified
    if category_id is None:
        category_id = random.randint(1, 20)
    
    # Generate operation type based on amount
    if amount < 0:
        operation_types = ["card", "direct_debit", "transfer"]
        weights = [0.7, 0.2, 0.1]
    else:
        operation_types = ["transfer", "deposit", "refund"]
        weights = [0.6, 0.3, 0.1]
    
    operation_type = random.choices(operation_types, weights=weights)[0]
    
    # Generate transaction
    return {
        "id": str(uuid.uuid4()),
        "bridge_transaction_id": random.randint(10000000, 99999999),
        "user_id": user_id,
        "account_id": account_id,
        "amount": amount,
        "currency_code": "EUR",
        "description": f"Payment to {merchant}",
        "clean_description": merchant,
        "normalized_merchant": merchant,
        "transaction_date": tx_date.strftime("%Y-%m-%d"),
        "booking_date": tx_date.strftime("%Y-%m-%d"),
        "value_date": (tx_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        "category_id": category_id,
        "operation_type": operation_type,
        "is_future": date_offset > 0,
        "is_deleted": False,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }


def generate_mock_transactions(
    count: int = 100,
    user_id: int = 1,
    account_id: int = 1,
    days_range: int = 90,
    include_recurring: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate a list of mock transactions.
    
    Args:
        count: Number of transactions to generate
        user_id: User ID for the transactions
        account_id: Account ID for the transactions
        days_range: Number of days in the past to spread transactions
        include_recurring: Whether to include recurring transactions
        
    Returns:
        List of mock transactions
    """
    transactions = []
    
    # Generate regular random transactions
    for _ in range(count):
        # Random date in the past
        date_offset = -random.randint(0, days_range)
        transactions.append(generate_mock_transaction(user_id, account_id, date_offset))
    
    # Add recurring transactions if requested
    if include_recurring:
        # Monthly subscriptions
        subscriptions = [
            {"merchant": "Netflix", "amount": -13.99, "category_id": 5},
            {"merchant": "Spotify", "amount": -9.99, "category_id": 5},
            {"merchant": "Gym Membership", "amount": -29.99, "category_id": 12},
            {"merchant": "Mobile Phone", "amount": -39.99, "category_id": 9},
            {"merchant": "Internet Provider", "amount": -45.00, "category_id": 9}
        ]
        
        # Add each subscription for multiple months
        for sub in subscriptions:
            # Add for last several months, same day each month
            for month in range(1, 7):  # Last 6 months
                date_offset = -month * 30 + random.randint(-2, 2)  # Add some variance
                transactions.append(generate_mock_transaction(
                    user_id=user_id,
                    account_id=account_id,
                    date_offset=date_offset,
                    amount=sub["amount"],
                    merchant=sub["merchant"],
                    category_id=sub["category_id"]
                ))
        
        # Weekly expenses
        weekly_expenses = [
            {"merchant": "Supermarket", "amount": -65.00, "category_id": 1},
            {"merchant": "Restaurant", "amount": -45.00, "category_id": 2}
        ]
        
        # Add weekly expenses
        for expense in weekly_expenses:
            for week in range(1, 13):  # Last 12 weeks
                date_offset = -week * 7 + random.randint(-1, 1)  # Add some variance
                # Random variation in amount
                amount = expense["amount"] * random.uniform(0.9, 1.1)
                transactions.append(generate_mock_transaction(
                    user_id=user_id,
                    account_id=account_id,
                    date_offset=date_offset,
                    amount=amount,
                    merchant=expense["merchant"],
                    category_id=expense["category_id"]
                ))
        
        # Monthly income
        for month in range(0, 6):  # Last 6 months
            # Salary around 28th of month
            date_offset = -month * 30 - 28 + random.randint(-1, 1)
            transactions.append(generate_mock_transaction(
                user_id=user_id,
                account_id=account_id,
                date_offset=date_offset,
                amount=round(random.uniform(2800, 3200), 2),
                merchant="Employer",
                category_id=15  # Income category
            ))
    
    # Sort by date
    transactions.sort(key=lambda x: x["transaction_date"], reverse=True)
    
    return transactions


def generate_mock_merchant(
    name: Optional[str] = None,
    category_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a mock merchant with random or specified values.
    
    Args:
        name: Merchant name (random if None)
        category_id: Category ID (random if None)
        
    Returns:
        Mock merchant dictionary
    """
    # Random merchant name if not specified
    if name is None:
        merchants = [
            "Amazon", "Carrefour", "Netflix", "Uber", "Starbucks",
            "Apple", "H&M", "IKEA", "McDonald's", "Shell",
            "Spotify", "Zara", "Decathlon", "Fnac", "Sephora"
        ]
        name = random.choice(merchants)
    
    # Random category if not specified
    if category_id is None:
        category_id = random.randint(1, 20)
    
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "display_name": name,
        "normalized_name": name.lower(),
        "category_id": category_id,
        "transaction_count": random.randint(5, 100),
        "logo_url": f"https://example.com/logos/{name.lower().replace(' ', '_')}.png",
        "website": f"https://www.{name.lower().replace(' ', '')}.com",
        "country_code": "FR",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }


def generate_mock_category(
    category_id: int,
    name: str,
    parent_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a mock category with specified values.
    
    Args:
        category_id: Category ID
        name: Category name
        parent_id: Parent category ID (None for top-level categories)
        
    Returns:
        Mock category dictionary
    """
    return {
        "id": category_id,
        "name": name,
        "display_name": name,
        "parent_id": parent_id,
        "level": "primary" if parent_id is None else "secondary",
        "type": "expense",
        "icon": f"icon-{name.lower().replace(' ', '-')}",
        "color": "#{:06x}".format(random.randint(0, 0xFFFFFF)),
        "transaction_count": random.randint(10, 200),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }


def generate_mock_categories() -> List[Dict[str, Any]]:
    """
    Generate a list of mock categories.
    
    Returns:
        List of mock categories
    """
    categories = []
    
    # Top-level categories
    parent_categories = [
        {"id": 1, "name": "Food & Dining"},
        {"id": 2, "name": "Shopping"},
        {"id": 3, "name": "Housing"},
        {"id": 4, "name": "Transportation"},
        {"id": 5, "name": "Entertainment"},
        {"id": 6, "name": "Health & Fitness"},
        {"id": 7, "name": "Personal Care"},
        {"id": 8, "name": "Education"},
        {"id": 9, "name": "Utilities"},
        {"id": 10, "name": "Transfers"},
        {"id": 15, "name": "Income"}
    ]
    
    # Subcategories
    subcategories = [
        {"id": 101, "name": "Groceries", "parent_id": 1},
        {"id": 102, "name": "Restaurants", "parent_id": 1},
        {"id": 103, "name": "Coffee Shops", "parent_id": 1},
        {"id": 201, "name": "Clothing", "parent_id": 2},
        {"id": 202, "name": "Electronics", "parent_id": 2},
        {"id": 203, "name": "Online Shopping", "parent_id": 2},
        {"id": 301, "name": "Rent", "parent_id": 3},
        {"id": 302, "name": "Mortgage", "parent_id": 3},
        {"id": 303, "name": "Home Insurance", "parent_id": 3},
        {"id": 401, "name": "Public Transport", "parent_id": 4},
        {"id": 402, "name": "Fuel", "parent_id": 4},
        {"id": 403, "name": "Parking", "parent_id": 4},
        {"id": 501, "name": "Movies", "parent_id": 5},
        {"id": 502, "name": "Streaming Services", "parent_id": 5},
        {"id": 503, "name": "Concerts", "parent_id": 5},
        {"id": 901, "name": "Electricity", "parent_id": 9},
        {"id": 902, "name": "Water", "parent_id": 9},
        {"id": 903, "name": "Internet", "parent_id": 9},
        {"id": 904, "name": "Phone", "parent_id": 9},
        {"id": 1501, "name": "Salary", "parent_id": 15},
        {"id": 1502, "name": "Freelance", "parent_id": 15},
        {"id": 1503, "name": "Investments", "parent_id": 15}
    ]
    
    # Add parent categories
    for cat in parent_categories:
        categories.append(generate_mock_category(cat["id"], cat["name"]))
    
    # Add subcategories
    for cat in subcategories:
        categories.append(generate_mock_category(cat["id"], cat["name"], cat["parent_id"]))
    
    return categories


def generate_mock_embedding(dimension: int = 1536) -> List[float]:
    """
    Generate a mock embedding vector.
    
    Args:
        dimension: Embedding dimension
        
    Returns:
        Random embedding vector
    """
    # Generate random vector
    vector = [random.normalvariate(0, 1) for _ in range(dimension)]
    
    # Normalize
    magnitude = sum(v**2 for v in vector) ** 0.5
    return [v / magnitude for v in vector]


def generate_mock_query(
    query_type: str = "random",
    date_range: Optional[Tuple[str, str]] = None,
    merchant: Optional[str] = None,
    category_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a mock query for testing search functionality.
    
    Args:
        query_type: Type of query to generate
        date_range: Optional date range tuple (start, end)
        merchant: Optional merchant name
        category_id: Optional category ID
        
    Returns:
        Mock query dictionary
    """
    # Base query
    query = {
        "limit": 50,
        "offset": 0,
        "sort_by": "transaction_date",
        "sort_order": "desc"
    }
    
    # Add query parameters based on type
    if query_type == "date_range":
        if date_range:
            query["start_date"] = date_range[0]
            query["end_date"] = date_range[1]
        else:
            # Last 30 days
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            query["start_date"] = start_date.strftime("%Y-%m-%d")
            query["end_date"] = end_date.strftime("%Y-%m-%d")
    
    elif query_type == "merchant":
        query["merchant_names"] = [merchant or "Supermarket"]
    
    elif query_type == "category":
        query["categories"] = [category_id or 1]
    
    elif query_type == "amount_range":
        query["min_amount"] = -100
        query["max_amount"] = -10
    
    elif query_type == "text_search":
        query["query"] = merchant or "restaurant"
    
    elif query_type == "complex":
        # Combine multiple filters
        query["start_date"] = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")
        query["end_date"] = date.today().strftime("%Y-%m-%d")
        query["categories"] = [category_id or 1]
        query["min_amount"] = -500
        query["max_amount"] = -20
    
    return query


def assert_transactions_equal(
    transaction1: Dict[str, Any],
    transaction2: Dict[str, Any],
    ignore_fields: List[str] = None
) -> bool:
    """
    Assert that two transactions are equal, ignoring specified fields.
    
    Args:
        transaction1: First transaction
        transaction2: Second transaction
        ignore_fields: Fields to ignore in comparison
        
    Returns:
        True if transactions are equal, False otherwise
    """
    if ignore_fields is None:
        ignore_fields = ["id", "created_at", "updated_at"]
    
    # Check all relevant fields
    for key, value in transaction1.items():
        if key in ignore_fields:
            continue
        
        if key not in transaction2:
            print(f"Field {key} missing in second transaction")
            return False
        
        if transaction2[key] != value:
            print(f"Field {key} different: {value} vs {transaction2[key]}")
            return False
    
    return True


def create_mock_database() -> Dict[str, List[Dict[str, Any]]]:
    """
    Create a mock database for testing.
    
    Returns:
        Dictionary containing mock data tables
    """
    db = {
        "transactions": [],
        "merchants": [],
        "categories": [],
        "embeddings": {}
    }
    
    # Generate categories
    db["categories"] = generate_mock_categories()
    
    # Generate merchants
    for _ in range(20):
        db["merchants"].append(generate_mock_merchant())
    
    # Generate transactions
    db["transactions"] = generate_mock_transactions(250, include_recurring=True)
    
    # Generate embeddings for transactions
    for tx in db["transactions"]:
        db["embeddings"][tx["id"]] = generate_mock_embedding()
    
    return db