# transaction_vector_service/api/endpoints/transactions.py
"""
API endpoints for transaction operations.

This module defines the FastAPI endpoints for searching, retrieving,
and managing financial transactions.
"""

import time
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, Request
from fastapi.responses import JSONResponse
from pydantic import UUID4
from datetime import date, datetime, timedelta

from ...models.transaction import (
    TransactionRead, 
    TransactionDetail, 
    TransactionSearch, 
    TransactionCreate, 
    TransactionBatchCreate, 
    TransactionStats, 
    TransactionSearchResults,
    SearchWeights,
    SearchMode
)
from ...services.transaction_service import TransactionService
from ...config.constants import DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT, SEARCH_WEIGHTS
from ...utils.search_utils import parse_natural_language_query, convert_nl_params_to_search_query
from ...utils.date_utils import parse_date, get_date_range
from ...search.hybrid_search import HybridSearch
from ..dependencies import get_current_user, get_rate_limiter, get_transaction_service

router = APIRouter()


@router.get("/transactions", response_model=TransactionSearchResults)
async def search_transactions(
    request: Request,
    current_user: Dict = Depends(get_current_user),
    rate_limiter: Any = Depends(get_rate_limiter),
    transaction_service: TransactionService = Depends(get_transaction_service),
    query: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    categories: List[int] = Query(None),
    merchant_names: List[str] = Query(None),
    operation_types: List[str] = Query(None),
    account_ids: List[int] = Query(None),
    include_future: bool = False,
    include_deleted: bool = False,
    limit: int = DEFAULT_SEARCH_LIMIT,
    offset: int = 0,
    sort_by: str = "transaction_date",
    sort_order: str = "desc",
    search_mode: SearchMode = SearchMode.HYBRID,
    include_explanation: bool = False,
    bm25_weight: Optional[float] = None,
    vector_weight: Optional[float] = None,
    cross_encoder_weight: Optional[float] = None,
    min_relevance: Optional[float] = None,
    natural_language: bool = False
):
    """
    Search for transactions with various filters.
    
    - **query**: Text search query
    - **start_date**: Filter transactions after this date
    - **end_date**: Filter transactions before this date
    - **min_amount**: Minimum transaction amount
    - **max_amount**: Maximum transaction amount
    - **categories**: List of category IDs to filter by
    - **merchant_names**: List of merchant names to filter by
    - **operation_types**: List of operation types to filter by
    - **account_ids**: List of account IDs to filter by
    - **include_future**: Whether to include future transactions
    - **include_deleted**: Whether to include deleted transactions
    - **limit**: Maximum number of results to return
    - **offset**: Offset for pagination
    - **sort_by**: Field to sort by
    - **sort_order**: Sort order, "asc" or "desc"
    - **search_mode**: Search mode ("hybrid", "bm25", "vector")
    - **include_explanation**: Whether to include an explanation of search results
    - **bm25_weight**: Weight of BM25 search in hybrid search
    - **vector_weight**: Weight of vector search in hybrid search
    - **cross_encoder_weight**: Weight of cross-encoder in hybrid search
    - **min_relevance**: Minimum relevance score for results
    - **natural_language**: Whether to parse the query as natural language
    """
    # Apply rate limiting
    await rate_limiter(request)
    
    # Get the user ID from the token
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
        
    # Limit the number of results
    if limit > MAX_SEARCH_LIMIT:
        limit = MAX_SEARCH_LIMIT
    
    # Handle natural language queries
    if natural_language and query:
        nl_params = parse_natural_language_query(query)
        search_query = convert_nl_params_to_search_query(nl_params)
        
        # Override parameters if found in the natural language query
        if "query" in search_query:
            query = search_query["query"]
        if "start_date" in search_query:
            start_date = search_query["start_date"]
        if "end_date" in search_query:
            end_date = search_query["end_date"]
        if "min_amount" in search_query:
            min_amount = search_query["min_amount"]
        if "max_amount" in search_query:
            max_amount = search_query["max_amount"]
        if "categories" in search_query:
            categories = search_query["categories"]
        if "merchant_names" in search_query:
            merchant_names = search_query["merchant_names"]
    
    # Configure search weights if provided
    search_weights = None
    if bm25_weight is not None and vector_weight is not None and cross_encoder_weight is not None:
        try:
            weights_sum = bm25_weight + vector_weight + cross_encoder_weight
            if abs(weights_sum - 1.0) > 0.001:  # Allow small floating point errors
                normalized_bm25 = bm25_weight / weights_sum
                normalized_vector = vector_weight / weights_sum
                normalized_cross_encoder = cross_encoder_weight / weights_sum
                search_weights = SearchWeights(
                    bm25_weight=normalized_bm25,
                    vector_weight=normalized_vector,
                    cross_encoder_weight=normalized_cross_encoder
                )
            else:
                search_weights = SearchWeights(
                    bm25_weight=bm25_weight,
                    vector_weight=vector_weight,
                    cross_encoder_weight=cross_encoder_weight
                )
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid search weights: {str(e)}"
            )
    
    # Prepare search parameters
    search_params = TransactionSearch(
        query=query,
        start_date=start_date,
        end_date=end_date,
        min_amount=min_amount,
        max_amount=max_amount,
        categories=categories if categories else None,
        merchant_names=merchant_names if merchant_names else None,
        operation_types=operation_types if operation_types else None,
        account_ids=account_ids if account_ids else None,
        include_future=include_future,
        include_deleted=include_deleted,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order,
        search_mode=search_mode,
        search_weights=search_weights,
        min_relevance=min_relevance,
        include_explanation=include_explanation
    )
    
    # Measure execution time
    start_time = time.time()
    
    # Execute the search
    transactions, total = await transaction_service.search_transactions(user_id, search_params)
    
    # Calculate execution time
    execution_time_ms = int((time.time() - start_time) * 1000)
    
    # Generate explanation if requested
    explanation = None
    if include_explanation and query:
        explanation = await transaction_service.explain_search_results(query, transactions)
    
    # Prepare filters applied information
    filters_applied = {
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "min_amount": min_amount,
        "max_amount": max_amount,
        "categories": categories,
        "merchant_names": merchant_names,
        "operation_types": operation_types,
        "account_ids": account_ids,
        "search_mode": search_mode.value,
        "sort_by": sort_by,
        "sort_order": sort_order
    }
    
    # Build response
    page = offset // limit + 1
    page_size = limit
    has_more = total > (offset + limit)
    
    # Convert transactions to ReadModel
    transaction_reads = []
    for tx in transactions:
        tx_dict = dict(tx)
        
        # Add category_name if we have category_id but no category_name
        if "category_id" in tx and tx["category_id"] and "category_name" not in tx:
            # In a real implementation, this would look up the category name
            tx_dict["category_name"] = f"Category {tx['category_id']}"
        
        # Add parent_category_name if we have parent_category_id but no parent_category_name
        if "parent_category_id" in tx and tx["parent_category_id"] and "parent_category_name" not in tx:
            # In a real implementation, this would look up the parent category name
            tx_dict["parent_category_name"] = f"Category {tx['parent_category_id']}"
        
        transaction_reads.append(TransactionRead(**tx_dict))
    
    return TransactionSearchResults(
        results=transaction_reads,
        total=total,
        page=page,
        page_size=page_size,
        has_more=has_more,
        query=query,
        search_mode=search_mode.value,
        execution_time_ms=execution_time_ms,
        explanation=explanation,
        filters_applied=filters_applied
    )


@router.get("/transactions/{transaction_id}", response_model=TransactionDetail)
async def get_transaction(
    transaction_id: UUID4,
    current_user: Dict = Depends(get_current_user),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Get a transaction by ID.
    
    - **transaction_id**: UUID of the transaction to retrieve
    """
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Get the transaction
    transaction = await transaction_service.get_transaction(transaction_id)
    
    if not transaction:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
    
    # Check if the transaction belongs to the authenticated user
    if transaction.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this transaction")
    
    # Add category_name if we have category_id but no category_name
    if "category_id" in transaction and transaction["category_id"] and "category_name" not in transaction:
        # In a real implementation, this would look up the category name
        transaction["category_name"] = f"Category {transaction['category_id']}"
    
    # Add parent_category_name if we have parent_category_id but no parent_category_name
    if "parent_category_id" in transaction and transaction["parent_category_id"] and "parent_category_name" not in transaction:
        # In a real implementation, this would look up the parent category name
        transaction["parent_category_name"] = f"Category {transaction['parent_category_id']}"
    
    return TransactionDetail(**transaction)


@router.get("/transactions/{transaction_id}/similar", response_model=List[TransactionRead])
async def get_similar_transactions(
    transaction_id: UUID4,
    current_user: Dict = Depends(get_current_user),
    transaction_service: TransactionService = Depends(get_transaction_service),
    limit: int = 5
):
    """
    Find transactions similar to the given transaction.
    
    - **transaction_id**: UUID of the reference transaction
    - **limit**: Maximum number of similar transactions to return
    """
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Get the reference transaction
    transaction = await transaction_service.get_transaction(transaction_id)
    
    if not transaction:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
    
    # Check if the transaction belongs to the authenticated user
    if transaction.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this transaction")
    
    # Find similar transactions
    similar_transactions = await transaction_service.find_similar_transactions(
        embedding=transaction.get("embedding"),
        user_id=user_id,
        limit=limit
    )
    
    # Convert to TransactionRead models
    result = []
    for tx in similar_transactions:
        tx_dict = dict(tx)
        
        # Add category_name if we have category_id but no category_name
        if "category_id" in tx and tx["category_id"] and "category_name" not in tx:
            # In a real implementation, this would look up the category name
            tx_dict["category_name"] = f"Category {tx['category_id']}"
        
        # Add parent_category_name if we have parent_category_id but no parent_category_name
        if "parent_category_id" in tx and tx["parent_category_id"] and "parent_category_name" not in tx:
            # In a real implementation, this would look up the parent category name
            tx_dict["parent_category_name"] = f"Category {tx['parent_category_id']}"
        
        result.append(TransactionRead(**tx_dict))
    
    return result


@router.post("/transactions", response_model=TransactionRead)
async def create_transaction(
    transaction: TransactionCreate,
    current_user: Dict = Depends(get_current_user),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Create a new transaction.
    
    The transaction will be processed, enriched, and stored.
    """
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Check if the transaction belongs to the authenticated user
    if transaction.user_id != user_id:
        raise HTTPException(status_code=403, detail="Cannot create transactions for other users")
    
    # Process and store the transaction
    result = await transaction_service.process_transaction(transaction)
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to process transaction")
    
    # Convert to TransactionRead model
    return TransactionRead.model_validate(result)


@router.post("/transactions/batch", response_model=List[TransactionRead])
async def create_transactions_batch(
    batch: TransactionBatchCreate,
    current_user: Dict = Depends(get_current_user),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Create multiple transactions in a batch.
    
    All transactions will be processed, enriched, and stored.
    """
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Check if all transactions belong to the authenticated user
    for tx in batch.transactions:
        if tx.user_id != user_id:
            raise HTTPException(status_code=403, detail="Cannot create transactions for other users")
    
    # Process and store the transactions
    results = await transaction_service.process_transactions_batch(batch.transactions)
    
    # Convert successful results to TransactionRead models
    return [TransactionRead.model_validate(result) for result in results if result]


@router.put("/transactions/{transaction_id}", response_model=TransactionRead)
async def update_transaction(
    transaction_id: UUID4,
    updates: Dict[str, Any] = Body(...),
    current_user: Dict = Depends(get_current_user),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Update a transaction.
    
    - **transaction_id**: UUID of the transaction to update
    - **updates**: Dictionary of fields to update
    """
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Get the transaction
    transaction = await transaction_service.get_transaction(transaction_id)
    
    if not transaction:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
    
    # Check if the transaction belongs to the authenticated user
    if transaction.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this transaction")
    
    # Remove fields that cannot be updated
    disallowed_fields = ["id", "user_id", "bridge_transaction_id", "created_at"]
    for field in disallowed_fields:
        if field in updates:
            del updates[field]
    
    # Update the transaction
    success = await transaction_service.update_transaction(transaction_id, updates)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update transaction")
    
    # Get the updated transaction
    updated_transaction = await transaction_service.get_transaction(transaction_id)
    
    # Convert to TransactionRead model
    return TransactionRead(**updated_transaction)


@router.delete("/transactions/{transaction_id}", response_model=Dict[str, Any])
async def delete_transaction(
    transaction_id: UUID4,
    current_user: Dict = Depends(get_current_user),
    transaction_service: TransactionService = Depends(get_transaction_service),
    hard_delete: bool = False
):
    """
    Delete a transaction.
    
    - **transaction_id**: UUID of the transaction to delete
    - **hard_delete**: Whether to permanently delete the transaction (default: false, soft delete)
    """
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Get the transaction
    transaction = await transaction_service.get_transaction(transaction_id)
    
    if not transaction:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
    
    # Check if the transaction belongs to the authenticated user
    if transaction.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this transaction")
    
    if hard_delete:
        # Permanently delete the transaction
        success = await transaction_service.delete_transaction(transaction_id)
    else:
        # Soft delete (mark as deleted)
        success = await transaction_service.mark_transaction_as_deleted(transaction_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete transaction")
    
    return {
        "id": str(transaction_id),
        "deleted": True,
        "hard_delete": hard_delete
    }


@router.get("/transactions/stats", response_model=TransactionStats)
async def get_transaction_stats(
    current_user: Dict = Depends(get_current_user),
    transaction_service: TransactionService = Depends(get_transaction_service),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    account_id: Optional[int] = None,
    category_id: Optional[int] = None,
    period: Optional[str] = None
):
    """
    Get statistical summary of transactions.
    
    - **start_date**: Start date for filtering
    - **end_date**: End date for filtering
    - **account_id**: Account ID for filtering
    - **category_id**: Category ID for filtering
    - **period**: Predefined period (e.g., "this_month", "last_month", "this_year")
    """
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Handle predefined periods
    if period:
        try:
            period_start, period_end = get_date_range(period)
            start_date = period_start
            end_date = period_end
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid period: {period}")
    
    # Get transaction stats
    stats = await transaction_service.get_transaction_stats(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        account_id=account_id,
        category_id=category_id
    )
    
    return stats