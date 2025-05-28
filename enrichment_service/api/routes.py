"""
Routes API pour le service d'enrichissement.

Ce module définit tous les endpoints pour l'enrichissement et le stockage
vectoriel des transactions financières.
"""
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from db_service.session import get_db
from user_service.api.deps import get_current_active_user
from db_service.models.user import User
from db_service.models.sync import RawTransaction

from enrichment_service.models import (
    TransactionInput,
    BatchTransactionInput,
    EnrichmentResult,
    BatchEnrichmentResult,
    SearchResponse,
    SearchResult
)
from enrichment_service.core.processor import TransactionProcessor
from enrichment_service.core.embeddings import embedding_service
from enrichment_service.storage.qdrant import QdrantStorage

logger = logging.getLogger(__name__)
router = APIRouter()

# Instances globales (initialisées dans main.py)
qdrant_storage = None
transaction_processor = None

def get_processor() -> TransactionProcessor:
    """Récupère l'instance du processeur de transactions."""
    global transaction_processor
    if not transaction_processor:
        global qdrant_storage
        if not qdrant_storage:
            qdrant_storage = QdrantStorage()
        transaction_processor = TransactionProcessor(qdrant_storage)
    return transaction_processor

@router.post("/enrich/transaction", response_model=EnrichmentResult)
async def enrich_single_transaction(
    transaction: TransactionInput,
    force_update: bool = Query(False, description="Force la mise à jour même si elle existe déjà"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Enrichit et stocke une transaction individuelle.
    
    Args:
        transaction: Données de la transaction à enrichir
        force_update: Force la mise à jour
        
    Returns:
        EnrichmentResult: Résultat de l'enrichissement
    """
    # Vérifier que la transaction appartient à l'utilisateur actuel
    if transaction.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Transaction does not belong to current user"
        )
    
    try:
        processor = get_processor()
        result = await processor.process_transaction(transaction, force_update)
        
        if result.status == "error":
            logger.warning(f"Enrichissement échoué pour transaction {transaction.bridge_transaction_id}: {result.error_message}")
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de l'enrichissement de la transaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enrich transaction: {str(e)}"
        )

@router.post("/enrich/batch", response_model=BatchEnrichmentResult)
async def enrich_batch_transactions(
    batch: BatchTransactionInput,
    force_update: bool = Query(False, description="Force la mise à jour même si elles existent déjà"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Enrichit et stocke un lot de transactions.
    
    Args:
        batch: Lot de transactions à enrichir
        force_update: Force la mise à jour
        
    Returns:
        BatchEnrichmentResult: Résultat de l'enrichissement du lot
    """
    # Vérifier que le lot appartient à l'utilisateur actuel
    if batch.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Batch does not belong to current user"
        )
    
    # Vérifier que toutes les transactions appartiennent au bon utilisateur
    for tx in batch.transactions:
        if tx.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Transaction {tx.bridge_transaction_id} does not belong to current user"
            )
    
    try:
        processor = get_processor()
        result = await processor.process_batch(batch, force_update)
        
        logger.info(f"Lot traité: {result.successful}/{result.total_transactions} succès")
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de l'enrichissement du lot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enrich batch: {str(e)}"
        )

@router.post("/sync/user/{user_id}", response_model=BatchEnrichmentResult)
async def sync_user_transactions(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Synchronise toutes les transactions d'un utilisateur depuis PostgreSQL vers Qdrant.
    
    Args:
        user_id: ID de l'utilisateur à synchroniser
        
    Returns:
        BatchEnrichmentResult: Résultat de la synchronisation
    """
    # Vérifier les permissions
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        # Récupérer toutes les transactions de l'utilisateur depuis PostgreSQL
        raw_transactions = db.query(RawTransaction).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.deleted == False
        ).all()
        
        if not raw_transactions:
            logger.info(f"Aucune transaction trouvée pour l'utilisateur {user_id}")
            return BatchEnrichmentResult(
                user_id=user_id,
                total_transactions=0,
                successful=0,
                failed=0,
                processing_time=0.0,
                results=[],
                errors=[]
            )
        
        # Convertir en TransactionInput
        transaction_inputs = []
        for raw_tx in raw_transactions:
            tx_input = TransactionInput(
                bridge_transaction_id=raw_tx.bridge_transaction_id,
                user_id=raw_tx.user_id,
                account_id=raw_tx.account_id,
                clean_description=raw_tx.clean_description,
                provider_description=raw_tx.provider_description,
                amount=raw_tx.amount,
                date=raw_tx.date,
                booking_date=raw_tx.booking_date,
                transaction_date=raw_tx.transaction_date,
                value_date=raw_tx.value_date,
                currency_code=raw_tx.currency_code,
                category_id=raw_tx.category_id,
                operation_type=raw_tx.operation_type,
                deleted=raw_tx.deleted,
                future=raw_tx.future
            )
            transaction_inputs.append(tx_input)
        
        logger.info(f"Synchronisation de {len(transaction_inputs)} transactions pour l'utilisateur {user_id}")
        
        # Synchroniser via le processeur
        processor = get_processor()
        result = await processor.sync_user_transactions(user_id, transaction_inputs)
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de la synchronisation de l'utilisateur {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync user transactions: {str(e)}"
        )

@router.get("/search", response_model=SearchResponse)
async def search_transactions(
    query: str = Query(..., description="Requête de recherche"),
    limit: int = Query(10, ge=1, le=100, description="Nombre maximum de résultats"),
    amount_min: float = Query(None, description="Montant minimum"),
    amount_max: float = Query(None, description="Montant maximum"),
    transaction_type: str = Query(None, description="Type de transaction (debit/credit)"),
    date_from: str = Query(None, description="Date de début (YYYY-MM-DD)"),
    date_to: str = Query(None, description="Date de fin (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Recherche des transactions par similarité sémantique.
    
    Args:
        query: Texte de recherche
        limit: Nombre de résultats
        amount_min: Montant minimum
        amount_max: Montant maximum
        transaction_type: Type de transaction
        date_from: Date de début
        date_to: Date de fin
        
    Returns:
        SearchResponse: Résultats de recherche
    """
    import time
    from datetime import datetime
    
    start_time = time.time()
    
    try:
        # Générer l'embedding de la requête
        query_embedding = await embedding_service.generate_embedding(query)
        
        # Construire les filtres
        filters = {}
        if amount_min is not None:
            filters["amount_min"] = amount_min
        if amount_max is not None:
            filters["amount_max"] = amount_max
        if transaction_type:
            filters["transaction_type"] = transaction_type
        if date_from:
            try:
                date_from_dt = datetime.fromisoformat(date_from)
                filters["date_from"] = date_from_dt.timestamp()
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid date_from format. Use YYYY-MM-DD"
                )
        if date_to:
            try:
                date_to_dt = datetime.fromisoformat(date_to)
                filters["date_to"] = date_to_dt.timestamp()
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid date_to format. Use YYYY-MM-DD"
                )
        
        # Effectuer la recherche
        processor = get_processor()
        results = await processor.qdrant_storage.search_transactions(
            query_vector=query_embedding,
            user_id=current_user.id,
            limit=limit,
            filters=filters
        )
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            query=query,
            results=results,
            total_found=len(results),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.delete("/user/{user_id}")
async def delete_user_data(
    user_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """
    Supprime toutes les données vectorielles d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        
    Returns:
        Dict: Résultat de la suppression
    """
    # Vérifier les permissions
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        processor = get_processor()
        success = await processor.delete_user_data(user_id)
        
        if success:
            return {
                "status": "success",
                "message": f"All data deleted for user {user_id}",
                "user_id": user_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user data"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la suppression pour l'utilisateur {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user data: {str(e)}"
        )

@router.get("/user/{user_id}/stats")
async def get_user_stats(
    user_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère les statistiques d'enrichissement d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        
    Returns:
        Dict: Statistiques de l'utilisateur
    """
    # Vérifier les permissions
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        processor = get_processor()
        stats = await processor.get_user_stats(user_id)
        return stats
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats pour {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user stats: {str(e)}"
        )

@router.get("/collection/info")
async def get_collection_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère les informations de la collection Qdrant.
    
    Returns:
        Dict: Informations de la collection
    """
    # Seuls les superusers peuvent voir les infos de collection
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        processor = get_processor()
        collection_info = await processor.qdrant_storage.get_collection_info()
        
        if collection_info:
            return {
                "collection_name": collection_info.config.params,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "vectors_count": collection_info.vectors_count
            }
        else:
            return {
                "status": "Collection not found or not accessible"
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos de collection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection info: {str(e)}"
        )