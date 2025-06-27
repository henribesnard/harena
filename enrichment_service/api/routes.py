"""
Routes API pour le service d'enrichissement avec dual storage.

Ce module définit UNIQUEMENT les endpoints pour l'enrichissement et le stockage
vectoriel des transactions financières dans Qdrant ET Elasticsearch.

RESPONSABILITÉ: ÉCRITURE/STOCKAGE UNIQUEMENT
- Enrichissement des transactions
- Synchronisation dual storage
- Gestion des données (CRUD)
- Diagnostics de stockage

SUPPRIMÉ: Endpoints de recherche (déplacés vers search_service)
"""
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from datetime import datetime

from db_service.session import get_db
from user_service.api.deps import get_current_active_user
from db_service.models.user import User
from db_service.models.sync import RawTransaction

from enrichment_service.models import (
    TransactionInput,
    BatchTransactionInput,
    EnrichmentResult,
    BatchEnrichmentResult
)
from enrichment_service.core.processor import TransactionProcessor, DualStorageTransactionProcessor
from enrichment_service.storage.qdrant import QdrantStorage

logger = logging.getLogger(__name__)
router = APIRouter()

# Instances globales (initialisées dans main.py)
qdrant_storage = None
elasticsearch_client = None
transaction_processor = None
dual_processor = None

def get_processor() -> TransactionProcessor:
    """Récupère l'instance du processeur de transactions legacy."""
    global transaction_processor
    if not transaction_processor:
        global qdrant_storage
        if not qdrant_storage:
            qdrant_storage = QdrantStorage()
        transaction_processor = TransactionProcessor(qdrant_storage)
    return transaction_processor

def get_dual_processor() -> DualStorageTransactionProcessor:
    """Récupère l'instance du processeur dual storage."""
    global dual_processor
    if not dual_processor:
        global qdrant_storage, elasticsearch_client
        if not qdrant_storage:
            qdrant_storage = QdrantStorage()
        dual_processor = DualStorageTransactionProcessor(qdrant_storage, elasticsearch_client)
    return dual_processor

# ==========================================
# ENDPOINTS LEGACY (QDRANT UNIQUEMENT)
# ==========================================

@router.post("/enrich/transaction", response_model=EnrichmentResult)
async def enrich_single_transaction(
    transaction: TransactionInput,
    force_update: bool = Query(False, description="Force la mise à jour même si elle existe déjà"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Enrichit et stocke une transaction individuelle (legacy - Qdrant uniquement).
    
    Args:
        transaction: Données de la transaction à enrichir
        force_update: Force la mise à jour
        
    Returns:
        EnrichmentResult: Résultat de l'enrichissement
    """
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
    Enrichit et stocke un lot de transactions (legacy - Qdrant uniquement).
    
    Args:
        batch: Lot de transactions à enrichir
        force_update: Force la mise à jour
        
    Returns:
        BatchEnrichmentResult: Résultat de l'enrichissement du lot
    """
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
        
        logger.info(f"Lot traité (legacy): {result.successful}/{result.total_transactions} succès")
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de l'enrichissement du lot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enrich batch: {str(e)}"
        )

@router.post("/sync/user/{user_id}", response_model=BatchEnrichmentResult)
async def sync_user_transactions_legacy(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Synchronise toutes les transactions d'un utilisateur depuis PostgreSQL vers Qdrant (legacy).
    
    DÉPRÉCIÉ: Utilisez /dual/sync-user pour le dual storage.
    
    Args:
        user_id: ID de l'utilisateur à synchroniser
        
    Returns:
        BatchEnrichmentResult: Résultat de la synchronisation
    """
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
        
        logger.info(f"Synchronisation legacy de {len(transaction_inputs)} transactions pour l'utilisateur {user_id}")
        
        # Synchroniser via le processeur legacy
        processor = get_processor()
        result = await processor.sync_user_transactions(user_id, transaction_inputs)
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de la synchronisation legacy de l'utilisateur {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync user transactions: {str(e)}"
        )

@router.delete("/user/{user_id}")
async def delete_user_data_legacy(
    user_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """
    Supprime toutes les données vectorielles d'un utilisateur (legacy - Qdrant uniquement).
    
    DÉPRÉCIÉ: Utilisez /dual/user-data/{user_id} pour le dual storage.
    
    Args:
        user_id: ID de l'utilisateur
        
    Returns:
        Dict: Résultat de la suppression
    """
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
                "message": f"All data deleted for user {user_id} (legacy - Qdrant only)",
                "user_id": user_id,
                "storage": "qdrant_only"
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

# ==========================================
# ENDPOINTS DUAL STORAGE (RECOMMANDÉS)
# ==========================================

@router.post("/dual/sync-user", response_model=BatchEnrichmentResult)
async def sync_user_dual_storage(
    user_id: int = Query(..., description="ID de l'utilisateur à synchroniser"),
    force_refresh: bool = Query(False, description="Force la suppression et recréation des données"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Synchronise toutes les transactions d'un utilisateur dans Qdrant ET Elasticsearch.
    
    Cette méthode:
    1. Lit les transactions depuis PostgreSQL
    2. Génère les embeddings OpenAI
    3. Stocke dans Qdrant (recherche sémantique)
    4. Indexe dans Elasticsearch (recherche lexicale)
    
    RECOMMANDÉ: Utilisez cet endpoint pour tous les nouveaux déploiements.
    """
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    logger.info(f"🔄 Synchronisation dual storage demandée pour user {user_id} (force_refresh: {force_refresh})")
    
    try:
        # Récupérer les transactions depuis PostgreSQL
        raw_transactions = db.query(RawTransaction).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.deleted == False
        ).all()
        
        if not raw_transactions:
            logger.warning(f"Aucune transaction trouvée pour l'utilisateur {user_id}")
            return BatchEnrichmentResult(
                user_id=user_id,
                total_transactions=0,
                successful=0,
                failed=0,
                processing_time=0.0,
                results=[],
                errors=["No transactions found in database"]
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
        
        logger.info(f"📊 Synchronisation dual de {len(transaction_inputs)} transactions pour l'utilisateur {user_id}")
        
        # Synchroniser via le processeur dual
        dual_processor = get_dual_processor()
        result = await dual_processor.sync_user_transactions(
            user_id=user_id,
            transactions=transaction_inputs,
            force_refresh=force_refresh
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la synchronisation dual pour l'utilisateur {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync user transactions: {str(e)}"
        )

@router.post("/dual/enrich-transaction", response_model=EnrichmentResult)
async def enrich_transaction_dual_storage(
    transaction: TransactionInput,
    force_update: bool = Query(False, description="Force la mise à jour même si elle existe déjà"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Enrichit et stocke une transaction dans Qdrant ET Elasticsearch.
    
    Utile pour traiter des transactions individuelles en temps réel.
    
    RECOMMANDÉ: Utilisez cet endpoint pour tous les nouveaux déploiements.
    """
    if transaction.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only enrich your own transactions"
        )
    
    try:
        dual_processor = get_dual_processor()
        result = await dual_processor.process_single_transaction(
            transaction=transaction,
            force_update=force_update
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur enrichissement dual transaction {transaction.bridge_transaction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enrich transaction: {str(e)}"
        )

@router.delete("/dual/user-data/{user_id}")
async def delete_user_dual_data(
    user_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """
    Supprime toutes les données d'un utilisateur des deux systèmes de stockage.
    
    ATTENTION: Cette opération est irréversible!
    
    RECOMMANDÉ: Utilisez cet endpoint pour tous les nouveaux déploiements.
    """
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        dual_processor = get_dual_processor()
        
        # Supprimer de Qdrant
        qdrant_success = False
        if dual_processor.qdrant_storage:
            try:
                qdrant_success = await dual_processor.qdrant_storage.delete_user_transactions(user_id)
            except Exception as e:
                logger.error(f"❌ Erreur suppression Qdrant user {user_id}: {e}")
        
        # Supprimer d'Elasticsearch
        elasticsearch_success = False
        if dual_processor.elasticsearch_client:
            try:
                elasticsearch_success = await dual_processor.elasticsearch_client.delete_user_transactions(user_id)
            except Exception as e:
                logger.error(f"❌ Erreur suppression Elasticsearch user {user_id}: {e}")
        
        return {
            "user_id": user_id,
            "deleted": qdrant_success and elasticsearch_success,
            "details": {
                "qdrant_deleted": qdrant_success,
                "elasticsearch_deleted": elasticsearch_success
            },
            "storage": "dual_storage",
            "message": "All data deleted successfully" if (qdrant_success and elasticsearch_success) else "Partial deletion"
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur suppression données user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user data: {str(e)}"
        )

# ==========================================
# ENDPOINTS DE DIAGNOSTIC ET MONITORING
# ==========================================

@router.get("/dual/sync-status/{user_id}")
async def get_dual_sync_status(
    user_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère le statut de synchronisation pour un utilisateur dans les deux systèmes.
    
    Retourne:
    - Nombre de documents dans Qdrant
    - Nombre de documents dans Elasticsearch  
    - Statut de cohérence entre les deux
    """
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        dual_processor = get_dual_processor()
        status_info = await dual_processor.get_sync_status(user_id)
        return status_info
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération statut sync user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync status: {str(e)}"
        )

@router.get("/dual/health")
async def dual_storage_health():
    """
    Vérifie la santé des deux systèmes de stockage.
    
    Retourne l'état de Qdrant et Elasticsearch.
    """
    try:
        dual_processor = get_dual_processor()
        
        # Tester Qdrant
        qdrant_healthy = False
        qdrant_error = None
        try:
            if dual_processor.qdrant_storage:
                qdrant_info = await dual_processor.qdrant_storage.get_collection_info()
                qdrant_healthy = qdrant_info is not None
        except Exception as e:
            qdrant_error = str(e)
        
        # Tester Elasticsearch
        elasticsearch_healthy = False
        elasticsearch_error = None
        try:
            if dual_processor.elasticsearch_client:
                elasticsearch_healthy = dual_processor.elasticsearch_client._initialized
                if not elasticsearch_healthy:
                    elasticsearch_error = "Client not initialized"
        except Exception as e:
            elasticsearch_error = str(e)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": qdrant_healthy and elasticsearch_healthy,
            "storage_systems": {
                "qdrant": {
                    "healthy": qdrant_healthy,
                    "error": qdrant_error,
                    "collection": dual_processor.qdrant_storage.collection_name if dual_processor.qdrant_storage else None
                },
                "elasticsearch": {
                    "healthy": elasticsearch_healthy,
                    "error": elasticsearch_error,
                    "index": dual_processor.elasticsearch_client.index_name if dual_processor.elasticsearch_client else None
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur health check dual storage: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": False,
            "error": str(e)
        }

@router.get("/legacy/collection-info")
async def get_collection_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère les informations de la collection Qdrant (legacy).
    
    DÉPRÉCIÉ: Informations disponibles via /dual/health.
    Réservé aux superusers pour diagnostics avancés.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions - superuser required"
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
                "vectors_count": collection_info.vectors_count,
                "note": "DÉPRÉCIÉ - Utilisez /dual/health pour les informations générales"
            }
        else:
            return {
                "status": "Collection not found or not accessible",
                "note": "DÉPRÉCIÉ - Utilisez /dual/health pour les informations générales"
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos de collection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection info: {str(e)}"
        )