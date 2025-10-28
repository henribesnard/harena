"""
Routes API pour le service d'enrichissement - Elasticsearch uniquement.

Ce module définit UNIQUEMENT les endpoints pour l'enrichissement et l'indexation
des transactions financières dans Elasticsearch.

RESPONSABILITÉ: ÉCRITURE/INDEXATION UNIQUEMENT
- Enrichissement des transactions
- Synchronisation Elasticsearch
- Gestion des données (CRUD)
- Diagnostics d'indexation

SUPPRIMÉ: Endpoints de recherche (déplacés vers search_service)
SUPPRIMÉ: Endpoints Qdrant et dual storage
"""
import logging
import aiohttp
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Header
from sqlalchemy.orm import Session
from datetime import datetime

from db_service.session import get_db
from user_service.api.deps import get_current_active_user
from db_service.models.user import User
from db_service.models.sync import RawTransaction, SyncAccount

from enrichment_service.models import (
    TransactionInput,
    UserSyncResult,
    ElasticsearchHealthStatus,
    BatchMerchantEnrichmentResult,
)
from enrichment_service.core.processor import ElasticsearchTransactionProcessor
from enrichment_service.core.account_enrichment_service import AccountEnrichmentService
from enrichment_service.core.merchant_batch_enrichment import MerchantBatchEnrichmentService
from config_service.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Instances globales (initialisées dans main.py)
elasticsearch_client = None
elasticsearch_processor = None
account_enrichment_service = None
merchant_batch_service = None


async def invalidate_search_cache(user_id: int, token: str) -> bool:
    """
    Invalide le cache du service de recherche pour un utilisateur après synchronisation.

    Args:
        user_id: ID de l'utilisateur
        token: Token JWT pour authentifier la requête

    Returns:
        bool: True si l'invalidation a réussi, False sinon
    """
    try:
        search_url = f"{settings.SEARCH_SERVICE_URL}/api/v1/search/cache/user/{user_id}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.delete(search_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Cache invalidé pour user {user_id}: {data.get('entries_deleted', 0)} entrées supprimées")
                    return True
                else:
                    error_text = await response.text()
                    logger.warning(f"⚠️ Échec invalidation cache user {user_id}: {response.status} - {error_text}")
                    return False
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'invalidation du cache pour user {user_id}: {e}")
        return False


def get_account_enrichment_service(
    db: Session = Depends(get_db),
) -> AccountEnrichmentService:
    """Retourne l'instance du service d'enrichissement de compte."""
    global account_enrichment_service
    if not account_enrichment_service:
        account_enrichment_service = AccountEnrichmentService(db)
    return account_enrichment_service


def get_elasticsearch_processor(
    account_service: AccountEnrichmentService = Depends(get_account_enrichment_service),
) -> ElasticsearchTransactionProcessor:
    """Récupère l'instance du processeur Elasticsearch."""
    global elasticsearch_processor
    if not elasticsearch_processor:
        global elasticsearch_client
        if not elasticsearch_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Elasticsearch client not available",
            )
        elasticsearch_processor = ElasticsearchTransactionProcessor(
            elasticsearch_client, account_service
        )
    return elasticsearch_processor


def get_merchant_batch_service(
    db: Session = Depends(get_db),
) -> MerchantBatchEnrichmentService:
    """Récupère l'instance du service d'enrichissement de marchands par lot."""
    global merchant_batch_service
    if not merchant_batch_service:
        global elasticsearch_client
        if not elasticsearch_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Elasticsearch client not available",
            )
        merchant_batch_service = MerchantBatchEnrichmentService(db, elasticsearch_client)
    return merchant_batch_service

# ==========================================
# ENDPOINTS ELASTICSEARCH PRINCIPAUX
# ==========================================


@router.post("/elasticsearch/sync-user/{user_id}", response_model=UserSyncResult)
async def sync_user_transactions(
    user_id: int,
    force_refresh: bool = Query(False, description="Force la suppression et recréation de tous les documents"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    processor: ElasticsearchTransactionProcessor = Depends(get_elasticsearch_processor),
    authorization: Optional[str] = Header(None),
):
    """
    Synchronise toutes les transactions d'un utilisateur depuis PostgreSQL vers Elasticsearch.
    
    Cette méthode:
    1. Lit les transactions depuis PostgreSQL
    2. Structure les données pour Elasticsearch
    3. Indexe en mode bulk pour optimiser les performances
    
    Args:
        user_id: ID de l'utilisateur à synchroniser
        force_refresh: Force la suppression et recréation
        
    Returns:
        UserSyncResult: Résultat de la synchronisation
    """
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    logger.info(f"🔄 Synchronisation Elasticsearch demandée pour user {user_id} (force_refresh: {force_refresh})")
    
    try:
        # Récupérer toutes les transactions de l'utilisateur depuis PostgreSQL
        raw_transactions = db.query(RawTransaction).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.deleted == False
        ).all()
        
        if not raw_transactions:
            logger.info(f"Aucune transaction trouvée pour l'utilisateur {user_id}")
            return UserSyncResult(
                user_id=user_id,
                total_transactions=0,
                transactions_indexed=0,
                accounts_indexed=0,
                updated=0,
                errors=0,
                with_account_metadata=0,
                accounts_synced=0,
                processing_time=0.0,
                status="success",
                error_details=[]
            )
        
        # Précharger les informations de compte pour toutes les transactions
        # On récupère les comptes par leur ID interne (FK de raw_transactions)
        account_ids = {tx.account_id for tx in raw_transactions}
        accounts = db.query(SyncAccount).filter(SyncAccount.id.in_(account_ids)).all()

        # DEBUG: Logger les comptes récupérés
        logger.info(f"📊 {len(accounts)} comptes récupérés pour user {user_id}")
        for acc in accounts[:3]:  # Logger les 3 premiers
            logger.info(f"   Compte {acc.id}: name='{acc.account_name}', type='{acc.account_type}', bridge_id={acc.bridge_account_id}")

        # Map interne pour retrouver rapidement le compte à partir de l'ID interne
        accounts_by_internal_id = {acc.id: acc for acc in accounts}

        # Map public pour le processeur, indexé par bridge_account_id (identifiant métier)
        accounts_map = {acc.bridge_account_id: acc for acc in accounts}

        # Convertir en TransactionInput avec métadonnées de compte
        transaction_inputs = []
        for raw_tx in raw_transactions:
            # Récupération via l'ID interne
            account = accounts_by_internal_id.get(raw_tx.account_id)

            # DEBUG: Logger quelques transactions pour vérifier
            if len(transaction_inputs) < 2:
                logger.info(f"   TX {raw_tx.bridge_transaction_id}: account_id={raw_tx.account_id}, found_account={account is not None}")
                if account:
                    logger.info(f"      -> account_name='{account.account_name}', account_type='{account.account_type}'")

            tx_input = TransactionInput(
                bridge_transaction_id=raw_tx.bridge_transaction_id,
                user_id=raw_tx.user_id,
                # Pour l'enrichment_service, account_id correspond désormais au bridge_account_id
                account_id=(account.bridge_account_id if account else raw_tx.account_id),
                account_name=account.account_name if account else None,
                account_type=account.account_type if account else None,
                account_balance=account.balance if account else None,
                account_currency=account.currency_code if account else None,
                account_last_sync=account.last_sync_timestamp if account else None,
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
                future=raw_tx.future,
            )
            transaction_inputs.append(tx_input)
        
        logger.info(f"📊 Synchronisation de {len(transaction_inputs)} transactions pour l'utilisateur {user_id}")

        # Synchroniser via le processeur Elasticsearch
        result = await processor.sync_user_transactions(
            user_id=user_id,
            transactions=transaction_inputs,
            accounts=accounts,
            accounts_map=accounts_map,
            force_refresh=force_refresh,
        )

        # ✅ LOG UNIQUE - Suppression des doublons
        logger.info(
            f"📈 Sync user {user_id} completed: {result.accounts_indexed} accounts, {result.transactions_indexed} transactions indexed in {result.processing_time:.3f}s"
        )

        # 🔄 Invalider le cache de recherche si la sync a réussi
        if result.status in ["success", "partial_success"] and result.transactions_indexed > 0:
            if authorization and authorization.startswith("Bearer "):
                token = authorization.replace("Bearer ", "")
                await invalidate_search_cache(user_id, token)
            else:
                logger.warning(f"⚠️ Impossible d'invalider le cache: token d'autorisation manquant")

        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la synchronisation pour l'utilisateur {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync user transactions: {str(e)}"
        )

@router.delete("/elasticsearch/user-data/{user_id}")
async def delete_user_data(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    processor: ElasticsearchTransactionProcessor = Depends(get_elasticsearch_processor),
):
    """
    Supprime toutes les données d'un utilisateur d'Elasticsearch.
    
    ATTENTION: Cette opération est irréversible!
    
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
        deletion_result = await processor.delete_user_data(user_id)
        
        if deletion_result["status"] == "success":
            return {
                "status": "success",
                "message": f"All data deleted for user {user_id}",
                "user_id": user_id,
                "deleted_count": deletion_result["deleted_count"],
                "storage": "elasticsearch",
                "timestamp": deletion_result["timestamp"]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete user data: {deletion_result.get('error', 'Unknown error')}"
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
# ENDPOINTS DE DIAGNOSTIC ET MONITORING
# ==========================================

@router.get("/elasticsearch/health")
async def elasticsearch_health(
    processor: ElasticsearchTransactionProcessor = Depends(get_elasticsearch_processor),
):
    """
    Vérifie la santé du service d'enrichissement Elasticsearch.
    
    Retourne l'état d'Elasticsearch et du processeur.
    """
    try:
        health_status = await processor.health_check()
        if health_status["status"] != "healthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=health_status,
            )

        return ElasticsearchHealthStatus(
            status=health_status["status"],
            timestamp=health_status["timestamp"],
            elasticsearch=health_status["elasticsearch"],
            database=health_status.get("database"),
            capabilities=health_status["capabilities"],
            performance_metrics=health_status.get("performance_metrics"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

@router.get("/elasticsearch/user-stats/{user_id}")
async def get_user_statistics(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    processor: ElasticsearchTransactionProcessor = Depends(get_elasticsearch_processor),
):
    """
    Récupère les statistiques d'un utilisateur dans Elasticsearch.
    
    Args:
        user_id: ID de l'utilisateur
        
    Returns:
        Dict: Statistiques de l'utilisateur
    """
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        stats = await processor.get_user_stats(user_id)
        
        if stats["status"] == "success":
            return stats
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get user statistics: {stats.get('error', 'Unknown error')}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur récupération stats user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user statistics: {str(e)}"
        )

@router.get("/elasticsearch/cluster-info")
async def get_cluster_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère les informations du cluster Elasticsearch.
    
    Réservé aux superusers pour diagnostics avancés.
    
    Returns:
        Dict: Informations du cluster
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions - superuser required"
        )
    
    try:
        global elasticsearch_client
        if not elasticsearch_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Elasticsearch client not available"
            )
        
        # Obtenir les infos du cluster
        cluster_info = await elasticsearch_client.get_cluster_info()
        
        return {
            "cluster_info": cluster_info,
            "index_name": elasticsearch_client.index_name,
            "client_initialized": elasticsearch_client._initialized,
            "timestamp": datetime.now().isoformat(),
            "service": "enrichment_service_elasticsearch"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos cluster: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster info: {str(e)}"
        )

@router.get("/elasticsearch/index-mapping")
async def get_index_mapping(
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère le mapping de l'index Elasticsearch.
    
    Réservé aux superusers pour diagnostics avancés.
    
    Returns:
        Dict: Mapping de l'index
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions - superuser required"
        )
    
    try:
        global elasticsearch_client
        if not elasticsearch_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Elasticsearch client not available"
            )
        
        # Récupérer le mapping
        async with elasticsearch_client.session.get(
            f"{elasticsearch_client.base_url}/{elasticsearch_client.index_name}/_mapping"
        ) as response:
            if response.status == 200:
                mapping = await response.json()
                return {
                    "index_name": elasticsearch_client.index_name,
                    "mapping": mapping,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Failed to get mapping: {response.status}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du mapping: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index mapping: {str(e)}"
        )

# ==========================================
# ENDPOINTS UTILITAIRES
# ==========================================

@router.post("/elasticsearch/reindex-user/{user_id}")
async def reindex_user_transactions(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    processor: ElasticsearchTransactionProcessor = Depends(get_elasticsearch_processor),
):
    """
    Force la ré-indexation complète d'un utilisateur.
    
    Équivalent à sync-user avec force_refresh=True.
    Utile pour corriger des problèmes d'indexation.
    
    Args:
        user_id: ID de l'utilisateur à ré-indexer
        
    Returns:
        UserSyncResult: Résultat de la ré-indexation
    """
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    logger.info(f"🔄 Ré-indexation complète demandée pour user {user_id}")
    
    # Utiliser l'endpoint de sync avec force_refresh=True
    return await sync_user_transactions(
        user_id=user_id,
        force_refresh=True,
        current_user=current_user,
        db=db,
        processor=processor,
    )

@router.get("/elasticsearch/document-exists/{user_id}/{transaction_id}")
async def check_document_exists(
    user_id: int,
    transaction_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """
    Vérifie si un document existe dans Elasticsearch.
    
    Utile pour déboguer des problèmes d'indexation.
    
    Args:
        user_id: ID de l'utilisateur
        transaction_id: ID de la transaction
        
    Returns:
        Dict: Existence du document
    """
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        global elasticsearch_client
        if not elasticsearch_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Elasticsearch client not available"
            )
        
        document_id = f"user_{user_id}_tx_{transaction_id}"
        exists = await elasticsearch_client.document_exists(document_id)
        
        return {
            "user_id": user_id,
            "transaction_id": transaction_id,
            "document_id": document_id,
            "exists": exists,
            "index_name": elasticsearch_client.index_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur vérification existence document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check document existence: {str(e)}"
        )

# ==========================================
# ENDPOINTS ENRICHISSEMENT MARCHANDS PAR LOT
# ==========================================

@router.post("/merchant-batch-enrichment/{user_id}", response_model=BatchMerchantEnrichmentResult)
async def enrich_user_merchants_batch(
    user_id: int,
    limit: Optional[int] = Query(None, description="Limite du nombre de transactions à traiter"),
    only_missing: bool = Query(True, description="Ne traiter que les transactions sans merchant_name"),
    confidence_threshold: Optional[float] = Query(None, description="Seuil de confiance (défaut 0.3)"),
    current_user: User = Depends(get_current_active_user),
    merchant_service: MerchantBatchEnrichmentService = Depends(get_merchant_batch_service),
):
    """
    Enrichit les noms de marchands pour un utilisateur par lot avec Deepseek LLM.
    
    Cette méthode est spécialement conçue pour l'enrichissement économique :
    1. Traite les transactions par petits lots pour éviter les timeouts
    2. Ajoute des délais entre requêtes pour éviter rate limiting
    3. Estime les coûts de traitement
    4. Met à jour uniquement les documents Elasticsearch avec succès
    
    Args:
        user_id: ID de l'utilisateur à traiter
        limit: Limite du nombre de transactions (optionnel)
        only_missing: Ne traiter que les transactions sans merchant_name
        confidence_threshold: Seuil de confiance personnalisé
        
    Returns:
        BatchMerchantEnrichmentResult: Résultat complet du traitement
    """
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    logger.info(f"🤖 Enrichissement marchands par lot demandé pour user {user_id}")
    
    try:
        result = await merchant_service.enrich_user_merchants(
            user_id=user_id,
            limit=limit,
            only_missing=only_missing,
            confidence_threshold=confidence_threshold
        )
        
        logger.info(
            f"🎉 Enrichissement terminé: {result.successful_extractions}/{result.total_transactions} "
            f"extractions réussies en {result.total_processing_time:.2f}s - "
            f"Coût estimé: ${result.cost_estimate['estimated_cost_usd']}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur enrichissement marchands user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enrich user merchants: {str(e)}"
        )

@router.get("/merchant-batch-enrichment/{user_id}/preview")
async def preview_merchant_enrichment(
    user_id: int,
    limit: int = Query(10, description="Nombre de transactions à prévisualiser"),
    only_missing: bool = Query(True, description="Ne traiter que les transactions sans merchant_name"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Prévisualise les transactions qui seraient traitées par l'enrichissement par lot.
    
    Utile pour estimer les coûts et vérifier les données avant traitement.
    
    Args:
        user_id: ID de l'utilisateur
        limit: Nombre de transactions à prévisualiser
        only_missing: Filtre pour transactions sans merchant_name
        
    Returns:
        Dict: Aperçu des transactions et estimation des coûts
    """
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        # Récupérer un échantillon de transactions
        query = db.query(RawTransaction).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.deleted == False
        )
        
        # Filtrer les transactions avec description
        query = query.filter(
            (RawTransaction.clean_description.isnot(None)) |
            (RawTransaction.provider_description.isnot(None))
        )
        
        if only_missing:
            # TODO: Ajouter filtre Elasticsearch pour ne prendre que celles sans merchant_name
            pass
        
        transactions = query.order_by(RawTransaction.date.desc()).limit(limit).all()
        
        # Compter le total sans limite
        total_count = query.count()
        
        # Estimation des coûts (approximative)
        cost_per_request = 0.0001  # ~0.01 centime par requête
        estimated_total_cost = total_count * cost_per_request
        
        preview_data = []
        for tx in transactions:
            description = tx.clean_description or tx.provider_description or ""
            preview_data.append({
                "transaction_id": tx.bridge_transaction_id,
                "description": description,
                "amount": tx.amount,
                "date": tx.date.isoformat(),
                "currency": tx.currency_code
            })
        
        return {
            "user_id": user_id,
            "preview_transactions": preview_data,
            "preview_count": len(preview_data),
            "total_eligible_transactions": total_count,
            "cost_estimate": {
                "total_transactions": total_count,
                "estimated_cost_usd": round(estimated_total_cost, 4),
                "cost_per_extraction": cost_per_request,
                "currency": "USD"
            },
            "filters_applied": {
                "only_missing": only_missing,
                "has_description": True
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur prévisualisation enrichissement user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to preview merchant enrichment: {str(e)}"
        )
