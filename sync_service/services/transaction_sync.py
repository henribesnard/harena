"""
Service de synchronisation des transactions financières.

Ce module gère la synchronisation et le stockage vectoriel des transactions
bancaires récupérées depuis l'API Bridge.
"""
import logging
import httpx
import re
import hashlib
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional, Union, Set
from fastapi import HTTPException

# Imports depuis user_service
from user_service.models.user import User, BridgeConnection
from user_service.services.bridge import get_bridge_token, get_bridge_transactions
from config_service.config import settings

# Import VectorStorageService avec gestion d'erreur
try:
    from sync_service.services.vector_storage import VectorStorageService
    VECTOR_STORAGE_AVAILABLE = True
except ImportError as e:
    VECTOR_STORAGE_AVAILABLE = False
    # Classe factice pour éviter les erreurs AttributeError
    class VectorStorageService:
        async def batch_store_transactions(self, *args, **kwargs): 
            logging.getLogger(__name__).warning("VectorStorage indispo: batch_store_transactions ignoré.")
            return {"status": "skipped"}
        async def find_merchant(self, *args, **kwargs): 
            logging.getLogger(__name__).warning("VectorStorage indispo: find_merchant ignoré.")
            return None
        async def store_merchant(self, *args, **kwargs): 
            logging.getLogger(__name__).warning("VectorStorage indispo: store_merchant ignoré.")
            return None
        async def get_user_statistics(self, *args, **kwargs): 
            logging.getLogger(__name__).warning("VectorStorage indispo: get_user_statistics ignoré.")
            return {"status": "unavailable"}

# Import des modèles sync
from sync_service.models.sync import SyncAccount, SyncItem

# Configuration du logger
logger = logging.getLogger(__name__)

# --- Fonction d'enrichissement marchand ---

def _normalize_merchant_name(description: str) -> Optional[str]:
    """
    Tente d'extraire et de normaliser un nom de marchand à partir d'une description.
    Logique basique, à améliorer en production.
    """
    if not description:
        return None
    # Supprimer les dates, numéros de carte, etc.
    name = re.sub(r'\d{2,}/\d{2,}/\d{2,}', '', description).strip()
    name = re.sub(r'CB\*?\s*\d+', '', name).strip()
    name = re.sub(r'CARTE \d+', '', name).strip()
    name = re.sub(r'PAIEMENT\s*(CB|CARTE)?\s*', '', name, flags=re.IGNORECASE).strip()
    name = re.sub(r'\b\d{4,}\b', '', name).strip()  # Enlever longs nombres
    name = re.sub(r'\s{2,}', ' ', name).strip()  # Espaces multiples

    # Mettre en majuscules pour normalisation
    normalized = name.upper()
    # Remplacer caractères spéciaux ou chaînes communes
    normalized = normalized.replace('*', '').replace('.', '').replace(',', '').strip()
    normalized = normalized.replace('AMAZON MKTPLACE', 'AMAZON')
    normalized = normalized.replace('AMZN', 'AMAZON')
    
    # Prendre les premiers mots jusqu'à une certaine longueur
    parts = normalized.split()
    max_len = 25
    result = ""
    for part in parts:
        if len(result) + len(part) + (1 if result else 0) <= max_len:
            result += (" " if result else "") + part
        else:
            break
    result = result.strip()

    if len(result) < 3:  # Éviter les noms trop courts
        return None

    return result if result else None


async def _get_or_create_merchant_vector(
    vector_storage: VectorStorageService,
    user_id: int,
    description: str,
    category_id: Optional[int]
) -> Optional[str]:
    """
    Tente de trouver ou créer un marchand dans la collection vectorielle.
    Retourne l'ID du point marchand Qdrant si trouvé/créé, sinon None.
    """
    if not VECTOR_STORAGE_AVAILABLE or not vector_storage:
        return None

    normalized_name = _normalize_merchant_name(description)
    if not normalized_name:
        return None

    # 1. Chercher le marchand existant par nom normalisé
    merchant_point_id = await vector_storage.find_merchant(normalized_name=normalized_name)

    if merchant_point_id:
        logger.debug(f"Marchand existant trouvé pour '{normalized_name}': {merchant_point_id}")
        return merchant_point_id
    else:
        # 2. Si non trouvé, créer un nouveau marchand
        logger.info(f"Création d'un nouveau marchand pour '{normalized_name}' (depuis: '{description[:30]}...')")
        merchant_data = {
            "normalized_name": normalized_name,
            "display_name": normalized_name,
            "category_id": category_id,
            "source": "inferred_from_transaction"
        }
        new_merchant_point_id = await vector_storage.store_merchant(merchant_data)
        if new_merchant_point_id:
            logger.info(f"Nouveau marchand créé avec point_id: {new_merchant_point_id}")
            return new_merchant_point_id
        else:
            logger.error(f"Échec de la création du nouveau marchand pour '{normalized_name}'")
            return None


# --- Méthodes Principales ---

async def sync_account_transactions(db: Session, sync_account: SyncAccount) -> Dict[str, Any]:
    """
    Synchronise les transactions d'un compte depuis Bridge API,
    enrichit potentiellement avec les marchands,
    et stocke les transactions dans le Vector Store.
    """
    bridge_account_id = sync_account.bridge_account_id
    ctx_logger = logging.LoggerAdapter(logger, {"bridge_account_id": bridge_account_id, "sql_account_id": sync_account.id})
    ctx_logger.info(f"Début synchronisation transactions pour compte {bridge_account_id}")

    result_summary = {
        "status": "pending",
        "new_transactions": 0,
        "updated_transactions": 0,
        "processed_transactions": 0,
        "vector_storage_status": "unknown",
        "merchant_enrichment_attempts": 0,
        "merchants_found": 0,
        "merchants_created": 0,
        "errors": None
    }

    try:
        # 1. Récupérer l'utilisateur et l'item
        sync_item = db.query(SyncItem).filter(SyncItem.id == sync_account.item_id).first()
        if not sync_item:
            ctx_logger.error(f"SyncItem non trouvé (id={sync_account.item_id}) pour le compte {bridge_account_id}")
            result_summary["status"] = "error"
            result_summary["errors"] = "Parent SyncItem not found"
            return result_summary
        
        user_id = sync_item.user_id
        ctx_logger = logging.LoggerAdapter(logger, {
            "bridge_account_id": bridge_account_id, 
            "sql_account_id": sync_account.id, 
            "user_id": user_id
        })
        ctx_logger.debug(f"SyncItem trouvé: id={sync_item.id}, user_id={user_id}")

        # 2. Récupérer le token Bridge
        token_data = await get_bridge_token(db, user_id)
        ctx_logger.debug(f"Token Bridge récupéré avec succès")

        # 3. Déterminer la date 'since' pour la requête Bridge
        since_date = None
        # Prendre une marge de sécurité (3 jours) pour récupérer les transactions potentiellement mises à jour
        safety_margin = timedelta(days=3)
        last_effective_date = sync_account.last_transaction_date or sync_account.last_sync_timestamp
        
        if last_effective_date:
            since_date = last_effective_date - safety_margin
            ctx_logger.info(f"Synchronisation depuis {since_date} (dernière date: {last_effective_date})")
        else:
            # Première synchronisation pour ce compte, prendre 90 jours par défaut
            since_date = datetime.now(timezone.utc) - timedelta(days=90)
            ctx_logger.info(f"Première synchronisation, récupération depuis {since_date}")

        # 4. Récupérer les transactions depuis Bridge
        ctx_logger.info(f"Appel API Bridge pour transactions du compte {bridge_account_id} depuis {since_date}")
        bridge_transactions = await get_bridge_transactions(db, user_id, bridge_account_id, since=since_date)
        result_summary["processed_transactions"] = len(bridge_transactions)
        ctx_logger.info(f"{len(bridge_transactions)} transactions récupérées de Bridge API.")

        if not bridge_transactions:
            result_summary["status"] = "success"  # Succès, mais aucune nouvelle transaction
            result_summary["vector_storage_status"] = "not_attempted"
            # Mettre à jour le timestamp de synchro même si 0 transaction
            sync_account.last_sync_timestamp = datetime.now(timezone.utc)
            db.add(sync_account)
            db.commit()
            ctx_logger.info(f"Aucune transaction à synchroniser, timestamp mis à jour: {sync_account.last_sync_timestamp}")
            return result_summary

        # 5. Préparer et stocker les transactions vectoriellement
        ctx_logger.info(f"Préparation des {len(bridge_transactions)} transactions pour le stockage vectoriel")
        vector_transactions_to_store = []
        failed_txs = []
        max_updated_at = last_effective_date or datetime.min.replace(tzinfo=timezone.utc)
        categories_set = set()  # Pour suivre les catégories trouvées
        operation_types = set()  # Pour suivre les types d'opérations

        vector_storage = None
        if VECTOR_STORAGE_AVAILABLE:
            vector_storage = VectorStorageService()
            ctx_logger.debug(f"Service de stockage vectoriel initialisé")
        else:
            ctx_logger.warning(f"Service de stockage vectoriel indisponible!")

        for tx in bridge_transactions:
            try:
                transaction_id_str = str(tx.get("id"))
                if not transaction_id_str or user_id is None:
                    ctx_logger.warning(f"ID transaction Bridge ou User ID manquant, skipping")
                    failed_txs.append({"id": tx.get("id"), "reason": "missing_id"})
                    continue

                # Collecter des statistiques
                category_id = tx.get("category_id")
                if category_id:
                    categories_set.add(category_id)
                
                op_type = tx.get("operation_type")
                if op_type:
                    operation_types.add(op_type)

                # Préparer la transaction pour le stockage vectoriel
                description = tx.get("clean_description") or tx.get("description", "")
                merchant_point_id = None
                # Si enrichissement marchand est activé:
                # result_summary["merchant_enrichment_attempts"] += 1
                # merchant_point_id = await _get_or_create_merchant_vector(
                #     vector_storage, user_id, description, category_id
                # )
                # if merchant_point_id:
                #     result_summary["merchants_found"] += 1

                vector_tx = {
                    "user_id": user_id,
                    "account_id": bridge_account_id,
                    "bridge_transaction_id": transaction_id_str,
                    "amount": tx.get("amount", 0.0),
                    "currency_code": tx.get("currency_code", "EUR"),
                    "description": tx.get("description", ""),
                    "clean_description": description,
                    "transaction_date": tx.get("date"),
                    "booking_date": tx.get("booking_date"),
                    "value_date": tx.get("value_date"),
                    "category_id": category_id,
                    "operation_type": op_type,
                    "is_recurring": tx.get("recurring", {}).get("is"),
                    "merchant_id": merchant_point_id,
                    "bridge_updated_at": tx.get("updated_at")
                }
                vector_transactions_to_store.append(vector_tx)

                # Mettre à jour la date max traitée
                try:
                    tx_updated_at_str = tx.get("updated_at")
                    if tx_updated_at_str:
                        tx_updated_at = datetime.fromisoformat(tx_updated_at_str.replace('Z', '+00:00'))
                        if tx_updated_at > max_updated_at:
                            max_updated_at = tx_updated_at
                except Exception as date_err:
                    ctx_logger.warning(f"Impossible de parser updated_at '{tx_updated_at_str}' pour tx {transaction_id_str}: {date_err}")
            except Exception as tx_prep_error:
                ctx_logger.error(f"Erreur lors de la préparation de la transaction {tx.get('id')}: {tx_prep_error}")
                failed_txs.append({"id": tx.get("id"), "reason": "preparation_error"})

        # Logging des statistiques de transactions préparées
        ctx_logger.info(f"Transactions préparées: {len(vector_transactions_to_store)} sur {len(bridge_transactions)}. " 
                      f"Catégories distinctes: {len(categories_set)}, Types d'opérations: {len(operation_types)}")

        # 6. Stocker les transactions dans le Vector Store
        if vector_storage and vector_transactions_to_store:
            ctx_logger.info(f"Stockage vectoriel de {len(vector_transactions_to_store)} transactions...")
            vector_result = await vector_storage.batch_store_transactions(vector_transactions_to_store)
            ctx_logger.info(f"Résultat stockage vectoriel: {vector_result}")
            result_summary["vector_storage_status"] = vector_result.get("status", "error")
            result_summary["new_transactions"] = vector_result.get("successful", 0)
            
            if vector_result.get("status") != "success":
                result_summary["status"] = "partial" if vector_result.get("successful", 0) > 0 else "error"
                result_summary["errors"] = f"Vector storage failed for {vector_result.get('failed', 0)} transactions. Details: {str(failed_txs[:5]) if failed_txs else 'None'}"
            else:
                result_summary["status"] = "success"
        elif not vector_storage:
            ctx_logger.warning("Vector storage indisponible - les transactions ne seront pas stockées vectoriellement")
            result_summary["vector_storage_status"] = "unavailable"
            result_summary["status"] = "success"  # Succès car pas d'erreur, mais pas de stockage vectoriel
        else:
            # Pas de transactions à stocker
            ctx_logger.info("Aucune transaction à stocker dans le stockage vectoriel")
            result_summary["vector_storage_status"] = "no_transactions_to_store"
            result_summary["status"] = "success"

        # 7. Mettre à jour les timestamps du compte SQL
        now = datetime.now(timezone.utc)
        sync_account.last_sync_timestamp = now
        # Utiliser max_updated_at trouvé comme date de dernière transaction
        if max_updated_at > (sync_account.last_transaction_date or datetime.min.replace(tzinfo=timezone.utc)):
            sync_account.last_transaction_date = max_updated_at
            ctx_logger.debug(f"Mise à jour de last_transaction_date à {max_updated_at}")

        db.add(sync_account)
        db.commit()
        db.refresh(sync_account)

        # Ajouter des statistiques de catégories et types d'opérations au résultat
        result_summary["stats"] = {
            "categories_count": len(categories_set),
            "operation_types_count": len(operation_types),
            "categories": list(categories_set)[:10] if len(categories_set) <= 10 else f"{len(categories_set)} categories",
            "operation_types": list(operation_types)
        }

        ctx_logger.info(f"Synchronisation terminée pour compte {bridge_account_id}: status={result_summary['status']}, "
                      f"nouvelles transactions={result_summary['new_transactions']}, "
                      f"catégories={len(categories_set)}, "
                      f"types d'opérations={len(operation_types)}")
        return result_summary

    except HTTPException as http_exc:
        # Erreur lors de l'appel à Bridge API
        ctx_logger.error(f"Erreur HTTP {http_exc.status_code} durant sync transactions pour compte {bridge_account_id}: {http_exc.detail}")
        result_summary["status"] = "error"
        result_summary["errors"] = f"API Error: {http_exc.status_code} - {http_exc.detail}"
        return result_summary
    except Exception as e:
        ctx_logger.error(f"Erreur inattendue durant sync transactions pour compte {bridge_account_id}: {e}", exc_info=True)
        result_summary["status"] = "error"
        result_summary["errors"] = f"Unexpected error: {str(e)}"
        db.rollback()  # Rollback des changements potentiels sur sync_account
        return result_summary


async def force_sync_all_accounts(db: Session, user_id: int) -> Dict[str, Any]:
    """Force la synchronisation des transactions pour tous les comptes actifs d'un utilisateur."""
    ctx_logger = logging.LoggerAdapter(logger, {"user_id": user_id})
    ctx_logger.info(f"Forçage de la synchronisation des transactions pour tous les comptes de l'utilisateur {user_id}")

    try:
        # Récupérer tous les comptes SQL actifs
        accounts_to_sync = db.query(SyncAccount).join(SyncItem).filter(
            SyncItem.user_id == user_id,
        ).all()

        if not accounts_to_sync:
            ctx_logger.warning(f"Aucun compte trouvé pour le forçage de synchronisation de l'utilisateur {user_id}")
            return {
                "status": "warning", 
                "message": "No accounts found for this user", 
                "accounts_processed": 0, 
                "details": []
            }

        ctx_logger.info(f"{len(accounts_to_sync)} comptes trouvés pour l'utilisateur {user_id}. Démarrage synchro individuelle.")

        results = []
        accounts_with_errors = 0
        total_new_tx = 0
        total_accounts_processed = 0
        categories_found = set()

        for account in accounts_to_sync:
            try:
                total_accounts_processed += 1
                ctx_logger.info(f"Synchronisation du compte {account.bridge_account_id} ({total_accounts_processed}/{len(accounts_to_sync)})")
                account_result = await sync_account_transactions(db, account)
                
                # Collecter les statistiques globales
                if account_result.get("stats", {}).get("categories"):
                    cats = account_result.get("stats", {}).get("categories")
                    if isinstance(cats, list): 
                        categories_found.update(cats)
                
                results.append({
                    "bridge_account_id": account.bridge_account_id,
                    "status": account_result.get("status"),
                    "new_transactions": account_result.get("new_transactions", 0),
                    "account_type": account.account_type,
                    "account_name": account.account_name
                })
                
                if account_result.get("status") != "success":
                    accounts_with_errors += 1
                    ctx_logger.warning(f"Synchronisation du compte {account.bridge_account_id} terminée avec des erreurs: {account_result.get('errors')}")
                else:
                    ctx_logger.info(f"Synchronisation du compte {account.bridge_account_id} réussie: {account_result.get('new_transactions', 0)} nouvelles transactions")
                
                total_new_tx += account_result.get("new_transactions", 0)
            except Exception as e:
                ctx_logger.error(f"Erreur lors de la synchronisation du compte {account.bridge_account_id}: {e}", exc_info=True)
                accounts_with_errors += 1
                results.append({
                    "bridge_account_id": account.bridge_account_id,
                    "status": "error",
                    "error": str(e),
                    "account_type": account.account_type,
                    "account_name": account.account_name
                })

        # Déterminer le statut global
        overall_status = "success"
        if accounts_with_errors > 0:
            overall_status = "partial" if accounts_with_errors < len(accounts_to_sync) else "error"

        ctx_logger.info(f"Forçage de synchronisation terminé pour user {user_id}. "
                      f"Statut: {overall_status}, "
                      f"{total_new_tx} nouvelles transactions stockées, "
                      f"{len(categories_found)} catégories distinctes trouvées, "
                      f"{accounts_with_errors}/{len(accounts_to_sync)} comptes avec erreurs.")

        return {
            "status": overall_status,
            "accounts_processed": len(accounts_to_sync),
            "accounts_with_errors": accounts_with_errors,
            "total_new_transactions_stored": total_new_tx,
            "categories_found": len(categories_found),
            "details": results
        }

    except Exception as e:
        ctx_logger.error(f"Erreur majeure lors du forçage de synchronisation pour user {user_id}: {str(e)}", exc_info=True)
        return {
            "status": "critical_error", 
            "message": str(e), 
            "accounts_processed": 0, 
            "details": []
        }


async def check_and_sync_missing_transactions(db: Session, user_id: int) -> Dict[str, Any]:
    """
    Vérifie l'état de synchronisation et lance une synchronisation forcée si nécessaire.
    """
    ctx_logger = logging.LoggerAdapter(logger, {"user_id": user_id})
    ctx_logger.info(f"Vérification et synchronisation potentielle des transactions manquantes pour l'utilisateur {user_id}")

    try:
        # 1. Vérifier s'il y a des comptes à synchroniser
        accounts_count = db.query(SyncAccount).join(SyncItem).filter(
            SyncItem.user_id == user_id
        ).count()
        
        if accounts_count == 0:
            ctx_logger.info(f"Aucun compte trouvé. Pas de synchronisation forcée nécessaire.")
            return {
                "status": "not_needed", 
                "message": "No accounts found."
            }

        # 2. Vérifier l'état de synchronisation actuel
        vector_stats = None
        if VECTOR_STORAGE_AVAILABLE:
            try:
                vector_storage = VectorStorageService()
                vector_stats = await vector_storage.get_user_statistics(user_id)
                ctx_logger.info(f"Statistiques vectorielles actuelles: {vector_stats}")
            except Exception as vs_error:
                ctx_logger.error(f"Erreur lors de la récupération des statistiques vectorielles: {str(vs_error)}")
                # Continuer quand même avec la synchronisation

        # 3. Lancer une synchronisation complète
        ctx_logger.info(f"Synchronisation forcée nécessaire. Appel de force_sync_all_accounts.")
        result = await force_sync_all_accounts(db, user_id)
        
        # Ajouter les statistiques vectorielles au résultat
        if vector_stats:
            result["vector_stats_before_sync"] = vector_stats
            
            # Récupérer les nouvelles statistiques après synchronisation
            try:
                after_stats = await vector_storage.get_user_statistics(user_id)
                result["vector_stats_after_sync"] = after_stats
                
                # Calculer la différence
                if "transactions_count" in after_stats and "transactions_count" in vector_stats:
                    tx_before = vector_stats.get("transactions_count", 0)
                    tx_after = after_stats.get("transactions_count", 0)
                    result["vector_transactions_added"] = tx_after - tx_before
                    ctx_logger.info(f"Différence de transactions: {result['vector_transactions_added']} (de {tx_before} à {tx_after})")
            except Exception as e:
                ctx_logger.error(f"Erreur lors de la récupération des statistiques après synchronisation: {e}")
        
        ctx_logger.info(f"Résultat de la synchronisation forcée: {result.get('status')}, {result.get('total_new_transactions_stored', 0)} nouvelles transactions")
        return result

    except Exception as e:
        ctx_logger.error(f"Erreur lors de check_and_sync_missing_transactions pour user {user_id}: {str(e)}", exc_info=True)
        return {
            "status": "error", 
            "message": f"Failed checking/syncing missing transactions: {str(e)}"
        }


async def get_user_vector_stats(db: Session, user_id: int) -> Dict[str, Any]:
    """Récupère les statistiques vectorielles pour un utilisateur via VectorStorageService."""
    ctx_logger = logging.LoggerAdapter(logger, {"user_id": user_id})
    ctx_logger.info(f"Récupération des statistiques vectorielles pour l'utilisateur {user_id}")

    if not VECTOR_STORAGE_AVAILABLE:
        ctx_logger.warning("Vector storage indisponible, impossible de récupérer les stats.")
        return {
            "user_id": user_id, 
            "status": "unavailable", 
            "message": "Vector storage service not available."
        }

    try:
        vector_storage = VectorStorageService()
        stats = await vector_storage.get_user_statistics(user_id)
        
        # Enrichir les statistiques avec des informations supplémentaires
        sql_info = {}
        try:
            # Compter les éléments SQL associés (items, accounts)
            items_count = db.query(SyncItem).filter(SyncItem.user_id == user_id).count()
            accounts_count = db.query(SyncAccount).join(SyncItem).filter(SyncItem.user_id == user_id).count()
            sql_info = {
                "sql_items_count": items_count,
                "sql_accounts_count": accounts_count
            }
        except Exception as sql_err:
            ctx_logger.warning(f"Erreur lors de la récupération des statistiques SQL: {sql_err}")
            
        # Combiner les statistiques
        combined_stats = {**stats, **sql_info}
        
        ctx_logger.info(f"Statistiques vectorielles récupérées: {combined_stats}")
        return combined_stats
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la récupération des statistiques vectorielles pour user {user_id}: {str(e)}", exc_info=True)
        return {
            "user_id": user_id,
            "status": "error",
            "message": f"Failed to retrieve vector stats: {str(e)}"
        }