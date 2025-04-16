# sync_service/services/transaction_sync.py
import logging
import httpx
import json
import re # Importé pour l'enrichissement marchand potentiel
import hashlib # Importé pour l'enrichissement marchand potentiel
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional, Union
from fastapi import HTTPException
import traceback

# Imports Models
from sync_service.models.sync import SyncAccount, SyncItem
from user_service.models.user import User, BridgeConnection

# Imports Services
from user_service.services import bridge as bridge_service
from user_service.core.config import settings
# Utiliser VectorStorageService avec gestion d'erreur
try:
    from .vector_storage import VectorStorageService
    VECTOR_STORAGE_AVAILABLE = True
    logger_vs = logging.getLogger(__name__)
except ImportError as e:
    VECTOR_STORAGE_AVAILABLE = False
    class VectorStorageService: # Classe factice
        async def batch_store_transactions(self, *args, **kwargs): logger_vs.warning("TS: VectorStorage indispo: batch_store_transactions ignoré."); return {"status": "skipped"}
        async def find_merchant(self, *args, **kwargs): logger_vs.warning("TS: VectorStorage indispo: find_merchant ignoré."); return None
        async def store_merchant(self, *args, **kwargs): logger_vs.warning("TS: VectorStorage indispo: store_merchant ignoré."); return None
        async def get_user_statistics(self, *args, **kwargs): logger_vs.warning("TS: VectorStorage indispo: get_user_statistics ignoré."); return {"status": "unavailable"}
        # Ajouter d'autres méthodes factices si nécessaire
    logger_vs = logging.getLogger(__name__)
    logger_vs.warning(f"TransactionSync: VectorStorageService non trouvé ({e}). Stockage vectoriel désactivé.")

# Importer sync_manager pour get_user_sync_status
from . import sync_manager

# Configuration du logger
logger = logging.getLogger(__name__)

# --- Fonction d'enrichissement marchand (Optionnelle) ---

def _normalize_merchant_name(description: str) -> Optional[str]:
    """
    Tente d'extraire et de normaliser un nom de marchand à partir d'une description.
    Logique très basique, à améliorer considérablement en production.
    """
    if not description:
        return None
    # Supprimer les dates, numéros de carte, etc. (simpliste)
    name = re.sub(r'\d{2,}/\d{2,}/\d{2,}', '', description).strip()
    name = re.sub(r'CB\*?\s*\d+', '', name).strip()
    name = re.sub(r'CARTE \d+', '', name).strip()
    name = re.sub(r'PAIEMENT\s*(CB|CARTE)?\s*', '', name, flags=re.IGNORECASE).strip()
    name = re.sub(r'\b\d{4,}\b', '', name).strip() # Enlever longs nombres
    name = re.sub(r'\s{2,}', ' ', name).strip() # Espaces multiples

    # Mettre en majuscules pour normalisation
    normalized = name.upper()
    # Remplacer caractères spéciaux ou chaînes communes (exemples)
    normalized = normalized.replace('*', '').replace('.', '').replace(',', '').strip()
    normalized = normalized.replace('AMAZON MKTPLACE', 'AMAZON')
    normalized = normalized.replace('AMZN', 'AMAZON')
    # Garder seulement les N premiers mots significatifs ? Ou limiter longueur ?
    parts = normalized.split()
    # Logique très basique : prendre les premiers mots jusqu'à une certaine longueur
    max_len = 25
    result = ""
    for part in parts:
         if len(result) + len(part) + (1 if result else 0) <= max_len:
              result += (" " if result else "") + part
         else:
              break
    result = result.strip()

    if len(result) < 3: # Éviter les noms trop courts/insignifiants
         return None

    return result if result else None

async def _get_or_create_merchant_vector(
    vector_storage: VectorStorageService,
    user_id: int, # Si les marchands sont spécifiques à l'utilisateur
    description: str,
    category_id: Optional[int]
) -> Optional[str]:
    """
    Tente de trouver ou créer un marchand dans la collection vectorielle.
    Retourne l'ID du point marchand Qdrant si trouvé/créé, sinon None.
    NOTE : C'est une implémentation basique. L'enrichissement est complexe.
    """
    if not VECTOR_STORAGE_AVAILABLE or not vector_storage:
        return None

    normalized_name = _normalize_merchant_name(description)
    if not normalized_name:
        return None

    # 1. Chercher le marchand existant par nom normalisé
    # Note: La recherche par nom seul peut être imprécise.
    # Idéalement, on utiliserait une combinaison nom + catégorie, ou un service externe.
    merchant_point_id = await vector_storage.find_merchant(normalized_name=normalized_name) # Adaptez find_merchant si user_id est nécessaire

    if merchant_point_id:
        logger.debug(f"Marchand existant trouvé pour '{normalized_name}': {merchant_point_id}")
        return merchant_point_id
    else:
        # 2. Si non trouvé, créer un nouveau marchand
        logger.info(f"Création d'un nouveau marchand pour '{normalized_name}' (depuis: '{description[:30]}...')")
        merchant_data = {
            "normalized_name": normalized_name,
            "display_name": normalized_name, # Utiliser le nom normalisé comme display name par défaut
            "category_id": category_id,
            # "user_id": user_id, # Décommenter si la collection Merchant a user_id
            "source": "inferred_from_transaction"
        }
        new_merchant_point_id = await vector_storage.store_merchant(merchant_data)
        if new_merchant_point_id:
             logger.info(f"Nouveau marchand créé avec point_id: {new_merchant_point_id}")
             return new_merchant_point_id
        else:
             logger.error(f"Échec de la création du nouveau marchand pour '{normalized_name}'")
             return None


# --- Fonctions Principales ---

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
        "updated_transactions": 0, # Non géré explicitement ici, upsert remplace
        "processed_transactions": 0,
        "vector_storage_status": "unknown",
        "merchant_enrichment_attempts": 0,
        "merchants_found": 0,
        "merchants_created": 0,
        "errors": None
    }

    try:
        # 1. Récupérer l'utilisateur et le token
        sync_item = db.query(SyncItem).filter(SyncItem.id == sync_account.item_id).first()
        if not sync_item:
            ctx_logger.error(f"SyncItem non trouvé (id={sync_account.item_id}) pour le compte {bridge_account_id}")
            result_summary["status"] = "error"
            result_summary["errors"] = "Parent SyncItem not found"
            return result_summary
        user_id = sync_item.user_id
        ctx_logger = logging.LoggerAdapter(logger, {"bridge_account_id": bridge_account_id, "sql_account_id": sync_account.id, "user_id": user_id})
        ctx_logger.debug(f"SyncItem trouvé: id={sync_item.id}, user_id={user_id}")

        # Note: get_bridge_token gère déjà la récupération/renouvellement
        token_data = await bridge_service.get_bridge_token(db, user_id)
        # access_token = token_data["access_token"] # Non utilisé directement ici si bridge_service le gère

        # 2. Déterminer la date 'since' pour la requête Bridge
        since_date = None
        # Utiliser last_transaction_date s'il est plus récent que last_sync_timestamp
        # Ou si last_sync_timestamp est null (première synchro)
        # Prendre une marge de sécurité (ex: 3 jours) pour récupérer les transactions potentiellement mises à jour
        safety_margin = timedelta(days=3)
        last_effective_date = sync_account.last_transaction_date or sync_account.last_sync_timestamp
        if last_effective_date:
             since_date = last_effective_date - safety_margin
             ctx_logger.info(f"Synchronisation depuis {since_date} (dernière date: {last_effective_date})")
        else:
            # Première synchronisation pour ce compte, prendre 90 jours par défaut
            since_date = datetime.now(timezone.utc) - timedelta(days=90)
            ctx_logger.info(f"Première synchronisation, récupération depuis {since_date}")

        # 3. Récupérer les transactions depuis Bridge
        ctx_logger.info(f"Appel API Bridge pour transactions du compte {bridge_account_id} depuis {since_date}")
        bridge_transactions = await bridge_service.get_bridge_transactions(db, user_id, bridge_account_id, since=since_date)
        result_summary["processed_transactions"] = len(bridge_transactions)
        ctx_logger.info(f"{len(bridge_transactions)} transactions récupérées de Bridge API.")

        if not bridge_transactions:
            result_summary["status"] = "success" # Succès, mais aucune nouvelle transaction
            result_summary["vector_storage_status"] = "not_attempted"
             # Mettre à jour le timestamp de synchro même si 0 transaction
            sync_account.last_sync_timestamp = datetime.now(timezone.utc)
            db.add(sync_account)
            db.commit()
            return result_summary

        # 4. Préparer et stocker les transactions vectoriellement (avec enrichissement marchand optionnel)
        vector_transactions_to_store = []
        max_updated_at = last_effective_date or datetime.min.replace(tzinfo=timezone.utc) # Pour suivre la date de la dernière transaction traitée

        vector_storage = None
        if VECTOR_STORAGE_AVAILABLE:
            vector_storage = VectorStorageService() # Instancier ici si nécessaire

        for tx in bridge_transactions:
            bridge_tx_id = tx.get("id")
            description = tx.get("description", "")
            clean_description = tx.get("clean_description", description) # Utiliser clean si dispo
            category_id = tx.get("category_id")

            # --- Enrichissement Marchand Optionnel ---
            merchant_point_id = None
            # Décommenter pour activer l'enrichissement
            # if vector_storage:
            #     result_summary["merchant_enrichment_attempts"] += 1
            #     merchant_point_id = await _get_or_create_merchant_vector(
            #         vector_storage, user_id, clean_description or description, category_id
            #     )
            #     if merchant_point_id:
            #         # Distinguer si trouvé ou créé pour les stats
            #         # (Nécessiterait que _get_or_create retourne plus d'infos)
            #         result_summary["merchants_found"] += 1 # Simplification: on compte comme trouvé/créé
            # --- Fin Enrichissement Marchand ---


            # Préparer la transaction pour le stockage vectoriel
            vector_tx = {
                "user_id": user_id,
                "account_id": bridge_account_id,
                "bridge_transaction_id": bridge_tx_id,
                "amount": tx.get("amount", 0.0),
                "currency_code": tx.get("currency_code", "EUR"),
                "description": description,
                "clean_description": clean_description,
                "transaction_date": tx.get("date"), # Utiliser 'date' fournie par Bridge
                "booking_date": tx.get("booking_date"),
                "value_date": tx.get("value_date"),
                "category_id": category_id,
                "operation_type": tx.get("operation_type"),
                "is_recurring": tx.get("recurring", {}).get("is"), # Vérifier structure exacte si récurrence gérée
                "merchant_id": merchant_point_id, # ID Qdrant du marchand (si enrichissement activé)
                "bridge_updated_at": tx.get("updated_at") # Date de mise à jour Bridge
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
                 ctx_logger.warning(f"Impossible de parser updated_at '{tx_updated_at_str}' pour tx {bridge_tx_id}: {date_err}")


        # Stocker les transactions préparées dans le Vector Store
        if vector_storage and vector_transactions_to_store:
            ctx_logger.info(f"Stockage vectoriel de {len(vector_transactions_to_store)} transactions...")
            vector_result = await vector_storage.batch_store_transactions(vector_transactions_to_store)
            ctx_logger.info(f"Résultat stockage vectoriel: {vector_result}")
            result_summary["vector_storage_status"] = vector_result.get("status", "error")
            result_summary["new_transactions"] = vector_result.get("successful", 0) # Utiliser le compte du résultat vectoriel
            if vector_result.get("status") != "success":
                 result_summary["status"] = "partial" if vector_result.get("successful", 0) > 0 else "error"
                 result_summary["errors"] = f"Vector storage failed for {vector_result.get('failed', 0)} transactions."
            else:
                result_summary["status"] = "success"
        elif not vector_storage:
             result_summary["vector_storage_status"] = "unavailable"
             result_summary["status"] = "success" # Succès car pas d'erreur, mais pas de stockage vectoriel
        else:
             # Pas de transactions à stocker (déjà loggué avant)
             result_summary["vector_storage_status"] = "no_transactions_to_store"
             result_summary["status"] = "success"


        # 5. Mettre à jour les timestamps du compte SQL
        now = datetime.now(timezone.utc)
        sync_account.last_sync_timestamp = now
        # Utiliser max_updated_at trouvé comme date de dernière transaction
        if max_updated_at > (sync_account.last_transaction_date or datetime.min.replace(tzinfo=timezone.utc)):
            sync_account.last_transaction_date = max_updated_at
            ctx_logger.debug(f"Mise à jour de last_transaction_date à {max_updated_at}")

        db.add(sync_account)
        db.commit()
        db.refresh(sync_account)

        ctx_logger.info(f"Synchronisation terminée pour compte {bridge_account_id}: status={result_summary['status']}")
        return result_summary

    except HTTPException as http_exc:
         # Erreur lors de l'appel à Bridge API (ex: token invalide)
         ctx_logger.error(f"Erreur HTTP {http_exc.status_code} durant sync transactions pour compte {bridge_account_id}: {http_exc.detail}")
         result_summary["status"] = "error"
         result_summary["errors"] = f"API Error: {http_exc.status_code} - {http_exc.detail}"
         # Pas de rollback ici car l'erreur vient de l'extérieur
         return result_summary
    except Exception as e:
        error_details = traceback.format_exc()
        ctx_logger.error(f"Erreur inattendue durant sync transactions pour compte {bridge_account_id}: {e}\n{error_details}")
        result_summary["status"] = "error"
        result_summary["errors"] = f"Unexpected error: {str(e)}"
        db.rollback() # Rollback des changements potentiels sur sync_account
        return result_summary


async def force_sync_all_accounts(db: Session, user_id: int) -> Dict[str, Any]:
    """Force la synchronisation des transactions pour tous les comptes actifs d'un utilisateur."""
    ctx_logger = logging.LoggerAdapter(logger, {"user_id": user_id})
    ctx_logger.info(f"Forçage de la synchronisation des transactions pour tous les comptes de l'utilisateur {user_id}")

    try:
        # Récupérer tous les comptes SQL actifs (associés à des items actifs/OK ?)
        # Pour l'instant, on prend tous les comptes liés à l'utilisateur
        accounts_to_sync = db.query(SyncAccount).join(SyncItem).filter(
            SyncItem.user_id == user_id,
            # SyncItem.status == 0 # Optionnel: ne synchroniser que les comptes liés à des items OK ?
        ).all()

        if not accounts_to_sync:
            ctx_logger.warning(f"Aucun compte trouvé pour le forçage de synchronisation de l'utilisateur {user_id}")
            return {"status": "warning", "message": "No accounts found for this user", "accounts_processed": 0, "details": []}

        ctx_logger.info(f"{len(accounts_to_sync)} comptes trouvés pour l'utilisateur {user_id}. Démarrage synchro individuelle.")

        results = []
        accounts_with_errors = 0
        total_new_tx = 0

        for account in accounts_to_sync:
            account_result = await sync_account_transactions(db, account)
            results.append({
                "bridge_account_id": account.bridge_account_id,
                "status": account_result.get("status"),
                "new_transactions": account_result.get("new_transactions", 0)
            })
            if account_result.get("status") != "success":
                accounts_with_errors += 1
            total_new_tx += account_result.get("new_transactions", 0)

        # Déterminer le statut global
        overall_status = "success"
        if accounts_with_errors > 0:
            overall_status = "partial" if accounts_with_errors < len(accounts_to_sync) else "error"

        ctx_logger.info(f"Forçage de synchronisation terminé pour user {user_id}. Statut: {overall_status}, {total_new_tx} nouvelles transactions stockées.")

        return {
            "status": overall_status,
            "accounts_processed": len(accounts_to_sync),
            "accounts_with_errors": accounts_with_errors,
            "total_new_transactions_stored": total_new_tx,
            "details": results # Peut être volumineux
        }

    except Exception as e:
        ctx_logger.error(f"Erreur majeure lors du forçage de synchronisation pour user {user_id}: {str(e)}", exc_info=True)
        return {"status": "critical_error", "message": str(e), "accounts_processed": 0, "details": []}

async def check_and_sync_missing_transactions(db: Session, user_id: int) -> Dict[str, Any]:
    """
    Vérifie l'état de synchronisation et lance une synchronisation forcée si nécessaire.
    NOTE : La logique exacte de "quand forcer" pourrait être affinée. Ici, on force si des comptes existent.
    """
    ctx_logger = logging.LoggerAdapter(logger, {"user_id": user_id})
    ctx_logger.info(f"Vérification et synchronisation potentielle des transactions manquantes pour l'utilisateur {user_id}")

    try:
        # 1. Récupérer l'état de synchronisation global (SQL et Vectoriel)
        # Utilisation de la fonction corrigée dans sync_manager
        sync_status_report = await sync_manager.get_user_sync_status(db, user_id)
        ctx_logger.info(f"État actuel de la synchronisation récupéré.")
        ctx_logger.debug(f"Rapport d'état: {sync_status_report}") # Attention, peut être gros

        # 2. Décider s'il faut forcer la synchronisation
        # Logique simple : s'il y a des comptes SQL, on force une synchro pour être sûr.
        # On pourrait ajouter des conditions plus fines basées sur sync_status_report
        # (ex: si vector_storage_status montre 0 transaction mais sql_status > 0 comptes).
        sql_status = sync_status_report.get("sql_status", {})
        needs_forced_sync = sql_status.get("total_accounts", 0) > 0

        if not needs_forced_sync:
             ctx_logger.info(f"Aucun compte trouvé ou autre condition non remplie. Pas de synchronisation forcée nécessaire.")
             return {"status": "not_needed", "message": "No accounts found or sync deemed up-to-date.", "sync_status": sync_status_report}

        # 3. Lancer une synchronisation complète si nécessaire
        ctx_logger.info(f"Synchronisation forcée nécessaire. Appel de force_sync_all_accounts.")
        result = await force_sync_all_accounts(db, user_id)
        ctx_logger.info(f"Résultat de la synchronisation forcée: {result.get('status')}")

        return result # Retourner le résultat de la synchronisation forcée

    except Exception as e:
        error_details = traceback.format_exc()
        ctx_logger.error(f"Erreur lors de check_and_sync_missing_transactions pour user {user_id}: {str(e)}\n{error_details}")
        return {"status": "error", "message": f"Failed checking/syncing missing transactions: {str(e)}"}

async def get_user_vector_stats(db: Session, user_id: int) -> Dict[str, Any]:
    """Récupère les statistiques vectorielles pour un utilisateur via VectorStorageService."""
    ctx_logger = logging.LoggerAdapter(logger, {"user_id": user_id})
    ctx_logger.info(f"Récupération des statistiques vectorielles pour l'utilisateur {user_id}")

    if not VECTOR_STORAGE_AVAILABLE:
         ctx_logger.warning("Vector storage indisponible, impossible de récupérer les stats.")
         return {"user_id": user_id, "status": "unavailable", "message": "Vector storage service not available."}

    try:
        vector_storage = VectorStorageService()
        stats = await vector_storage.get_user_statistics(user_id)
        ctx_logger.info(f"Statistiques vectorielles récupérées: {stats}")
        return stats
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la récupération des statistiques vectorielles pour user {user_id}: {str(e)}", exc_info=True)
        return {
            "user_id": user_id,
            "status": "error",
            "message": f"Failed to retrieve vector stats: {str(e)}"
        }