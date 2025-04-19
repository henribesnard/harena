"""
Service Manager de synchronisation pour Harena.

Ce module coordonne la synchronisation des données bancaires depuis Bridge API
vers la base de données SQL et le stockage vectoriel.
"""

import logging
import httpx
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional, Tuple
from fastapi import HTTPException, status

# Imports Models
from sync_service.models.sync import SyncItem, SyncAccount
from user_service.models.user import User, BridgeConnection

# Imports Services
from user_service.services.bridge import get_bridge_token, get_bridge_accounts, get_bridge_categories, get_bridge_insights, get_bridge_stocks
from config_service.config import settings
# Import forward-only pour éviter les dépendances circulaires
# Le transaction_sync sera importé au besoin dans les fonctions spécifiques

# Import Vector Storage Service avec gestion d'erreur
try:
    from sync_service.services.vector_storage import VectorStorageService
    VECTOR_STORAGE_AVAILABLE = True
except ImportError as e:
    VECTOR_STORAGE_AVAILABLE = False
    # Définir une classe factice pour éviter les erreurs AttributeError
    class VectorStorageService:
        async def batch_store_accounts(self, *args, **kwargs): 
            logging.getLogger(__name__).warning("VectorStorage non dispo: batch_store_accounts ignoré.")
            return {"status": "skipped"}
        async def batch_store_categories(self, *args, **kwargs): 
            logging.getLogger(__name__).warning("VectorStorage non dispo: batch_store_categories ignoré.")
            return {"status": "skipped"}
        async def batch_store_insights(self, *args, **kwargs): 
            logging.getLogger(__name__).warning("VectorStorage non dispo: batch_store_insights ignoré.")
            return {"status": "skipped"}
        async def batch_store_stocks(self, *args, **kwargs): 
            logging.getLogger(__name__).warning("VectorStorage non dispo: batch_store_stocks ignoré.")
            return {"status": "skipped"}
        async def check_user_storage_initialized(self, *args, **kwargs): return False
        async def initialize_user_storage(self, *args, **kwargs): pass
        async def get_user_statistics(self, *args, **kwargs): return {"status": "unavailable"}
        async def get_user_storage_metadata(self, *args, **kwargs): return {"status": "unavailable"}

# Imports Utilities
from sync_service.utils.logging import get_contextual_logger

# Configuration du logger
logger = logging.getLogger(__name__)

# Liste des statuts qui requièrent une action utilisateur
ACTION_REQUIRED_STATUSES = [402, 429, 1010]

# --- Fonctions d'aide ---

def parse_iso_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse une date ISO 8601, gérant la présence ou l'absence de 'Z'."""
    if not date_str:
        return None
    try:
        # Gestion des formats avec ou sans 'Z'
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'
        return datetime.fromisoformat(date_str)
    except ValueError:
        logger.warning(f"Impossible de parser la date ISO: {date_str}")
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue lors du parsing de la date {date_str}: {e}")
        return None


# --- Fonctions du Sync Manager ---

async def create_or_update_sync_item(db: Session, user_id: int, bridge_item_id: int, item_data: Dict[str, Any]) -> SyncItem:
    """Créer ou mettre à jour un item de synchronisation dans la base SQL."""
    ctx_logger = get_contextual_logger("sync_service.sync_manager", user_id=user_id, bridge_item_id=bridge_item_id)
    ctx_logger.info(f"Création ou mise à jour de l'item de synchronisation SQL")

    try:
        sync_item = db.query(SyncItem).filter(SyncItem.bridge_item_id == bridge_item_id).first()

        status = item_data.get("status", 0)
        status_code_info = item_data.get("status_code_info")
        needs_action = status in ACTION_REQUIRED_STATUSES

        # Utilisation de la fonction helper parse_iso_date
        last_successful_refresh_dt = parse_iso_date(item_data.get("last_successful_refresh"))
        last_try_refresh_dt = parse_iso_date(item_data.get("last_try_refresh"))

        if sync_item:
            ctx_logger.info(f"Item SQL existant trouvé: id={sync_item.id}. Mise à jour.")
            sync_item.status = status
            sync_item.status_code_info = status_code_info
            sync_item.needs_user_action = needs_action
            sync_item.status_description = item_data.get("status_code_description", sync_item.status_description)
            sync_item.last_successful_refresh = last_successful_refresh_dt or sync_item.last_successful_refresh
            sync_item.last_try_refresh = last_try_refresh_dt or sync_item.last_try_refresh
            sync_item.provider_id = item_data.get("provider_id", sync_item.provider_id)
            sync_item.account_types = item_data.get("account_types", sync_item.account_types)
        else:
            ctx_logger.info(f"Création d'un nouvel item de synchronisation SQL.")
            sync_item = SyncItem(
                user_id=user_id,
                bridge_item_id=bridge_item_id,
                status=status,
                status_code_info=status_code_info,
                status_description=item_data.get("status_code_description"),
                provider_id=item_data.get("provider_id"),
                account_types=item_data.get("account_types"),
                needs_user_action=needs_action,
                last_successful_refresh=last_successful_refresh_dt,
                last_try_refresh=last_try_refresh_dt
            )

        db.add(sync_item)
        db.commit()
        db.refresh(sync_item)
        ctx_logger.info(f"Item SQL enregistré/mis à jour avec succès: id={sync_item.id}")

        # Initialiser le stockage vectoriel pour l'utilisateur SI NÉCESSAIRE
        if VECTOR_STORAGE_AVAILABLE:
            try:
                vector_storage = VectorStorageService()
                is_initialized = await vector_storage.check_user_storage_initialized(user_id)
                if not is_initialized:
                    ctx_logger.info(f"Initialisation du stockage vectoriel (métadata) pour l'utilisateur {user_id}")
                    await vector_storage.initialize_user_storage(user_id)
            except Exception as e:
                ctx_logger.error(f"Erreur lors de l'initialisation du stockage vectoriel: {e}", exc_info=True)

        # Récupérer les comptes associés pour la base SQL
        try:
            token_data = await get_bridge_token(db, user_id)
            await fetch_and_update_sql_accounts(db, sync_item, token_data["access_token"])
        except Exception as e:
            ctx_logger.error(f"Erreur lors de la récupération initiale des comptes SQL: {e}", exc_info=True)

        return sync_item
    except Exception as e:
        ctx_logger.error(f"Erreur générale lors de create_or_update_sync_item: {e}", exc_info=True)
        db.rollback()
        raise  # Renvoyer l'exception pour que l'appelant sache qu'il y a eu un problème


async def update_item_status(db: Session, sync_item: SyncItem, status_code: int, status_code_info: Optional[str] = None, status_description: Optional[str] = None) -> SyncItem:
    """Mettre à jour le statut d'un item dans la base SQL."""
    ctx_logger = get_contextual_logger("sync_service.sync_manager", user_id=sync_item.user_id, bridge_item_id=sync_item.bridge_item_id)
    ctx_logger.info(f"Mise à jour du statut SQL de l'item: status={status_code}, code={status_code_info}")

    try:
        sync_item.status = status_code
        if status_code_info:
            sync_item.status_code_info = status_code_info
        if status_description:
            sync_item.status_description = status_description
        sync_item.needs_user_action = status_code in ACTION_REQUIRED_STATUSES

        now = datetime.now(timezone.utc)
        if status_code == 0:
            sync_item.last_successful_refresh = now
            ctx_logger.info(f"Statut OK, mise à jour de last_successful_refresh SQL.")
        sync_item.last_try_refresh = now  # Toujours mettre à jour la dernière tentative

        db.add(sync_item)
        db.commit()
        db.refresh(sync_item)
        ctx_logger.info(f"Statut SQL de l'item mis à jour avec succès.")
        return sync_item
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la mise à jour du statut SQL: {e}", exc_info=True)
        db.rollback()
        raise


async def get_bridge_item_info(db: Session, sync_item: SyncItem, access_token: str) -> Optional[Dict[str, Any]]:
    """Récupérer les informations détaillées d'un item depuis Bridge API."""
    ctx_logger = get_contextual_logger("sync_service.sync_manager", user_id=sync_item.user_id, bridge_item_id=sync_item.bridge_item_id)
    ctx_logger.info(f"Récupération des informations détaillées de l'item {sync_item.bridge_item_id} depuis Bridge API")
    url = f"{settings.BRIDGE_API_URL}/aggregation/items/{sync_item.bridge_item_id}"
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "Client-Id": settings.BRIDGE_CLIENT_ID,
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
        "authorization": f"Bearer {access_token}"
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            item_data = response.json()
            ctx_logger.debug(f"Réponse API Bridge pour item info reçue")
            return item_data
    except httpx.HTTPStatusError as e:
        ctx_logger.error(f"Erreur API Bridge ({e.response.status_code}) lors de la récupération des infos de l'item: {e.response.text}")
        return None
    except httpx.RequestError as e:
        ctx_logger.error(f"Erreur de connexion lors de la récupération des infos de l'item: {e}", exc_info=True)
        return None
    except Exception as e:
        ctx_logger.error(f"Erreur inattendue lors de la récupération des infos de l'item: {e}", exc_info=True)
        return None


async def fetch_and_update_sql_accounts(db: Session, sync_item: SyncItem, access_token: str) -> Tuple[List[SyncAccount], List[Dict[str, Any]]]:
    """
    Récupère les comptes depuis Bridge API, met à jour la base SQL (SyncAccount),
    et retourne les objets SQL ET les données brutes de l'API pour le stockage vectoriel.
    """
    ctx_logger = get_contextual_logger("sync_service.sync_manager", user_id=sync_item.user_id, bridge_item_id=sync_item.bridge_item_id)
    ctx_logger.info(f"Récupération et mise à jour des comptes SQL pour l'item {sync_item.bridge_item_id}")

    sql_accounts_updated: List[SyncAccount] = []
    bridge_accounts_data: List[Dict[str, Any]] = []

    try:
        # Récupérer les comptes depuis Bridge API
        bridge_accounts_data = await get_bridge_accounts(db, sync_item.user_id, item_id=sync_item.bridge_item_id)
        ctx_logger.info(f"Récupération de {len(bridge_accounts_data)} comptes depuis Bridge API pour l'item {sync_item.bridge_item_id}")

        if not bridge_accounts_data:
            ctx_logger.warning(f"Aucun compte trouvé via Bridge API pour l'item {sync_item.bridge_item_id}.")
            return [], []

        bridge_account_ids = {acc_data["id"] for acc_data in bridge_accounts_data}

        # Mettre à jour ou créer les comptes dans la base SQL
        for account_data in bridge_accounts_data:
            bridge_account_id = account_data.get("id")
            if bridge_account_id is None:
                ctx_logger.warning(f"Donnée de compte sans ID reçue de Bridge")
                continue

            sync_account = db.query(SyncAccount).filter(
                SyncAccount.bridge_account_id == bridge_account_id
            ).first()

            # Préparer les champs pour SQL
            account_name = account_data.get("name")
            account_type = account_data.get("type")

            if sync_account:
                # Mettre à jour le compte SQL existant si nécessaire
                needs_update = False
                if sync_account.account_name != account_name:
                    sync_account.account_name = account_name
                    needs_update = True
                if sync_account.account_type != account_type:
                    sync_account.account_type = account_type
                    needs_update = True

                if needs_update:
                    ctx_logger.debug(f"Mise à jour du compte SQL existant {bridge_account_id}")
                    db.add(sync_account)
            else:
                # Créer un nouveau compte SQL
                ctx_logger.info(f"Création d'un nouveau compte SQL {bridge_account_id} lié à l'item {sync_item.id}")
                sync_account = SyncAccount(
                    item_id=sync_item.id,
                    bridge_account_id=bridge_account_id,
                    account_name=account_name,
                    account_type=account_type
                )
                db.add(sync_account)

            sql_accounts_updated.append(sync_account)

        # Supprimer les comptes SQL qui n'existent plus dans Bridge pour cet item
        existing_sql_accounts = db.query(SyncAccount).filter(SyncAccount.item_id == sync_item.id).all()
        for sql_acc in existing_sql_accounts:
            if sql_acc.bridge_account_id not in bridge_account_ids:
                ctx_logger.warning(f"Suppression du compte SQL {sql_acc.bridge_account_id} car non trouvé dans Bridge pour l'item {sync_item.id}")
                db.delete(sql_acc)

        db.commit()
        
        # Rafraîchir les objets SQL après commit
        for acc in sql_accounts_updated:
            try:
                db.refresh(acc)
            except Exception:
                ctx_logger.warning(f"Impossible de rafraîchir le compte SQL id={acc.id}, potentiellement supprimé.")

        ctx_logger.info(f"Mise à jour de {len(sql_accounts_updated)} comptes SQL terminée.")
        return sql_accounts_updated, bridge_accounts_data

    except HTTPException as http_exc:
        ctx_logger.error(f"Erreur HTTP lors de la récupération des comptes Bridge: {http_exc.detail}", exc_info=True)
        db.rollback()
        raise
    except Exception as e:
        ctx_logger.error(f"Erreur générale lors de fetch_and_update_sql_accounts: {e}", exc_info=True)
        db.rollback()
        raise


# --- Fonction Principale de Synchronisation ---

async def trigger_full_sync_for_item(db: Session, sync_item: SyncItem) -> Dict[str, Any]:
    """
    Déclenche une synchronisation complète pour un item:
    Met à jour le statut, récupère et stocke vectoriellement les comptes, catégories,
    insights, actions, puis synchronise les transactions.
    """
    user_id = sync_item.user_id
    bridge_item_id = sync_item.bridge_item_id
    ctx_logger = get_contextual_logger("sync_service.sync_manager", user_id=user_id, bridge_item_id=bridge_item_id)
    ctx_logger.info(f"--- Début synchronisation COMPLÈTE pour item {bridge_item_id} ---")

    overall_status = "success"
    sync_report: Dict[str, Any] = {
        "item_id": bridge_item_id,
        "user_id": user_id,
        "status": "pending",
        "steps": {}
    }

    vector_storage = None
    if VECTOR_STORAGE_AVAILABLE:
        vector_storage = VectorStorageService()
    else:
        ctx_logger.warning("Stockage vectoriel non disponible. Certaines étapes seront ignorées.")

    try:
        # 1. Récupérer le token Bridge
        ctx_logger.info("Étape 1: Récupération du token Bridge")
        token_data = await get_bridge_token(db, user_id)
        access_token = token_data["access_token"]
        sync_report["steps"]["get_token"] = {"status": "success"}

        # 2. Récupérer les informations à jour de l'item et mettre à jour le statut SQL
        ctx_logger.info("Étape 2: Mise à jour du statut de l'item")
        item_info = await get_bridge_item_info(db, sync_item, access_token)
        if item_info:
            sync_item = await update_item_status(
                db, sync_item,
                item_info.get("status", sync_item.status),
                item_info.get("status_code_info", sync_item.status_code_info),
                item_info.get("status_code_description", sync_item.status_description)
            )
            sync_report["steps"]["update_item_status"] = {"status": "success", "item_status": sync_item.status}
        else:
            ctx_logger.warning("Impossible de récupérer les informations à jour de l'item depuis Bridge.")
            sync_report["steps"]["update_item_status"] = {"status": "warning", "message": "Could not fetch item info"}

        # 3. Vérifier si l'item est en erreur
        if sync_item.status != 0:
            ctx_logger.warning(f"Item en erreur (status={sync_item.status}), synchronisation annulée.")
            sync_report["status"] = "error_item_status"
            sync_report["message"] = f"Item in error state: {sync_item.status} ({sync_item.status_code_info})"
            sync_report["steps"]["check_item_error"] = {"status": "failed", "item_status": sync_item.status}
            return sync_report

        sync_report["steps"]["check_item_error"] = {"status": "success", "item_status": 0}

        # 4. Récupérer les comptes (SQL + Données Brutes)
        ctx_logger.info("Étape 4: Récupération et mise à jour SQL des comptes")
        sql_accounts, bridge_accounts_list = await fetch_and_update_sql_accounts(db, sync_item, access_token)
        sync_report["steps"]["fetch_sql_accounts"] = {"status": "success", "count": len(sql_accounts)}

        if not sql_accounts:
            ctx_logger.warning(f"Aucun compte associé à l'item {bridge_item_id}. Synchronisation terminée.")
            sync_report["status"] = "warning_no_accounts"
            sync_report["message"] = "No accounts found for this item."
            return sync_report

        # 5. Stocker les comptes dans le Vector Store
        if vector_storage:
            ctx_logger.info("Étape 5: Stockage vectoriel des comptes")
            accounts_to_store = []
            for acc_data in bridge_accounts_list:
                acc_data_with_context = {
                    **acc_data,
                    "user_id": user_id,
                    "item_id": bridge_item_id,
                    "bridge_account_id": acc_data.get("id"),
                    "bridge_updated_at": acc_data.get("updated_at")
                }
                accounts_to_store.append(acc_data_with_context)

            if accounts_to_store:
                vector_result = await vector_storage.batch_store_accounts(accounts_to_store)
                sync_report["steps"]["store_vector_accounts"] = vector_result
                if vector_result.get("status") != "success":
                    overall_status = "partial"
                    ctx_logger.warning(f"Stockage vectoriel des comptes partiel/échoué: {vector_result}")
            else:
                sync_report["steps"]["store_vector_accounts"] = {"status": "skipped", "message": "No accounts data"}
        else:
            sync_report["steps"]["store_vector_accounts"] = {"status": "skipped", "message": "Vector storage unavailable"}

        # 6. Récupérer et Stocker les Catégories
        if vector_storage:
            ctx_logger.info("Étape 6: Récupération et stockage vectoriel des catégories")
            try:
                categories_list = await get_bridge_categories(db, user_id)
                if categories_list:
                    ctx_logger.info(f"Récupération de {len(categories_list)} catégories depuis Bridge API")

                    #logging détaillé des catégories recupérées
                    for i, cat in enumerate(categories_list[:5]):
                        ctx_logger.debug(f"Catégorie {i+1}: {cat.get('name')} (ID: {cat.get('bridge_category_id')})")

                    vector_result = await vector_storage.batch_store_categories(categories_list)
                    sync_report["steps"]["store_vector_categories"] = vector_result

                    # Log détaillé du résultat 
                    ctx_logger.info(f"Stockage vectoriel des catégories: {vector_result.get('status')}, "
                           f"succès: {vector_result.get('successful', 0)}/{len(categories_list)}")

                    if vector_result.get("status") != "success":
                        overall_status = "partial"
                        ctx_logger.warning(f"Stockage vectoriel des catégories partiel/échoué: {vector_result}")
                else:
                    sync_report["steps"]["store_vector_categories"] = {"status": "skipped", "message": "No categories returned"}
            except Exception as cat_error:
                ctx_logger.error(f"Erreur lors de la récupération/stockage des catégories: {cat_error}", exc_info=True)
                sync_report["steps"]["store_vector_categories"] = {"status": "error", "message": str(cat_error)}
                overall_status = "partial"
        else:
            ctx_logger.warning("Vector storage indisponible pour la synchronisation des catégories")
            sync_report["steps"]["store_vector_categories"] = {"status": "skipped", "message": "Vector storage unavailable"}

        # 7. Récupérer et Stocker les Insights
        if vector_storage:
            ctx_logger.info("Étape 7: Récupération et stockage vectoriel des insights")
            try:
                insights_data = await get_bridge_insights(db, user_id)
                if insights_data:
                    insights_to_store = []
                    monthly_kpis = insights_data.get("kpis", {}).get("monthly", [])
                    for month_data in monthly_kpis:
                        period_start_str = month_data.get("from")
                        period_end_str = month_data.get("to")
                        period_start_dt = parse_iso_date(period_start_str)
                        period_end_dt = parse_iso_date(period_end_str)

                        if not period_start_dt:
                            ctx_logger.warning(f"Date de début invalide pour insight mensuel: {period_start_str}")
                            continue

                        for cat_insight in month_data.get("categories", []):
                            insights_to_store.append({
                                "user_id": user_id,
                                "category_id": cat_insight.get("id"),
                                "category_name": cat_insight.get("name", "Unknown"),
                                "period_type": "monthly",
                                "period_start": period_start_dt,
                                "period_end": period_end_dt,
                                "aggregates": cat_insight.get("aggregates")
                            })

                    if insights_to_store:
                        ctx_logger.info(f"Stockage vectoriel de {len(insights_to_store)} insights")
                        vector_result = await vector_storage.batch_store_insights(insights_to_store)
                        sync_report["steps"]["store_vector_insights"] = vector_result
                        ctx_logger.info(f"Résultat stockage vectoriel des insights: {vector_result.get('status')}, {vector_result.get('successful', 0)}/{len(insights_to_store)} réussis")
                        if vector_result.get("status") != "success":
                            overall_status = "partial"
                            ctx_logger.warning(f"Stockage vectoriel des insights partiel/échoué: {vector_result}")
                    else:
                        ctx_logger.warning("Aucun insight à traiter")
                        sync_report["steps"]["store_vector_insights"] = {"status": "skipped", "message": "No processable insights"}
                else:
                    ctx_logger.warning("Aucune données d'insights retournée par Bridge API")
                    sync_report["steps"]["store_vector_insights"] = {"status": "skipped", "message": "No insights returned or endpoint unavailable"}
            except Exception as insight_error:
                ctx_logger.error(f"Erreur lors de la récupération/stockage des insights: {insight_error}", exc_info=True)
                sync_report["steps"]["store_vector_insights"] = {"status": "error", "message": str(insight_error)}
                overall_status = "partial"
        else:
            sync_report["steps"]["store_vector_insights"] = {"status": "skipped", "message": "Vector storage unavailable"}

        # 8. Récupérer et Stocker les Actions (Stocks)
        if vector_storage:
            ctx_logger.info("Étape 8: Récupération et stockage vectoriel des actions/stocks")
            INVESTMENT_ACCOUNT_TYPES = ["stock", "savings", "life_insurance", "market", "pea", "cryptocurrency"]
            investment_account_ids = [
                acc.bridge_account_id for acc in sql_accounts if acc.account_type in INVESTMENT_ACCOUNT_TYPES
            ]
            stocks_results = []
            total_stocks_stored = 0
            
            if investment_account_ids:
                ctx_logger.info(f"Comptes potentiels d'investissement trouvés: {investment_account_ids}")
                all_stocks_to_store = []
                
                for acc_id in investment_account_ids:
                    try:
                        stocks_list = await get_bridge_stocks(db, user_id, account_id=acc_id)
                        if stocks_list:
                            for stock in stocks_list:
                                all_stocks_to_store.append({
                                    **stock,
                                    "user_id": user_id,
                                    "account_id": acc_id,
                                    "bridge_stock_id": stock.get("id"),
                                    "bridge_updated_at": stock.get("updated_at")
                                })
                        else:
                            ctx_logger.info(f"Aucune action trouvée pour le compte {acc_id}")
                    except Exception as stock_fetch_error:
                        ctx_logger.error(f"Erreur lors de la récupération des stocks pour compte {acc_id}: {stock_fetch_error}", exc_info=True)
                        stocks_results.append({"account_id": acc_id, "status": "error", "message": f"Fetch error: {str(stock_fetch_error)}"})
                        overall_status = "partial"

                # Stocker toutes les actions récupérées en un seul batch
                if all_stocks_to_store:
                    try:
                        vector_result = await vector_storage.batch_store_stocks(all_stocks_to_store)
                        stocks_results.append(vector_result)
                        if vector_result.get("status") != "success":
                            overall_status = "partial"
                        total_stocks_stored += vector_result.get("successful", 0)
                    except Exception as stock_store_error:
                        ctx_logger.error(f"Erreur lors du stockage en batch des stocks: {stock_store_error}", exc_info=True)
                        stocks_results.append({"status": "error", "message": f"Store error: {str(stock_store_error)}"})
                        overall_status = "partial"

                sync_report["steps"]["store_vector_stocks"] = {"status": overall_status, "details": stocks_results, "total_stored": total_stocks_stored}
            else:
                ctx_logger.info("Aucun compte de type investissement trouvé pour cet item.")
                sync_report["steps"]["store_vector_stocks"] = {"status": "skipped", "message": "No investment accounts"}
        else:
            sync_report["steps"]["store_vector_stocks"] = {"status": "skipped", "message": "Vector storage unavailable"}

        # 9. Synchroniser les Transactions (après les autres données pour cohérence)
        ctx_logger.info("Étape 9: Synchronisation des transactions (inclut stockage vectoriel)")
        
        # Import transaction_sync uniquement ici pour éviter les dépendances circulaires
        from sync_service.services.transaction_sync import sync_account_transactions
        
        transaction_sync_results = []
        accounts_with_errors = 0
        total_new_transactions = 0
        
        for account in sql_accounts:
            ctx_logger.info(f"Synchronisation des transactions pour le compte SQL id={account.id}, bridge_id={account.bridge_account_id}")
            try:
                account_result = await sync_account_transactions(db, account)
                transaction_sync_results.append({
                    "bridge_account_id": account.bridge_account_id,
                    "result": account_result
                })
                total_new_transactions += account_result.get("new_transactions", 0)
                if account_result.get("status") != "success":
                    accounts_with_errors += 1
                    overall_status = "partial"  # Synchro partielle si un compte échoue
            except Exception as tx_sync_error:
                ctx_logger.error(f"Erreur lors de la synchronisation des transactions pour compte {account.bridge_account_id}: {tx_sync_error}", exc_info=True)
                transaction_sync_results.append({
                    "bridge_account_id": account.bridge_account_id,
                    "result": {"status": "error", "errors": str(tx_sync_error)}
                })
                accounts_with_errors += 1
                overall_status = "partial"

        sync_report["steps"]["sync_transactions"] = {
            "status": "success" if accounts_with_errors == 0 else ("partial" if accounts_with_errors < len(sql_accounts) else "error"),
            "accounts_processed": len(sql_accounts),
            "accounts_with_errors": accounts_with_errors,
            "total_new_transactions_stored": total_new_transactions,
        }

        # 10. Mettre à jour les statistiques vectorielles globales (optionnel)
        if vector_storage:
            ctx_logger.info("Étape 10: Mise à jour des statistiques vectorielles")
            try:
                vector_stats = await vector_storage.get_user_statistics(user_id)
                sync_report["final_vector_stats"] = vector_stats
                sync_report["steps"]["update_vector_stats"] = {"status": "success"}
            except Exception as vs_error:
                ctx_logger.error(f"Erreur lors de la récupération des statistiques vectorielles finales: {vs_error}", exc_info=True)
                sync_report["steps"]["update_vector_stats"] = {"status": "error", "message": str(vs_error)}

        # Finaliser le rapport
        sync_report["status"] = overall_status
        sync_report["message"] = f"Full sync completed with status: {overall_status}"
        ctx_logger.info(f"--- Fin synchronisation COMPLÈTE pour item {bridge_item_id}. Statut final: {overall_status} ---")
        return sync_report

    except Exception as e:
        ctx_logger.error(f"Erreur majeure lors de trigger_full_sync_for_item: {e}", exc_info=True)
        sync_report["status"] = "critical_error"
        sync_report["message"] = f"Critical error during sync: {str(e)}"
        # Tenter de mettre l'item en erreur dans SQL
        try:
            await update_item_status(db, sync_item, status_code=1003, status_code_info="sync_error", status_description=f"Sync failed: {str(e)[:100]}")
        except Exception as status_update_error:
            ctx_logger.error(f"Impossible de mettre à jour le statut de l'item après erreur critique: {status_update_error}")
        return sync_report


# --- Autres fonctions du Sync Manager ---

async def get_user_sync_status(db: Session, user_id: int) -> Dict[str, Any]:
    """Obtenir l'état de synchronisation SQL et vectoriel pour un utilisateur."""
    ctx_logger = get_contextual_logger("sync_service.sync_manager", user_id=user_id)
    ctx_logger.info(f"Récupération de l'état de synchronisation global pour l'utilisateur {user_id}")

    try:
        # Statut SQL
        sync_items = db.query(SyncItem).filter(SyncItem.user_id == user_id).all()
        # Récupérer les comptes via la relation pour éviter une requête séparée
        all_sql_accounts_count = sum(len(item.accounts) for item in sync_items)

        items_info = []
        needs_action = False
        last_sync_sql = None
        for item in sync_items:
            if item.needs_user_action:
                needs_action = True
            if item.last_successful_refresh and (last_sync_sql is None or item.last_successful_refresh > last_sync_sql):
                last_sync_sql = item.last_successful_refresh

            items_info.append({
                "bridge_item_id": item.bridge_item_id,
                "status": item.status,
                "status_code_info": item.status_code_info,
                "status_description": item.status_description,
                "needs_user_action": item.needs_user_action,
                "last_successful_refresh": item.last_successful_refresh.isoformat() if item.last_successful_refresh else None,
                "last_try_refresh": item.last_try_refresh.isoformat() if item.last_try_refresh else None,
                "sql_account_count": len(item.accounts)
            })

        items_needing_action = [item for item in items_info if item["needs_user_action"]]

        # Statut Vectoriel
        vector_stats = {"status": "unavailable"}
        if VECTOR_STORAGE_AVAILABLE:
            try:
                vector_storage = VectorStorageService()
                vector_stats = await vector_storage.get_user_statistics(user_id)
            except Exception as vs_error:
                ctx_logger.error(f"Erreur lors de la récupération des statistiques vectorielles: {vs_error}", exc_info=True)
                vector_stats = {"status": "error", "message": str(vs_error)}
        else:
            vector_stats = {"status": "unavailable", "message": "Vector storage service not imported"}

        # Combiner les informations
        days_since_last_sync = None
        if last_sync_sql:
            days_since_last_sync = (datetime.now(timezone.utc) - last_sync_sql).days
            
        status_response = {
            "user_id": user_id,
            "sql_status": {
                "total_items": len(items_info),
                "total_accounts": all_sql_accounts_count,
                "needs_user_action": needs_action,
                "last_successful_sync": last_sync_sql.isoformat() if last_sync_sql else None,
                "days_since_last_sync": days_since_last_sync,
                "items_needing_action": items_needing_action,
                "items": items_info,
            },
            "vector_storage_status": vector_stats
        }
        ctx_logger.info(f"État de synchronisation récupéré avec succès pour l'utilisateur {user_id}")
        return status_response

    except Exception as e:
        ctx_logger.error(f"Erreur lors de la récupération de l'état de synchronisation global: {e}", exc_info=True)
        return {
            "user_id": user_id,
            "error": f"Failed to get sync status: {str(e)}",
            "sql_status": {"error": "Could not retrieve SQL status"},
            "vector_storage_status": {"error": "Could not retrieve vector status"}
        }


async def create_reconnect_session(db: Session, user_id: int, bridge_item_id: int) -> str:
    """Créer une session de reconnexion Bridge pour un item."""
    ctx_logger = get_contextual_logger("sync_service.sync_manager", user_id=user_id, bridge_item_id=bridge_item_id)
    ctx_logger.info(f"Création d'une session de reconnexion pour l'item {bridge_item_id}")

    try:
        # Vérifier que l'item appartient bien à l'utilisateur
        sync_item = db.query(SyncItem).filter(
            SyncItem.user_id == user_id,
            SyncItem.bridge_item_id == bridge_item_id
        ).first()

        if not sync_item:
            ctx_logger.error(f"Item {bridge_item_id} non trouvé pour l'utilisateur {user_id}")
            raise ValueError(f"Item {bridge_item_id} not found or does not belong to user {user_id}")

        # Importer bridge uniquement ici pour éviter les dépendances circulaires
        from user_service.services.bridge import create_connect_session
        
        base_url = settings.WEBHOOK_BASE_URL
        # L'URL de callback doit être gérée par votre application
        callback_url = f"{base_url}/reconnection/callback?item_id={bridge_item_id}"
        ctx_logger.info(f"URL de callback pour la reconnexion: {callback_url}")

        connect_url = await create_connect_session(
            db,
            user_id,
            item_id=bridge_item_id,  # Mode "Manage existing item"
            callback_url=callback_url,
        )
        ctx_logger.info(f"Session de reconnexion créée avec succès. URL: {connect_url}")
        return connect_url
        
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        ctx_logger.error(f"Erreur inattendue lors de la création de la session de reconnexion: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create reconnect session: {e}")