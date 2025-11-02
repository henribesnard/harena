"""
Orchestrateur de synchronisation.

Ce module coordonne l'ensemble du processus de synchronisation, 
en assurant l'exécution des différentes étapes dans le bon ordre.
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional, Tuple
from fastapi import HTTPException

# Import des modèles
from db_service.models.sync import SyncItem, SyncAccount
from db_service.models.user import User, BridgeConnection

# Import des services
from config_service.config import settings
from sync_service.utils.logging import get_contextual_logger

# Import des gestionnaires spécifiques
from sync_service.sync_manager.item_handler import get_bridge_item_info, update_item_status
from sync_service.sync_manager.account_handler import fetch_and_update_sql_accounts
from sync_service.sync_manager.transaction_handler import force_sync_all_accounts

logger = logging.getLogger(__name__)

async def create_or_update_sync_item(db: Session, user_id: int, bridge_item_id: int, item_data: Dict[str, Any]) -> SyncItem:
    """
    Crée ou met à jour un item de synchronisation dans la base SQL.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        bridge_item_id: ID de l'item Bridge
        item_data: Données de l'item à enregistrer
        
    Returns:
        SyncItem: Item de synchronisation créé ou mis à jour
    """
    ctx_logger = get_contextual_logger("sync_service.sync_manager", user_id=user_id, bridge_item_id=bridge_item_id)
    ctx_logger.info(f"Création ou mise à jour de l'item de synchronisation SQL")

    try:
        sync_item = db.query(SyncItem).filter(SyncItem.bridge_item_id == bridge_item_id).first()

        status = item_data.get("status", 0)
        status_code_info = item_data.get("status_code_info")
        needs_action = status in [402, 429, 1010]  # Statuts nécessitant une action utilisateur

        # Vérification des dates
        last_successful_refresh_str = item_data.get("last_successful_refresh")
        last_successful_refresh_dt = None
        if last_successful_refresh_str:
            try:
                last_successful_refresh_dt = datetime.fromisoformat(last_successful_refresh_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                ctx_logger.warning(f"Format de date invalide pour last_successful_refresh: {last_successful_refresh_str}")
        
        last_try_refresh_str = item_data.get("last_try_refresh")
        last_try_refresh_dt = None
        if last_try_refresh_str:
            try:
                last_try_refresh_dt = datetime.fromisoformat(last_try_refresh_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                ctx_logger.warning(f"Format de date invalide pour last_try_refresh: {last_try_refresh_str}")

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

        # Récupérer les comptes associés pour la base SQL si nécessaire
        try:
            if sync_item.status == 0:  # Uniquement si statut OK
                from user_service.services.bridge import get_bridge_token
                token_data = await get_bridge_token(db, user_id)
                await fetch_and_update_sql_accounts(db, sync_item, token_data["access_token"])
        except Exception as e:
            ctx_logger.error(f"Erreur lors de la récupération initiale des comptes SQL: {e}", exc_info=True)

        return sync_item
    except Exception as e:
        ctx_logger.error(f"Erreur générale lors de create_or_update_sync_item: {e}", exc_info=True)
        db.rollback()
        raise  

async def trigger_full_sync_for_item(db: Session, sync_item: SyncItem) -> Dict[str, Any]:
    """
    Déclenche une synchronisation complète pour un item:
    Met à jour le statut, récupère les données brutes.
    
    Args:
        db: Session de base de données
        sync_item: Item de synchronisation à synchroniser
        
    Returns:
        Dict: Rapport détaillé de la synchronisation
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
        "steps": {},
        "raw_transactions": []  # Pour stocker les transactions pour enrichissement
    }

    try:
        # 1. Récupérer le token Bridge
        ctx_logger.info("Étape 1: Récupération du token Bridge")
        from user_service.services.bridge import get_bridge_token
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

        # 4. Récupérer les comptes (SQL)
        ctx_logger.info("Étape 4: Récupération et mise à jour SQL des comptes")
        sql_accounts, bridge_accounts_list = await fetch_and_update_sql_accounts(db, sync_item, access_token)
        sync_report["steps"]["fetch_sql_accounts"] = {"status": "success", "count": len(sql_accounts)}

        if not sql_accounts:
            ctx_logger.warning(f"Aucun compte associé à l'item {bridge_item_id}. Synchronisation terminée.")
            sync_report["status"] = "warning_no_accounts"
            sync_report["message"] = "No accounts found for this item."
            return sync_report

        # 5. Récupérer les informations des comptes si disponible (IBAN, etc.)
        ctx_logger.info("Étape 5: Récupération et stockage des informations des comptes (IBAN, etc.)")
        try:
            # Import au besoin pour éviter les dépendances circulaires
            from user_service.services.bridge import get_accounts_information
            accounts_info = await get_accounts_information(db, user_id, access_token)
            
            if accounts_info:
                # Traiter et stocker les informations des comptes
                from sync_service.sync_manager.account_handler import store_accounts_information
                info_result = await store_accounts_information(db, user_id, accounts_info)
                sync_report["steps"]["fetch_account_info"] = info_result
            else:
                sync_report["steps"]["fetch_account_info"] = {"status": "skipped", "message": "No account information available"}
        except Exception as info_error:
            ctx_logger.warning(f"Erreur lors de la récupération des informations de compte: {info_error}")
            sync_report["steps"]["fetch_account_info"] = {"status": "failed", "message": str(info_error)}

        # 6. Récupérer et Stocker les Catégories
        ctx_logger.info("Étape 6: Récupération et stockage des catégories")
        try:
            # Tenter de récupérer les catégories avec gestion d'erreur 403
            categories_list = []
            try:
                from user_service.services.bridge import get_bridge_categories
                categories_list = await get_bridge_categories(db, user_id)
            except HTTPException as http_exc:
                if http_exc.status_code == 403:
                    ctx_logger.warning("Accès aux catégories refusé (403). Endpoint probablement non disponible dans votre formule Bridge.")
                    sync_report["steps"]["store_categories"] = {"status": "skipped", "message": "Categories API endpoint not available in your subscription"}
                else:
                    raise  # Re-raise toute autre erreur HTTP

            # Stocker les catégories dans la BDD SQL
            if categories_list:
                from sync_service.sync_manager.category_handler import store_bridge_categories
                cat_result = await store_bridge_categories(db, categories_list)
                sync_report["steps"]["store_categories"] = cat_result
            elif "store_categories" not in sync_report["steps"]:
                sync_report["steps"]["store_categories"] = {"status": "skipped", "message": "No categories data"}
        except Exception as e:
            ctx_logger.error(f"Erreur lors de la récupération/stockage des catégories: {e}", exc_info=True)
            sync_report["steps"]["store_categories"] = {"status": "failed", "message": str(e)}
            overall_status = "partial"

        # 7. Récupérer les transactions pour tous les comptes
        ctx_logger.info("Étape 7: Synchronisation des transactions")
        try:
            # Synchroniser les transactions pour tous les comptes de cet utilisateur
            tx_sync_result = await force_sync_all_accounts(db, user_id)

            # Stocker les résultats de synchronisation
            sync_report["steps"]["sync_transactions"] = tx_sync_result
            sync_report["raw_transactions"] = tx_sync_result.get("transactions", [])
            
        except Exception as tx_error:
            ctx_logger.error(f"Erreur lors de la synchronisation des transactions: {tx_error}", exc_info=True)
            sync_report["steps"]["sync_transactions"] = {
                "status": "error",
                "message": str(tx_error)
            }
            overall_status = "partial"

        # 8. Récupérer les stocks/titres si applicables
        ctx_logger.info("Étape 8: Récupération des stocks/titres si disponibles")
        try:
            from sync_service.sync_manager.stock_handler import sync_all_stocks
            stocks_result = await sync_all_stocks(db, user_id)
            sync_report["steps"]["sync_stocks"] = stocks_result
        except Exception as stocks_error:
            ctx_logger.error(f"Erreur lors de la récupération des stocks: {stocks_error}", exc_info=True)
            sync_report["steps"]["sync_stocks"] = {"status": "error", "message": str(stocks_error)}
            # Ne pas changer le statut global car c'est une fonctionnalité optionnelle

        # 9. Récupérer les insights si disponibles
        ctx_logger.info("Étape 9: Récupération des insights si disponibles")
        try:
            from user_service.services.bridge import get_bridge_insights
            from sync_service.sync_manager.insight_handler import store_bridge_insights

            insights = await get_bridge_insights(db, user_id)
            if insights:
                insights_result = await store_bridge_insights(db, user_id, insights)
                sync_report["steps"]["sync_insights"] = insights_result
            else:
                sync_report["steps"]["sync_insights"] = {"status": "skipped", "message": "No insights available"}
        except Exception as insights_error:
            ctx_logger.error(f"Erreur lors de la récupération des insights: {insights_error}", exc_info=True)
            sync_report["steps"]["sync_insights"] = {"status": "error", "message": str(insights_error)}
            # Ne pas changer le statut global car c'est une fonctionnalité optionnelle

        # 10. Synchroniser les transactions dans Elasticsearch via enrichment_service
        ctx_logger.info("Étape 10: Synchronisation Elasticsearch via enrichment_service")
        try:
            import aiohttp
            from user_service.core.security import create_access_token
            from datetime import timedelta

            enrichment_url = f"http://harena_enrichment_service:3005/api/v1/enrichment/elasticsearch/sync-user/{user_id}"

            # Générer un JWT utilisateur Harena pour l'authentification
            user_jwt = create_access_token(
                subject=user_id,
                permissions=["chat:write"],
                expires_delta=timedelta(minutes=30)
            )
            headers = {
                "Authorization": f"Bearer {user_jwt}",
                "Content-Type": "application/json"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(enrichment_url, headers=headers, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        enrichment_result = await response.json()
                        sync_report["steps"]["sync_elasticsearch"] = {
                            "status": "success",
                            "transactions_indexed": enrichment_result.get("transactions_indexed", 0),
                            "accounts_indexed": enrichment_result.get("accounts_indexed", 0)
                        }
                        ctx_logger.info(f"✅ Elasticsearch sync réussi: {enrichment_result.get('transactions_indexed', 0)} transactions indexées")
                    else:
                        error_text = await response.text()
                        ctx_logger.warning(f"⚠️ Échec sync Elasticsearch: {response.status} - {error_text}")
                        sync_report["steps"]["sync_elasticsearch"] = {
                            "status": "warning",
                            "message": f"HTTP {response.status}: {error_text}"
                        }
        except Exception as es_error:
            ctx_logger.error(f"Erreur lors de la synchronisation Elasticsearch: {es_error}", exc_info=True)
            sync_report["steps"]["sync_elasticsearch"] = {"status": "error", "message": str(es_error)}
            # Ne pas changer le statut global car la sync PostgreSQL a réussi

        # 11. Calculer le profil budgétaire via budget_profiling_service
        ctx_logger.info("Étape 11: Calcul du profil budgétaire via budget_profiling_service")
        try:
            import aiohttp
            from user_service.core.security import create_access_token
            from datetime import timedelta

            budget_url = f"http://harena_budget_profiling_service:3006/api/v1/budget/profile/analyze"

            # Générer un JWT utilisateur Harena pour l'authentification
            user_jwt = create_access_token(
                subject=user_id,
                permissions=["chat:write"],
                expires_delta=timedelta(minutes=30)
            )
            headers = {
                "Authorization": f"Bearer {user_jwt}",
                "Content-Type": "application/json"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(budget_url, headers=headers, timeout=aiohttp.ClientTimeout(total=180)) as response:
                    if response.status == 200:
                        budget_result = await response.json()
                        sync_report["steps"]["calculate_budget_profile"] = {
                            "status": "success",
                            "user_segment": budget_result.get("user_segment"),
                            "avg_monthly_income": budget_result.get("avg_monthly_income"),
                            "avg_monthly_expenses": budget_result.get("avg_monthly_expenses"),
                            "savings_rate": budget_result.get("savings_rate")
                        }
                        ctx_logger.info(f"✅ Profil budgétaire calculé: segment={budget_result.get('user_segment')}, taux d'épargne={budget_result.get('savings_rate', 0):.1f}%")
                    else:
                        error_text = await response.text()
                        ctx_logger.warning(f"⚠️ Échec calcul profil budgétaire: {response.status} - {error_text}")
                        sync_report["steps"]["calculate_budget_profile"] = {
                            "status": "warning",
                            "message": f"HTTP {response.status}: {error_text}"
                        }
        except Exception as budget_error:
            ctx_logger.error(f"Erreur lors du calcul du profil budgétaire: {budget_error}", exc_info=True)
            sync_report["steps"]["calculate_budget_profile"] = {"status": "error", "message": str(budget_error)}
            # Ne pas changer le statut global car c'est une fonctionnalité optionnelle

        # Finaliser le rapport
        sync_report["status"] = overall_status
        ctx_logger.info(f"--- Fin synchronisation COMPLÈTE pour item {bridge_item_id} --- Statut final: {overall_status}")
        return sync_report

    except HTTPException as http_exc:
        ctx_logger.error(f"Erreur HTTP lors de trigger_full_sync_for_item: {http_exc.status_code} - {http_exc.detail}", exc_info=True)
        sync_report["status"] = "failed"
        sync_report["message"] = f"HTTP Error {http_exc.status_code}: {http_exc.detail}"
        return sync_report
    except Exception as e:
        ctx_logger.error(f"Erreur générale majeure lors de trigger_full_sync_for_item: {e}", exc_info=True)
        sync_report["status"] = "failed"
        sync_report["message"] = f"Unexpected error during sync: {str(e)}"
        return sync_report