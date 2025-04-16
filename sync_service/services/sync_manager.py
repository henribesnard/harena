# sync_service/services/sync_manager.py
import logging
import traceback
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional, Tuple

from sync_service.models.sync import SyncItem, SyncAccount
from user_service.models.user import User, BridgeConnection
from user_service.services import bridge as bridge_service
from sync_service.services import transaction_sync
from sync_service.utils.logging import get_contextual_logger
from user_service.core.config import settings

# Configuration du logger
logger = logging.getLogger(__name__)

# Liste des statuts qui requièrent une action utilisateur
ACTION_REQUIRED_STATUSES = [402, 429, 1010]

async def create_or_update_sync_item(db: Session, user_id: int, bridge_item_id: int, item_data: Dict[str, Any]) -> SyncItem:
    """Créer ou mettre à jour un item de synchronisation."""
    # Logger avec contexte
    ctx_logger = get_contextual_logger("sync_service.sync_manager", user_id=user_id, bridge_item_id=bridge_item_id)
    ctx_logger.info(f"Création ou mise à jour de l'item de synchronisation")
    
    try:
        # Vérifier si l'item existe déjà
        sync_item = db.query(SyncItem).filter(SyncItem.bridge_item_id == bridge_item_id).first()
        
        status = item_data.get("status", 0)
        status_code_info = item_data.get("status_code_info")
        
        if sync_item:
            ctx_logger.info(f"Item existant trouvé: id={sync_item.id}")
            # Mettre à jour l'item existant
            sync_item.status = status
            sync_item.status_code_info = status_code_info
            sync_item.needs_user_action = status in ACTION_REQUIRED_STATUSES
            
            if "status_code_description" in item_data:
                sync_item.status_description = item_data["status_code_description"]
            
            if "last_successful_refresh" in item_data and item_data["last_successful_refresh"]:
                try:
                    sync_item.last_successful_refresh = datetime.fromisoformat(
                        item_data["last_successful_refresh"].replace('Z', '+00:00')
                    )
                except (ValueError, TypeError) as e:
                    ctx_logger.warning(f"Impossible de parser last_successful_refresh: {e}")
            
            if "last_try_refresh" in item_data and item_data["last_try_refresh"]:
                try:
                    sync_item.last_try_refresh = datetime.fromisoformat(
                        item_data["last_try_refresh"].replace('Z', '+00:00')
                    )
                except (ValueError, TypeError) as e:
                    ctx_logger.warning(f"Impossible de parser last_try_refresh: {e}")
                    
            if "provider_id" in item_data:
                sync_item.provider_id = item_data["provider_id"]
                
            if "account_types" in item_data:
                sync_item.account_types = item_data["account_types"]
        else:
            # Créer un nouvel item
            ctx_logger.info(f"Création d'un nouvel item de synchronisation")
            sync_item = SyncItem(
                user_id=user_id,
                bridge_item_id=bridge_item_id,
                status=status,
                status_code_info=status_code_info,
                status_description=item_data.get("status_code_description"),
                provider_id=item_data.get("provider_id"),
                account_types=item_data.get("account_types"),
                needs_user_action=status in ACTION_REQUIRED_STATUSES
            )
            
            if "last_successful_refresh" in item_data and item_data["last_successful_refresh"]:
                try:
                    sync_item.last_successful_refresh = datetime.fromisoformat(
                        item_data["last_successful_refresh"].replace('Z', '+00:00')
                    )
                except (ValueError, TypeError) as e:
                    ctx_logger.warning(f"Impossible de parser last_successful_refresh: {e}")
                
            if "last_try_refresh" in item_data and item_data["last_try_refresh"]:
                try:
                    sync_item.last_try_refresh = datetime.fromisoformat(
                        item_data["last_try_refresh"].replace('Z', '+00:00')
                    )
                except (ValueError, TypeError) as e:
                    ctx_logger.warning(f"Impossible de parser last_try_refresh: {e}")
        
        db.add(sync_item)
        db.commit()
        db.refresh(sync_item)
        
        ctx_logger.info(f"Item enregistré avec succès: id={sync_item.id}")
        
        # Si c'est un nouvel item, récupérer les comptes associés
        if not sync_item.accounts:
            ctx_logger.info(f"Récupération des comptes pour le nouvel item")
            try:
                token_data = await bridge_service.get_bridge_token(db, user_id)
                access_token = token_data["access_token"]
                
                await fetch_and_update_accounts(db, sync_item, access_token)
            except Exception as e:
                ctx_logger.error(f"Erreur lors de la récupération des comptes: {str(e)}")
                ctx_logger.error(traceback.format_exc())
        
        return sync_item
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la création/mise à jour de l'item: {str(e)}")
        ctx_logger.error(traceback.format_exc())
        db.rollback()
        raise

async def update_item_status(db: Session, sync_item: SyncItem, status: int, status_code_info: Optional[str] = None) -> SyncItem:
    """Mettre à jour le statut d'un item."""
    # Logger avec contexte
    ctx_logger = get_contextual_logger("sync_service.sync_manager", 
                                      user_id=sync_item.user_id, 
                                      bridge_item_id=sync_item.bridge_item_id)
    ctx_logger.info(f"Mise à jour du statut de l'item: status={status}, status_code_info={status_code_info}")
    
    try:
        # Mettre à jour les informations de statut
        sync_item.status = status
        if status_code_info:
            sync_item.status_code_info = status_code_info
        
        # Déterminer si une action utilisateur est requise
        sync_item.needs_user_action = status in ACTION_REQUIRED_STATUSES
        
        # Si le statut est OK, mettre à jour la date de dernière synchronisation réussie
        if status == 0:
            sync_item.last_successful_refresh = datetime.now(timezone.utc)
            ctx_logger.info(f"Statut OK, mise à jour de last_successful_refresh")
        
        # Dans tous les cas, mettre à jour la date de dernière tentative
        sync_item.last_try_refresh = datetime.now(timezone.utc)
        
        db.add(sync_item)
        db.commit()
        db.refresh(sync_item)
        
        ctx_logger.info(f"Statut de l'item mis à jour avec succès")
        return sync_item
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la mise à jour du statut: {str(e)}")
        ctx_logger.error(traceback.format_exc())
        db.rollback()
        raise

async def trigger_full_sync_for_item(db: Session, sync_item: SyncItem) -> Dict[str, Any]:
    """Déclencher une synchronisation complète pour un item."""
    # Logger avec contexte
    ctx_logger = get_contextual_logger("sync_service.sync_manager", 
                                      user_id=sync_item.user_id, 
                                      bridge_item_id=sync_item.bridge_item_id)
    ctx_logger.info(f"Déclenchement d'une synchronisation complète pour l'item")
    
    try:
        # Récupérer les informations d'authentification Bridge
        user_id = sync_item.user_id
        token_data = await bridge_service.get_bridge_token(db, user_id)
        
        # Mettre à jour les informations de l'item depuis Bridge API
        ctx_logger.info(f"Récupération des informations à jour de l'item depuis Bridge API")
        item_info = await get_bridge_item_info(db, sync_item, token_data["access_token"])
        
        if item_info:
            # Mettre à jour le statut de l'item
            sync_item = await update_item_status(
                db, 
                sync_item, 
                item_info.get("status", sync_item.status),
                item_info.get("status_code_info", sync_item.status_code_info)
            )
        
        # Si l'item est en erreur, ne pas continuer
        if sync_item.status != 0:
            ctx_logger.warning(f"L'item est en erreur (status={sync_item.status}), synchronisation annulée")
            return {
                "status": "error",
                "message": f"Item in error state: {sync_item.status_code_info}",
                "item_id": sync_item.bridge_item_id
            }
        
        # Vérifier et mettre à jour les comptes de l'item
        ctx_logger.info(f"Récupération et mise à jour des comptes de l'item")
        accounts = await fetch_and_update_accounts(db, sync_item, token_data["access_token"])
        
        if not accounts:
            ctx_logger.warning(f"Aucun compte trouvé pour l'item, synchronisation annulée")
            return {
                "status": "warning",
                "message": "No accounts found for item",
                "item_id": sync_item.bridge_item_id
            }
        
        # Synchroniser les transactions pour chaque compte
        ctx_logger.info(f"Démarrage de la synchronisation des transactions pour {len(accounts)} comptes")
        
        sync_results = []
        for account in accounts:
            ctx_logger.info(f"Synchronisation du compte {account.bridge_account_id}")
            try:
                account_result = await transaction_sync.sync_account_transactions(db, account)
                sync_results.append({
                    "account_id": account.bridge_account_id,
                    "result": account_result
                })
            except Exception as acc_error:
                ctx_logger.error(f"Erreur lors de la synchronisation du compte {account.bridge_account_id}: {str(acc_error)}")
                ctx_logger.error(traceback.format_exc())
                sync_results.append({
                    "account_id": account.bridge_account_id,
                    "result": {"status": "error", "errors": str(acc_error)}
                })
        
        # Déterminer le statut global
        overall_status = "success"
        errors_count = 0
        for result in sync_results:
            if result["result"].get("status") == "error":
                overall_status = "partial"
                errors_count += 1
        
        if errors_count == len(sync_results):
            overall_status = "error"
        
        ctx_logger.info(f"Synchronisation terminée avec statut: {overall_status}")
        
        return {
            "status": overall_status,
            "message": f"Sync completed with {errors_count} errors out of {len(sync_results)} accounts",
            "item_id": sync_item.bridge_item_id,
            "accounts_synced": len(sync_results),
            "errors": errors_count,
            "details": sync_results
        }
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la synchronisation complète: {str(e)}")
        ctx_logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "item_id": sync_item.bridge_item_id
        }

async def fetch_and_update_accounts(db: Session, sync_item: SyncItem, access_token: str) -> List[SyncAccount]:
    """Récupérer et mettre à jour les comptes d'un item."""
    # Logger avec contexte
    ctx_logger = get_contextual_logger("sync_service.sync_manager", 
                                     user_id=sync_item.user_id, 
                                     bridge_item_id=sync_item.bridge_item_id)
    ctx_logger.info(f"Récupération et mise à jour des comptes de l'item")
    
    try:
        import httpx
        
        # Appel à l'API Bridge pour récupérer les comptes
        url = f"{settings.BRIDGE_API_URL}/aggregation/accounts?item_id={sync_item.bridge_item_id}"
        headers = {
            "accept": "application/json",
            "Bridge-Version": settings.BRIDGE_API_VERSION,
            "Client-Id": settings.BRIDGE_CLIENT_ID,
            "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
            "authorization": f"Bearer {access_token}"
        }
        
        ctx_logger.debug(f"Requête API Bridge: {url}")
        
        accounts = []
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code != 200:
                ctx_logger.error(f"Erreur API Bridge: {response.status_code} {response.text}")
                return accounts
                
            accounts_data = response.json()
            
            if "resources" not in accounts_data:
                ctx_logger.error(f"Format de réponse invalide: {accounts_data}")
                return accounts
            
            ctx_logger.info(f"Récupération de {len(accounts_data['resources'])} comptes depuis Bridge API")
                
            # Traiter chaque compte
            for account_data in accounts_data["resources"]:
                bridge_account_id = account_data["id"]
                ctx_logger.debug(f"Traitement du compte {bridge_account_id}")
                
                # Vérifier si le compte existe déjà
                sync_account = db.query(SyncAccount).filter(
                    SyncAccount.bridge_account_id == bridge_account_id
                ).first()
                
                if sync_account:
                    # Mettre à jour le compte existant
                    ctx_logger.debug(f"Mise à jour du compte existant {bridge_account_id}")
                    sync_account.account_name = account_data.get("name")
                    sync_account.account_type = account_data.get("type")
                else:
                    # Créer un nouveau compte
                    ctx_logger.info(f"Création d'un nouveau compte {bridge_account_id}")
                    sync_account = SyncAccount(
                        item_id=sync_item.id,
                        bridge_account_id=bridge_account_id,
                        account_name=account_data.get("name"),
                        account_type=account_data.get("type")
                    )
                
                db.add(sync_account)
                accounts.append(sync_account)
            
            db.commit()
            for account in accounts:
                db.refresh(account)
            
            ctx_logger.info(f"Mise à jour de {len(accounts)} comptes terminée avec succès")
            return accounts
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la récupération des comptes: {str(e)}")
        ctx_logger.error(traceback.format_exc())
        db.rollback()
        return []

async def get_bridge_item_info(db: Session, sync_item: SyncItem, access_token: str) -> Optional[Dict[str, Any]]:
    """Récupérer les informations détaillées d'un item depuis Bridge API."""
    # Logger avec contexte
    ctx_logger = get_contextual_logger("sync_service.sync_manager", 
                                     user_id=sync_item.user_id, 
                                     bridge_item_id=sync_item.bridge_item_id)
    ctx_logger.info(f"Récupération des informations détaillées de l'item depuis Bridge API")
    
    try:
        import httpx
        
        # Appel à l'API Bridge pour récupérer les informations de l'item
        url = f"{settings.BRIDGE_API_URL}/aggregation/items/{sync_item.bridge_item_id}"
        headers = {
            "accept": "application/json",
            "Bridge-Version": settings.BRIDGE_API_VERSION,
            "Client-Id": settings.BRIDGE_CLIENT_ID,
            "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
            "authorization": f"Bearer {access_token}"
        }
        
        ctx_logger.debug(f"Requête API Bridge: {url}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code != 200:
                ctx_logger.error(f"Erreur API Bridge: {response.status_code} {response.text}")
                return None
                
            item_data = response.json()
            ctx_logger.debug(f"Réponse API Bridge: {item_data}")
            
            return item_data
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la récupération des informations de l'item: {str(e)}")
        ctx_logger.error(traceback.format_exc())
        return None

async def get_user_sync_status(db: Session, user_id: int) -> Dict[str, Any]:
    """Obtenir l'état de synchronisation pour un utilisateur."""
    # Logger avec contexte
    ctx_logger = get_contextual_logger("sync_service.sync_manager", user_id=user_id)
    ctx_logger.info(f"Récupération de l'état de synchronisation pour l'utilisateur")
    
    try:
        # Récupérer tous les items de l'utilisateur
        sync_items = db.query(SyncItem).filter(SyncItem.user_id == user_id).all()
        ctx_logger.info(f"Nombre d'items trouvés: {len(sync_items)}")
        
        # Récupérer tous les comptes
        account_ids = []
        for item in sync_items:
            account_ids.extend([account.id for account in item.accounts])
        
        sync_accounts = db.query(SyncAccount).filter(SyncAccount.id.in_(account_ids)).all() if account_ids else []
        ctx_logger.info(f"Nombre de comptes trouvés: {len(sync_accounts)}")
        
        # Récupérer les comptes par item pour inclusion dans la réponse
        item_accounts = {}
        for account in sync_accounts:
            if account.item_id not in item_accounts:
                item_accounts[account.item_id] = []
            
            last_sync_iso = None
            if account.last_sync_timestamp:
                last_sync_iso = account.last_sync_timestamp.isoformat()
            
            item_accounts[account.item_id].append({
                "id": account.id,
                "bridge_account_id": account.bridge_account_id,
                "name": account.account_name,
                "type": account.account_type,
                "last_sync": last_sync_iso
            })
        
        # Déterminer l'état global
        needs_action = any(item.needs_user_action for item in sync_items)
        
        # Trouver la dernière synchronisation réussie
        last_sync = None
        successful_refreshes = [item.last_successful_refresh for item in sync_items if item.last_successful_refresh]
        if successful_refreshes:
            last_sync = max(successful_refreshes)
        
        # Calculer le nombre de jours depuis la dernière synchro
        days_since_last_sync = None
        if last_sync:
            days_since_last_sync = (datetime.now(timezone.utc) - last_sync).days
        
        # Préparer les informations sur tous les items
        all_items = []
        for item in sync_items:
            # Convertir les timestamps en ISO pour la sérialisation JSON
            last_successful_refresh_iso = None
            if item.last_successful_refresh:
                last_successful_refresh_iso = item.last_successful_refresh.isoformat()
            
            last_try_refresh_iso = None
            if item.last_try_refresh:
                last_try_refresh_iso = item.last_try_refresh.isoformat()
            
            all_items.append({
                "id": item.id,
                "bridge_item_id": item.bridge_item_id,
                "status": item.status,
                "status_code_info": item.status_code_info,
                "status_description": item.status_description,
                "provider_id": item.provider_id,
                "account_types": item.account_types,
                "needs_user_action": item.needs_user_action,
                "last_successful_refresh": last_successful_refresh_iso,
                "last_try_refresh": last_try_refresh_iso,
                "accounts": item_accounts.get(item.id, [])
            })
        
        # Filtrer les items nécessitant une action
        items_needing_action = [
            item for item in all_items if item["needs_user_action"]
        ]
        
        # Préparer la réponse
        status_response = {
            "total_items": len(sync_items),
            "total_accounts": len(sync_accounts),
            "needs_user_action": needs_action,
            "items_needing_action": items_needing_action,
            "items": all_items,
            "last_successful_sync": last_sync.isoformat() if last_sync else None,
            "days_since_last_sync": days_since_last_sync
        }
        
        ctx_logger.info(f"État de synchronisation récupéré avec succès")
        return status_response
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la récupération de l'état de synchronisation: {str(e)}")
        ctx_logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "total_items": 0,
            "total_accounts": 0,
            "needs_user_action": False,
            "items_needing_action": [],
            "items": []
        }

async def create_reconnect_session(db: Session, user_id: int, bridge_item_id: int) -> str:
    """Créer une session de reconnexion pour un item."""
    # Logger avec contexte
    ctx_logger = get_contextual_logger("sync_service.sync_manager", 
                                     user_id=user_id, 
                                     bridge_item_id=bridge_item_id)
    ctx_logger.info(f"Création d'une session de reconnexion pour l'item")
    
    try:
        # Vérifier que l'item appartient bien à l'utilisateur
        sync_item = db.query(SyncItem).filter(
            SyncItem.user_id == user_id,
            SyncItem.bridge_item_id == bridge_item_id
        ).first()
        
        if not sync_item:
            ctx_logger.error(f"Item non trouvé pour l'utilisateur")
            raise ValueError(f"Item {bridge_item_id} not found for user {user_id}")
        
        # Utiliser l'URL de base de l'application depuis les paramètres
        base_url = settings.WEBHOOK_BASE_URL or "https://harenabackend-ab1b255e55c6.herokuapp.com"
        callback_url = f"{base_url}/reconnect-callback"
        ctx_logger.info(f"URL de callback: {callback_url}")
        
        # Créer une session Connect avec l'item_id
        connect_url = await bridge_service.create_connect_session(
            db,
            user_id,
            item_id=bridge_item_id,
            callback_url=callback_url,
        )
        
        ctx_logger.info(f"Session de reconnexion créée avec succès")
        return connect_url
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la création de la session de reconnexion: {str(e)}")
        ctx_logger.error(traceback.format_exc())
        raise

async def check_stale_items(db: Session) -> List[Dict[str, Any]]:
    """Vérifie les items qui n'ont pas été mis à jour depuis longtemps."""
    logger.info("Vérification des items obsolètes")
    
    try:
        # Définir le seuil d'obsolescence (7 jours)
        stale_threshold = datetime.now(timezone.utc) - timedelta(days=7)
        
        # Rechercher les items qui n'ont pas été mis à jour depuis le seuil
        stale_items = db.query(SyncItem).filter(
            (SyncItem.last_successful_refresh < stale_threshold) | 
            (SyncItem.last_successful_refresh == None)
        ).all()
        
        logger.info(f"Nombre d'items obsolètes trouvés: {len(stale_items)}")
        
        results = []
        # Pour chaque item obsolète, déconnecter ou forcer un rafraîchissement
        for item in stale_items:
            ctx_logger = get_contextual_logger("sync_service.sync_manager", 
                                            user_id=item.user_id, 
                                            bridge_item_id=item.bridge_item_id)
            try:
                # Si l'item est en erreur, le marquer pour reconnexion
                if item.status != 0:
                    ctx_logger.info(f"Item en erreur (status={item.status}), marqué pour reconnexion")
                    results.append({
                        "user_id": item.user_id,
                        "bridge_item_id": item.bridge_item_id,
                        "status": "needs_reconnect",
                        "message": f"Item in error: {item.status_code_info}"
                    })
                else:
                    # Sinon, forcer un rafraîchissement
                    ctx_logger.info(f"Forçage d'un rafraîchissement pour l'item obsolète")
                    await trigger_full_sync_for_item(db, item)
                    results.append({
                        "user_id": item.user_id,
                        "bridge_item_id": item.bridge_item_id,
                        "status": "refresh_triggered",
                        "message": "Refresh triggered for stale item"
                    })
            except Exception as item_error:
                ctx_logger.error(f"Erreur lors du traitement de l'item obsolète: {str(item_error)}")
                results.append({
                    "user_id": item.user_id,
                    "bridge_item_id": item.bridge_item_id,
                    "status": "error",
                    "message": str(item_error)
                })
        
        return results
    except Exception as e:
        logger.error(f"Erreur lors de la vérification des items obsolètes: {str(e)}")
        logger.error(traceback.format_exc())
        return [{
            "status": "error",
            "message": str(e)
        }]