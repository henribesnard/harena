"""
Gestionnaire des comptes bancaires.

Ce module gère la récupération et la mise à jour des comptes bancaires
depuis Bridge API vers la base de données SQL.
"""

import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional, Tuple

from sync_service.models.sync import SyncItem, SyncAccount, AccountInformation, LoanDetail
from sync_service.utils.logging import get_contextual_logger

logger = logging.getLogger(__name__)

async def fetch_and_update_sql_accounts(
    db: Session,
    sync_item: SyncItem,
    access_token: str
) -> Tuple[List[SyncAccount], List[Dict[str, Any]]]:
    """
    Récupère et met à jour les comptes dans la base SQL.
    
    Args:
        db: Session de base de données
        sync_item: Item de synchronisation
        access_token: Token d'accès Bridge
        
    Returns:
        Tuple: (Liste des comptes SQL, Liste des données de comptes brutes)
    """
    user_id = sync_item.user_id
    bridge_item_id = sync_item.bridge_item_id
    ctx_logger = get_contextual_logger("sync_service.account_handler", user_id=user_id, bridge_item_id=bridge_item_id)
    ctx_logger.info(f"Récupération et mise à jour des comptes pour l'item {bridge_item_id}")
    
    try:
        # Récupérer les comptes depuis Bridge API
        from user_service.services.bridge import get_bridge_accounts
        bridge_accounts = await get_bridge_accounts(db, user_id, access_token, bridge_item_id)
        
        if not bridge_accounts:
            ctx_logger.warning(f"Aucun compte récupéré pour l'item {bridge_item_id}")
            return [], []
            
        ctx_logger.info(f"Récupération de {len(bridge_accounts)} comptes depuis Bridge API")
        
        # Mettre à jour les comptes SQL
        sql_accounts = await update_sql_accounts(db, sync_item, bridge_accounts)
        
        return sql_accounts, bridge_accounts
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la récupération et mise à jour des comptes: {e}", exc_info=True)
        return [], []

async def update_sql_accounts(
    db: Session,
    sync_item: SyncItem,
    bridge_accounts: List[Dict[str, Any]]
) -> List[SyncAccount]:
    """
    Met à jour les comptes dans la base SQL.
    
    Args:
        db: Session de base de données
        sync_item: Item de synchronisation
        bridge_accounts: Liste des comptes depuis Bridge API
        
    Returns:
        List: Liste des comptes SQL mis à jour
    """
    user_id = sync_item.user_id
    item_id = sync_item.id
    bridge_item_id = sync_item.bridge_item_id
    ctx_logger = get_contextual_logger("sync_service.account_handler", user_id=user_id, bridge_item_id=bridge_item_id)
    
    # Récupérer les comptes existants
    existing_accounts = db.query(SyncAccount).filter(SyncAccount.item_id == item_id).all()
    existing_account_map = {acc.bridge_account_id: acc for acc in existing_accounts}
    
    # Liste des comptes mis à jour
    updated_accounts = []
    
    for bridge_acc in bridge_accounts:
        bridge_acc_id = bridge_acc.get("id")
        if not bridge_acc_id:
            ctx_logger.warning(f"ID de compte manquant dans les données Bridge, ignoré")
            continue
            
        # Mise à jour ou création
        if bridge_acc_id in existing_account_map:
            # Mise à jour
            account = existing_account_map[bridge_acc_id]
            account.account_name = bridge_acc.get("name", account.account_name)
            account.account_type = bridge_acc.get("type", account.account_type)
            account.last_sync_timestamp = datetime.now(timezone.utc)
            account.balance = bridge_acc.get("balance", account.balance)
            account.currency_code = bridge_acc.get("currency_code", account.currency_code)
            
            # Vérifier si une date de transaction est fournie
            if "last_transaction_date" in bridge_acc:
                try:
                    last_tx_date = bridge_acc["last_transaction_date"]
                    if isinstance(last_tx_date, str):
                        account.last_transaction_date = datetime.fromisoformat(last_tx_date.replace('Z', '+00:00'))
                    else:
                        account.last_transaction_date = last_tx_date
                except (ValueError, TypeError):
                    ctx_logger.warning(f"Format de date invalide pour last_transaction_date")
            
            # Mise à jour des détails de prêt si applicable
            if bridge_acc.get("type") == "loan" and "loan_details" in bridge_acc:
                await update_loan_details(db, account, bridge_acc["loan_details"])
                
        else:
            # Création
            ctx_logger.info(f"Création du compte SQL pour Bridge account {bridge_acc_id}")
            
            # Extraire la date de dernière transaction si disponible
            last_tx_date = None
            if "last_transaction_date" in bridge_acc:
                try:
                    last_tx_date_val = bridge_acc["last_transaction_date"]
                    if isinstance(last_tx_date_val, str):
                        last_tx_date = datetime.fromisoformat(last_tx_date_val.replace('Z', '+00:00'))
                    else:
                        last_tx_date = last_tx_date_val
                except (ValueError, TypeError):
                    ctx_logger.warning(f"Format de date invalide pour last_transaction_date")
            
            account = SyncAccount(
                item_id=item_id,
                bridge_account_id=bridge_acc_id,
                account_name=bridge_acc.get("name", f"Compte {bridge_acc_id}"),
                account_type=bridge_acc.get("type", "unknown"),
                last_sync_timestamp=datetime.now(timezone.utc),
                last_transaction_date=last_tx_date,
                balance=bridge_acc.get("balance"),
                currency_code=bridge_acc.get("currency_code")
            )
            db.add(account)
            
            # Commit pour avoir un ID et pouvoir ajouter les détails de prêt
            db.flush()
            
            # Ajouter les détails de prêt si applicable
            if bridge_acc.get("type") == "loan" and "loan_details" in bridge_acc:
                await create_loan_details(db, account, bridge_acc["loan_details"])
            
        updated_accounts.append(account)
    
    # Commit des changements
    try:
        db.commit()
        ctx_logger.info(f"Mise à jour SQL de {len(updated_accounts)} comptes terminée avec succès")
    except Exception as e:
        db.rollback()
        ctx_logger.error(f"Erreur lors du commit des comptes SQL: {e}", exc_info=True)
        
    return updated_accounts

async def update_loan_details(db: Session, account: SyncAccount, loan_data: Dict[str, Any]) -> None:
    """
    Met à jour les détails d'un prêt existant.
    
    Args:
        db: Session de base de données
        account: Compte associé au prêt
        loan_data: Données du prêt depuis Bridge API
    """
    try:
        # Vérifier si des détails de prêt existent déjà
        loan_details = db.query(LoanDetail).filter(LoanDetail.account_id == account.id).first()
        
        if loan_details:
            # Mise à jour
            loan_details.interest_rate = loan_data.get("interest_rate", loan_details.interest_rate)
            loan_details.next_payment_date = parse_date(loan_data.get("next_payment_date"))
            loan_details.next_payment_amount = loan_data.get("next_payment_amount", loan_details.next_payment_amount)
            loan_details.maturity_date = parse_date(loan_data.get("maturity_date"))
            loan_details.opening_date = parse_date(loan_data.get("opening_date"))
            loan_details.borrowed_capital = loan_data.get("borrowed_capital", loan_details.borrowed_capital)
            loan_details.repaid_capital = loan_data.get("repaid_capital", loan_details.repaid_capital)
            loan_details.remaining_capital = loan_data.get("remaining_capital", loan_details.remaining_capital)
            loan_details.total_estimated_interests = loan_data.get("total_estimated_interests", loan_details.total_estimated_interests)
            
            db.add(loan_details)
        else:
            # Création (ce cas ne devrait pas se produire car create_loan_details est appelé à la création)
            await create_loan_details(db, account, loan_data)
            
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des détails de prêt pour le compte {account.id}: {e}", exc_info=True)

async def create_loan_details(db: Session, account: SyncAccount, loan_data: Dict[str, Any]) -> None:
    """
    Crée les détails d'un nouveau prêt.
    
    Args:
        db: Session de base de données
        account: Compte associé au prêt
        loan_data: Données du prêt depuis Bridge API
    """
    try:
        loan_details = LoanDetail(
            account_id=account.id,
            interest_rate=loan_data.get("interest_rate"),
            next_payment_date=parse_date(loan_data.get("next_payment_date")),
            next_payment_amount=loan_data.get("next_payment_amount"),
            maturity_date=parse_date(loan_data.get("maturity_date")),
            opening_date=parse_date(loan_data.get("opening_date")),
            borrowed_capital=loan_data.get("borrowed_capital"),
            repaid_capital=loan_data.get("repaid_capital"),
            remaining_capital=loan_data.get("remaining_capital"),
            total_estimated_interests=loan_data.get("total_estimated_interests")
        )
        
        db.add(loan_details)
    except Exception as e:
        logger.error(f"Erreur lors de la création des détails de prêt pour le compte {account.id}: {e}", exc_info=True)

def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse une date depuis une chaîne ISO.
    
    Args:
        date_str: Chaîne de date ISO ou None
        
    Returns:
        datetime: Date parsée ou None si erreur
    """
    if not date_str:
        return None
        
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except (ValueError, TypeError, AttributeError):
        logger.warning(f"Format de date invalide: {date_str}")
        return None

async def store_accounts_information(db: Session, user_id: int, accounts_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stocke les informations de compte (IBAN, identité) dans la base SQL.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        accounts_info: Informations des comptes depuis Bridge API
        
    Returns:
        Dict: Résultat de l'opération
    """
    ctx_logger = get_contextual_logger("sync_service.account_handler", user_id=user_id)
    ctx_logger.info(f"Stockage des informations de compte pour l'utilisateur {user_id}")
    
    result = {
        "status": "pending",
        "items_processed": 0,
        "accounts_updated": 0
    }
    
    try:
        for item_info in accounts_info:
            item_id = item_info.get("item_id")
            if not item_id:
                ctx_logger.warning("Information d'item sans item_id, ignorée")
                continue
                
            # Trouver l'item correspondant
            sync_item = db.query(SyncItem).filter(
                SyncItem.bridge_item_id == item_id,
                SyncItem.user_id == user_id
            ).first()
            
            if not sync_item:
                ctx_logger.warning(f"SyncItem {item_id} non trouvé pour user {user_id}, information ignorée")
                continue
                
            # Créer ou mettre à jour l'information de compte
            first_name = item_info.get("first_name")
            last_name = item_info.get("last_name")
            accounts = item_info.get("accounts", [])
            
            # Rechercher une entrée existante ou en créer une nouvelle
            account_info = db.query(AccountInformation).filter(
                AccountInformation.item_id == sync_item.id,
                AccountInformation.user_id == user_id
            ).first()
            
            if account_info:
                # Mise à jour
                account_info.first_name = first_name
                account_info.last_name = last_name
                account_info.account_details = accounts
            else:
                # Création
                account_info = AccountInformation(
                    item_id=sync_item.id,
                    user_id=user_id,
                    first_name=first_name,
                    last_name=last_name,
                    account_details=accounts
                )
                db.add(account_info)
                
            # Mettre à jour les IBANs dans les comptes SQL
            for acc_detail in accounts:
                acc_id = acc_detail.get("id")
                iban = acc_detail.get("iban")
                
                if acc_id and iban:
                    sync_account = db.query(SyncAccount).filter(SyncAccount.bridge_account_id == acc_id).first()
                    if sync_account:
                        # Utiliser un champ séparé pour l'IBAN si ajouté au modèle SyncAccount
                        # pour l'instant, on peut le mettre dans les métadonnées si besoin
                        result["accounts_updated"] += 1
            
            result["items_processed"] += 1
            
        # Commit des changements
        db.commit()
        result["status"] = "success"
        ctx_logger.info(f"Informations de compte stockées avec succès: {result['items_processed']} items, {result['accounts_updated']} comptes")
        
        return result
    except Exception as e:
        db.rollback()
        ctx_logger.error(f"Erreur lors du stockage des informations de compte: {e}", exc_info=True)
        result["status"] = "error"
        result["error"] = str(e)
        return result

async def find_account_by_bridge_id(db: Session, bridge_account_id: int) -> Optional[SyncAccount]:
    """
    Recherche un compte par son ID Bridge.
    
    Args:
        db: Session de base de données
        bridge_account_id: ID du compte Bridge
        
    Returns:
        SyncAccount: Compte trouvé ou None si non trouvé
    """
    try:
        account = db.query(SyncAccount).filter(SyncAccount.bridge_account_id == bridge_account_id).first()
        return account
    except Exception as e:
        logger.error(f"Erreur lors de la recherche du compte {bridge_account_id}: {e}", exc_info=True)
        return None