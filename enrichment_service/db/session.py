"""
Gestionnaire de session de base de données pour le service d'enrichissement.

Ce module fournit l'accès à la base de données SQL partagée et gère
les sessions pour les opérations d'enrichissement.
"""

import logging
from contextlib import contextmanager
from sqlalchemy.orm import Session
from typing import Generator

# Utilisation de la session centralisée de db_service
from db_service.session import get_db as get_db_from_service, get_db_context as get_db_context_from_service

logger = logging.getLogger(__name__)

def get_db() -> Generator[Session, None, None]:
    """
    Obtient une session de base de données pour les endpoints FastAPI.
    
    Cette fonction est un alias vers la session centralisée de db_service
    pour maintenir la cohérence dans l'enrichissement service.
    
    Yields:
        Session: Session SQLAlchemy
    """
    yield from get_db_from_service()

@contextmanager
def get_db_context():
    """
    Gestionnaire de contexte pour les opérations de base de données.
    
    Utilise la session centralisée avec gestion automatique des transactions.
    Idéal pour les opérations d'enrichissement en arrière-plan.
    
    Yields:
        Session: Session SQLAlchemy avec gestion automatique des transactions
    """
    with get_db_context_from_service() as db:
        yield db

class EnrichmentDatabaseManager:
    """
    Gestionnaire spécialisé pour les opérations de base de données d'enrichissement.
    
    Cette classe fournit des méthodes utilitaires spécifiques aux besoins
    du service d'enrichissement, comme la gestion des lots et des transactions longues.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @contextmanager
    def get_batch_session(self, batch_size: int = 1000):
        """
        Gestionnaire de contexte pour les opérations par lots.
        
        Optimise les sessions pour les insertions/mises à jour en masse
        en configurant des paramètres appropriés.
        
        Args:
            batch_size: Taille des lots pour les commits périodiques
            
        Yields:
            Session: Session optimisée pour les opérations par lots
        """
        with get_db_context() as db:
            try:
                # Configuration pour les opérations par lots
                db.execute("SET session_replication_role = replica")  # Désactive les triggers temporairement
                
                self.logger.debug(f"Session par lots initialisée (batch_size={batch_size})")
                yield db
                
            except Exception as e:
                self.logger.error(f"Erreur dans la session par lots: {e}")
                db.rollback()
                raise
            finally:
                try:
                    # Restaurer la configuration normale
                    db.execute("SET session_replication_role = DEFAULT")
                    db.commit()
                except Exception as e:
                    self.logger.warning(f"Impossible de restaurer la configuration normale: {e}")
    
    @contextmanager
    def get_read_only_session(self):
        """
        Gestionnaire de contexte pour les opérations de lecture seule.
        
        Optimise la session pour les requêtes de lecture en définissant
        la transaction en mode lecture seule.
        
        Yields:
            Session: Session en mode lecture seule
        """
        with get_db_context() as db:
            try:
                # Configuration en lecture seule
                db.execute("SET TRANSACTION READ ONLY")
                self.logger.debug("Session lecture seule initialisée")
                yield db
                
            except Exception as e:
                self.logger.error(f"Erreur dans la session lecture seule: {e}")
                raise
    
    def execute_with_retry(self, operation, max_retries: int = 3, delay: float = 1.0):
        """
        Exécute une opération de base de données avec retry automatique.
        
        Args:
            operation: Fonction à exécuter (doit accepter une session db)
            max_retries: Nombre maximum de tentatives
            delay: Délai entre les tentatives (en secondes)
            
        Returns:
            Résultat de l'opération
            
        Raises:
            Exception: Si toutes les tentatives échouent
        """
        import time
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                with get_db_context() as db:
                    result = operation(db)
                    self.logger.debug(f"Opération réussie à la tentative {attempt + 1}")
                    return result
                    
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Tentative {attempt + 1} échouée: {e}")
                
                if attempt < max_retries:
                    time.sleep(delay * (2 ** attempt))  # Backoff exponentiel
                else:
                    self.logger.error(f"Toutes les tentatives ont échoué après {max_retries + 1} essais")
        
        raise last_exception
    
    def get_user_data_stats(self, user_id: int) -> dict:
        """
        Récupère des statistiques sur les données d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            dict: Statistiques des données utilisateur
        """
        with self.get_read_only_session() as db:
            from db_service.models.sync import RawTransaction, SyncAccount, SyncItem
            from sqlalchemy import func
            
            stats = {}
            
            # Statistiques des transactions
            transaction_stats = db.query(
                func.count(RawTransaction.id).label('total_transactions'),
                func.min(RawTransaction.date).label('oldest_transaction'),
                func.max(RawTransaction.date).label('newest_transaction')
            ).filter(RawTransaction.user_id == user_id).first()
            
            stats['transactions'] = {
                'total': transaction_stats.total_transactions or 0,
                'oldest_date': transaction_stats.oldest_transaction.isoformat() if transaction_stats.oldest_transaction else None,
                'newest_date': transaction_stats.newest_transaction.isoformat() if transaction_stats.newest_transaction else None
            }
            
            # Statistiques des comptes
            account_count = db.query(func.count(SyncAccount.id)).join(
                SyncAccount.item
            ).filter(
                SyncAccount.item.has(user_id=user_id)
            ).scalar()
            
            stats['accounts'] = {
                'total': account_count or 0
            }
            
            # Statistiques des items
            item_stats = db.query(
                func.count(SyncItem.id).label('total_items'),
                func.sum(func.case([(SyncItem.status == 0, 1)], else_=0)).label('active_items')
            ).filter(SyncItem.user_id == user_id).first()
            
            stats['items'] = {
                'total': item_stats.total_items or 0,
                'active': item_stats.active_items or 0
            }
            
            return stats
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> dict:
        """
        Nettoie les anciennes données selon une politique de rétention.
        
        Args:
            days_to_keep: Nombre de jours de données à conserver
            
        Returns:
            dict: Statistiques de nettoyage
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        cleanup_stats = {}
        
        with get_db_context() as db:
            # Pour l'instant, on ne supprime pas les données de transaction
            # mais on pourrait nettoyer les logs ou données temporaires
            
            self.logger.info(f"Nettoyage des données antérieures au {cutoff_date.isoformat()}")
            
            # Exemple: nettoyer les anciennes entrées de webhook_events
            from db_service.models.sync import WebhookEvent
            
            old_webhooks = db.query(WebhookEvent).filter(
                WebhookEvent.created_at < cutoff_date,
                WebhookEvent.processed == True
            ).count()
            
            if old_webhooks > 0:
                db.query(WebhookEvent).filter(
                    WebhookEvent.created_at < cutoff_date,
                    WebhookEvent.processed == True
                ).delete()
                
                cleanup_stats['webhook_events_deleted'] = old_webhooks
                self.logger.info(f"Supprimé {old_webhooks} anciens événements webhook")
            
            db.commit()
        
        return cleanup_stats
    
    def health_check(self) -> dict:
        """
        Vérifie l'état de santé de la base de données.
        
        Returns:
            dict: État de santé de la base de données
        """
        health_status = {
            "status": "unknown",
            "connection": False,
            "tables_accessible": False,
            "response_time_ms": None
        }
        
        import time
        start_time = time.time()
        
        try:
            with self.get_read_only_session() as db:
                # Test de connexion basique
                db.execute("SELECT 1")
                health_status["connection"] = True
                
                # Test d'accès aux tables principales
                from db_service.models.sync import RawTransaction
                db.query(RawTransaction).limit(1).first()
                health_status["tables_accessible"] = True
                
                # Calculer le temps de réponse
                response_time = (time.time() - start_time) * 1000
                health_status["response_time_ms"] = round(response_time, 2)
                
                health_status["status"] = "healthy"
                
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return health_status

# Instance globale du gestionnaire
db_manager = EnrichmentDatabaseManager()

# Fonctions utilitaires
def get_user_transaction_count(user_id: int) -> int:
    """
    Récupère rapidement le nombre de transactions d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        
    Returns:
        int: Nombre de transactions
    """
    with get_db_context() as db:
        from db_service.models.sync import RawTransaction
        from sqlalchemy import func
        
        count = db.query(func.count(RawTransaction.id)).filter(
            RawTransaction.user_id == user_id
        ).scalar()
        
        return count or 0

def get_latest_transactions(user_id: int, limit: int = 10):
    """
    Récupère les dernières transactions d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        limit: Nombre maximum de transactions à retourner
        
    Returns:
        List: Liste des dernières transactions
    """
    with get_db_context() as db:
        from db_service.models.sync import RawTransaction
        
        transactions = db.query(RawTransaction).filter(
            RawTransaction.user_id == user_id
        ).order_by(
            RawTransaction.date.desc()
        ).limit(limit).all()
        
        return transactions

def check_user_has_data(user_id: int) -> bool:
    """
    Vérifie rapidement si un utilisateur a des données à enrichir.
    
    Args:
        user_id: ID de l'utilisateur
        
    Returns:
        bool: True si l'utilisateur a des données
    """
    count = get_user_transaction_count(user_id)
    return count > 0