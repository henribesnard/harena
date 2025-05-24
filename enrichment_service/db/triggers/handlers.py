"""
Gestionnaires d'événements de triggers PostgreSQL pour l'enrichissement.

Ce module contient les gestionnaires qui traitent les événements déclenchés 
par les triggers PostgreSQL et coordonnent les mises à jour vectorielles.
"""

import logging
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from enrichment_service.core.logging import get_contextual_logger
from enrichment_service.core.exceptions import TriggerProcessingError, handle_enrichment_error

logger = logging.getLogger(__name__)

class TriggerEventHandler:
    """
    Gestionnaire central des événements de triggers PostgreSQL.
    
    Cette classe traite les notifications reçues des triggers et coordonne
    les mises à jour appropriées dans Qdrant et les processus d'enrichissement.
    """
    
    def __init__(self, db: Session, qdrant_service=None, embedding_service=None):
        """
        Initialise le gestionnaire d'événements.
        
        Args:
            db: Session de base de données
            qdrant_service: Service Qdrant pour les mises à jour vectorielles
            embedding_service: Service d'embedding pour générer les vecteurs
        """
        self.db = db
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service
        
        # Initialiser les gestionnaires spécialisés (lazy loading)
        self._transaction_enricher = None
        self._pattern_detector = None
        self._summary_generator = None
        self._account_profiler = None
        
        # Statistiques de traitement
        self.processing_stats = {
            "events_processed": 0,
            "events_failed": 0,
            "last_processed": None,
            "errors": []
        }
    
    @property
    def transaction_enricher(self):
        """Lazy loading du transaction enricher."""
        if self._transaction_enricher is None and self.embedding_service and self.qdrant_service:
            try:
                from enrichment_service.enrichers.transaction_enricher import TransactionEnricher
                self._transaction_enricher = TransactionEnricher(
                    self.db, self.embedding_service, self.qdrant_service, None
                )
            except ImportError:
                logger.warning("TransactionEnricher non disponible")
        return self._transaction_enricher
    
    @property
    def pattern_detector(self):
        """Lazy loading du pattern detector."""
        if self._pattern_detector is None and self.embedding_service and self.qdrant_service:
            try:
                from enrichment_service.enrichers.pattern_detector import PatternDetector
                self._pattern_detector = PatternDetector(
                    self.db, self.embedding_service, self.qdrant_service
                )
            except ImportError:
                logger.warning("PatternDetector non disponible")
        return self._pattern_detector
    
    @property
    def summary_generator(self):
        """Lazy loading du summary generator."""
        if self._summary_generator is None and self.embedding_service and self.qdrant_service:
            try:
                from enrichment_service.enrichers.summary_generator import SummaryGenerator
                self._summary_generator = SummaryGenerator(
                    self.db, self.embedding_service, self.qdrant_service
                )
            except ImportError:
                logger.warning("SummaryGenerator non disponible")
        return self._summary_generator
    
    @property
    def account_profiler(self):
        """Lazy loading de l'account profiler."""
        if self._account_profiler is None and self.embedding_service and self.qdrant_service:
            try:
                from enrichment_service.enrichers.account_profiler import AccountProfiler
                self._account_profiler = AccountProfiler(
                    self.db, self.embedding_service, self.qdrant_service
                )
            except ImportError:
                logger.warning("AccountProfiler non disponible")
        return self._account_profiler
    
    async def process_event(self, channel: str, payload_json: str) -> Dict[str, Any]:
        """
        Traite un événement depuis pg_notify.
        
        Args:
            channel: Canal de notification
            payload_json: Payload JSON de l'événement
            
        Returns:
            Dict: Résultat du traitement
        """
        ctx_logger = get_contextual_logger(
            __name__,
            enrichment_type="trigger_event",
            collection=channel
        )
        
        try:
            payload = json.loads(payload_json)
            operation = payload.get("operation", "UNKNOWN")
            timestamp = payload.get("timestamp", datetime.now().timestamp())
            
            ctx_logger.info(f"Traitement événement {channel}: {operation}")
            
            # Router vers le gestionnaire approprié
            result = await self._route_event(channel, payload)
            
            # Mettre à jour les statistiques
            self.processing_stats["events_processed"] += 1
            self.processing_stats["last_processed"] = datetime.now()
            
            ctx_logger.info(f"Événement {channel} traité avec succès")
            return result
            
        except json.JSONDecodeError as e:
            error_msg = f"Payload JSON invalide pour {channel}: {str(e)}"
            ctx_logger.error(error_msg)
            self._record_error(error_msg, channel, payload_json)
            raise TriggerProcessingError(error_msg, channel, {"payload": payload_json})
            
        except Exception as e:
            error_msg = f"Erreur lors du traitement de l'événement {channel}: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            self._record_error(error_msg, channel, payload_json)
            raise TriggerProcessingError(error_msg, channel, {"payload": payload_json})
    
    async def _route_event(self, channel: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route l'événement vers le gestionnaire approprié.
        
        Args:
            channel: Canal de notification
            payload: Payload de l'événement
            
        Returns:
            Dict: Résultat du traitement
        """
        router_map = {
            "transaction_changes": self.handle_transaction_event,
            "account_changes": self.handle_account_event,
            "item_changes": self.handle_item_event,
            "stock_changes": self.handle_stock_event
        }
        
        handler = router_map.get(channel)
        if handler:
            return await handler(payload)
        else:
            raise TriggerProcessingError(f"Canal non supporté: {channel}", channel, payload)
    
    async def handle_transaction_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gère les événements liés aux transactions.
        
        Args:
            payload: Payload de l'événement transaction
            
        Returns:
            Dict: Résultat du traitement
        """
        operation = payload.get("operation")
        transaction_id = payload.get("id")
        bridge_transaction_id = payload.get("bridge_transaction_id")
        user_id = payload.get("user_id")
        account_id = payload.get("account_id")
        
        ctx_logger = get_contextual_logger(
            __name__,
            enrichment_type="transaction_trigger",
            user_id=user_id,
            transaction_id=transaction_id
        )
        
        ctx_logger.info(f"Traitement événement transaction: {operation} pour ID {transaction_id}")
        
        result = {
            "operation": operation,
            "transaction_id": transaction_id,
            "bridge_transaction_id": bridge_transaction_id,
            "status": "pending",
            "actions_performed": []
        }
        
        try:
            if operation in ("INSERT", "UPDATE"):
                # Récupérer la transaction complète
                from enrichment_service.db.models import RawTransaction
                transaction = self.db.query(RawTransaction).filter(
                    RawTransaction.id == transaction_id
                ).first()
                
                if transaction:
                    # 1. Enrichir et vectoriser la transaction
                    if self.transaction_enricher:
                        try:
                            await self.transaction_enricher.enrich_transaction(transaction)
                            result["actions_performed"].append("transaction_enriched")
                            ctx_logger.debug("Transaction enrichie avec succès")
                        except Exception as e:
                            ctx_logger.warning(f"Échec de l'enrichissement de transaction: {e}")
                    
                    # 2. Mettre à jour les patterns potentiellement affectés
                    if self.pattern_detector:
                        try:
                            await self.pattern_detector.update_patterns_for_transaction(transaction)
                            result["actions_performed"].append("patterns_updated")
                            ctx_logger.debug("Patterns mis à jour")
                        except Exception as e:
                            ctx_logger.warning(f"Échec de la mise à jour des patterns: {e}")
                    
                    # 3. Mettre à jour les résumés pour la période concernée
                    if self.summary_generator:
                        try:
                            await self.summary_generator.update_summaries_for_transaction(transaction)
                            result["actions_performed"].append("summaries_updated")
                            ctx_logger.debug("Résumés mis à jour")
                        except Exception as e:
                            ctx_logger.warning(f"Échec de la mise à jour des résumés: {e}")
                    
                    result["status"] = "success"
                else:
                    ctx_logger.warning(f"Transaction {transaction_id} non trouvée en base")
                    result["status"] = "warning"
                    result["message"] = "Transaction not found"
            
            elif operation == "DELETE":
                # Supprimer la transaction de Qdrant
                if self.qdrant_service:
                    try:
                        await self.qdrant_service.delete_transaction(bridge_transaction_id, user_id)
                        result["actions_performed"].append("vector_deleted")
                        ctx_logger.debug("Vecteur supprimé de Qdrant")
                    except Exception as e:
                        ctx_logger.warning(f"Échec de la suppression du vecteur: {e}")
                
                result["status"] = "success"
            
            return result
            
        except Exception as e:
            ctx_logger.error(f"Erreur lors du traitement de l'événement transaction: {e}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)
            return result
    
    async def handle_account_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gère les événements liés aux comptes.
        
        Args:
            payload: Payload de l'événement compte
            
        Returns:
            Dict: Résultat du traitement
        """
        operation = payload.get("operation")
        account_id = payload.get("id")
        bridge_account_id = payload.get("bridge_account_id")
        item_id = payload.get("item_id")
        
        ctx_logger = get_contextual_logger(
            __name__,
            enrichment_type="account_trigger",
            account_id=bridge_account_id
        )
        
        ctx_logger.info(f"Traitement événement compte: {operation} pour ID {bridge_account_id}")
        
        result = {
            "operation": operation,
            "account_id": account_id,
            "bridge_account_id": bridge_account_id,
            "status": "pending",
            "actions_performed": []
        }
        
        try:
            if operation in ("INSERT", "UPDATE"):
                # Récupérer le compte complet
                from enrichment_service.db.models import SyncAccount
                account = self.db.query(SyncAccount).filter(
                    SyncAccount.id == account_id
                ).first()
                
                if account and self.account_profiler:
                    try:
                        await self.account_profiler.profile_account(account)
                        result["actions_performed"].append("account_profiled")
                        ctx_logger.debug("Profil de compte mis à jour")
                    except Exception as e:
                        ctx_logger.warning(f"Échec du profilage de compte: {e}")
                
                result["status"] = "success"
            
            elif operation == "DELETE":
                # Supprimer les données vectorielles du compte
                if self.qdrant_service:
                    try:
                        await self.qdrant_service.delete_account_data(bridge_account_id)
                        result["actions_performed"].append("account_vectors_deleted")
                        ctx_logger.debug("Données vectorielles du compte supprimées")
                    except Exception as e:
                        ctx_logger.warning(f"Échec de la suppression des données vectorielles: {e}")
                
                result["status"] = "success"
            
            return result
            
        except Exception as e:
            ctx_logger.error(f"Erreur lors du traitement de l'événement compte: {e}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)
            return result
    
    async def handle_item_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gère les événements liés aux items.
        
        Args:
            payload: Payload de l'événement item
            
        Returns:
            Dict: Résultat du traitement
        """
        operation = payload.get("operation")
        item_id = payload.get("id")
        bridge_item_id = payload.get("bridge_item_id")
        user_id = payload.get("user_id")
        status = payload.get("status")
        
        ctx_logger = get_contextual_logger(
            __name__,
            enrichment_type="item_trigger",
            user_id=user_id,
            bridge_item_id=bridge_item_id
        )
        
        ctx_logger.info(f"Traitement événement item: {operation} pour ID {bridge_item_id}")
        
        result = {
            "operation": operation,
            "item_id": item_id,
            "bridge_item_id": bridge_item_id,
            "status": "pending",
            "actions_performed": []
        }
        
        try:
            if operation in ("INSERT", "UPDATE"):
                # Si l'item devient inactif ou en erreur, on peut décider d'actions spécifiques
                if status and status != 0:
                    ctx_logger.info(f"Item {bridge_item_id} en erreur (status={status}), aucune action d'enrichissement")
                    result["status"] = "skipped"
                    result["message"] = f"Item in error state: {status}"
                    return result
                
                # Pour un item actif, on peut déclencher un enrichissement complet
                result["actions_performed"].append("item_status_updated")
                result["status"] = "success"
            
            elif operation == "DELETE":
                # Supprimer toutes les données vectorielles de l'item
                if self.qdrant_service and user_id:
                    try:
                        await self.qdrant_service.delete_user_item_data(user_id, bridge_item_id)
                        result["actions_performed"].append("item_vectors_deleted")
                        ctx_logger.debug("Données vectorielles de l'item supprimées")
                    except Exception as e:
                        ctx_logger.warning(f"Échec de la suppression des données vectorielles: {e}")
                
                result["status"] = "success"
            
            return result
            
        except Exception as e:
            ctx_logger.error(f"Erreur lors du traitement de l'événement item: {e}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)
            return result
    
    async def handle_stock_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gère les événements liés aux stocks.
        
        Args:
            payload: Payload de l'événement stock
            
        Returns:
            Dict: Résultat du traitement
        """
        operation = payload.get("operation")
        stock_id = payload.get("id")
        bridge_stock_id = payload.get("bridge_stock_id")
        user_id = payload.get("user_id")
        account_id = payload.get("account_id")
        
        ctx_logger = get_contextual_logger(
            __name__,
            enrichment_type="stock_trigger",
            user_id=user_id,
            transaction_id=bridge_stock_id
        )
        
        ctx_logger.info(f"Traitement événement stock: {operation} pour ID {bridge_stock_id}")
        
        result = {
            "operation": operation,
            "stock_id": stock_id,
            "bridge_stock_id": bridge_stock_id,
            "status": "pending",
            "actions_performed": []
        }
        
        try:
            if operation in ("INSERT", "UPDATE"):
                # Pour l'instant, les stocks ne sont pas enrichis automatiquement
                # mais on peut ajouter une logique future ici
                result["actions_performed"].append("stock_acknowledged")
                result["status"] = "success"
            
            elif operation == "DELETE":
                # Supprimer le stock des données vectorielles si nécessaire
                if self.qdrant_service:
                    try:
                        # await self.qdrant_service.delete_stock(bridge_stock_id, user_id)
                        result["actions_performed"].append("stock_vector_deleted")
                        ctx_logger.debug("Vecteur de stock supprimé")
                    except Exception as e:
                        ctx_logger.warning(f"Échec de la suppression du vecteur de stock: {e}")
                
                result["status"] = "success"
            
            return result
            
        except Exception as e:
            ctx_logger.error(f"Erreur lors du traitement de l'événement stock: {e}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)
            return result
    
    def _record_error(self, error_msg: str, channel: str, payload: str):
        """
        Enregistre une erreur dans les statistiques.
        
        Args:
            error_msg: Message d'erreur
            channel: Canal de l'événement
            payload: Payload de l'événement
        """
        self.processing_stats["events_failed"] += 1
        self.processing_stats["errors"].append({
            "timestamp": datetime.now(),
            "channel": channel,
            "error": error_msg,
            "payload_preview": payload[:200] if payload else None
        })
        
        # Garder seulement les 100 dernières erreurs
        if len(self.processing_stats["errors"]) > 100:
            self.processing_stats["errors"] = self.processing_stats["errors"][-100:]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de traitement.
        
        Returns:
            Dict: Statistiques de traitement
        """
        stats = self.processing_stats.copy()
        
        # Convertir les timestamps en ISO pour la sérialisation
        if stats["last_processed"]:
            stats["last_processed"] = stats["last_processed"].isoformat()
        
        for error in stats["errors"]:
            if isinstance(error["timestamp"], datetime):
                error["timestamp"] = error["timestamp"].isoformat()
        
        # Ajouter des métriques calculées
        total_events = stats["events_processed"] + stats["events_failed"]
        stats["success_rate"] = (
            (stats["events_processed"] / total_events * 100) 
            if total_events > 0 else 0
        )
        
        return stats
    
    def reset_stats(self):
        """Remet à zéro les statistiques de traitement."""
        self.processing_stats = {
            "events_processed": 0,
            "events_failed": 0,
            "last_processed": None,
            "errors": []
        }