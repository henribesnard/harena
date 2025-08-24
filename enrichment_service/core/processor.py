"""
Processeur principal pour l'enrichissement des transactions - Elasticsearch uniquement.

Ce module coordonne la structuration et l'indexation des transactions financières 
dans Elasticsearch, sans vectorisation ni embeddings.
"""
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from enrichment_service.models import (
    TransactionInput,
    BatchTransactionInput,
    ElasticsearchEnrichmentResult,
    BatchEnrichmentResult,
    UserSyncResult,
    StructuredTransaction,
)
from enrichment_service.core.data_quality_validator import DataQualityValidator

logger = logging.getLogger(__name__)

class ElasticsearchTransactionProcessor:
    """
    Processeur de transactions pour Elasticsearch uniquement.
    Gère la structuration et l'indexation des données financières.
    """
    
    def __init__(self, elasticsearch_client):
        """
        Initialise le processeur avec le client Elasticsearch.
        
        Args:
            elasticsearch_client: Instance du client Elasticsearch
        """
        self.elasticsearch_client = elasticsearch_client
        self.validator = DataQualityValidator()
        logger.info("ElasticsearchTransactionProcessor initialisé")
    
    


    async def process_single_transaction(
        self,
        transaction: TransactionInput,
        force_update: bool = False,
    ) -> ElasticsearchEnrichmentResult:
        """
        Traite et indexe une transaction unique dans Elasticsearch.

        Args:
            transaction: Transaction à traiter
            force_update: Force la mise à jour même si elle existe

        Returns:
            ElasticsearchEnrichmentResult: Résultat du traitement
        """
        start_time = time.time()
        user_id = transaction.user_id
        transaction_id = transaction.bridge_transaction_id

        logger.info(f"🔄 Traitement transaction {transaction_id} pour user {user_id}")

        try:
            # 1. Structurer la transaction
            structured_tx = StructuredTransaction.from_transaction_input(transaction)
            logger.debug(f"📋 Transaction structurée: {structured_tx.searchable_text[:100]}...")

            # 2. Valider la qualité des données
            is_valid, quality_score, flags = self.validator.evaluate(transaction)
            structured_tx.quality_score = quality_score
            if not is_valid:
                processing_time = time.time() - start_time
                logger.warning(
                    f"🚫 Qualité insuffisante pour transaction {transaction_id}: {flags}"
                )
                return ElasticsearchEnrichmentResult(
                    transaction_id=transaction_id,
                    user_id=user_id,
                    searchable_text=structured_tx.searchable_text,
                    document_id="",
                    indexed=False,
                    metadata={
                        "action": "skipped",
                        "reason": "data_quality",
                        "quality_score": quality_score,
                        "flags": flags,
                    },
                    processing_time=processing_time,
                    status="skipped",
                )

            # 3. Générer l'ID du document
            document_id = structured_tx.get_document_id()

            # 4. Vérifier si le document existe déjà (si pas de force_update)
            if not force_update:
                exists = await self.elasticsearch_client.document_exists(document_id)
                if exists:
                    logger.debug(f"⏭️ Transaction {transaction_id} existe déjà, ignorée")
                    processing_time = time.time() - start_time
                    return ElasticsearchEnrichmentResult(
                        transaction_id=transaction_id,
                        user_id=user_id,
                        searchable_text=structured_tx.searchable_text,
                        document_id=document_id,
                        indexed=False,
                        metadata={
                            "action": "skipped",
                            "reason": "document_exists",
                            "quality_score": quality_score,
                        },
                        processing_time=processing_time,
                        status="skipped",
                    )

            # 5. Indexer dans Elasticsearch
            success = await self.elasticsearch_client.index_document(
                document_id=document_id,
                document=structured_tx.to_elasticsearch_document(),
            )

            processing_time = time.time() - start_time

            if success:
                logger.debug(
                    f"✅ Transaction {transaction_id} indexée avec succès ({processing_time:.3f}s)"
                )

                return ElasticsearchEnrichmentResult(
                    transaction_id=transaction_id,
                    user_id=user_id,
                    searchable_text=structured_tx.searchable_text,
                    document_id=document_id,
                    indexed=True,
                    metadata={
                        "action": "created" if not force_update else "updated",
                        "elasticsearch_index": self.elasticsearch_client.index_name,
                        "document_size": len(
                            str(structured_tx.to_elasticsearch_document())
                        ),
                        "quality_score": quality_score,
                    },
                    processing_time=processing_time,
                    status="success",
                )
            else:
                logger.error(f"❌ Échec indexation transaction {transaction_id}")

                return ElasticsearchEnrichmentResult(
                    transaction_id=transaction_id,
                    user_id=user_id,
                    searchable_text=structured_tx.searchable_text,
                    document_id=document_id,
                    indexed=False,
                    metadata={"action": "failed", "quality_score": quality_score},
                    processing_time=processing_time,
                    status="error",
                    error_message="Elasticsearch indexing failed",
                )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"💥 Exception traitement transaction {transaction_id}: {e}")

            return ElasticsearchEnrichmentResult(
                transaction_id=transaction_id,
                user_id=user_id,
                searchable_text="",
                document_id="",
                indexed=False,
                metadata={"action": "error"},
                processing_time=processing_time,
                status="error",
                error_message=str(e),
            )

    async def process_transactions_batch(
        self,
        batch_input: BatchTransactionInput,
        force_update: bool = False,
    ) -> BatchEnrichmentResult:
        """
        Traite un lot de transactions en mode bulk Elasticsearch.

        Args:
            batch_input: Lot de transactions à traiter
            force_update: Force la mise à jour même si elles existent

        Returns:
            BatchEnrichmentResult: Résultat du traitement en lot
        """
        start_time = time.time()
        user_id = batch_input.user_id
        transactions = batch_input.transactions

        logger.info(f"🔄 Traitement en lot: {len(transactions)} transactions pour user {user_id}")

        try:
            structured_transactions = []
            documents_to_index = []
            results = []
            errors = []
            valid_pairs = []

            for tx in transactions:
                structured_tx = StructuredTransaction.from_transaction_input(tx)
                is_valid, quality_score, flags = self.validator.evaluate(tx)
                structured_tx.quality_score = quality_score
                if not is_valid:
                    results.append(
                        ElasticsearchEnrichmentResult(
                            transaction_id=tx.bridge_transaction_id,
                            user_id=user_id,
                            searchable_text=structured_tx.searchable_text,
                            document_id="",
                            indexed=False,
                            metadata={
                                "batch_processing": True,
                                "reason": "data_quality",
                                "quality_score": quality_score,
                                "flags": flags,
                            },
                            processing_time=0,
                            status="skipped",
                        )
                    )
                    errors.append(f"Transaction {tx.bridge_transaction_id}: data_quality")
                    continue
                structured_transactions.append(structured_tx)
                valid_pairs.append((tx, structured_tx))
                document_id = structured_tx.get_document_id()
                documents_to_index.append({
                    "id": document_id,
                    "document": structured_tx.to_elasticsearch_document(),
                    "transaction_id": tx.bridge_transaction_id,
                })

            logger.debug(f"📋 {len(structured_transactions)} transactions structurées")

            bulk_result = await self.elasticsearch_client.bulk_index_documents(
                documents_to_index,
                force_update=force_update,
            )

            processing_time = time.time() - start_time
            successful = bulk_result.get("indexed", 0)
            failed = bulk_result.get("errors", 0) + len([r for r in results if r.status == "skipped"])
            total = len(transactions)

            bulk_responses = bulk_result.get("responses", [])

            for i, (tx, structured_tx) in enumerate(valid_pairs):
                document_id = structured_tx.get_document_id()
                if i < len(bulk_responses):
                    response = bulk_responses[i]
                    indexed = response.get("success", False)
                    status = "success" if indexed else "error"
                    error_msg = response.get("error") if not indexed else None
                else:
                    indexed = False
                    status = "error"
                    error_msg = "No response from bulk operation"

                result = ElasticsearchEnrichmentResult(
                    transaction_id=tx.bridge_transaction_id,
                    user_id=user_id,
                    searchable_text=structured_tx.searchable_text if indexed else "",
                    document_id=document_id,
                    indexed=indexed,
                    metadata={
                        "batch_processing": True,
                        "batch_size": total,
                        "force_update": force_update,
                        "quality_score": structured_tx.quality_score,
                    },
                    processing_time=processing_time / max(len(valid_pairs), 1),
                    status=status,
                    error_message=error_msg,
                )
                results.append(result)
                if not indexed and error_msg:
                    errors.append(f"Transaction {tx.bridge_transaction_id}: {error_msg}")

            logger.info(
                f"🎉 Traitement en lot terminé: {successful}/{total} succès en {processing_time:.2f}s"
            )

            return BatchEnrichmentResult(
                user_id=user_id,
                total_transactions=total,
                successful=successful,
                failed=failed,
                processing_time=processing_time,
                results=results,
                errors=errors,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"💥 Erreur traitement en lot: {e}")

            results = []
            for tx in transactions:
                result = ElasticsearchEnrichmentResult(
                    transaction_id=tx.bridge_transaction_id,
                    user_id=user_id,
                    searchable_text="",
                    document_id="",
                    indexed=False,
                    metadata={"batch_processing": True, "error": "batch_failed"},
                    processing_time=0,
                    status="error",
                    error_message=str(e),
                )
                results.append(result)

            return BatchEnrichmentResult(
                user_id=user_id,
                total_transactions=len(transactions),
                successful=0,
                failed=len(transactions),
                processing_time=processing_time,
                results=results,
                errors=[f"Batch processing failed: {str(e)}"],
            )
async def sync_user_transactions(
        self, 
        user_id: int, 
        transactions: List[TransactionInput],
        force_refresh: bool = False
    ) -> UserSyncResult:
        """
        Synchronise toutes les transactions d'un utilisateur dans Elasticsearch.
        
        Args:
            user_id: ID de l'utilisateur
            transactions: Liste complète des transactions de l'utilisateur
            force_refresh: Force la suppression et recréation de tous les documents
            
        Returns:
            UserSyncResult: Résultat de la synchronisation
        """
        start_time = time.time()
        logger.info(f"🔄 Synchronisation user {user_id}: {len(transactions)} transactions (force_refresh={force_refresh})")
        
        try:
            # 1. Optionnel: Nettoyer les données existantes si force_refresh
            if force_refresh:
                logger.info(f"🧹 Nettoyage des données existantes pour user {user_id}")
                
                deleted_count = await self.elasticsearch_client.delete_user_transactions(user_id)
                logger.info(f"🗑️ {deleted_count} documents supprimés pour user {user_id}")
            
            # 2. Créer un batch input et traiter
            batch_input = BatchTransactionInput(
                user_id=user_id,
                transactions=transactions
            )
            
            batch_result = await self.process_transactions_batch(
                batch_input, 
                force_update=force_refresh
            )
            
            # 3. Convertir le résultat batch en résultat de sync utilisateur
            processing_time = time.time() - start_time
            
            # Compter les actions spécifiques
            indexed_count = sum(1 for result in batch_result.results if result.status == "success" and result.indexed)
            updated_count = sum(1 for result in batch_result.results 
                              if result.status == "success" and result.metadata.get("action") == "updated")
            error_count = len(batch_result.errors)
            
            # Collecter les détails d'erreurs
            error_details = batch_result.errors.copy()
            
            return UserSyncResult(
                user_id=user_id,
                total_transactions=len(transactions),
                indexed=indexed_count,
                updated=updated_count,
                errors=error_count,
                processing_time=processing_time,
                status="success" if error_count == 0 else "partial_success" if indexed_count > 0 else "failed",
                error_details=error_details
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"💥 Erreur synchronisation user {user_id}: {e}")
            
            return UserSyncResult(
                user_id=user_id,
                total_transactions=len(transactions),
                indexed=0,
                updated=0,
                errors=len(transactions),
                processing_time=processing_time,
                status="failed",
                error_details=[f"Sync failed: {str(e)}"]
            )
    
    async def delete_user_data(self, user_id: int) -> Dict[str, Any]:
        """
        Supprime toutes les données d'un utilisateur d'Elasticsearch.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Résultat de la suppression
        """
        logger.info(f"🗑️ Suppression des données pour user {user_id}")
        
        try:
            deleted_count = await self.elasticsearch_client.delete_user_transactions(user_id)
            
            logger.info(f"✅ {deleted_count} documents supprimés pour user {user_id}")
            
            return {
                "user_id": user_id,
                "deleted_count": deleted_count,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur suppression user {user_id}: {e}")
            
            return {
                "user_id": user_id,
                "deleted_count": 0,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Récupère les statistiques d'un utilisateur dans Elasticsearch.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Statistiques de l'utilisateur
        """
        try:
            stats = await self.elasticsearch_client.get_user_statistics(user_id)
            
            return {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "elasticsearch_stats": stats,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats user {user_id}: {e}")
            
            return {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "elasticsearch_stats": {},
                "status": "error",
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé du processeur et d'Elasticsearch.
        
        Returns:
            Dict: Statut de santé
        """
        try:
            # Vérifier la disponibilité d'Elasticsearch
            es_healthy = await self.elasticsearch_client.health_check()
            
            # Obtenir des métriques de base
            es_info = await self.elasticsearch_client.get_cluster_info() if es_healthy else {}
            
            return {
                "processor": "ElasticsearchTransactionProcessor",
                "version": "2.0.0",
                "status": "healthy" if es_healthy else "degraded",
                "timestamp": datetime.now().isoformat(),
                "elasticsearch": {
                    "available": es_healthy,
                    "cluster_info": es_info,
                    "index_name": self.elasticsearch_client.index_name if hasattr(self.elasticsearch_client, 'index_name') else "unknown"
                },
                "capabilities": {
                    "single_transaction_processing": True,
                    "batch_processing": True,
                    "user_sync": True,
                    "data_deletion": True,
                    "statistics": True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur health check: {e}")
            
            return {
                "processor": "ElasticsearchTransactionProcessor", 
                "version": "2.0.0",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "elasticsearch": {"available": False},
                "capabilities": {}
            }