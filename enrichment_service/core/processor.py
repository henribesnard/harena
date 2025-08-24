"""
Processeur principal pour l'enrichissement des transactions - Elasticsearch uniquement.

Ce module coordonne la structuration et l'indexation des transactions financières 
dans Elasticsearch, sans vectorisation ni embeddings.
"""
import logging
import time
import asyncio
import uuid

from typing import List, Dict, Any, Optional
from datetime import datetime

from prometheus_client import Counter, Histogram

from enrichment_service.models import (
    TransactionInput,
    BatchTransactionInput,
    ElasticsearchEnrichmentResult,
    BatchEnrichmentResult,
    UserSyncResult,
    StructuredTransaction,

)
from enrichment_service.core.account_enrichment_service import (
    AccountEnrichmentService,
)
from enrichment_service.core.data_quality_validator import DataQualityValidator

logger = logging.getLogger(__name__)

# Métriques Prometheus
TRANSACTION_PROCESSING_TIME = Histogram(
    "enrichment_processing_seconds",
    "Temps de traitement d'une transaction"
)
CACHE_HITS = Counter(
    "enrichment_cache_hits_total",
    "Nombre de transactions ignorées car déjà indexées"
)
PROCESSING_ERRORS = Counter(
    "enrichment_errors_total",
    "Nombre d'erreurs lors du traitement"
)
BATCH_PROCESSING_TIME = Histogram(
    "enrichment_batch_processing_seconds",
    "Temps de traitement d'un lot de transactions"
)

class ElasticsearchTransactionProcessor:
    """
    Processeur de transactions pour Elasticsearch uniquement.
    Gère la structuration et l'indexation des données financières.
    """
    
    def __init__(
        self,
        elasticsearch_client,
        account_enrichment_service: Optional[AccountEnrichmentService] = None,
    ):
        """Initialise le processeur avec le client Elasticsearch et les services externes."""
        self.elasticsearch_client = elasticsearch_client
        self.validator = DataQualityValidator()
        self.account_enrichment_service = account_enrichment_service
        logger.info("ElasticsearchTransactionProcessor initialisé")

    def _apply_enrichment(self, structured_tx: StructuredTransaction, data: Dict[str, Optional[str]]):
        """Append enrichment information to the searchable text."""
        if not data:
            return
        parts = []
        account = data.get("account_name")
        category = data.get("category_name")
        merchant = data.get("merchant_name")
        if account:
            parts.append(f"Compte: {account}")
        if category:
            parts.append(f"Catégorie: {category}")
        if merchant:
            parts.append(f"Marchand: {merchant}")
        if parts:
            structured_tx.searchable_text += " | " + " | ".join(parts)
    
    


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
        start_time = time.perf_counter()
        user_id = transaction.user_id
        transaction_id = transaction.bridge_transaction_id

        correlation_id = f"{user_id}-{transaction_id}"
        log = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
        log.info(f"🔄 Traitement transaction {transaction_id} pour user {user_id}")
        
        logger.info(f"🔄 Traitement transaction {transaction_id} pour user {user_id}")

        try:
            # 1. Enrichir les données du compte avant structuration
            enriched = {}
            if self.account_enrichment_service:
                try:
                    enriched = await self.account_enrichment_service.enrich_with_account_data(transaction)
                except Exception as e:  # pragma: no cover - logging only
                    logger.warning(f"Enrichment failed for tx {transaction_id}: {e}")

            # 2. Structurer la transaction
            structured_tx = StructuredTransaction.from_transaction_input(transaction)
            log.debug(f"📋 Transaction structurée: {structured_tx.searchable_text[:100]}...")
            
            # 2. Générer l'ID du document
            self._apply_enrichment(structured_tx, enriched)
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
                    CACHE_HITS.inc()
                    log.debug(f"⏭️ Transaction {transaction_id} existe déjà, ignorée")
                    processing_time = time.perf_counter() - start_time
                    TRANSACTION_PROCESSING_TIME.observe(processing_time)

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
            
            processing_time = time.perf_counter() - start_time
            TRANSACTION_PROCESSING_TIME.observe(processing_time)

            if success:
                log.debug(f"✅ Transaction {transaction_id} indexée avec succès ({processing_time:.3f}s)")
                

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
                log.error(f"❌ Échec indexation transaction {transaction_id}")
                PROCESSING_ERRORS.inc()
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
            processing_time = time.perf_counter() - start_time
            TRANSACTION_PROCESSING_TIME.observe(processing_time)
            PROCESSING_ERRORS.inc()
            log.error(f"💥 Exception traitement transaction {transaction_id}: {e}")
            
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
        start_time = time.perf_counter()
        user_id = batch_input.user_id
        transactions = batch_input.transactions

        correlation_id = f"batch-{user_id}-{uuid.uuid4()}"
        log = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
        log.info(f"🔄 Traitement en lot: {len(transactions)} transactions pour user {user_id}")
        
        logger.info(f"🔄 Traitement en lot: {len(transactions)} transactions pour user {user_id}")

        try:
            structured_transactions = []

            documents_to_index = []
            results = []
            errors = []
            valid_pairs = []
            for tx in transactions:
                enriched = {}
                if self.account_enrichment_service:
                    try:
                        enriched = await self.account_enrichment_service.enrich_with_account_data(tx)
                    except Exception as e:  # pragma: no cover - logging only
                        logger.warning(f"Enrichment failed for tx {tx.bridge_transaction_id}: {e}")
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
                self._apply_enrichment(structured_tx, enriched)
                structured_transactions.append(structured_tx)

            logger.debug(f"📋 {len(structured_transactions)} transactions structurées")

            # 2. Indexation en parallèle via des sous-batches
            chunk_size = self.elasticsearch_client.default_batch_size * 2
            batches = [
                structured_transactions[i:i + chunk_size]
                for i in range(0, len(structured_transactions), chunk_size)
            ]

            tasks = [
                asyncio.create_task(
                    self.elasticsearch_client.index_transactions_batch(batch)
                ) for batch in batches
            ]

            batch_summaries = await asyncio.gather(*tasks)

            # 3. Agréger les résultats
            processing_time = time.time() - start_time
            total = len(transactions)
            successful = sum(s.get("indexed", 0) for s in batch_summaries)
            failed = sum(s.get("errors", 0) for s in batch_summaries)

            responses = {}
            for summary in batch_summaries:
                for resp in summary.get("responses", []):
                    responses[resp["transaction_id"]] = resp

            # 4. Créer les résultats individuels
            results = []
            errors = []
            for tx, structured_tx in zip(transactions, structured_transactions):
                document_id = structured_tx.get_document_id()
                resp = responses.get(tx.bridge_transaction_id, {"success": False, "error": "unknown"})
                indexed = resp.get("success", False)
                status = "success" if indexed else "error"
                error_msg = resp.get("error") if not indexed else None

                valid_pairs.append((tx, structured_tx))
                document_id = structured_tx.get_document_id()
                documents_to_index.append({
                    "id": document_id,
                    "document": structured_tx.to_elasticsearch_document(),
                    "transaction_id": tx.bridge_transaction_id,
                })
            
            log.debug(f"📋 {len(structured_transactions)} transactions structurées")
            
            # 2. Indexation bulk dans Elasticsearch

            logger.debug(f"📋 {len(structured_transactions)} transactions structurées")

            bulk_result = await self.elasticsearch_client.bulk_index_documents(
                documents_to_index,
                force_update=force_update,
            )
            
            # 3. Analyser les résultats du bulk
            processing_time = time.perf_counter() - start_time

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
                    processing_time=processing_time / total if total else 0,

                    status=status,
                    error_message=error_msg,
                )
                results.append(result)

                if not indexed and error_msg:
                    errors.append(f"Transaction {tx.bridge_transaction_id}: {error_msg}")

            cache_hits = len([r for r in results if r.status == "skipped"])
            if cache_hits:
                CACHE_HITS.inc(cache_hits)

            BATCH_PROCESSING_TIME.observe(processing_time)

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
            processing_time = time.perf_counter() - start_time
            BATCH_PROCESSING_TIME.observe(processing_time)
            PROCESSING_ERRORS.inc()
            log.error(f"💥 Erreur traitement en lot: {e}")
            
            # Créer des résultats d'erreur pour toutes les transactions
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
    
