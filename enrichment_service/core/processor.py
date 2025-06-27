"""
Processeur principal pour l'enrichissement des transactions avec dual storage.

Ce module coordonne la structuration, la vectorisation et le stockage
des transactions financières dans Qdrant ET Elasticsearch.
"""
import logging
import time
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from enrichment_service.models import (
    TransactionInput, 
    BatchTransactionInput,
    EnrichmentResult, 
    BatchEnrichmentResult,
    StructuredTransaction,
    VectorizedTransaction
)
from enrichment_service.core.embeddings import embedding_service
from enrichment_service.storage.qdrant import QdrantStorage

logger = logging.getLogger(__name__)

def generate_qdrant_id(user_id: int, transaction_id: int) -> str:
    """
    Génère un UUID déterministe pour Qdrant basé sur user_id et transaction_id.
    
    Args:
        user_id: ID de l'utilisateur
        transaction_id: ID de la transaction Bridge
        
    Returns:
        str: UUID au format string compatible avec Qdrant
    """
    # Créer une chaîne unique basée sur les IDs
    unique_string = f"user_{user_id}_tx_{transaction_id}"
    
    # Générer un UUID déterministe basé sur cette chaîne
    namespace = uuid.UUID('12345678-1234-5678-1234-123456789abc')  # UUID namespace fixe
    point_uuid = uuid.uuid5(namespace, unique_string)
    
    return str(point_uuid)

class TransactionProcessor:
    """Processeur legacy pour compatibilité (Qdrant uniquement)."""
    
    def __init__(self, qdrant_storage: QdrantStorage):
        self.qdrant_storage = qdrant_storage
        self.embedding_service = embedding_service
    
    async def process_transaction(
        self, 
        transaction: TransactionInput,
        force_update: bool = False
    ) -> EnrichmentResult:
        """
        Traite une transaction individuelle (legacy - Qdrant uniquement).
        
        Args:
            transaction: Transaction à traiter
            force_update: Force la mise à jour même si elle existe déjà
            
        Returns:
            EnrichmentResult: Résultat du traitement
        """
        start_time = time.time()
        
        # Générer un UUID compatible Qdrant au lieu d'un string
        vector_id = generate_qdrant_id(transaction.user_id, transaction.bridge_transaction_id)
        original_id = f"user_{transaction.user_id}_tx_{transaction.bridge_transaction_id}"
        
        try:
            logger.debug(f"Traitement de la transaction {transaction.bridge_transaction_id}")
            
            # 1. Structurer la transaction
            structured_tx = StructuredTransaction.from_transaction_input(transaction)
            
            # 2. Générer l'embedding
            embedding = await self.embedding_service.generate_embedding(
                structured_tx.searchable_text
            )
            
            # 3. Créer la transaction vectorisée avec UUID et métadonnées enrichies
            vectorized_tx = VectorizedTransaction(
                id=vector_id,  # UUID compatible Qdrant
                vector=embedding,
                payload={
                    **structured_tx.to_qdrant_payload(),
                    # Ajouter les IDs pour la recherche et le debug
                    "original_id": original_id,
                    "user_id": transaction.user_id,
                    "transaction_id": transaction.bridge_transaction_id,
                    "uuid": vector_id
                }
            )
            
            # 4. Stocker dans Qdrant
            success = await self.qdrant_storage.store_transaction(vectorized_tx)
            
            if not success:
                raise Exception("Failed to store transaction in Qdrant")
            
            processing_time = time.time() - start_time
            
            return EnrichmentResult(
                transaction_id=transaction.bridge_transaction_id,
                user_id=transaction.user_id,
                searchable_text=structured_tx.searchable_text,
                vector_id=vector_id,  # Retourner l'UUID
                metadata=structured_tx.to_qdrant_payload(),
                processing_time=processing_time,
                status="success"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Erreur lors du traitement de la transaction {transaction.bridge_transaction_id}: {e}")
            
            return EnrichmentResult(
                transaction_id=transaction.bridge_transaction_id,
                user_id=transaction.user_id,
                searchable_text="",
                vector_id=vector_id,
                metadata={},
                processing_time=processing_time,
                status="error",
                error_message=str(e)
            )
    
    async def process_batch(
        self, 
        batch_input: BatchTransactionInput,
        force_update: bool = False
    ) -> BatchEnrichmentResult:
        """
        Traite un lot de transactions (legacy - Qdrant uniquement).
        
        Args:
            batch_input: Lot de transactions à traiter
            force_update: Force la mise à jour même si elles existent déjà
            
        Returns:
            BatchEnrichmentResult: Résultat du traitement en lot
        """
        start_time = time.time()
        user_id = batch_input.user_id
        transactions = batch_input.transactions
        
        logger.info(f"Traitement en lot (legacy): {len(transactions)} transactions pour l'utilisateur {user_id}")
        
        try:
            # 1. Structurer toutes les transactions
            structured_transactions = []
            searchable_texts = []
            
            for tx in transactions:
                structured_tx = StructuredTransaction.from_transaction_input(tx)
                structured_transactions.append(structured_tx)
                searchable_texts.append(structured_tx.searchable_text)
            
            logger.debug(f"Transactions structurées: {len(structured_transactions)}")
            
            # 2. Générer tous les embeddings en lot
            embeddings = await self.embedding_service.generate_batch_embeddings(
                searchable_texts, 
                batch_id=f"user_{user_id}_batch"
            )
            
            if len(embeddings) != len(structured_transactions):
                raise Exception(f"Mismatch entre embeddings ({len(embeddings)}) et transactions ({len(structured_transactions)})")
            
            # 3. Créer les transactions vectorisées avec UUIDs compatibles Qdrant
            vectorized_transactions = []
            for i, (structured_tx, embedding) in enumerate(zip(structured_transactions, embeddings)):
                # Générer UUID compatible Qdrant
                vector_id = generate_qdrant_id(user_id, structured_tx.transaction_id)
                original_id = f"user_{user_id}_tx_{structured_tx.transaction_id}"
                
                vectorized_tx = VectorizedTransaction(
                    id=vector_id,  # UUID compatible Qdrant
                    vector=embedding,
                    payload={
                        **structured_tx.to_qdrant_payload(),
                        # Enrichir avec les métadonnées de recherche
                        "original_id": original_id,
                        "user_id": user_id,
                        "transaction_id": structured_tx.transaction_id,
                        "uuid": vector_id
                    }
                )
                vectorized_transactions.append(vectorized_tx)
            
            logger.debug(f"Transactions vectorisées: {len(vectorized_transactions)}")
            
            # 4. Stocker en lot dans Qdrant
            storage_result = await self.qdrant_storage.store_transactions_batch(vectorized_transactions)
            
            # 5. Construire les résultats
            results = []
            errors = []
            
            successful = storage_result.get("stored", 0)
            failed = storage_result.get("errors", 0)
            
            # Créer les résultats pour chaque transaction
            for i, (tx, structured_tx, vectorized_tx) in enumerate(zip(
                transactions, structured_transactions, vectorized_transactions
            )):
                if i < successful:
                    result = EnrichmentResult(
                        transaction_id=tx.bridge_transaction_id,
                        user_id=tx.user_id,
                        searchable_text=structured_tx.searchable_text,
                        vector_id=vectorized_tx.id,  # UUID
                        metadata=structured_tx.to_qdrant_payload(),
                        processing_time=0.0,  # Temps individuel non calculé en lot
                        status="success"
                    )
                    results.append(result)
                else:
                    result = EnrichmentResult(
                        transaction_id=tx.bridge_transaction_id,
                        user_id=tx.user_id,
                        searchable_text="",
                        vector_id=vectorized_tx.id,
                        metadata={},
                        processing_time=0.0,
                        status="error",
                        error_message="Failed to store in Qdrant"
                    )
                    results.append(result)
                    errors.append(f"Transaction {tx.bridge_transaction_id}: Failed to store")
            
            processing_time = time.time() - start_time
            
            logger.info(f"Traitement en lot terminé: {successful}/{len(transactions)} succès en {processing_time:.2f}s")
            
            return BatchEnrichmentResult(
                user_id=user_id,
                total_transactions=len(transactions),
                successful=successful,
                failed=failed,
                processing_time=processing_time,
                results=results,
                errors=errors
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Erreur lors du traitement en lot: {e}")
            
            # Créer des résultats d'erreur pour toutes les transactions
            results = []
            for tx in transactions:
                vector_id = generate_qdrant_id(user_id, tx.bridge_transaction_id)
                result = EnrichmentResult(
                    transaction_id=tx.bridge_transaction_id,
                    user_id=user_id,
                    searchable_text="",
                    vector_id=vector_id,
                    metadata={},
                    processing_time=0,
                    status="error",
                    error_message=str(e)
                )
                results.append(result)
            
            return BatchEnrichmentResult(
                user_id=user_id,
                total_transactions=len(transactions),
                successful=0,
                failed=len(transactions),
                processing_time=processing_time,
                results=results,
                errors=[f"Batch processing failed: {str(e)}"]
            )
    
    async def sync_user_transactions(
        self, 
        user_id: int, 
        transactions: List[TransactionInput]
    ) -> BatchEnrichmentResult:
        """
        Synchronise toutes les transactions d'un utilisateur (legacy).
        
        Args:
            user_id: ID de l'utilisateur
            transactions: Liste complète des transactions de l'utilisateur
            
        Returns:
            BatchEnrichmentResult: Résultat de la synchronisation
        """
        logger.info(f"Synchronisation complète (legacy): {len(transactions)} transactions pour l'utilisateur {user_id}")
        
        # Créer un batch input
        batch_input = BatchTransactionInput(
            user_id=user_id,
            transactions=transactions
        )
        
        # Traiter en lot avec force update
        return await self.process_batch(batch_input, force_update=True)
    
    async def delete_user_data(self, user_id: int) -> bool:
        """
        Supprime toutes les données d'un utilisateur (legacy - Qdrant uniquement).
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si la suppression a réussi
        """
        logger.info(f"Suppression des données (legacy) pour l'utilisateur {user_id}")
        
        try:
            success = await self.qdrant_storage.delete_user_transactions(user_id)
            
            if success:
                logger.info(f"Données supprimées avec succès pour l'utilisateur {user_id}")
            else:
                logger.error(f"Échec de la suppression pour l'utilisateur {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression pour l'utilisateur {user_id}: {e}")
            return False

class DualStorageTransactionProcessor:
    """
    Processeur de transactions avec dual storage (Qdrant + Elasticsearch).
    Assure la cohérence entre recherche sémantique et lexicale.
    """
    
    def __init__(self, qdrant_storage: QdrantStorage, elasticsearch_client):
        self.qdrant_storage = qdrant_storage
        self.elasticsearch_client = elasticsearch_client
        self.embedding_service = embedding_service
        
    async def process_single_transaction(
        self, 
        transaction: TransactionInput,
        force_update: bool = False
    ) -> EnrichmentResult:
        """
        Traite une transaction unique et l'indexe dans Qdrant ET Elasticsearch.
        
        Args:
            transaction: Transaction à traiter
            force_update: Force la mise à jour même si elle existe
            
        Returns:
            EnrichmentResult: Résultat du traitement
        """
        start_time = time.time()
        user_id = transaction.user_id
        transaction_id = transaction.bridge_transaction_id
        
        logger.info(f"🔄 Traitement dual storage transaction {transaction_id} pour user {user_id}")
        
        try:
            # 1. Structurer la transaction
            structured_tx = StructuredTransaction.from_transaction_input(transaction)
            logger.debug(f"📋 Transaction structurée: {structured_tx.searchable_text[:100]}...")
            
            # 2. Générer l'embedding
            embedding = await self.embedding_service.generate_embedding(
                structured_tx.searchable_text,
                text_id=f"user_{user_id}_tx_{transaction_id}"
            )
            
            if not embedding:
                raise Exception("Failed to generate embedding")
            
            # 3. Créer la transaction vectorisée pour Qdrant
            vector_id = generate_qdrant_id(user_id, transaction_id)
            vectorized_tx = VectorizedTransaction(
                id=vector_id,
                vector=embedding,
                payload={
                    **structured_tx.to_qdrant_payload(),
                    "uuid": vector_id,
                    "original_id": f"user_{user_id}_tx_{transaction_id}"
                }
            )
            
            # 4. Stockage dual (Qdrant + Elasticsearch)
            qdrant_success = False
            elasticsearch_success = False
            errors = []
            
            # Stocker dans Qdrant
            if self.qdrant_storage:
                try:
                    qdrant_success = await self.qdrant_storage.store_transaction(vectorized_tx)
                    if qdrant_success:
                        logger.debug(f"✅ Qdrant: Transaction {transaction_id} stockée")
                    else:
                        errors.append("Qdrant storage failed")
                except Exception as e:
                    logger.error(f"❌ Erreur Qdrant: {e}")
                    errors.append(f"Qdrant error: {str(e)}")
            else:
                errors.append("Qdrant client not available")
            
            # Stocker dans Elasticsearch
            if self.elasticsearch_client:
                try:
                    elasticsearch_success = await self.elasticsearch_client.index_transaction(structured_tx)
                    if elasticsearch_success:
                        logger.debug(f"✅ Elasticsearch: Transaction {transaction_id} indexée")
                    else:
                        errors.append("Elasticsearch indexing failed")
                except Exception as e:
                    logger.error(f"❌ Erreur Elasticsearch: {e}")
                    errors.append(f"Elasticsearch error: {str(e)}")
            else:
                errors.append("Elasticsearch client not available")
            
            # 5. Évaluer le succès global
            processing_time = time.time() - start_time
            
            if qdrant_success and elasticsearch_success:
                status = "success"
                logger.info(f"🎉 Transaction {transaction_id} traitée avec succès ({processing_time:.3f}s)")
            elif qdrant_success or elasticsearch_success:
                status = "partial_success"
                logger.warning(f"⚠️ Transaction {transaction_id} partiellement traitée ({processing_time:.3f}s)")
            else:
                status = "failed"
                logger.error(f"❌ Échec traitement transaction {transaction_id} ({processing_time:.3f}s)")
            
            return EnrichmentResult(
                transaction_id=transaction_id,
                user_id=user_id,
                searchable_text=structured_tx.searchable_text,
                vector_id=vector_id,
                metadata={
                    "qdrant_success": qdrant_success,
                    "elasticsearch_success": elasticsearch_success,
                    "storage_backends": {
                        "qdrant": "success" if qdrant_success else "failed",
                        "elasticsearch": "success" if elasticsearch_success else "failed"
                    }
                },
                processing_time=processing_time,
                status=status,
                error_message="; ".join(errors) if errors else None
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"💥 Exception traitement transaction {transaction_id}: {e}")
            
            return EnrichmentResult(
                transaction_id=transaction_id,
                user_id=user_id,
                searchable_text="",
                vector_id="",
                metadata={},
                processing_time=processing_time,
                status="error",
                error_message=str(e)
            )
    
    async def sync_user_transactions(
        self, 
        user_id: int, 
        transactions: List[TransactionInput],
        force_refresh: bool = False
    ) -> BatchEnrichmentResult:
        """
        Synchronise toutes les transactions d'un utilisateur dans les deux systèmes.
        
        Args:
            user_id: ID de l'utilisateur
            transactions: Liste des transactions à traiter
            force_refresh: Force la suppression et recréation
            
        Returns:
            BatchEnrichmentResult: Résultat de la synchronisation
        """
        start_time = time.time()
        logger.info(f"🔄 Synchronisation dual storage de {len(transactions)} transactions pour user {user_id}")
        
        try:
            # 1. Optionnel: Nettoyer les données existantes si force_refresh
            if force_refresh:
                logger.info(f"🧹 Nettoyage des données existantes pour user {user_id}")
                
                # Supprimer de Qdrant
                if self.qdrant_storage:
                    try:
                        await self.qdrant_storage.delete_user_transactions(user_id)
                        logger.debug("✅ Données Qdrant supprimées")
                    except Exception as e:
                        logger.error(f"❌ Erreur suppression Qdrant: {e}")
                
                # Supprimer d'Elasticsearch
                if self.elasticsearch_client:
                    try:
                        await self.elasticsearch_client.delete_user_transactions(user_id)
                        logger.debug("✅ Données Elasticsearch supprimées")
                    except Exception as e:
                        logger.error(f"❌ Erreur suppression Elasticsearch: {e}")
            
            # 2. Structurer toutes les transactions
            structured_transactions = []
            searchable_texts = []
            
            for tx in transactions:
                structured_tx = StructuredTransaction.from_transaction_input(tx)
                structured_transactions.append(structured_tx)
                searchable_texts.append(structured_tx.searchable_text)
            
            logger.debug(f"📋 {len(structured_transactions)} transactions structurées")
            
            # 3. Générer tous les embeddings en lot
            embeddings = await self.embedding_service.generate_batch_embeddings(
                searchable_texts, 
                batch_id=f"user_{user_id}_dual_sync"
            )
            
            if len(embeddings) != len(structured_transactions):
                raise Exception(f"Mismatch embeddings ({len(embeddings)}) vs transactions ({len(structured_transactions)})")
            
            # 4. Préparer les données pour les deux systèmes
            vectorized_transactions = []
            
            for structured_tx, embedding in zip(structured_transactions, embeddings):
                vector_id = generate_qdrant_id(user_id, structured_tx.transaction_id)
                
                vectorized_tx = VectorizedTransaction(
                    id=vector_id,
                    vector=embedding,
                    payload={
                        **structured_tx.to_qdrant_payload(),
                        "uuid": vector_id,
                        "original_id": f"user_{user_id}_tx_{structured_tx.transaction_id}"
                    }
                )
                vectorized_transactions.append(vectorized_tx)
            
            # 5. Stockage en lot dans les deux systèmes
            qdrant_result = {"stored": 0, "errors": 0}
            elasticsearch_result = {"indexed": 0, "errors": 0}
            
            # Qdrant bulk storage
            if self.qdrant_storage:
                try:
                    qdrant_result = await self.qdrant_storage.store_transactions_batch(vectorized_transactions)
                    logger.info(f"📦 Qdrant: {qdrant_result['stored']}/{qdrant_result['total']} stockées")
                except Exception as e:
                    logger.error(f"❌ Erreur stockage Qdrant en lot: {e}")
                    qdrant_result = {"stored": 0, "errors": len(vectorized_transactions), "total": len(vectorized_transactions)}
            
            # Elasticsearch bulk indexing
            if self.elasticsearch_client:
                try:
                    elasticsearch_result = await self.elasticsearch_client.index_transactions_batch(structured_transactions)
                    logger.info(f"📦 Elasticsearch: {elasticsearch_result['indexed']}/{elasticsearch_result['total']} indexées")
                except Exception as e:
                    logger.error(f"❌ Erreur indexation Elasticsearch en lot: {e}")
                    elasticsearch_result = {"indexed": 0, "errors": len(structured_transactions), "total": len(structured_transactions)}
            
            # 6. Compiler le résultat final
            processing_time = time.time() - start_time
            total_transactions = len(transactions)
            
            # Calculer les succès (intersection des deux systèmes)
            successful_qdrant = qdrant_result.get("stored", 0)
            successful_elasticsearch = elasticsearch_result.get("indexed", 0)
            successful_both = min(successful_qdrant, successful_elasticsearch)
            
            failed_count = total_transactions - successful_both
            
            # Créer les résultats individuels (simplifiés pour le lot)
            results = []
            for i, tx in enumerate(transactions):
                if i < successful_both:
                    result = EnrichmentResult(
                        transaction_id=tx.bridge_transaction_id,
                        user_id=user_id,
                        searchable_text=searchable_texts[i][:100] + "...",
                        vector_id=str(vectorized_transactions[i].id),
                        metadata={
                            "batch_processing": True,
                            "qdrant_success": True,
                            "elasticsearch_success": True
                        },
                        processing_time=processing_time / total_transactions,
                        status="success"
                    )
                else:
                    result = EnrichmentResult(
                        transaction_id=tx.bridge_transaction_id,
                        user_id=user_id,
                        searchable_text="",
                        vector_id="",
                        metadata={"batch_processing": True},
                        processing_time=0,
                        status="failed",
                        error_message="Batch processing failed"
                    )
                results.append(result)
            
            batch_result = BatchEnrichmentResult(
                user_id=user_id,
                total_transactions=total_transactions,
                successful=successful_both,
                failed=failed_count,
                processing_time=processing_time,
                results=results,
                errors=[
                    f"Qdrant errors: {qdrant_result.get('errors', 0)}",
                    f"Elasticsearch errors: {elasticsearch_result.get('errors', 0)}"
                ] if (qdrant_result.get('errors', 0) > 0 or elasticsearch_result.get('errors', 0) > 0) else []
            )
            
            logger.info(f"🎉 Synchronisation terminée: {successful_both}/{total_transactions} réussies ({processing_time:.3f}s)")
            return batch_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"💥 Erreur synchronisation user {user_id}: {e}")
            
            return BatchEnrichmentResult(
                user_id=user_id,
                total_transactions=len(transactions),
                successful=0,
                failed=len(transactions),
                processing_time=processing_time,
                results=[],
                errors=[str(e)]
            )
    
    async def get_sync_status(self, user_id: int) -> Dict[str, Any]:
        """
        Récupère le statut de synchronisation pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Statut de synchronisation des deux systèmes
        """
        try:
            # Compter dans Qdrant (approximatif)
            qdrant_count = "unknown"
            if self.qdrant_storage:
                try:
                    # Pour Qdrant, on ne peut pas facilement compter par user_id
                    # On utilise une méthode approximative
                    qdrant_count = "available_but_not_countable"
                except Exception:
                    qdrant_count = "error"
            
            # Compter dans Elasticsearch
            elasticsearch_count = 0
            if self.elasticsearch_client:
                try:
                    elasticsearch_count = await self.elasticsearch_client.get_user_transaction_count(user_id)
                except Exception as e:
                    logger.error(f"Erreur comptage Elasticsearch: {e}")
                    elasticsearch_count = "error"
            
            return {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "storage_status": {
                    "qdrant": {
                        "available": self.qdrant_storage is not None,
                        "user_documents": qdrant_count,
                        "collection": self.qdrant_storage.collection_name if self.qdrant_storage else None
                    },
                    "elasticsearch": {
                        "available": self.elasticsearch_client is not None,
                        "user_documents": elasticsearch_count,
                        "index": self.elasticsearch_client.index_name if self.elasticsearch_client else None
                    }
                },
                "sync_consistency": {
                    "elasticsearch_count": elasticsearch_count,
                    "qdrant_count": qdrant_count
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération statut sync user {user_id}: {e}")
            return {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "storage_status": {
                    "qdrant": {"available": False, "error": "Status check failed"},
                    "elasticsearch": {"available": False, "error": "Status check failed"}
                }
            }