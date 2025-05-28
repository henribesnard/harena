"""
Processeur principal pour l'enrichissement des transactions.

Ce module coordonne la structuration, la vectorisation et le stockage
des transactions financières.
"""
import logging
import time
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

class TransactionProcessor:
    """Processeur principal pour l'enrichissement des transactions."""
    
    def __init__(self, qdrant_storage: QdrantStorage):
        self.qdrant_storage = qdrant_storage
        self.embedding_service = embedding_service
    
    async def process_transaction(
        self, 
        transaction: TransactionInput,
        force_update: bool = False
    ) -> EnrichmentResult:
        """
        Traite une transaction individuelle.
        
        Args:
            transaction: Transaction à traiter
            force_update: Force la mise à jour même si elle existe déjà
            
        Returns:
            EnrichmentResult: Résultat du traitement
        """
        start_time = time.time()
        vector_id = f"user_{transaction.user_id}_tx_{transaction.bridge_transaction_id}"
        
        try:
            logger.debug(f"Traitement de la transaction {transaction.bridge_transaction_id}")
            
            # 1. Structurer la transaction
            structured_tx = StructuredTransaction.from_transaction_input(transaction)
            
            # 2. Générer l'embedding
            embedding = await self.embedding_service.generate_embedding(
                structured_tx.searchable_text
            )
            
            # 3. Créer la transaction vectorisée
            vectorized_tx = VectorizedTransaction(
                id=vector_id,
                vector=embedding,
                payload=structured_tx.to_qdrant_payload()
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
                vector_id=vector_id,
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
        Traite un lot de transactions.
        
        Args:
            batch_input: Lot de transactions à traiter
            force_update: Force la mise à jour même si elles existent déjà
            
        Returns:
            BatchEnrichmentResult: Résultat du traitement en lot
        """
        start_time = time.time()
        user_id = batch_input.user_id
        transactions = batch_input.transactions
        
        logger.info(f"Traitement en lot: {len(transactions)} transactions pour l'utilisateur {user_id}")
        
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
            
            # 3. Créer les transactions vectorisées
            vectorized_transactions = []
            for i, (structured_tx, embedding) in enumerate(zip(structured_transactions, embeddings)):
                vector_id = f"user_{user_id}_tx_{structured_tx.transaction_id}"
                
                vectorized_tx = VectorizedTransaction(
                    id=vector_id,
                    vector=embedding,
                    payload=structured_tx.to_qdrant_payload()
                )
                vectorized_transactions.append(vectorized_tx)
            
            logger.debug(f"Transactions vectorisées: {len(vectorized_transactions)}")
            
            # 4. Stocker en lot dans Qdrant
            storage_result = await self.qdrant_storage.store_transactions_batch(vectorized_transactions)
            
            # 5. Créer les résultats individuels
            results = []
            errors = []
            successful = 0
            failed = 0
            
            for i, (tx, structured_tx) in enumerate(zip(transactions, structured_transactions)):
                vector_id = f"user_{user_id}_tx_{tx.bridge_transaction_id}"
                
                # Déterminer le statut basé sur le résultat du stockage
                if i < storage_result["stored"]:
                    status = "success"
                    successful += 1
                    error_msg = None
                else:
                    status = "error"
                    failed += 1
                    error_msg = "Storage failed"
                    errors.append(f"Transaction {tx.bridge_transaction_id}: Storage failed")
                
                result = EnrichmentResult(
                    transaction_id=tx.bridge_transaction_id,
                    user_id=user_id,
                    searchable_text=structured_tx.searchable_text,
                    vector_id=vector_id,
                    metadata=structured_tx.to_qdrant_payload(),
                    processing_time=0,  # Temps individuel non calculé en lot
                    status=status,
                    error_message=error_msg
                )
                results.append(result)
            
            processing_time = time.time() - start_time
            
            batch_result = BatchEnrichmentResult(
                user_id=user_id,
                total_transactions=len(transactions),
                successful=successful,
                failed=failed,
                processing_time=processing_time,
                results=results,
                errors=errors
            )
            
            logger.info(f"Traitement en lot terminé: {successful}/{len(transactions)} succès en {processing_time:.2f}s")
            return batch_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Erreur lors du traitement en lot: {e}")
            
            # Créer un résultat d'erreur pour tout le lot
            results = []
            for tx in transactions:
                result = EnrichmentResult(
                    transaction_id=tx.bridge_transaction_id,
                    user_id=user_id,
                    searchable_text="",
                    vector_id=f"user_{user_id}_tx_{tx.bridge_transaction_id}",
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
        Synchronise toutes les transactions d'un utilisateur.
        Cette méthode est optimisée pour la synchronisation complète.
        
        Args:
            user_id: ID de l'utilisateur
            transactions: Liste complète des transactions de l'utilisateur
            
        Returns:
            BatchEnrichmentResult: Résultat de la synchronisation
        """
        logger.info(f"Synchronisation complète: {len(transactions)} transactions pour l'utilisateur {user_id}")
        
        # Créer un batch input
        batch_input = BatchTransactionInput(
            user_id=user_id,
            transactions=transactions
        )
        
        # Traiter en lot avec force update
        return await self.process_batch(batch_input, force_update=True)
    
    async def delete_user_data(self, user_id: int) -> bool:
        """
        Supprime toutes les données d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si la suppression a réussi
        """
        logger.info(f"Suppression des données pour l'utilisateur {user_id}")
        
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
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Récupère les statistiques d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Statistiques de l'utilisateur
        """
        try:
            # Pour l'instant, on ne peut pas facilement compter les documents par utilisateur dans Qdrant
            # Cette fonctionnalité pourrait être ajoutée plus tard avec un compteur en cache
            
            collection_info = await self.qdrant_storage.get_collection_info()
            total_points = collection_info.points_count if collection_info else 0
            
            return {
                "user_id": user_id,
                "total_transactions_in_collection": total_points,
                "collection_status": "active" if collection_info else "inactive"
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats pour l'utilisateur {user_id}: {e}")
            return {
                "user_id": user_id,
                "error": str(e)
            }