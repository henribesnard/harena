"""
Processeur principal pour l'enrichissement des transactions.

Ce module coordonne la structuration, la vectorisation et le stockage
des transactions financières.
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

def generate_qdrant_id_hash(user_id: int, transaction_id: int) -> str:
    """
    Génère un UUID basé sur un hash SHA256 (alternative).
    
    Args:
        user_id: ID de l'utilisateur
        transaction_id: ID de la transaction Bridge
        
    Returns:
        str: UUID au format string
    """
    unique_string = f"user_{user_id}_tx_{transaction_id}"
    hash_object = hashlib.sha256(unique_string.encode())
    hash_hex = hash_object.hexdigest()
    
    # Prendre les 32 premiers caractères et formater comme UUID
    uuid_str = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
    return uuid_str

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
        logger.debug(f"Récupération des statistiques pour l'utilisateur {user_id}")
        
        try:
            stats = await self.qdrant_storage.get_user_stats(user_id)
            
            return {
                "user_id": user_id,
                "total_transactions": stats.get("count", 0),
                "last_update": stats.get("last_update"),
                "storage_size_mb": stats.get("storage_size_mb", 0),
                "embedding_dimension": self.embedding_service.get_embedding_dimension(),
                "collection_name": self.qdrant_storage.collection_name
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats pour l'utilisateur {user_id}: {e}")
            return {
                "user_id": user_id,
                "total_transactions": 0,
                "error": str(e)
            }
    
    async def search_transactions(
        self,
        user_id: int,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche des transactions pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            query: Requête de recherche
            limit: Nombre maximum de résultats
            filters: Filtres additionnels
            
        Returns:
            List: Résultats de recherche
        """
        logger.debug(f"Recherche pour utilisateur {user_id}: '{query}' (limit: {limit})")
        
        try:
            # 1. Générer l'embedding de la requête
            query_vector = await self.embedding_service.generate_embedding(query)
            
            # 2. Rechercher dans Qdrant
            results = await self.qdrant_storage.search_transactions(
                query_vector=query_vector,
                user_id=user_id,
                limit=limit,
                filters=filters
            )
            
            logger.debug(f"Trouvé {len(results)} résultats pour la recherche")
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche pour l'utilisateur {user_id}: {e}")
            return []
    
    async def get_transaction_by_id(
        self,
        user_id: int,
        transaction_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère une transaction spécifique.
        
        Args:
            user_id: ID de l'utilisateur
            transaction_id: ID de la transaction
            
        Returns:
            Dict: Données de la transaction ou None
        """
        try:
            # Générer l'UUID de la transaction
            vector_id = generate_qdrant_id(user_id, transaction_id)
            
            # Récupérer depuis Qdrant
            transaction = await self.qdrant_storage.get_transaction_by_id(vector_id)
            
            return transaction
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la transaction {transaction_id}: {e}")
            return None
    
    async def update_transaction(
        self,
        transaction: TransactionInput,
        force_update: bool = True
    ) -> EnrichmentResult:
        """
        Met à jour une transaction existante.
        
        Args:
            transaction: Nouvelles données de la transaction
            force_update: Force la mise à jour
            
        Returns:
            EnrichmentResult: Résultat de la mise à jour
        """
        logger.debug(f"Mise à jour de la transaction {transaction.bridge_transaction_id}")
        
        # Utiliser la méthode de traitement standard avec force_update
        return await self.process_transaction(transaction, force_update=force_update)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé du processeur.
        
        Returns:
            Dict: État de santé
        """
        health_status = {
            "processor": "ok",
            "embedding_service": "unknown",
            "qdrant_storage": "unknown",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Vérifier le service d'embeddings
            if hasattr(self.embedding_service, 'client') and self.embedding_service.client:
                health_status["embedding_service"] = "ok"
            else:
                health_status["embedding_service"] = "not_initialized"
            
            # Vérifier Qdrant
            if hasattr(self.qdrant_storage, 'client') and self.qdrant_storage.client:
                health_status["qdrant_storage"] = "ok"
            else:
                health_status["qdrant_storage"] = "not_initialized"
            
            # Test rapide d'embedding
            test_embedding = await self.embedding_service.generate_embedding("test")
            if len(test_embedding) > 0:
                health_status["embedding_test"] = "ok"
            else:
                health_status["embedding_test"] = "failed"
                
        except Exception as e:
            health_status["error"] = str(e)
            health_status["processor"] = "error"
        
        return health_status