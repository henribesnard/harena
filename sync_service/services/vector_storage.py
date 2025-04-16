"""
Service pour le stockage vectoriel des transactions.

Ce module fournit des fonctionnalités pour stocker et gérer
les transactions dans une base de données vectorielle.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from uuid import UUID, uuid4

# Import des dépendances Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Import du service d'embedding local
from sync_service.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class VectorStorageService:
    """Service pour le stockage vectoriel des transactions."""

    def __init__(self):
        """Initialise le service de stockage vectoriel."""
        # Configuration Qdrant
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Dimension des vecteurs d'embedding
        self.vector_size = 1536  # Taille par défaut pour text-embedding-3-small
        
        # Noms des collections
        self.transactions_collection = "transactions"
        self.merchants_collection = "merchants"
        
        # Initialiser le client Qdrant
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=30.0  # secondes
        )
        
        # Service d'embedding
        self.embedding_service = EmbeddingService()
        
        # Initialiser les collections
        self._ensure_collections()
        
        logger.info("Service de stockage vectoriel initialisé")

    def _ensure_collections(self):
        """S'assure que les collections requises existent et ont la configuration appropriée."""
        try:
            # Vérifier les collections existantes
            collections_list = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections_list]
            
            # Créer la collection de transactions si elle n'existe pas
            if self.transactions_collection not in collection_names:
                self.client.create_collection(
                    collection_name=self.transactions_collection,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=qmodels.Distance.COSINE,
                    ),
                    optimizers_config=qmodels.OptimizersConfigDiff(
                        indexing_threshold=10000,  # Indexer après ce nombre de vecteurs
                    )
                )
                
                # Créer des index de payload pour les champs de recherche courants
                indexes = [
                    ("user_id", "keyword"),
                    ("account_id", "keyword"),
                    ("transaction_date", "datetime"),
                    ("category_id", "keyword"),
                    ("merchant_id", "keyword"),
                    ("amount", "float"),
                    ("operation_type", "keyword"),
                    ("is_recurring", "bool"),
                ]
                
                for field_name, field_type in indexes:
                    self.client.create_payload_index(
                        collection_name=self.transactions_collection,
                        field_name=field_name,
                        field_schema=field_type
                    )
                
                logger.info(f"Collection créée: {self.transactions_collection}")
            else:
                logger.info(f"Collection existante: {self.transactions_collection}")
            
            # Créer la collection de marchands si elle n'existe pas
            if self.merchants_collection not in collection_names:
                self.client.create_collection(
                    collection_name=self.merchants_collection,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=qmodels.Distance.COSINE,
                    ),
                    optimizers_config=qmodels.OptimizersConfigDiff(
                        indexing_threshold=10000,
                    )
                )
                
                # Créer des index pour les champs de marchands
                indexes = [
                    ("name", "text"),
                    ("normalized_name", "keyword"),
                    ("category_id", "keyword"),
                ]
                
                for field_name, field_type in indexes:
                    self.client.create_payload_index(
                        collection_name=self.merchants_collection,
                        field_name=field_name,
                        field_schema=field_type
                    )
                
                logger.info(f"Collection créée: {self.merchants_collection}")
            else:
                logger.info(f"Collection existante: {self.merchants_collection}")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des collections: {str(e)}")
            raise

    async def store_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Stocke une transaction dans la base de données vectorielle.
        
        Args:
            transaction: Données de la transaction
            
        Returns:
            True si l'opération a réussi
        """
        try:
            # Générer un identifiant unique pour la transaction
            transaction_id = str(uuid4())
            
            # Extraire ou générer le texte pour l'embedding
            description = transaction.get("clean_description") or transaction.get("description", "")
            if not description:
                logger.warning(f"Description vide pour la transaction, utilisation d'un vecteur nul")
            
            # Générer l'embedding
            embedding = await self.embedding_service.get_embedding(description)
            
            # S'assurer que le payload est JSON-serializable
            payload = {
                "user_id": transaction.get("user_id"),
                "account_id": transaction.get("account_id"),
                "bridge_transaction_id": transaction.get("bridge_transaction_id"),
                "amount": transaction.get("amount", 0.0),
                "currency_code": transaction.get("currency_code", "EUR"),
                "description": transaction.get("description", ""),
                "clean_description": transaction.get("clean_description", ""),
                "transaction_date": transaction.get("transaction_date"),
                "category_id": transaction.get("category_id"),
                "operation_type": transaction.get("operation_type"),
                "is_recurring": transaction.get("is_recurring", False),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Convertir les objets datetime en chaînes ISO
            if isinstance(payload["transaction_date"], datetime):
                payload["transaction_date"] = payload["transaction_date"].isoformat()
            
            # S'assurer que le payload est sérialisable
            payload = json.loads(json.dumps(payload, default=str))
            
            # Upsert dans Qdrant
            self.client.upsert(
                collection_name=self.transactions_collection,
                points=[
                    qmodels.PointStruct(
                        id=transaction_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            logger.info(f"Transaction stockée avec succès: {transaction_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du stockage de la transaction: {str(e)}")
            return False

    async def batch_store_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stocke un lot de transactions dans la base de données vectorielle.
        
        Args:
            transactions: Liste des transactions à stocker
            
        Returns:
            Résultat du stockage par lot
        """
        if not transactions:
            return {
                "status": "success",
                "total": 0,
                "successful": 0,
                "failed": 0
            }
        
        successful = 0
        failed = 0
        
        # Traiter chaque transaction individuellement
        # Note: Dans une implémentation production, on pourrait batcher les upserts
        for transaction in transactions:
            try:
                result = await self.store_transaction(transaction)
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Erreur lors du stockage d'une transaction: {str(e)}")
                failed += 1
        
        return {
            "status": "success" if failed == 0 else "partial",
            "total": len(transactions),
            "successful": successful,
            "failed": failed
        }

    async def check_transaction_exists(self, bridge_transaction_id: str, user_id: int) -> bool:
        """
        Vérifie si une transaction avec l'ID Bridge donné existe déjà.
        
        Args:
            bridge_transaction_id: ID de la transaction Bridge
            user_id: ID de l'utilisateur
            
        Returns:
            True si la transaction existe
        """
        try:
            # Construire la requête
            filter_query = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="bridge_transaction_id",
                        match=qmodels.MatchValue(value=bridge_transaction_id)
                    ),
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id)
                    )
                ]
            )
            
            # Rechercher dans la collection
            results = self.client.scroll(
                collection_name=self.transactions_collection,
                scroll_filter=filter_query,
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            
            # Vérifier s'il y a des résultats
            return len(results[0]) > 0
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'existence de la transaction: {str(e)}")
            return False
    
    async def check_user_storage_initialized(self, user_id: int) -> bool:
        """
        Vérifie si le stockage vectoriel est initialisé pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            True si le stockage est initialisé
        """
        try:
            # Cette vérification dépend de la façon dont vous stockez les informations utilisateur
            # Dans cet exemple, nous cherchons simplement si des transactions existent
            filter_query = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id)
                    )
                ]
            )
            
            # Vérifier s'il y a des transactions
            count_response = self.client.count(
                collection_name=self.transactions_collection,
                count_filter=filter_query
            )
            
            # Considérer que le stockage est initialisé s'il existe des transactions
            # Dans une implémentation plus complète, vous pourriez avoir une collection
            # spécifique pour les métadonnées utilisateur
            return count_response.count > 0
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'initialisation du stockage: {str(e)}")
            return False

    async def initialize_user_storage(self, user_id: int) -> bool:
        """
        Initialise le stockage vectoriel pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            True si l'initialisation a réussi
        """
        try:
            # Dans une implémentation complète, vous pourriez créer des métadonnées 
            # utilisateur ou des structures spécifiques
            
            # Pour cet exemple, nous stockons simplement un marqueur d'initialisation
            # Cette approche est simplifiée - dans un système réel, vous pourriez avoir
            # une collection dédiée aux métadonnées utilisateur
            metadata = {
                "user_id": user_id,
                "initialized_at": datetime.now().isoformat(),
                "version": "1.0",
                "status": "active"
            }
            
            # Générer un ID pour le point de métadonnées
            metadata_id = f"user_{user_id}_metadata"
            
            # Créer un vecteur nul pour les métadonnées (ce n'est pas un vrai embedding)
            null_vector = [0.0] * self.vector_size
            
            # Stocker les métadonnées dans la collection de transactions
            # Dans un système réel, vous auriez une collection séparée
            self.client.upsert(
                collection_name=self.transactions_collection,
                points=[
                    qmodels.PointStruct(
                        id=metadata_id,
                        vector=null_vector,
                        payload={
                            "type": "user_metadata",
                            "user_id": user_id,
                            **metadata
                        }
                    )
                ]
            )
            
            logger.info(f"Stockage vectoriel initialisé pour l'utilisateur {user_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du stockage: {str(e)}")
            return False

    async def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """
        Récupère les statistiques du stockage vectoriel pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Statistiques du stockage vectoriel
        """
        try:
            # Filtrer par utilisateur
            filter_query = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id)
                    )
                ]
            )
            
            # Exclure les métadonnées
            filter_with_type = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id)
                    )
                ],
                must_not=[
                    qmodels.FieldCondition(
                        key="type",
                        match=qmodels.MatchValue(value="user_metadata")
                    )
                ]
            )
            
            # Compter le nombre total de transactions
            count_response = self.client.count(
                collection_name=self.transactions_collection,
                count_filter=filter_with_type
            )
            transaction_count = count_response.count
            
            # Récupérer les statistiques par catégorie
            categories = {}
            if transaction_count > 0:
                # Dans une implémentation complète, vous calculeriez des statistiques
                # par catégorie, par compte, etc.
                pass
            
            # Récupérer les statistiques par compte
            accounts = {}
            if transaction_count > 0:
                # Dans une implémentation complète, vous calculeriez des statistiques
                # par compte
                pass
            
            # Récupérer les métadonnées
            metadata = await self.get_user_storage_metadata(user_id)
            
            return {
                "user_id": user_id,
                "transaction_count": transaction_count,
                "initialized": transaction_count > 0 or metadata is not None,
                "last_updated": metadata.get("updated_at") if metadata else None,
                "categories": categories,
                "accounts": accounts
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques: {str(e)}")
            return {
                "user_id": user_id,
                "transaction_count": 0,
                "initialized": False,
                "error": str(e)
            }

    async def get_user_storage_metadata(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Récupère les métadonnées du stockage vectoriel pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Métadonnées du stockage ou None si non initialisé
        """
        try:
            # Filtrer pour trouver les métadonnées
            filter_query = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id)
                    ),
                    qmodels.FieldCondition(
                        key="type",
                        match=qmodels.MatchValue(value="user_metadata")
                    )
                ]
            )
            
            # Rechercher les métadonnées
            results = self.client.scroll(
                collection_name=self.transactions_collection,
                scroll_filter=filter_query,
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            # Vérifier si des métadonnées ont été trouvées
            if results and results[0] and len(results[0]) > 0:
                metadata = results[0][0].payload
                return metadata
            
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métadonnées: {str(e)}")
            return None

    async def update_user_storage_metadata(self, user_id: int, metadata_updates: Dict[str, Any]) -> bool:
        """
        Met à jour les métadonnées du stockage vectoriel pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            metadata_updates: Mises à jour à appliquer
            
        Returns:
            True si la mise à jour a réussi
        """
        try:
            # Récupérer les métadonnées actuelles
            current_metadata = await self.get_user_storage_metadata(user_id)
            
            # Si aucune métadonnée n'existe, initialiser le stockage
            if not current_metadata:
                return await self.initialize_user_storage(user_id)
            
            # Mettre à jour les métadonnées
            updated_metadata = {
                **current_metadata,
                **metadata_updates,
                "updated_at": datetime.now().isoformat()
            }
            
            # ID du point de métadonnées
            metadata_id = f"user_{user_id}_metadata"
            
            # Vecteur nul (les métadonnées n'ont pas besoin de vecteur réel)
            null_vector = [0.0] * self.vector_size
            
            # Mettre à jour les métadonnées
            self.client.upsert(
                collection_name=self.transactions_collection,
                points=[
                    qmodels.PointStruct(
                        id=metadata_id,
                        vector=null_vector,
                        payload=updated_metadata
                    )
                ]
            )
            
            logger.info(f"Métadonnées mises à jour pour l'utilisateur {user_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métadonnées: {str(e)}")
            return False