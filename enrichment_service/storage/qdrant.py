"""
Interface de stockage vectoriel avec Qdrant.

Ce module gère les opérations de stockage et de recherche dans Qdrant
pour les transactions enrichies.
"""
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    CollectionInfo, VectorParams, Distance, PointStruct,
    Filter, FieldCondition, Match, Range, SearchRequest
)
from qdrant_client.http.exceptions import UnexpectedResponse

from config_service.config import settings
from enrichment_service.models import VectorizedTransaction, SearchResult
from enrichment_service.core.embeddings import embedding_service

logger = logging.getLogger(__name__)

class QdrantStorage:
    """Interface pour le stockage vectoriel avec Qdrant."""
    
    def __init__(self):
        self.client = None
        self.collection_name = "financial_transactions"
        self.vector_dimension = None  # Sera défini dynamiquement
    
    async def initialize(self):
        """Initialise la connexion Qdrant et crée les collections."""
        if not settings.QDRANT_URL:
            raise ValueError("QDRANT_URL is required")
        
        # Récupérer la dimension des embeddings depuis le service
        self.vector_dimension = embedding_service.get_embedding_dimension()
        logger.info(f"Dimension des vecteurs configurée: {self.vector_dimension}")
        
        # Initialiser le client
        if settings.QDRANT_API_KEY:
            self.client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=30.0
            )
        else:
            self.client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                timeout=30.0
            )
        
        logger.info(f"Connexion Qdrant établie: {settings.QDRANT_URL}")
        
        # Créer la collection si elle n'existe pas
        await self._setup_collection()
    
    async def _setup_collection(self):
        """Crée la collection pour les transactions financières."""
        try:
            # Vérifier si la collection existe
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Création de la collection {self.collection_name}")
                
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dimension,
                        distance=Distance.COSINE
                    )
                )
                
                logger.info(f"Collection {self.collection_name} créée avec succès (dimension: {self.vector_dimension})")
            else:
                logger.info(f"Collection {self.collection_name} existe déjà")
                
                # Vérifier que la dimension est correcte
                collection_info = await self.client.get_collection(self.collection_name)
                existing_dimension = collection_info.config.params.vectors.size
                
                if existing_dimension != self.vector_dimension:
                    logger.warning(
                        f"Dimension mismatch: collection existante ({existing_dimension}) "
                        f"vs configuration ({self.vector_dimension})"
                    )
                
        except Exception as e:
            logger.error(f"Erreur lors de la création de la collection: {e}")
            raise
    
    async def store_transaction(self, transaction: VectorizedTransaction) -> bool:
        """
        Stocke une transaction vectorisée dans Qdrant.
        
        Args:
            transaction: Transaction vectorisée à stocker
            
        Returns:
            bool: True si le stockage a réussi
        """
        if not self.client:
            raise ValueError("QdrantStorage not initialized")
        
        try:
            point = PointStruct(
                id=transaction.id,
                vector=transaction.vector,
                payload=transaction.payload
            )
            
            result = await self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Transaction {transaction.id} stockée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du stockage de la transaction {transaction.id}: {e}")
            return False
    
    async def store_transactions_batch(self, transactions: List[VectorizedTransaction]) -> Dict[str, Any]:
        """
        Stocke un lot de transactions dans Qdrant.
        
        Args:
            transactions: Liste des transactions à stocker
            
        Returns:
            Dict: Résumé du stockage
        """
        if not self.client:
            raise ValueError("QdrantStorage not initialized")
        
        if not transactions:
            return {"stored": 0, "errors": 0, "total": 0}
        
        try:
            # Convertir en PointStruct
            points = []
            for tx in transactions:
                point = PointStruct(
                    id=tx.id,
                    vector=tx.vector,
                    payload=tx.payload
                )
                points.append(point)
            
            # Stocker par lots pour éviter les timeouts
            batch_size = min(100, len(points))  # Limiter à 100 par lot
            stored_count = 0
            error_count = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                try:
                    result = await self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    stored_count += len(batch)
                    logger.debug(f"Lot {i//batch_size + 1} stocké: {len(batch)} transactions")
                    
                except Exception as e:
                    logger.error(f"Erreur lors du stockage du lot {i//batch_size + 1}: {e}")
                    error_count += len(batch)
            
            result_summary = {
                "stored": stored_count,
                "errors": error_count,
                "total": len(transactions)
            }
            
            logger.info(f"Stockage en lot terminé: {result_summary}")
            return result_summary
            
        except Exception as e:
            logger.error(f"Erreur générale lors du stockage en lot: {e}")
            return {"stored": 0, "errors": len(transactions), "total": len(transactions)}
    
    async def search_transactions(
        self,
        query_vector: List[float],
        user_id: int,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Recherche des transactions par similarité vectorielle.
        
        Args:
            query_vector: Vecteur de la requête
            user_id: ID de l'utilisateur (filtrage obligatoire)
            limit: Nombre maximum de résultats
            filters: Filtres additionnels
            
        Returns:
            List[SearchResult]: Résultats de recherche
        """
        if not self.client:
            raise ValueError("QdrantStorage not initialized")
        
        try:
            # Construire les filtres
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=Match(value=user_id)
                    ),
                    FieldCondition(
                        key="is_deleted",
                        match=Match(value=False)
                    )
                ]
            )
            
            # Ajouter des filtres additionnels si fournis
            if filters:
                if "amount_min" in filters:
                    search_filter.must.append(
                        FieldCondition(
                            key="amount_abs",
                            range=Range(gte=filters["amount_min"])
                        )
                    )
                
                if "amount_max" in filters:
                    search_filter.must.append(
                        FieldCondition(
                            key="amount_abs",
                            range=Range(lte=filters["amount_max"])
                        )
                    )
                
                if "transaction_type" in filters:
                    search_filter.must.append(
                        FieldCondition(
                            key="transaction_type",
                            match=Match(value=filters["transaction_type"])
                        )
                    )
                
                if "date_from" in filters:
                    search_filter.must.append(
                        FieldCondition(
                            key="timestamp",
                            range=Range(gte=filters["date_from"])
                        )
                    )
                
                if "date_to" in filters:
                    search_filter.must.append(
                        FieldCondition(
                            key="timestamp",
                            range=Range(lte=filters["date_to"])
                        )
                    )
            
            # Effectuer la recherche
            search_result = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Convertir en SearchResult
            results = []
            for point in search_result:
                result = SearchResult(
                    transaction_id=point.payload["transaction_id"],
                    user_id=point.payload["user_id"],
                    score=point.score,
                    primary_description=point.payload["primary_description"],
                    amount=point.payload["amount"],
                    date=point.payload["date"],
                    transaction_type=point.payload["transaction_type"],
                    metadata=point.payload
                )
                results.append(result)
            
            logger.debug(f"Recherche effectuée: {len(results)} résultats trouvés")
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            return []
    
    async def delete_user_transactions(self, user_id: int) -> bool:
        """
        Supprime toutes les transactions d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si la suppression a réussi
        """
        if not self.client:
            raise ValueError("QdrantStorage not initialized")
        
        try:
            # Supprimer par filtre
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=Match(value=user_id)
                        )
                    ]
                )
            )
            
            logger.info(f"Transactions supprimées pour l'utilisateur {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression pour l'utilisateur {user_id}: {e}")
            return False
    
    async def get_collection_info(self) -> Optional[CollectionInfo]:
        """Récupère les informations de la collection."""
        if not self.client:
            return None
        
        try:
            return await self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos de collection: {e}")
            return None
    
    async def close(self):
        """Ferme la connexion Qdrant."""
        if self.client:
            await self.client.close()
            logger.info("Connexion Qdrant fermée")
