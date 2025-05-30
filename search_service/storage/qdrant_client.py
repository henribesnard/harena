"""
Client Qdrant pour la recherche sémantique.

Ce module fournit une interface simplifiée pour la recherche vectorielle
dans Qdrant, réutilisant la collection existante.
"""
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, Match, Range

from config_service.config import settings

logger = logging.getLogger(__name__)


class QdrantClient:
    """Client simplifié pour la recherche dans Qdrant."""
    
    def __init__(self):
        self.client = None
        self.collection_name = "financial_transactions"
        self._initialized = False
        
    async def initialize(self):
        """Initialise la connexion Qdrant."""
        if not settings.QDRANT_URL:
            logger.warning("QDRANT_URL non configurée")
            return
        
        try:
            # Créer le client
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
            
            # Vérifier la connexion
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Connecté à Qdrant, collection {self.collection_name} disponible")
                self._initialized = True
            else:
                logger.warning(f"Collection {self.collection_name} non trouvée dans Qdrant")
                self._initialized = False
                
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation Qdrant: {e}")
            self.client = None
            self._initialized = False
    
    async def search(
        self,
        query_vector: List[float],
        user_id: int,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Effectue une recherche vectorielle dans Qdrant.
        
        Args:
            query_vector: Vecteur de la requête
            user_id: ID de l'utilisateur
            limit: Nombre de résultats
            filters: Filtres additionnels
            
        Returns:
            List[Dict]: Résultats de recherche
        """
        if not self.client or not self._initialized:
            return []
        
        try:
            # Construire les filtres
            must_conditions = [
                FieldCondition(
                    key="user_id",
                    match=Match(value=user_id)
                ),
                FieldCondition(
                    key="is_deleted",
                    match=Match(value=False)
                )
            ]
            
            # Ajouter les filtres additionnels
            if filters:
                if "date_from" in filters:
                    must_conditions.append(
                        FieldCondition(
                            key="timestamp",
                            range=Range(gte=filters["date_from"])
                        )
                    )
                
                if "date_to" in filters:
                    must_conditions.append(
                        FieldCondition(
                            key="timestamp",
                            range=Range(lte=filters["date_to"])
                        )
                    )
                
                if "amount_min" in filters:
                    must_conditions.append(
                        FieldCondition(
                            key="amount_abs",
                            range=Range(gte=filters["amount_min"])
                        )
                    )
                
                if "amount_max" in filters:
                    must_conditions.append(
                        FieldCondition(
                            key="amount_abs",
                            range=Range(lte=filters["amount_max"])
                        )
                    )
                
                if "categories" in filters:
                    # Pour plusieurs catégories, utiliser should avec des Match
                    category_conditions = [
                        FieldCondition(
                            key="category_id",
                            match=Match(value=cat_id)
                        )
                        for cat_id in filters["categories"]
                    ]
                    # Qdrant ne supporte pas OR dans must, donc on doit restructurer
                    # Pour l'instant, on prend seulement la première catégorie
                    if category_conditions:
                        must_conditions.append(category_conditions[0])
                
                if "transaction_types" in filters:
                    # Même limitation pour les types
                    if filters["transaction_types"]:
                        must_conditions.append(
                            FieldCondition(
                                key="transaction_type",
                                match=Match(value=filters["transaction_types"][0])
                            )
                        )
            
            # Créer le filtre final
            search_filter = Filter(must=must_conditions)
            
            # Effectuer la recherche
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Formater les résultats
            formatted_results = []
            for point in results:
                formatted_results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche Qdrant: {e}")
            return []
    
    async def is_healthy(self) -> bool:
        """Vérifie l'état de santé du client."""
        if not self.client:
            return False
        
        try:
            # Vérifier que la collection existe
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return self.collection_name in collection_names
        except:
            return False
    
    async def close(self):
        """Ferme la connexion Qdrant."""
        if self.client:
            await self.client.close()
            self._initialized = False
            logger.info("Connexion Qdrant fermée")