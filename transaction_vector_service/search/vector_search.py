# transaction_vector_service/services/vector_search.py
"""
Module de recherche vectorielle pour les transactions.
"""

from typing import List, Dict, Any, Optional, Tuple, Union

from ..config.logging_config import get_logger
from ..models.transaction import TransactionSearch
from ..services.embedding_service import EmbeddingService
from ..services.qdrant_client import QdrantService

logger = get_logger(__name__)

class VectorSearch:
    """
    Service de recherche vectorielle utilisant des embeddings pour les transactions.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        qdrant_service: Optional[QdrantService] = None,
        transaction_service = None
    ):
        """
        Initialise le service de recherche vectorielle.
        
        Args:
            embedding_service: Service d'embedding
            qdrant_service: Service Qdrant
            transaction_service: Service de transaction
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_service = qdrant_service or QdrantService()
        self.transaction_service = transaction_service
        
        logger.info("Service de recherche vectorielle initialisé")

    def set_transaction_service(self, transaction_service):
        """
        Définit le service de transaction à utiliser.
        Cette méthode permet l'injection de dépendances après construction.
        
        Args:
            transaction_service: Service de transaction
        """
        self.transaction_service = transaction_service
        logger.info("Service de transaction injecté dans VectorSearch")

    async def search(
        self, 
        user_id: int,
        query: str,
        search_params: TransactionSearch,
        top_k: int = 50
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Exécute une recherche vectorielle sur les transactions.
        
        Args:
            user_id: ID de l'utilisateur
            query: Requête de recherche
            search_params: Paramètres de recherche supplémentaires
            top_k: Nombre maximum de résultats
            
        Returns:
            Tuple de (résultats de recherche, nombre total)
        """
        # Vérifier que transaction_service est défini
        if not self.transaction_service:
            logger.error("Le service de transaction n'est pas initialisé. Appelez set_transaction_service d'abord.")
            return [], 0
            
        logger.info(f"Exécution d'une recherche vectorielle pour '{query}' (utilisateur {user_id})")
        
        # Générer l'embedding pour la requête
        query_embedding = await self.embedding_service.get_embedding(query)
        
        # Construire les conditions de filtre basées sur les paramètres de recherche
        filter_conditions = {"user_id": user_id}
        
        # Ajouter les filtres de date
        if search_params.start_date or search_params.end_date:
            date_range = {}
            if search_params.start_date:
                date_range["start_date"] = search_params.start_date
            if search_params.end_date:
                date_range["end_date"] = search_params.end_date
            filter_conditions["date_range"] = date_range
        
        # Ajouter les filtres de montant
        if search_params.min_amount is not None or search_params.max_amount is not None:
            amount_range = {}
            if search_params.min_amount is not None:
                amount_range["min_amount"] = search_params.min_amount
            if search_params.max_amount is not None:
                amount_range["max_amount"] = search_params.max_amount
            filter_conditions["amount_range"] = amount_range
        
        # Ajouter les filtres de catégorie
        if search_params.categories:
            filter_conditions["categories"] = search_params.categories
        
        # Ajouter les filtres de compte
        if search_params.account_ids:
            filter_conditions["account_ids"] = search_params.account_ids
        
        # Ajouter les filtres de type d'opération
        if search_params.operation_types:
            filter_conditions["operation_types"] = search_params.operation_types
            
        # Ajouter les filtres d'inclusion
        if not search_params.include_future:
            filter_conditions["is_future"] = False
        if not search_params.include_deleted:
            filter_conditions["is_deleted"] = False
        
        # Exécuter la recherche vectorielle
        vector_results = await self.qdrant_service.search_similar_transactions(
            embedding=query_embedding,
            user_id=user_id,
            limit=top_k,
            score_threshold=0.5,  # Seuil plus bas pour la recherche
            filter_conditions=filter_conditions
        )
        
        logger.info(f"Recherche vectorielle terminée: {len(vector_results)} résultats")
        
        return vector_results, len(vector_results)

    async def filter_transactions(
        self,
        user_id: int,
        search_params: TransactionSearch
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Filtre les transactions sans recherche textuelle.
        
        Args:
            user_id: ID de l'utilisateur
            search_params: Paramètres de filtrage
            
        Returns:
            Tuple de (transactions filtrées, nombre total)
        """
        # Vérifier que transaction_service est défini
        if not self.transaction_service:
            logger.error("Le service de transaction n'est pas initialisé. Appelez set_transaction_service d'abord.")
            return [], 0
            
        return await self.transaction_service.search_transactions(user_id, search_params)