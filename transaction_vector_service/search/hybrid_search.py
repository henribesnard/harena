"""
Module de recherche hybride combinant recherche lexicale (BM25), 
vectorielle et reranking pour les transactions.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING

from ..config.logging_config import get_logger
from ..models.interfaces import SearchServiceInterface
from ..common.types import SIMILARITY_THRESHOLD, DEFAULT_SEARCH_LIMIT

# Importation de types seulement pour les annotations, pas d'importation circulaire
if TYPE_CHECKING:
    from ..models.transaction import TransactionSearch

logger = get_logger(__name__)


class HybridSearch(SearchServiceInterface):
    """
    Service de recherche hybride qui combine recherche lexicale BM25,
    recherche vectorielle et reranking via cross-encoder.
    """

    def __init__(
        self,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.3,
        cross_encoder_weight: float = 0.4
    ):
        """
        Initialise le service de recherche hybride.
        
        Args:
            bm25_weight: Poids des scores BM25 dans le score final
            vector_weight: Poids des scores vectoriels dans le score final
            cross_encoder_weight: Poids des scores cross-encoder dans le score final
        """
        # On n'importe pas les implémentations concrètes ici
        # On les recevra via l'injection de dépendances
        self.bm25_search = None
        self.vector_search = None
        self.cross_encoder = None
        
        # Poids pour la combinaison finale des scores
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.cross_encoder_weight = cross_encoder_weight
        
        logger.info(
            f"Service de recherche hybride initialisé avec poids: "
            f"BM25={bm25_weight}, Vectoriel={vector_weight}, CrossEncoder={cross_encoder_weight}"
        )

    def set_search_components(self, bm25_search, vector_search, cross_encoder):
        """
        Définit les composants de recherche à utiliser.
        Cette méthode permet l'injection de dépendances.
        
        Args:
            bm25_search: Service de recherche BM25
            vector_search: Service de recherche vectorielle
            cross_encoder: Service de reranking cross-encoder
        """
        self.bm25_search = bm25_search
        self.vector_search = vector_search
        self.cross_encoder = cross_encoder

    async def search(
        self, 
        user_id: int,
        query: str,
        search_params: Any,
        top_k_initial: int = 100,
        top_k_final: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Exécute une recherche hybride sur les transactions.
        
        Args:
            user_id: ID de l'utilisateur
            query: Requête de recherche textuelle
            search_params: Paramètres de recherche supplémentaires
            top_k_initial: Nombre de résultats à récupérer pour chaque méthode
            top_k_final: Nombre final de résultats à retourner
            
        Returns:
            Tuple de (résultats de recherche, nombre total)
        """
        # Vérifier que les composants sont définis
        if not all([self.bm25_search, self.vector_search, self.cross_encoder]):
            logger.error("Search components not initialized. Call set_search_components first.")
            return [], 0
        
        if not query.strip():
            # Si la requête est vide, utiliser les filtres standards sans recherche textuelle
            logger.info(f"Requête vide, exécution d'une recherche par filtres pour l'utilisateur {user_id}")
            return await self.vector_search.filter_transactions(user_id, search_params)
        
        # Prétraiter la requête
        from ..utils.text_processors import clean_transaction_description
        clean_query = clean_transaction_description(query)
        logger.info(f"Exécution d'une recherche hybride pour '{clean_query}' (utilisateur {user_id})")
        
        # Exécuter les recherches en parallèle
        bm25_results_task = asyncio.create_task(
            self.bm25_search.search(user_id, clean_query, search_params, top_k_initial)
        )
        vector_results_task = asyncio.create_task(
            self.vector_search.search(user_id, clean_query, search_params, top_k_initial)
        )
        
        # Récupérer les résultats
        bm25_results, _ = await bm25_results_task
        vector_results, _ = await vector_results_task
        
        # Fusionner les résultats préliminaires
        merged_results = self._merge_initial_results(bm25_results, vector_results)
        
        # Limiter le nombre de résultats pour le reranking
        candidates = merged_results[:min(len(merged_results), top_k_initial)]
        
        if not candidates:
            logger.info(f"Aucun résultat trouvé pour la requête '{clean_query}'")
            return [], 0
        
        # Appliquer le reranking avec cross-encoder
        reranked_results = await self.cross_encoder.rerank(clean_query, candidates)
        
        # Limiter au nombre final de résultats demandés
        final_results = reranked_results[:min(len(reranked_results), top_k_final)]
        
        logger.info(f"Recherche hybride terminée: {len(final_results)} résultats sur un total de {len(merged_results)}")
        
        return final_results, len(merged_results)

    def _merge_initial_results(
        self, 
        bm25_results: List[Dict[str, Any]], 
        vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fusionne les résultats des recherches BM25 et vectorielle.
        
        Args:
            bm25_results: Résultats de la recherche BM25
            vector_results: Résultats de la recherche vectorielle
            
        Returns:
            Liste fusionnée de résultats uniques
        """
        # Créer un dictionnaire pour stocker les résultats fusionnés
        merged_dict = {}
        
        # Traiter les résultats BM25
        for item in bm25_results:
            item_id = item.get("id")
            if not item_id:
                continue
                
            merged_dict[item_id] = {
                **item,
                "bm25_score": item.get("score", 0),
                "vector_score": 0,
                "combined_initial_score": item.get("score", 0) * self.bm25_weight
            }
        
        # Traiter les résultats vectoriels
        for item in vector_results:
            item_id = item.get("id")
            if not item_id:
                continue
                
            if item_id in merged_dict:
                # Mettre à jour un résultat existant
                merged_dict[item_id]["vector_score"] = item.get("score", 0)
                merged_dict[item_id]["combined_initial_score"] += item.get("score", 0) * self.vector_weight
            else:
                # Ajouter un nouveau résultat
                merged_dict[item_id] = {
                    **item,
                    "bm25_score": 0,
                    "vector_score": item.get("score", 0),
                    "combined_initial_score": item.get("score", 0) * self.vector_weight
                }
        
        # Convertir le dictionnaire en liste et trier par score combiné
        merged_list = list(merged_dict.values())
        merged_list.sort(key=lambda x: x.get("combined_initial_score", 0), reverse=True)
        
        return merged_list