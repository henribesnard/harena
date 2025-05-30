"""
Service de reranking pour améliorer la pertinence des résultats.

Ce module utilise Cohere Rerank pour réordonner les résultats de recherche
en fonction de leur pertinence sémantique avec la requête.
"""
import logging
from typing import List, Dict, Any, Optional
import cohere
from tenacity import retry, stop_after_attempt, wait_exponential

from config_service.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """Service de reranking utilisant Cohere."""
    
    def __init__(self):
        self.client = None
        self.model = "rerank-multilingual-v2.0"  # Supporte le français
        self.max_documents = 100  # Limite Cohere
        self._initialized = False
        
    async def initialize(self):
        """Initialise le client Cohere."""
        if not settings.COHERE_KEY:
            logger.warning("COHERE_KEY non définie, reranking désactivé")
            return
            
        try:
            self.client = cohere.AsyncClient(
                api_key=settings.COHERE_KEY,
                timeout=30.0
            )
            
            # Tester la connexion
            await self.client.tokenize(text="test", model="command")
            
            self._initialized = True
            logger.info(f"RerankerService initialisé avec le modèle {self.model}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de Cohere: {e}")
            self.client = None
            self._initialized = False
    
    def is_initialized(self) -> bool:
        """Vérifie si le service est initialisé."""
        return self._initialized and self.client is not None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def rerank(
        self, 
        query: str, 
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[float]:
        """
        Réordonne les documents selon leur pertinence avec la requête.
        
        Args:
            query: Requête de recherche
            documents: Liste des documents à reranker
            top_k: Nombre de documents à retourner (None = tous)
            
        Returns:
            List[float]: Scores de reranking dans l'ordre des documents
        """
        if not self.client:
            logger.warning("RerankerService non initialisé")
            # Retourner des scores uniformes
            return [1.0] * len(documents)
        
        if not documents:
            return []
        
        try:
            # Limiter le nombre de documents si nécessaire
            truncated = False
            if len(documents) > self.max_documents:
                logger.warning(
                    f"Trop de documents ({len(documents)}), "
                    f"limitation à {self.max_documents}"
                )
                documents = documents[:self.max_documents]
                truncated = True
            
            logger.debug(f"Reranking {len(documents)} documents pour la requête: {query[:100]}...")
            
            # Appeler l'API Cohere
            response = await self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k if top_k else len(documents),
                return_documents=False
            )
            
            # Créer un mapping index -> score
            score_map = {}
            for result in response.results:
                score_map[result.index] = result.relevance_score
            
            # Construire la liste des scores dans l'ordre original
            scores = []
            for i in range(len(documents)):
                if i in score_map:
                    scores.append(score_map[i])
                else:
                    # Document non retourné par Cohere (si top_k < len(documents))
                    scores.append(0.0)
            
            # Ajouter des scores nuls pour les documents tronqués
            if truncated:
                scores.extend([0.0] * (len(documents) - self.max_documents))
            
            logger.debug(f"Reranking terminé, scores: {scores[:10]}...")
            return scores
            
        except Exception as e:
            logger.error(f"Erreur lors du reranking: {e}")
            # Retourner des scores uniformes en cas d'erreur
            return [1.0] * len(documents)
    
    async def close(self):
        """Nettoie les ressources du service."""
        self.client = None
        self._initialized = False
        logger.info("RerankerService fermé")


# Instance globale
reranker_service = RerankerService()