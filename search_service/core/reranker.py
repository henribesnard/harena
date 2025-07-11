"""
Service de reranking pour améliorer la pertinence des résultats de recherche.

Ce module optionnel utilise des modèles de reranking (comme Cohere) pour
réorganiser les résultats de recherche en fonction de leur pertinence.
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class RerankerModel(str, Enum):
    """Modèles de reranking supportés."""
    COHERE_RERANK_V3 = "rerank-english-v3.0"
    COHERE_RERANK_MULTILINGUAL = "rerank-multilingual-v3.0"
    LOCAL_CROSS_ENCODER = "local-cross-encoder"


@dataclass
class RerankerConfig:
    """Configuration pour le service de reranking."""
    model: RerankerModel = RerankerModel.COHERE_RERANK_V3
    api_key: Optional[str] = None
    max_chunks_per_doc: int = 10
    top_n: int = 10
    relevance_threshold: float = 0.0
    timeout: int = 30
    max_retries: int = 3


class SearchResult:
    """Représente un résultat de recherche à reranker."""
    
    def __init__(
        self,
        id: str,
        text: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.text = text
        self.score = score
        self.metadata = metadata or {}
        self.rerank_score: Optional[float] = None
        self.combined_score: Optional[float] = None


class Reranker:
    """
    Service de reranking pour améliorer la pertinence des résultats.
    
    Implémentation de base qui peut être étendue avec Cohere ou d'autres services.
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self._initialized = False
        
        # Métriques
        self.total_requests = 0
        self.successful_requests = 0
        
    async def initialize(self):
        """Initialise le service de reranking."""
        if self._initialized:
            return
            
        if self.config.model in [RerankerModel.COHERE_RERANK_V3, RerankerModel.COHERE_RERANK_MULTILINGUAL]:
            await self._initialize_cohere()
        elif self.config.model == RerankerModel.LOCAL_CROSS_ENCODER:
            await self._initialize_local()
        
        self._initialized = True
        logger.info(f"Reranker initialized with model: {self.config.model}")
    
    async def _initialize_cohere(self):
        """Initialise le client Cohere (nécessite cohere package)."""
        try:
            import cohere
            if not self.config.api_key:
                raise ValueError("Cohere API key is required")
            self.client = cohere.AsyncClient(api_key=self.config.api_key)
            logger.info("Cohere reranker initialized")
        except ImportError:
            logger.warning("Cohere package not installed, falling back to local reranking")
            self.config.model = RerankerModel.LOCAL_CROSS_ENCODER
            await self._initialize_local()
        except Exception as e:
            logger.error(f"Failed to initialize Cohere: {e}")
            raise
    
    async def _initialize_local(self):
        """Initialise le reranker local (implémentation simple)."""
        self.client = None
        logger.info("Local reranker initialized (simple scoring)")
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Reranke une liste de résultats de recherche.
        
        Args:
            query: La requête de recherche originale
            results: Liste des résultats à reranker
            top_n: Nombre de résultats à retourner (défaut: config.top_n)
            
        Returns:
            Liste des résultats rerankés et triés par pertinence
        """
        if not self._initialized:
            await self.initialize()
        
        if not results:
            return []
        
        top_n = top_n or self.config.top_n
        self.total_requests += 1
        
        try:
            if self.config.model in [RerankerModel.COHERE_RERANK_V3, RerankerModel.COHERE_RERANK_MULTILINGUAL]:
                reranked_results = await self._rerank_with_cohere(query, results, top_n)
            else:
                reranked_results = await self._rerank_local(query, results, top_n)
            
            self.successful_requests += 1
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: retourner les résultats originaux triés par score
            return sorted(results, key=lambda x: x.score, reverse=True)[:top_n]
    
    async def _rerank_with_cohere(
        self,
        query: str,
        results: List[SearchResult],
        top_n: int
    ) -> List[SearchResult]:
        """Reranke avec l'API Cohere."""
        if not self.client:
            raise ValueError("Cohere client not initialized")
        
        # Préparer les documents pour Cohere
        documents = [result.text for result in results]
        
        # Appel à l'API Cohere
        response = await self.client.rerank(
            model=self.config.model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            max_chunks_per_doc=self.config.max_chunks_per_doc
        )
        
        # Mapper les résultats rerankés
        reranked_results = []
        for result in response.results:
            original_result = results[result.index]
            original_result.rerank_score = result.relevance_score
            
            # Score combiné (moyenne pondérée)
            original_result.combined_score = (
                0.3 * original_result.score + 0.7 * result.relevance_score
            )
            
            # Filtrer par seuil de pertinence
            if result.relevance_score >= self.config.relevance_threshold:
                reranked_results.append(original_result)
        
        # Trier par score combiné
        reranked_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return reranked_results[:top_n]
    
    async def _rerank_local(
        self,
        query: str,
        results: List[SearchResult],
        top_n: int
    ) -> List[SearchResult]:
        """Reranking local simple basé sur la correspondance textuelle."""
        query_terms = set(query.lower().split())
        
        for result in results:
            # Score simple basé sur le nombre de termes de requête trouvés
            result_terms = set(result.text.lower().split())
            term_overlap = len(query_terms.intersection(result_terms))
            
            # Score de reranking simple
            result.rerank_score = term_overlap / len(query_terms) if query_terms else 0
            
            # Score combiné
            result.combined_score = 0.5 * result.score + 0.5 * result.rerank_score
        
        # Trier par score combiné
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results[:top_n]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du reranker."""
        success_rate = (
            self.successful_requests / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": round(success_rate, 2),
            "model": self.config.model.value,
            "top_n": self.config.top_n,
            "relevance_threshold": self.config.relevance_threshold
        }
    
    async def close(self):
        """Nettoie les ressources du reranker."""
        if hasattr(self, 'client') and self.client:
            if hasattr(self.client, 'close'):
                await self.client.close()
        logger.info("Reranker closed")


# Factory function
def create_reranker(config: Optional[RerankerConfig] = None) -> Reranker:
    """Factory function pour créer un reranker."""
    return Reranker(config)


# Exports
__all__ = [
    "RerankerModel",
    "RerankerConfig", 
    "SearchResult",
    "Reranker",
    "create_reranker"
]