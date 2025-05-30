"""
Service de génération d'embeddings pour la recherche.

Ce module gère la génération d'embeddings via OpenAI pour les requêtes
de recherche sémantique.
"""
import logging
import hashlib
from typing import List, Dict, Any, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from config_service.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service pour générer des embeddings via OpenAI."""
    
    def __init__(self):
        self.client = None
        self.model = settings.EMBEDDING_MODEL
        self.cache = {}  # Cache simple en mémoire
        self.max_cache_size = 1000
        self._initialized = False
        
    async def initialize(self):
        """Initialise le client OpenAI."""
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY non définie")
            raise ValueError("OpenAI API key is required")
            
        self.client = openai.AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=30.0
        )
        
        self._initialized = True
        logger.info(f"EmbeddingService initialisé avec le modèle {self.model}")
    
    def is_initialized(self) -> bool:
        """Vérifie si le service est initialisé."""
        return self._initialized and self.client is not None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Génère un embedding pour un texte donné.
        
        Args:
            text: Texte à vectoriser
            
        Returns:
            List[float]: Vecteur d'embedding
        """
        if not self.client:
            raise ValueError("EmbeddingService not initialized")
            
        if not text or not text.strip():
            logger.warning("Texte vide fourni pour l'embedding")
            return self._get_zero_vector()
        
        # Vérifier le cache
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            logger.debug(f"Cache hit pour embedding: {text[:50]}...")
            return self.cache[cache_key]
        
        try:
            logger.debug(f"Génération embedding pour: {text[:100]}...")
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=text.strip(),
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Embedding généré: dimension {len(embedding)}")
            
            # Mettre en cache
            self._cache_embedding(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'embedding: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Génère une clé de cache pour un texte."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    def _cache_embedding(self, key: str, embedding: List[float]):
        """Met en cache un embedding."""
        # Limiter la taille du cache
        if len(self.cache) >= self.max_cache_size:
            # Supprimer le plus ancien (FIFO simple)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[key] = embedding
    
    def _get_zero_vector(self) -> List[float]:
        """Retourne un vecteur zéro de la bonne dimension."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return [0.0] * dimensions.get(self.model, 1536)
    
    def get_embedding_dimension(self) -> int:
        """Retourne la dimension des embeddings."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model, 1536)
    
    async def close(self):
        """Nettoie les ressources du service."""
        if self.client:
            await self.client.close()
            self._initialized = False
            logger.info("EmbeddingService fermé")
    
    def clear_cache(self):
        """Vide le cache des embeddings."""
        self.cache.clear()
        logger.info("Cache des embeddings vidé")


# Instance globale
embedding_service = EmbeddingService()