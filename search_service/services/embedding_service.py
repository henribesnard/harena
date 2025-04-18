"""
Service pour la génération d'embeddings.

Ce module fournit des fonctionnalités pour transformer du texte
en vecteurs d'embeddings qui peuvent être stockés et recherchés.
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Import conditionnel d'OpenAI
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from search_service.core.config import settings
from search_service.storage.cache import get_cache, set_cache

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service pour la génération d'embeddings à partir de texte."""

    def __init__(self):
        """Initialise le service d'embeddings."""
        self.model_name = settings.EMBEDDING_MODEL
        self.vector_dim = 1536  # Dimension des vecteurs
        self.api_key = settings.OPENAI_API_KEY
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY n'a pas été trouvée dans les variables d'environnement.")
        
        # Initialiser le client OpenAI si disponible
        if OPENAI_AVAILABLE and self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"Client OpenAI initialisé avec le modèle: {self.model_name}")
        else:
            self.client = None
            logger.warning("OpenAI API non disponible. Utilisation du mode fallback pour les embeddings.")
        
        logger.info(f"Service d'embedding initialisé.")

    def _generate_cache_key(self, text: str) -> str:
        """
        Génère une clé de cache pour le texte donné.
        
        Args:
            text: Texte pour générer une clé de cache
            
        Returns:
            Clé de cache sous forme de chaîne
        """
        # Créer un hash du texte
        return f"emb:{hashlib.md5(text.encode('utf-8')).hexdigest()}"

    async def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Récupère un embedding du cache s'il est disponible.
        
        Args:
            text: Texte pour lequel récupérer l'embedding
            
        Returns:
            Embedding en cache ou None si pas en cache
        """
        cache_key = self._generate_cache_key(text)
        return await get_cache(cache_key)

    async def _add_to_cache(self, text: str, embedding: List[float], ttl: int = 604800):  # 7 jours
        """
        Ajoute un embedding au cache.
        
        Args:
            text: Texte qui a été converti en embedding
            embedding: Vecteur d'embedding
            ttl: Durée de vie en secondes (par défaut: 7 jours)
        """
        cache_key = self._generate_cache_key(text)
        await set_cache(cache_key, embedding, ttl=ttl)

    async def get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Obtient le vecteur d'embedding pour une chaîne de texte.
        
        Args:
            text: Texte à convertir en vecteur
            use_cache: Utiliser le cache (par défaut: True)
            
        Returns:
            Vecteur d'embedding sous forme de liste de floats
        """
        if not text:
            # Renvoyer un vecteur de zéros pour un texte vide
            return [0.0] * self.vector_dim
        
        # Nettoyer et normaliser le texte d'entrée
        text = text.strip().lower()
        
        # Vérifier le cache d'abord si activé
        if use_cache:
            cached_embedding = await self._get_from_cache(text)
            if cached_embedding is not None:
                logger.debug(f"Embedding trouvé en cache pour '{text[:30]}...'")
                return cached_embedding
        
        # Si OpenAI n'est pas disponible, utiliser le fallback
        if not OPENAI_AVAILABLE or not self.client:
            return self._get_fallback_embedding(text)
        
        try:
            # Générer l'embedding en utilisant l'API OpenAI
            logger.debug(f"Génération d'embedding pour '{text[:30]}...'")
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.vector_dim
            )
            
            # Extraire l'embedding de la réponse
            embedding = response.data[0].embedding
            
            # Ajouter au cache si activé
            if use_cache:
                await self._add_to_cache(text, embedding)
                
            return embedding
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding avec OpenAI: {str(e)}")
            # Utiliser l'embedding de repli en cas d'erreur
            return self._get_fallback_embedding(text)

    def _get_fallback_embedding(self, text: str) -> List[float]:
        """
        Génère un embedding de repli quand OpenAI n'est pas disponible.
        C'est une méthode très basique et ne doit être utilisée qu'en dernier recours.
        
        Args:
            text: Texte à convertir en vecteur
            
        Returns:
            Embedding de repli (méthode simpliste)
        """
        logger.warning(f"Utilisation de l'embedding de repli pour '{text[:30]}...'")
        
        # Initialiser un vecteur de zéros
        embedding = [0.0] * self.vector_dim
        
        # Cette méthode est très simpliste et ne produit pas de bons embeddings!
        # Elle est fournie uniquement pour le développement et les tests.
        text_bytes = text.encode('utf-8')
        for i, byte in enumerate(text_bytes[:min(len(text_bytes), self.vector_dim)]):
            embedding[i] = float(byte) / 255.0
        
        return embedding