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

# Import de la bibliothèque OpenAI
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service pour la génération d'embeddings à partir de texte."""

    def __init__(self):
        """Initialise le service d'embeddings."""
        self.model_name = "text-embedding-3-small"  # Modèle d'embedding OpenAI
        self.vector_dim = 1536  # Dimension des vecteurs
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY n'a pas été trouvée dans les variables d'environnement.")
        
        # Initialiser le client OpenAI
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Cache pour les embeddings
        self._cache = {}
        self._cache_expiry = {}
        
        logger.info(f"Service d'embedding initialisé avec le modèle: {self.model_name}")

    def _generate_cache_key(self, text: str) -> str:
        """
        Génère une clé de cache pour le texte donné.
        
        Args:
            text: Texte pour générer une clé de cache
            
        Returns:
            Clé de cache sous forme de chaîne
        """
        # Créer un hash du texte
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Récupère un embedding du cache s'il est disponible et non expiré.
        
        Args:
            text: Texte pour lequel récupérer l'embedding
            
        Returns:
            Embedding en cache ou None si pas en cache ou expiré
        """
        cache_key = self._generate_cache_key(text)
        
        if cache_key in self._cache:
            # Vérifier si l'entrée de cache a expiré
            if cache_key in self._cache_expiry and datetime.now() < self._cache_expiry[cache_key]:
                return self._cache[cache_key]
            else:
                # Supprimer les entrées expirées
                if cache_key in self._cache_expiry:
                    del self._cache_expiry[cache_key]
                if cache_key in self._cache:
                    del self._cache[cache_key]
        
        return None

    def _add_to_cache(self, text: str, embedding: List[float], ttl: int = 604800):  # 7 jours
        """
        Ajoute un embedding au cache avec un temps d'expiration.
        
        Args:
            text: Texte qui a été converti en embedding
            embedding: Vecteur d'embedding
            ttl: Durée de vie en secondes (par défaut: 7 jours)
        """
        cache_key = self._generate_cache_key(text)
        self._cache[cache_key] = embedding
        self._cache_expiry[cache_key] = datetime.now() + timedelta(seconds=ttl)
        
        # Nettoyage du cache s'il devient trop grand (stratégie simple)
        if len(self._cache) > 10000:
            # Supprimer les 1000 entrées les plus anciennes
            expired_keys = sorted(
                self._cache_expiry.keys(),
                key=lambda k: self._cache_expiry[k]
            )[:1000]
            
            for key in expired_keys:
                if key in self._cache:
                    del self._cache[key]
                if key in self._cache_expiry:
                    del self._cache_expiry[key]

    async def get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Obtient le vecteur d'embedding pour une chaîne de texte en utilisant l'API OpenAI.
        
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
            cached_embedding = self._get_from_cache(text)
            if cached_embedding is not None:
                return cached_embedding
        
        try:
            # Générer l'embedding en utilisant l'API OpenAI
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.vector_dim
            )
            
            # Extraire l'embedding de la réponse
            embedding = response.data[0].embedding
            
            # Ajouter au cache si activé
            if use_cache:
                self._add_to_cache(text, embedding)
                
            return embedding
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding avec OpenAI: {str(e)}")
            # Renvoyer un vecteur de zéros en cas d'erreur
            return [0.0] * self.vector_dim