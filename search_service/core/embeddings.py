"""
Service de génération d'embeddings pour la recherche sémantique.

Ce module gère la génération d'embeddings via OpenAI API
avec cache intelligent et optimisations de performance.
"""
import asyncio
import hashlib
import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import aiohttp

from search_service.clients.base_client import BaseClient, RetryConfig, CircuitBreakerConfig, HealthCheckConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration pour le service d'embeddings."""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    max_tokens: int = 8191
    batch_size: int = 100
    cache_ttl_seconds: int = 3600  # 1 heure
    max_cache_size: int = 10000


class EmbeddingCache:
    """Cache LRU pour les embeddings avec TTL."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        
        # Métriques
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _generate_key(self, text: str, model: str) -> str:
        """Génère une clé de cache unique."""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Récupère un embedding du cache."""
        key = self._generate_key(text, model)
        
        if key in self._cache:
            entry = self._cache[key]
            
            # Vérifier l'expiration
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                # Mettre à jour l'ordre d'accès
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                
                self.hits += 1
                return entry["embedding"]
            else:
                # Entrée expirée
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
        
        self.misses += 1
        return None
    
    def put(self, text: str, model: str, embedding: List[float]):
        """Stocke un embedding dans le cache."""
        key = self._generate_key(text, model)
        
        # Éviction si nécessaire
        while len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        self._cache[key] = {
            "embedding": embedding,
            "timestamp": time.time()
        }
        
        # Mettre à jour l'ordre d'accès
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_oldest(self):
        """Évince l'entrée la plus ancienne."""
        if self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
                self.evictions += 1
    
    def clear(self):
        """Vide le cache."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "utilization": len(self._cache) / self.max_size
        }


class EmbeddingService(BaseClient):
    """
    Service de génération d'embeddings via OpenAI API.
    
    Fonctionnalités:
    - Génération d'embeddings avec cache intelligent
    - Traitement par batch pour l'efficacité
    - Retry logic et circuit breaker
    - Monitoring des tokens et coûts
    - Optimisations de performance
    """
    
    def __init__(
        self,
        api_key: str,
        config: Optional[EmbeddingConfig] = None,
        timeout: float = 10.0,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        self.api_key = api_key
        self.config = config or EmbeddingConfig()
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        health_check_config = HealthCheckConfig(
            enabled=True,
            interval_seconds=60.0,
            timeout_seconds=8.0,
            endpoint="/models"
        )
        
        super().__init__(
            base_url="https://api.openai.com/v1",
            service_name="openai_embeddings",
            timeout=timeout,
            retry_config=retry_config,
            circuit_breaker_config=circuit_breaker_config,
            health_check_config=health_check_config,
            headers=headers
        )
        
        # Cache des embeddings
        self.cache = EmbeddingCache(
            max_size=self.config.max_cache_size,
            ttl_seconds=self.config.cache_ttl_seconds
        )
        
        # Métriques
        self.tokens_used = 0
        self.api_calls = 0
        self.cache_saves = 0
        
        logger.info(f"Embedding service initialized with model: {self.config.model}")
    
    async def test_connection(self) -> bool:
        """Teste la connectivité à l'API OpenAI."""
        try:
            async def _test():
                async with self.session.get(f"{self.base_url}/models") as response:
                    return response.status == 200
            
            return await self.execute_with_retry(_test, "connection_test")
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Effectue une vérification de santé spécifique à OpenAI."""
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    
                    # Vérifier que notre modèle est disponible
                    available_models = [model["id"] for model in models_data.get("data", [])]
                    model_available = self.config.model in available_models
                    
                    return {
                        "model": self.config.model,
                        "model_available": model_available,
                        "total_models": len(available_models),
                        "status": "healthy" if model_available else "degraded",
                        "cache_stats": self.cache.get_stats()
                    }
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[List[float]]:
        """
        Génère un embedding pour un texte donné.
        
        Args:
            text: Texte à encoder
            model: Modèle à utiliser (par défaut config.model)
            use_cache: Utiliser le cache
            
        Returns:
            Vecteur d'embedding ou None en cas d'erreur
        """
        if not text or not text.strip():
            return None
        
        text = text.strip()
        model = model or self.config.model
        
        # Vérifier le cache d'abord
        if use_cache:
            cached_embedding = self.cache.get(text, model)
            if cached_embedding is not None:
                self.cache_saves += 1
                return cached_embedding
        
        # Générer l'embedding via l'API
        try:
            embedding = await self._call_embedding_api(text, model)
            
            # Mettre en cache si réussi
            if embedding and use_cache:
                self.cache.put(text, model, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Génère des embeddings pour un lot de textes.
        
        Args:
            texts: Liste de textes
            model: Modèle à utiliser
            use_cache: Utiliser le cache
            
        Returns:
            Liste d'embeddings (None pour les échecs)
        """
        if not texts:
            return []
        
        model = model or self.config.model
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        # Vérifier le cache pour tous les textes
        if use_cache:
            for i, text in enumerate(texts):
                if text and text.strip():
                    cached_embedding = self.cache.get(text.strip(), model)
                    if cached_embedding is not None:
                        results[i] = cached_embedding
                        self.cache_saves += 1
                    else:
                        uncached_indices.append(i)
                        uncached_texts.append(text.strip())
                else:
                    uncached_indices.append(i)
                    uncached_texts.append("")
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = [t.strip() if t else "" for t in texts]
        
        # Traiter les textes non cachés par batch
        if uncached_texts:
            batch_results = await self._generate_embeddings_batch_api(uncached_texts, model)
            
            # Mapper les résultats et mettre en cache
            for i, embedding in enumerate(batch_results):
                original_index = uncached_indices[i]
                results[original_index] = embedding
                
                # Mettre en cache si réussi
                if embedding and use_cache and uncached_texts[i]:
                    self.cache.put(uncached_texts[i], model, embedding)
        
        return results
    
    async def _call_embedding_api(self, text: str, model: str) -> Optional[List[float]]:
        """Appel API pour un seul embedding."""
        payload = {
            "input": text,
            "model": model
        }
        
        if self.config.dimensions and self.config.model == "text-embedding-3-small":
            payload["dimensions"] = self.config.dimensions
        
        async def _api_call():
            async with self.session.post(
                f"{self.base_url}/embeddings",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.api_calls += 1
                    self.tokens_used += data.get("usage", {}).get("total_tokens", 0)
                    
                    embeddings = data.get("data", [])
                    if embeddings:
                        return embeddings[0]["embedding"]
                    return None
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_api_call, "single_embedding")
    
    async def _generate_embeddings_batch_api(
        self,
        texts: List[str],
        model: str
    ) -> List[Optional[List[float]]]:
        """Génère des embeddings par batch via l'API."""
        results = [None] * len(texts)
        
        # Filtrer les textes vides
        valid_texts = [(i, text) for i, text in enumerate(texts) if text]
        
        if not valid_texts:
            return results
        
        # Traiter par chunks selon la taille de batch configurée
        for chunk_start in range(0, len(valid_texts), self.config.batch_size):
            chunk_end = min(chunk_start + self.config.batch_size, len(valid_texts))
            chunk = valid_texts[chunk_start:chunk_end]
            
            chunk_texts = [text for _, text in chunk]
            
            try:
                chunk_embeddings = await self._call_embeddings_batch_api(chunk_texts, model)
                
                # Mapper les résultats
                for j, (original_index, _) in enumerate(chunk):
                    if j < len(chunk_embeddings):
                        results[original_index] = chunk_embeddings[j]
                
            except Exception as e:
                logger.error(f"Batch embedding failed for chunk {chunk_start}-{chunk_end}: {e}")
                # Fallback: essayer individuellement
                for original_index, text in chunk:
                    try:
                        embedding = await self._call_embedding_api(text, model)
                        results[original_index] = embedding
                    except Exception as individual_error:
                        logger.error(f"Individual embedding failed for text: {individual_error}")
            
            # Délai entre les chunks pour respecter les rate limits
            if chunk_end < len(valid_texts):
                await asyncio.sleep(0.1)
        
        return results
    
    async def _call_embeddings_batch_api(
        self,
        texts: List[str],
        model: str
    ) -> List[Optional[List[float]]]:
        """Appel API pour un batch d'embeddings."""
        payload = {
            "input": texts,
            "model": model
        }
        
        if self.config.dimensions and self.config.model == "text-embedding-3-small":
            payload["dimensions"] = self.config.dimensions
        
        async def _batch_api_call():
            async with self.session.post(
                f"{self.base_url}/embeddings",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.api_calls += 1
                    self.tokens_used += data.get("usage", {}).get("total_tokens", 0)
                    
                    embeddings_data = data.get("data", [])
                    
                    # Créer une liste ordonnée selon l'index
                    embeddings = [None] * len(texts)
                    for item in embeddings_data:
                        index = item.get("index", 0)
                        if 0 <= index < len(embeddings):
                            embeddings[index] = item["embedding"]
                    
                    return embeddings
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI batch API error: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_batch_api_call, "batch_embeddings")
    
    def preprocess_text(self, text: str) -> str:
        """
        Préprocesse le texte avant génération d'embedding.
        
        Args:
            text: Texte brut
            
        Returns:
            Texte préprocessé
        """
        if not text:
            return ""
        
        # Nettoyer le texte
        text = text.strip()
        
        # Limiter la longueur selon le modèle
        # Note: approximation de 4 caractères par token
        max_chars = self.config.max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters for embedding")
        
        return text
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du service d'embeddings.
        
        Returns:
            Dictionnaire avec les métriques
        """
        cache_stats = self.cache.get_stats()
        
        return {
            "config": {
                "model": self.config.model,
                "dimensions": self.config.dimensions,
                "max_tokens": self.config.max_tokens,
                "batch_size": self.config.batch_size
            },
            "usage": {
                "api_calls": self.api_calls,
                "tokens_used": self.tokens_used,
                "cache_saves": self.cache_saves,
                "estimated_cost_usd": self._estimate_cost()
            },
            "cache": cache_stats,
            "performance": {
                "avg_response_time_ms": (
                    self.total_response_time / (self.request_count - self.error_count) * 1000
                    if self.request_count > self.error_count
                    else 0
                ),
                "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0
            }
        }
    
    def _estimate_cost(self) -> float:
        """
        Estime le coût des embeddings générés.
        
        Returns:
            Coût estimé en USD
        """
        # Prix approximatifs OpenAI (peut changer)
        price_per_1k_tokens = {
            "text-embedding-3-small": 0.00002,
            "text-embedding-3-large": 0.00013,
            "text-embedding-ada-002": 0.00010
        }
        
        price = price_per_1k_tokens.get(self.config.model, 0.00002)
        return (self.tokens_used / 1000) * price
    
    def clear_cache(self):
        """Vide le cache des embeddings."""
        self.cache.clear()
        logger.info("Embedding service cache cleared")
    
    def reset_stats(self):
        """Remet à zéro les statistiques."""
        self.tokens_used = 0
        self.api_calls = 0
        self.cache_saves = 0
        self.reset_metrics()
        logger.info("Embedding service stats reset")


class EmbeddingManager:
    """
    Gestionnaire d'embeddings avec support multi-modèles et fallbacks.
    
    Permet de gérer plusieurs services d'embeddings et de basculer
    automatiquement en cas de problème.
    """
    
    def __init__(self, primary_service: EmbeddingService):
        self.primary_service = primary_service
        self.fallback_services: List[EmbeddingService] = []
        
        # Métriques globales
        self.total_requests = 0
        self.primary_failures = 0
        self.fallback_usage = 0
        
        logger.info("Embedding manager initialized")
    
    def add_fallback_service(self, service: EmbeddingService):
        """Ajoute un service de fallback."""
        self.fallback_services.append(service)
        logger.info(f"Added fallback embedding service: {service.config.model}")
    
    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True,
        max_retries: int = 2
    ) -> Optional[List[float]]:
        """
        Génère un embedding avec fallback automatique.
        
        Args:
            text: Texte à encoder
            use_cache: Utiliser le cache
            max_retries: Nombre max de tentatives par service
            
        Returns:
            Vecteur d'embedding ou None
        """
        self.total_requests += 1
        
        # Essayer le service principal
        try:
            embedding = await self.primary_service.generate_embedding(text, use_cache=use_cache)
            if embedding is not None:
                return embedding
        except Exception as e:
            logger.warning(f"Primary embedding service failed: {e}")
            self.primary_failures += 1
        
        # Essayer les services de fallback
        for i, fallback_service in enumerate(self.fallback_services):
            try:
                logger.info(f"Trying fallback service {i+1}: {fallback_service.config.model}")
                embedding = await fallback_service.generate_embedding(text, use_cache=use_cache)
                if embedding is not None:
                    self.fallback_usage += 1
                    logger.info(f"Fallback service {i+1} succeeded")
                    return embedding
            except Exception as e:
                logger.warning(f"Fallback service {i+1} failed: {e}")
        
        logger.error("All embedding services failed")
        return None
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Génère des embeddings en batch avec fallback.
        
        Args:
            texts: Liste de textes
            use_cache: Utiliser le cache
            
        Returns:
            Liste d'embeddings
        """
        # Essayer le service principal
        try:
            embeddings = await self.primary_service.generate_embeddings_batch(texts, use_cache=use_cache)
            
            # Vérifier si des embeddings ont échoué
            failed_indices = [i for i, emb in enumerate(embeddings) if emb is None]
            
            if not failed_indices:
                return embeddings
            
            # Retry les échecs avec les services de fallback
            for fallback_service in self.fallback_services:
                if not failed_indices:
                    break
                
                failed_texts = [texts[i] for i in failed_indices]
                retry_embeddings = await fallback_service.generate_embeddings_batch(
                    failed_texts, use_cache=use_cache
                )
                
                # Remplacer les échecs par les succès
                for j, embedding in enumerate(retry_embeddings):
                    if embedding is not None:
                        original_index = failed_indices[j]
                        embeddings[original_index] = embedding
                        self.fallback_usage += 1
                
                # Mettre à jour la liste des échecs
                failed_indices = [
                    failed_indices[j] for j, emb in enumerate(retry_embeddings)
                    if emb is None
                ]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding completely failed: {e}")
            return [None] * len(texts)
    
    async def start_all_services(self):
        """Démarre tous les services d'embeddings."""
        await self.primary_service.start()
        
        for service in self.fallback_services:
            try:
                await service.start()
            except Exception as e:
                logger.warning(f"Failed to start fallback service: {e}")
    
    async def close_all_services(self):
        """Ferme tous les services d'embeddings."""
        await self.primary_service.close()
        
        for service in self.fallback_services:
            try:
                await service.close()
            except Exception as e:
                logger.warning(f"Failed to close fallback service: {e}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du gestionnaire."""
        primary_stats = self.primary_service.get_embedding_stats()
        
        fallback_stats = []
        for service in self.fallback_services:
            fallback_stats.append(service.get_embedding_stats())
        
        return {
            "manager": {
                "total_requests": self.total_requests,
                "primary_failures": self.primary_failures,
                "fallback_usage": self.fallback_usage,
                "primary_success_rate": (
                    (self.total_requests - self.primary_failures) / self.total_requests
                    if self.total_requests > 0 else 0
                ),
                "fallback_rate": (
                    self.fallback_usage / self.total_requests
                    if self.total_requests > 0 else 0
                )
            },
            "primary": primary_stats,
            "fallbacks": fallback_stats
        }
    
    def clear_all_caches(self):
        """Vide tous les caches d'embeddings."""
        self.primary_service.clear_cache()
        
        for service in self.fallback_services:
            service.clear_cache()
        
        logger.info("All embedding caches cleared")


# Fonctions utilitaires

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs.
    
    Args:
        vec1: Premier vecteur
        vec2: Second vecteur
        
    Returns:
        Similarité cosinus (-1 à 1)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalise un vecteur (norme L2).
    
    Args:
        vector: Vecteur à normaliser
        
    Returns:
        Vecteur normalisé
    """
    magnitude = sum(x * x for x in vector) ** 0.5
    
    if magnitude == 0:
        return vector
    
    return [x / magnitude for x in vector]


def vector_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcule la distance euclidienne entre deux vecteurs.
    
    Args:
        vec1: Premier vecteur
        vec2: Second vecteur
        
    Returns:
        Distance euclidienne
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5