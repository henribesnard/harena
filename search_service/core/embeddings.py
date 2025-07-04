"""
Service d'embeddings pour la recherche sémantique.

Ce module gère la génération d'embeddings via OpenAI API avec cache intelligent,
batch processing et gestion avancée des erreurs pour optimiser les performances.

CORRECTION: Restructuration pour éviter les imports circulaires.
"""
import asyncio
import hashlib
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

import openai
from openai import AsyncOpenAI

# Import du cache avec gestion des erreurs
try:
    from search_service.utils.cache import SearchCache
except ImportError:
    # Fallback si le cache n'est pas disponible
    class SearchCache:
        def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
            self._cache = {}
            self.max_size = max_size
            self.ttl_seconds = ttl_seconds
        
        def get(self, key: str) -> Any:
            return self._cache.get(key)
        
        def put(self, key: str, value: Any) -> None:
            if len(self._cache) < self.max_size:
                self._cache[key] = value
        
        def clear(self) -> None:
            self._cache.clear()
        
        def get_stats(self) -> Dict[str, Any]:
            return {"size": len(self._cache), "max_size": self.max_size}

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Modèles d'embeddings disponibles."""
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large" 
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


@dataclass
class EmbeddingConfig:
    """Configuration pour le service d'embeddings."""
    # Modèle et dimensions
    model: EmbeddingModel = EmbeddingModel.TEXT_EMBEDDING_3_SMALL
    dimensions: int = 1536
    
    # Traitement par lots
    batch_size: int = 100
    max_concurrent_batches: int = 3
    
    # Cache
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 heure
    cache_max_size: int = 10000
    
    # Retry et timeouts
    max_retries: int = 3
    timeout_seconds: float = 30.0
    retry_delay_base: float = 1.0
    
    # Rate limiting
    requests_per_minute: int = 3000
    tokens_per_minute: int = 1000000
    
    # Optimisations
    enable_text_preprocessing: bool = True
    max_text_length: int = 8192
    truncation_strategy: str = "end"  # "start", "middle", "end"
    
    # Monitoring
    enable_metrics: bool = True
    log_expensive_requests: bool = True
    expensive_request_threshold: int = 50  # tokens


@dataclass
class EmbeddingResult:
    """Résultat d'une génération d'embedding."""
    embedding: List[float]
    text: str
    model: str
    dimensions: int
    tokens_used: int
    processing_time_ms: float
    from_cache: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchEmbeddingResult:
    """Résultat d'un traitement par lots."""
    results: List[EmbeddingResult]
    total_tokens: int
    total_processing_time_ms: float
    cache_hits: int
    api_calls: int
    failed_items: List[Tuple[str, str]] = field(default_factory=list)  # (text, error)


class EmbeddingService:
    """
    Service de génération d'embeddings avec OpenAI.
    
    Responsabilités:
    - Génération d'embeddings via OpenAI API
    - Cache intelligent avec TTL
    - Traitement par lots pour efficacité
    - Gestion des erreurs et retry
    - Rate limiting et monitoring
    """
    
    def __init__(
        self,
        api_key: str,
        config: Optional[EmbeddingConfig] = None
    ):
        self.config = config or EmbeddingConfig()
        
        # Client OpenAI
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=self.config.timeout_seconds
        )
        
        # Cache pour les embeddings
        self.cache = SearchCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds
        ) if self.config.enable_cache else None
        
        # Rate limiting
        self._request_times = []
        self._token_usage = []
        
        # Métriques
        self.total_requests = 0
        self.total_tokens = 0
        self.cache_hits = 0
        self.api_errors = 0
        self.total_processing_time = 0.0
        self.model_usage = {model.value: 0 for model in EmbeddingModel}
        
        # Semaphore pour limiter la concurrence
        self._batch_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        
        logger.info(f"Embedding service initialized with model {self.config.model.value}")
    
    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True,
        model: Optional[EmbeddingModel] = None,
        dimensions: Optional[int] = None
    ) -> Optional[List[float]]:
        """
        Génère un embedding pour un texte donné.
        
        Args:
            text: Texte à encoder
            use_cache: Utiliser le cache si disponible
            model: Modèle à utiliser (défaut: config)
            dimensions: Dimensions de l'embedding (défaut: config)
            
        Returns:
            Vecteur d'embedding ou None si erreur
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return None
        
        start_time = time.time()
        self.total_requests += 1
        
        # Utiliser les paramètres par défaut si non spécifiés
        model = model or self.config.model
        dimensions = dimensions or self.config.dimensions
        
        # Préprocesser le texte
        processed_text = self._preprocess_text(text)
        
        # Vérifier le cache
        cache_key = None
        if use_cache and self.cache:
            cache_key = self._generate_cache_key(processed_text, model, dimensions)
            cached_embedding = self.cache.get(cache_key)
            
            if cached_embedding:
                self.cache_hits += 1
                logger.debug(f"Cache hit for embedding: {cache_key[:16]}...")
                return cached_embedding
        
        try:
            # Générer l'embedding via API
            result = await self._generate_single_embedding(
                processed_text, model, dimensions
            )
            
            if result and result.embedding:
                # Mettre en cache
                if use_cache and self.cache and cache_key:
                    self.cache.put(cache_key, result.embedding)
                
                # Mettre à jour les métriques
                processing_time = (time.time() - start_time) * 1000
                self.total_processing_time += processing_time
                self.total_tokens += result.tokens_used
                self.model_usage[model.value] += 1
                
                return result.embedding
            
            return None
            
        except Exception as e:
            self.api_errors += 1
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _preprocess_text(self, text: str) -> str:
        """Préprocesse le texte avant génération d'embedding."""
        if not self.config.enable_text_preprocessing:
            return text
        
        # Nettoyer le texte
        processed = text.strip()
        
        # Supprimer les caractères de contrôle
        processed = "".join(char for char in processed if ord(char) >= 32 or char in "\n\t")
        
        # Normaliser les espaces
        processed = " ".join(processed.split())
        
        # Tronquer si nécessaire
        if len(processed) > self.config.max_text_length:
            if self.config.truncation_strategy == "start":
                processed = processed[-self.config.max_text_length:]
            elif self.config.truncation_strategy == "middle":
                half = self.config.max_text_length // 2
                processed = processed[:half] + processed[-half:]
            else:  # "end"
                processed = processed[:self.config.max_text_length]
        
        return processed
    
    def _generate_cache_key(
        self,
        text: str,
        model: EmbeddingModel,
        dimensions: int
    ) -> str:
        """Génère une clé de cache unique."""
        cache_data = f"{text}|{model.value}|{dimensions}"
        return hashlib.sha256(cache_data.encode()).hexdigest()
    
    async def _generate_single_embedding(
        self,
        text: str,
        model: EmbeddingModel,
        dimensions: int
    ) -> Optional[EmbeddingResult]:
        """Génère un embedding pour un seul texte."""
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                # Paramètres de la requête
                params = {
                    "model": model.value,
                    "input": text,
                    "encoding_format": "float"
                }
                
                # Ajouter dimensions si supporté par le modèle
                if model in [EmbeddingModel.TEXT_EMBEDDING_3_SMALL, EmbeddingModel.TEXT_EMBEDDING_3_LARGE]:
                    params["dimensions"] = dimensions
                
                # Appeler l'API OpenAI
                response = await self.client.embeddings.create(**params)
                
                processing_time = (time.time() - start_time) * 1000
                
                # Extraire les données
                if response.data:
                    embedding_data = response.data[0]
                    tokens_used = response.usage.total_tokens
                    
                    return EmbeddingResult(
                        embedding=embedding_data.embedding,
                        text=text,
                        model=model.value,
                        dimensions=len(embedding_data.embedding),
                        tokens_used=tokens_used,
                        processing_time_ms=processing_time,
                        from_cache=False
                    )
                
                return None
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to generate embedding after {self.config.max_retries} attempts: {e}")
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(f"API error, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du service."""
        avg_processing_time = (
            self.total_processing_time / self.total_requests
            if self.total_requests > 0 else 0
        )
        
        cache_hit_rate = (
            self.cache_hits / self.total_requests
            if self.total_requests > 0 else 0
        )
        
        error_rate = (
            self.api_errors / self.total_requests
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "api_errors": self.api_errors,
            "error_rate": error_rate,
            "average_processing_time_ms": avg_processing_time,
            "model_usage": self.model_usage,
            "cache_stats": self.cache.get_stats() if self.cache else None
        }


class EmbeddingManager:
    """
    Gestionnaire d'embeddings de haut niveau avec multiple services.
    
    Permet la gestion de plusieurs services d'embeddings avec fallback
    et optimisations avancées.
    """
    
    def __init__(self, primary_service: EmbeddingService):
        self.primary_service = primary_service
        self.fallback_services: List[EmbeddingService] = []
        
        # Métriques globales
        self.total_requests = 0
        self.successful_requests = 0
        self.fallback_usage = 0
        
        logger.info("Embedding manager initialized")
    
    def add_fallback_service(self, service: EmbeddingService) -> None:
        """Ajoute un service de fallback."""
        self.fallback_services.append(service)
        logger.info(f"Added fallback embedding service (total: {len(self.fallback_services)})")
    
    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True,
        model: Optional[EmbeddingModel] = None,
        dimensions: Optional[int] = None
    ) -> Optional[List[float]]:
        """
        Génère un embedding avec fallback automatique.
        
        Essaie le service primaire, puis les services de fallback si échec.
        """
        self.total_requests += 1
        
        # Essayer le service primaire
        try:
            embedding = await self.primary_service.generate_embedding(
                text, use_cache, model, dimensions
            )
            if embedding:
                self.successful_requests += 1
                return embedding
        except Exception as e:
            logger.warning(f"Primary embedding service failed: {e}")
        
        # Essayer les services de fallback
        for i, fallback_service in enumerate(self.fallback_services):
            try:
                logger.info(f"Trying fallback service {i + 1}")
                embedding = await fallback_service.generate_embedding(
                    text, use_cache, model, dimensions
                )
                if embedding:
                    self.successful_requests += 1
                    self.fallback_usage += 1
                    return embedding
            except Exception as e:
                logger.warning(f"Fallback service {i + 1} failed: {e}")
        
        logger.error(f"All embedding services failed for text: {text[:100]}...")
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du gestionnaire."""
        success_rate = (
            self.successful_requests / self.total_requests
            if self.total_requests > 0 else 0
        )
        
        fallback_rate = (
            self.fallback_usage / self.successful_requests
            if self.successful_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": success_rate,
            "fallback_usage": self.fallback_usage,
            "fallback_rate": fallback_rate,
            "fallback_services_count": len(self.fallback_services),
            "primary_service_metrics": self.primary_service.get_metrics()
        }


# Factory functions et exports
def create_embedding_service(
    api_key: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 1536,
    enable_cache: bool = True,
    **kwargs
) -> EmbeddingService:
    """Factory function pour créer un service d'embeddings."""
    # Convertir le nom du modèle en enum
    model_enum = EmbeddingModel.TEXT_EMBEDDING_3_SMALL
    for enum_model in EmbeddingModel:
        if enum_model.value == model:
            model_enum = enum_model
            break
    
    # Créer la configuration
    config = EmbeddingConfig(
        model=model_enum,
        dimensions=dimensions,
        enable_cache=enable_cache,
        **kwargs
    )
    
    return EmbeddingService(api_key=api_key, config=config)


# Exports principaux
__all__ = [
    "EmbeddingService",
    "EmbeddingManager",
    "EmbeddingConfig",
    "EmbeddingModel",
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "create_embedding_service"
]