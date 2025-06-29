"""
Service d'embeddings pour la recherche sémantique.

Ce module gère la génération d'embeddings via OpenAI API avec cache intelligent,
batch processing et gestion avancée des erreurs pour optimiser les performances.
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

from search_service.utils.cache import SearchCache

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
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        model: Optional[EmbeddingModel] = None,
        dimensions: Optional[int] = None
    ) -> BatchEmbeddingResult:
        """
        Génère des embeddings en lot pour optimiser les performances.
        
        Args:
            texts: Liste de textes à encoder
            use_cache: Utiliser le cache si disponible
            model: Modèle à utiliser (défaut: config)
            dimensions: Dimensions des embeddings (défaut: config)
            
        Returns:
            Résultats du traitement par lots
        """
        if not texts:
            return BatchEmbeddingResult(
                results=[], total_tokens=0, total_processing_time_ms=0.0,
                cache_hits=0, api_calls=0
            )
        
        start_time = time.time()
        model = model or self.config.model
        dimensions = dimensions or self.config.dimensions
        
        # Préprocesser et dédupliquer les textes
        processed_texts = []
        text_indices = {}  # Map texte -> indices originaux
        
        for i, text in enumerate(texts):
            if text and text.strip():
                processed = self._preprocess_text(text)
                if processed not in text_indices:
                    text_indices[processed] = []
                    processed_texts.append(processed)
                text_indices[processed].append(i)
        
        # Vérifier le cache pour tous les textes
        cache_results = {}
        uncached_texts = []
        cache_hits = 0
        
        if use_cache and self.cache:
            for text in processed_texts:
                cache_key = self._generate_cache_key(text, model, dimensions)
                cached_embedding = self.cache.get(cache_key)
                
                if cached_embedding:
                    cache_results[text] = cached_embedding
                    cache_hits += 1
                else:
                    uncached_texts.append(text)
        else:
            uncached_texts = processed_texts
        
        # Traiter les textes non cachés par lots
        api_results = {}
        total_tokens = 0
        api_calls = 0
        failed_items = []
        
        if uncached_texts:
            # Diviser en batches
            batches = [
                uncached_texts[i:i + self.config.batch_size]
                for i in range(0, len(uncached_texts), self.config.batch_size)
            ]
            
            # Traiter les batches en parallèle (avec limite de concurrence)
            batch_tasks = [
                self._process_batch(batch, model, dimensions, use_cache)
                for batch in batches
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Consolider les résultats
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    logger.error(f"Batch processing failed: {batch_result}")
                    continue
                
                if batch_result:
                    api_results.update(batch_result.get("results", {}))
                    total_tokens += batch_result.get("tokens", 0)
                    api_calls += batch_result.get("api_calls", 0)
                    failed_items.extend(batch_result.get("failed", []))
        
        # Construire les résultats finaux
        final_results = []
        processing_time = (time.time() - start_time) * 1000
        
        for text in processed_texts:
            # Obtenir l'embedding (cache ou API)
            embedding = cache_results.get(text) or api_results.get(text)
            
            if embedding:
                # Créer les résultats pour tous les indices originaux
                for original_idx in text_indices[text]:
                    result = EmbeddingResult(
                        embedding=embedding,
                        text=texts[original_idx],
                        model=model.value,
                        dimensions=dimensions,
                        tokens_used=len(text.split()),  # Estimation
                        processing_time_ms=processing_time / len(processed_texts),
                        from_cache=text in cache_results
                    )
                    final_results.append(result)
        
        # Mettre à jour les métriques globales
        self.total_tokens += total_tokens
        self.cache_hits += cache_hits
        self.total_processing_time += processing_time
        
        return BatchEmbeddingResult(
            results=final_results,
            total_tokens=total_tokens,
            total_processing_time_ms=processing_time,
            cache_hits=cache_hits,
            api_calls=api_calls,
            failed_items=failed_items
        )
    
    async def _generate_single_embedding(
        self,
        text: str,
        model: EmbeddingModel,
        dimensions: int
    ) -> Optional[EmbeddingResult]:
        """Génère un embedding pour un seul texte."""
        
        # Attendre le rate limiting
        await self._wait_for_rate_limit()
        
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
                    
                    # Log si requête coûteuse
                    if (self.config.log_expensive_requests and 
                        tokens_used > self.config.expensive_request_threshold):
                        logger.info(f"Expensive embedding request: {tokens_used} tokens")
                    
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
                
            except openai.RateLimitError as e:
                wait_time = self._calculate_retry_delay(attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
                
            except openai.APITimeoutError as e:
                wait_time = self._calculate_retry_delay(attempt)
                logger.warning(f"API timeout, retrying in {wait_time}s (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to generate embedding after {self.config.max_retries} attempts: {e}")
                    raise
                
                wait_time = self._calculate_retry_delay(attempt)
                logger.warning(f"API error, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
        
        return None
    
    async def _process_batch(
        self,
        batch_texts: List[str],
        model: EmbeddingModel,
        dimensions: int,
        use_cache: bool
    ) -> Optional[Dict[str, Any]]:
        """Traite un lot de textes."""
        async with self._batch_semaphore:
            try:
                # Attendre le rate limiting
                await self._wait_for_rate_limit()
                
                start_time = time.time()
                
                # Paramètres de la requête batch
                params = {
                    "model": model.value,
                    "input": batch_texts,
                    "encoding_format": "float"
                }
                
                if model in [EmbeddingModel.TEXT_EMBEDDING_3_SMALL, EmbeddingModel.TEXT_EMBEDDING_3_LARGE]:
                    params["dimensions"] = dimensions
                
                # Appeler l'API
                response = await self.client.embeddings.create(**params)
                
                processing_time = (time.time() - start_time) * 1000
                
                # Traiter la réponse
                results = {}
                if response.data:
                    for i, embedding_data in enumerate(response.data):
                        if i < len(batch_texts):
                            text = batch_texts[i]
                            embedding = embedding_data.embedding
                            
                            # Mettre en cache
                            if use_cache and self.cache:
                                cache_key = self._generate_cache_key(text, model, dimensions)
                                self.cache.put(cache_key, embedding)
                            
                            results[text] = embedding
                
                return {
                    "results": results,
                    "tokens": response.usage.total_tokens,
                    "api_calls": 1,
                    "processing_time": processing_time,
                    "failed": []
                }
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                return {
                    "results": {},
                    "tokens": 0,
                    "api_calls": 0,
                    "processing_time": 0,
                    "failed": [(text, str(e)) for text in batch_texts]
                }
    
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
    
    async def _wait_for_rate_limit(self) -> None:
        """Attendre si nécessaire pour respecter les limites de taux."""
        current_time = time.time()
        
        # Nettoyer les anciens timestamps
        self._request_times = [t for t in self._request_times if current_time - t < 60]
        
        # Vérifier la limite de requêtes par minute
        if len(self._request_times) >= self.config.requests_per_minute:
            wait_time = 60 - (current_time - self._request_times[0])
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Enregistrer cette requête
        self._request_times.append(current_time)
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calcule le délai de retry avec backoff exponentiel."""
        return self.config.retry_delay_base * (2 ** attempt) + (attempt * 0.1)
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du service d'embeddings."""
        try:
            # Test avec un texte simple
            test_embedding = await self.generate_embedding(
                "test query for health check",
                use_cache=False
            )
            
            return {
                "status": "healthy",
                "model": self.config.model.value,
                "dimensions": self.config.dimensions,
                "test_embedding_generated": test_embedding is not None,
                "cache_enabled": self.config.enable_cache,
                "metrics": self.get_metrics()
            }
            
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "metrics": self.get_metrics()
            }
    
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
    
    def clear_cache(self) -> None:
        """Vide le cache des embeddings."""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")
    
    def reset_metrics(self) -> None:
        """Remet à zéro les métriques."""
        self.total_requests = 0
        self.total_tokens = 0
        self.cache_hits = 0
        self.api_errors = 0
        self.total_processing_time = 0.0
        self.model_usage = {model.value: 0 for model in EmbeddingModel}
        
        logger.info("Embedding service metrics reset")


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
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        model: Optional[EmbeddingModel] = None,
        dimensions: Optional[int] = None
    ) -> BatchEmbeddingResult:
        """Génère des embeddings en lot avec fallback."""
        # Essayer le service primaire
        try:
            return await self.primary_service.generate_embeddings_batch(
                texts, use_cache, model, dimensions
            )
        except Exception as e:
            logger.warning(f"Primary batch service failed: {e}")
        
        # Fallback vers génération individuelle
        results = []
        for text in texts:
            embedding = await self.generate_embedding(text, use_cache, model, dimensions)
            if embedding:
                result = EmbeddingResult(
                    embedding=embedding,
                    text=text,
                    model=(model or self.primary_service.config.model).value,
                    dimensions=dimensions or self.primary_service.config.dimensions,
                    tokens_used=len(text.split()),
                    processing_time_ms=0.0,
                    from_cache=False
                )
                results.append(result)
        
        return BatchEmbeddingResult(
            results=results,
            total_tokens=sum(r.tokens_used for r in results),
            total_processing_time_ms=0.0,
            cache_hits=0,
            api_calls=len(results),
            failed_items=[]
        )
    
    async def precompute_embeddings(
        self,
        texts: List[str],
        model: Optional[EmbeddingModel] = None,
        dimensions: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Précalcule les embeddings pour une liste de textes.
        
        Utile pour le warm-up du cache ou la préparation de données.
        """
        start_time = time.time()
        
        batch_size = batch_size or self.primary_service.config.batch_size
        total_texts = len(texts)
        processed = 0
        successful = 0
        failed = 0
        
        logger.info(f"Starting precomputation for {total_texts} texts")
        
        # Traiter par batches
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                result = await self.generate_embeddings_batch(
                    batch, use_cache=True, model=model, dimensions=dimensions
                )
                
                successful += len(result.results)
                failed += len(result.failed_items)
                processed += len(batch)
                
                # Log du progrès
                if processed % (batch_size * 10) == 0 or processed == total_texts:
                    logger.info(f"Precomputation progress: {processed}/{total_texts} "
                              f"({successful} successful, {failed} failed)")
                
            except Exception as e:
                logger.error(f"Batch precomputation failed: {e}")
                failed += len(batch)
                processed += len(batch)
        
        processing_time = time.time() - start_time
        
        result_summary = {
            "total_texts": total_texts,
            "successful": successful,
            "failed": failed,
            "processing_time_seconds": processing_time,
            "texts_per_second": total_texts / processing_time if processing_time > 0 else 0
        }
        
        logger.info(f"Precomputation completed: {result_summary}")
        return result_summary
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé de tous les services."""
        primary_health = await self.primary_service.health_check()
        
        fallback_healths = []
        for i, service in enumerate(self.fallback_services):
            try:
                health = await service.health_check()
                health["service_index"] = i
                fallback_healths.append(health)
            except Exception as e:
                fallback_healths.append({
                    "service_index": i,
                    "status": "unhealthy",
                    "error": str(e)
                })
        
        return {
            "primary_service": primary_health,
            "fallback_services": fallback_healths,
            "manager_metrics": self.get_metrics()
        }
    
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
    
    def reset_metrics(self) -> None:
        """Remet à zéro toutes les métriques."""
        self.total_requests = 0
        self.successful_requests = 0
        self.fallback_usage = 0
        
        self.primary_service.reset_metrics()
        for service in self.fallback_services:
            service.reset_metrics()
        
        logger.info("All embedding service metrics reset")