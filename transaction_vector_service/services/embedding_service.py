# transaction_vector_service/services/embedding_service.py
"""
Service for generating vector embeddings.

This module provides functionality for transforming textual data into
vector embeddings that can be used for semantic search and similarity matching.
"""

import os
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import aiohttp

from openai import AsyncOpenAI

from ..config.settings import settings
from ..config.constants import EMBEDDING_CACHE_TTL
from ..config.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating vector embeddings from text using OpenAI."""

    def __init__(self):
        """Initialize the embedding service with OpenAI API."""
        self.model_name = "text-embedding-3-small"  # OpenAI's embedding model
        self.vector_dim = settings.EMBEDDING_DIMENSION
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables.")
        
        # Initialize the OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Cache for embeddings
        self._cache = {}
        self._cache_expiry = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Embedding service initialized with OpenAI model: {self.model_name}")

    def _generate_cache_key(self, text: str) -> str:
        """
        Generate a cache key for the given text.
        
        Args:
            text: Text to generate a cache key for
            
        Returns:
            Cache key as a string
        """
        # Create a hash of the text
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Retrieve an embedding from the cache if available and not expired.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not in cache or expired
        """
        cache_key = self._generate_cache_key(text)
        
        if cache_key in self._cache:
            # Check if cache entry has expired
            if cache_key in self._cache_expiry and datetime.now() < self._cache_expiry[cache_key]:
                self._cache_hits += 1
                return self._cache[cache_key]
            else:
                # Clear expired entries
                if cache_key in self._cache_expiry:
                    del self._cache_expiry[cache_key]
                if cache_key in self._cache:
                    del self._cache[cache_key]
        
        self._cache_misses += 1
        return None

    def _add_to_cache(self, text: str, embedding: List[float], ttl: int = EMBEDDING_CACHE_TTL):
        """
        Add an embedding to the cache with an expiry time.
        
        Args:
            text: Text that was embedded
            embedding: The embedding vector
            ttl: Time to live in seconds (default: 7 days)
        """
        cache_key = self._generate_cache_key(text)
        self._cache[cache_key] = embedding
        self._cache_expiry[cache_key] = datetime.now() + timedelta(seconds=ttl)
        
        # Cleanup cache if it gets too large (simple strategy)
        if len(self._cache) > 10000:
            # Remove oldest 1000 entries
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
        Get the embedding vector for a text string using OpenAI API.
        
        Args:
            text: Text to embed
            use_cache: Whether to use the cache (default: True)
            
        Returns:
            Embedding vector as a list of floats
        """
        if not text:
            # Return zero vector for empty text
            return [0.0] * self.vector_dim
        
        # Clean and normalize the input text
        text = text.strip().lower()
        
        # Check cache first if enabled
        if use_cache:
            cached_embedding = self._get_from_cache(text)
            if cached_embedding is not None:
                return cached_embedding
        
        try:
            # Generate embedding using OpenAI API
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.vector_dim
            )
            
            # Extract embedding from response
            embedding = response.data[0].embedding
            
            # Add to cache if enabled
            if use_cache:
                self._add_to_cache(text, embedding)
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding with OpenAI: {str(e)}")
            # Return zero vector in case of error
            return [0.0] * self.vector_dim

    async def get_embeddings(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use the cache (default: True)
            
        Returns:
            List of embedding vectors
        """
        # For empty list, return empty list
        if not texts:
            return []
            
        # Check cache first for all texts
        cached_embeddings = {}
        texts_to_embed = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cached = self._get_from_cache(text)
                if cached is not None:
                    cached_embeddings[i] = cached
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = [(i, text) for i, text in enumerate(texts)]
        
        # If all embeddings were in cache, return them
        if not texts_to_embed:
            return [cached_embeddings[i] for i in range(len(texts))]
        
        # Prepare batch request to OpenAI
        try:
            batch_texts = [text for _, text in texts_to_embed]
            
            # OpenAI has a limit on batch size, process in chunks if needed
            batch_size = 1000  # OpenAI's limit might be different
            embeddings_map = {}
            
            for i in range(0, len(batch_texts), batch_size):
                chunk = batch_texts[i:i+batch_size]
                
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=chunk,
                    dimensions=self.vector_dim
                )
                
                # Add to embeddings map
                for j, embedding_data in enumerate(response.data):
                    original_idx = texts_to_embed[i+j][0]
                    embeddings_map[original_idx] = embedding_data.embedding
                    
                    # Add to cache
                    if use_cache:
                        self._add_to_cache(batch_texts[j], embedding_data.embedding)
            
            # Combine cached and new embeddings
            result = []
            for i in range(len(texts)):
                if i in cached_embeddings:
                    result.append(cached_embeddings[i])
                else:
                    result.append(embeddings_map[i])
            
            return result
        except Exception as e:
            logger.error(f"Error generating batch embeddings with OpenAI: {str(e)}")
            # Return zero vectors in case of error
            return [[0.0] * self.vector_dim for _ in texts]

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_ratio": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_expiry.clear()
        logger.info("Embedding cache cleared")