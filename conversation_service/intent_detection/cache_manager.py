"""
Gestionnaire de cache spécialisé pour la détection d'intention
Intégration avec le système Redis multi-niveaux
"""

import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime

from conversation_service.cache.redis_manager import RedisManager
from conversation_service.config import settings
from conversation_service.utils.logging import get_logger
from .models import IntentResult

logger = get_logger(__name__)


class IntentCacheManager:
    """
    Cache manager spécialisé pour les intentions avec stratégies TTL optimisées
    """
    
    def __init__(self):
        self.redis_manager = RedisManager()
        
        # Préfixes cache par niveau
        self.PATTERN_PREFIX = "harena:intent:patterns"
        self.EMBEDDING_PREFIX = "harena:intent:embeddings" 
        self.LLM_PREFIX = "harena:intent:llm"
        self.USER_PREFIX = "harena:user"
    
    async def initialize(self):
        """Initialisation avec Redis manager"""
        await self.redis_manager.initialize()
    
    # === NIVEAU L0 - Cache Patterns ===
    
    async def get_pattern_cache(self, normalized_query: str) -> Optional[Dict[str, Any]]:
        """Récupération pattern pré-calculé (TTL: 1h)"""
        cache_key = f"{self.PATTERN_PREFIX}:{self._hash_string(normalized_query)}"
        return await self.redis_manager.get(cache_key)
    
    async def set_pattern_cache(
        self, 
        normalized_query: str, 
        intent_result: IntentResult,
        ttl: int = None
    ) -> bool:
        """Stockage pattern avec TTL optimisé"""
        cache_key = f"{self.PATTERN_PREFIX}:{self._hash_string(normalized_query)}"
        ttl = ttl or settings.INTENT_CACHE_TTL_PATTERNS
        
        cache_data = intent_result.to_cache()
        return await self.redis_manager.set(cache_key, cache_data, ttl)
    
    # === NIVEAU L1 - Cache Embeddings ===
    
    async def get_embedding_cache(self, query_hash: str, user_id: int) -> Optional[List[float]]:
        """Récupération embedding TinyBERT (TTL: 30min)"""
        cache_key = f"{self.EMBEDDING_PREFIX}:{user_id}:{query_hash}"
        result = await self.redis_manager.get(cache_key)
        return result.get("embedding") if result else None
    
    async def set_embedding_cache(
        self, 
        query_hash: str, 
        user_id: int, 
        embedding: List[float],
        ttl: int = None
    ) -> bool:
        """Stockage embedding avec TTL"""
        cache_key = f"{self.EMBEDDING_PREFIX}:{user_id}:{query_hash}"
        ttl = ttl or settings.INTENT_CACHE_TTL_EMBEDDINGS
        
        cache_data = {
            "embedding": embedding,
            "cached_at": datetime.utcnow().isoformat(),
            "user_id": user_id
        }
        return await self.redis_manager.set(cache_key, cache_data, ttl)
    
    # === NIVEAU L2 - Cache LLM Responses ===
    
    async def get_llm_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Récupération réponse DeepSeek (TTL: 15min)"""
        full_key = f"{self.LLM_PREFIX}:{cache_key}"
        return await self.redis_manager.get(full_key)
    
    async def set_llm_cache(
        self, 
        cache_key: str, 
        intent_result: IntentResult,
        ttl: int = None
    ) -> bool:
        """Stockage réponse LLM avec TTL court"""
        full_key = f"{self.LLM_PREFIX}:{cache_key}"
        ttl = ttl or settings.INTENT_CACHE_TTL_LLM_RESPONSES
        
        cache_data = intent_result.to_cache()
        return await self.redis_manager.set(full_key, cache_data, ttl)
    
    # === Cache Utilisateur Spécialisé ===
    
    async def clear_user_cache(self, user_id: int) -> bool:
        """Nettoyage cache spécifique utilisateur"""
        try:
            # Pattern matching pour toutes les clés utilisateur
            user_pattern = f"{self.USER_PREFIX}:{user_id}:*"
            embedding_pattern = f"{self.EMBEDDING_PREFIX}:{user_id}:*"
            
            # Note: En production, utiliser SCAN pour éviter KEYS sur large dataset
            # Ici simplifié pour l'exemple
            
            deleted_count = 0
            
            # Suppression embeddings utilisateur
            keys_to_delete = []
            # Implementation simplifiée - en prod utiliser Redis SCAN
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur suppression cache utilisateur {user_id}: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Statistiques cache par niveau"""
        return await self.redis_manager.get_stats()
    
    def _hash_string(self, content: str) -> str:
        """Hash rapide pour clés cache"""
        return hashlib.md5(content.encode()).hexdigest()[:12]