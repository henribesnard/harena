"""
Stratégies de cache pour optimisation performance
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """Entrée cache avec métadonnées"""
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        """Vérification expiration TTL"""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> int:
        """Age en secondes"""
        return int((datetime.utcnow() - self.created_at).total_seconds())
    
    def mark_accessed(self):
        """Marquer comme accédé"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class CacheStrategy(ABC):
    """Interface stratégie de cache"""
    
    @abstractmethod
    async def should_evict(self, entry: CacheEntry) -> bool:
        """Détermine si une entrée doit être évincée"""
        pass
    
    @abstractmethod
    async def get_eviction_priority(self, entry: CacheEntry) -> float:
        """Score priorité éviction (plus élevé = éviction prioritaire)"""
        pass


class LRUStrategy(CacheStrategy):
    """Stratégie Least Recently Used"""
    
    async def should_evict(self, entry: CacheEntry) -> bool:
        """LRU ne force pas l'éviction - basé sur priorité"""
        return entry.is_expired
    
    async def get_eviction_priority(self, entry: CacheEntry) -> float:
        """Plus ancien accès = priorité éviction plus haute"""
        seconds_since_access = (datetime.utcnow() - entry.last_accessed).total_seconds()
        return seconds_since_access


class TTLStrategy(CacheStrategy):
    """Stratégie Time To Live"""
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
    
    async def should_evict(self, entry: CacheEntry) -> bool:
        """Éviction basée sur TTL"""
        return entry.is_expired
    
    async def get_eviction_priority(self, entry: CacheEntry) -> float:
        """Entrées expirées ont priorité maximale"""
        if entry.is_expired:
            return float('inf')
        
        # Score basé sur proximité expiration
        ttl = entry.ttl_seconds or self.default_ttl
        remaining_ratio = 1 - (entry.age_seconds / ttl)
        return 1 - remaining_ratio  # Proche expiration = priorité haute


class HybridStrategy(CacheStrategy):
    """Stratégie hybride LRU + TTL + fréquence"""
    
    def __init__(self, ttl_weight: float = 0.5, lru_weight: float = 0.3, frequency_weight: float = 0.2):
        self.ttl_weight = ttl_weight
        self.lru_weight = lru_weight
        self.frequency_weight = frequency_weight
        
        self.ttl_strategy = TTLStrategy()
        self.lru_strategy = LRUStrategy()
    
    async def should_evict(self, entry: CacheEntry) -> bool:
        """Éviction si TTL expiré"""
        return entry.is_expired
    
    async def get_eviction_priority(self, entry: CacheEntry) -> float:
        """Score composite TTL + LRU + fréquence d'accès"""
        
        # Composante TTL
        ttl_score = await self.ttl_strategy.get_eviction_priority(entry)
        if ttl_score == float('inf'):
            return float('inf')  # Expiré = éviction immédiate
        
        # Composante LRU  
        lru_score = await self.lru_strategy.get_eviction_priority(entry)
        lru_normalized = min(lru_score / 3600, 1.0)  # Normalisation sur 1h
        
        # Composante fréquence (inverse)
        frequency_score = 1 / (1 + entry.access_count)  # Moins accédé = score plus haut
        
        # Score composite
        composite_score = (
            self.ttl_weight * ttl_score +
            self.lru_weight * lru_normalized +
            self.frequency_weight * frequency_score
        )
        
        return composite_score