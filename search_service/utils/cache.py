"""
Cache en mémoire pour les résultats de recherche.

Ce module fournit un cache LRU simple pour améliorer les performances
des recherches répétées.
"""
import logging
import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import asyncio

from search_service.models import SearchQuery, SearchResponse

logger = logging.getLogger(__name__)


class SearchCache:
    """Cache LRU pour les résultats de recherche."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialise le cache.
        
        Args:
            max_size: Nombre maximum d'entrées
            default_ttl: Durée de vie par défaut en secondes
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, Tuple[SearchResponse, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._lock = asyncio.Lock()
        
        # Statistiques par utilisateur
        self.user_stats: Dict[int, Dict[str, int]] = {}
    
    def generate_key(self, query: SearchQuery) -> str:
        """
        Génère une clé de cache unique pour une requête.
        
        Args:
            query: Requête de recherche
            
        Returns:
            str: Clé de cache
        """
        # Créer un dict ordonné des paramètres importants
        key_parts = {
            "user_id": query.user_id,
            "query": query.query.lower(),
            "search_type": query.search_type.value,
            "limit": query.limit,
            "offset": query.offset,
            "lexical_weight": query.lexical_weight,
            "semantic_weight": query.semantic_weight,
            "use_reranking": query.use_reranking
        }
        
        # Ajouter les filtres s'ils existent
        if query.date_from:
            key_parts["date_from"] = query.date_from.isoformat()
        if query.date_to:
            key_parts["date_to"] = query.date_to.isoformat()
        if query.amount_min is not None:
            key_parts["amount_min"] = query.amount_min
        if query.amount_max is not None:
            key_parts["amount_max"] = query.amount_max
        if query.categories:
            key_parts["categories"] = sorted(query.categories)
        if query.account_ids:
            key_parts["account_ids"] = sorted(query.account_ids)
        if query.transaction_types:
            key_parts["transaction_types"] = sorted(query.transaction_types)
        
        # Générer un hash MD5
        key_str = json.dumps(key_parts, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[SearchResponse]:
        """
        Récupère une entrée du cache.
        
        Args:
            key: Clé de cache
            
        Returns:
            SearchResponse ou None si non trouvé ou expiré
        """
        async with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Vérifier l'expiration
            response, expiry = self.cache[key]
            if time.time() > expiry:
                # Entrée expirée
                del self.cache[key]
                self.misses += 1
                return None
            
            # Déplacer en fin (LRU)
            self.cache.move_to_end(key)
            self.hits += 1
            
            # Mettre à jour les stats utilisateur
            user_id = response.results[0].user_id if response.results else None
            if user_id:
                self._update_user_stats(user_id, "hits")
            
            return response
    
    async def set(self, key: str, response: SearchResponse, ttl: Optional[int] = None):
        """
        Ajoute une entrée au cache.
        
        Args:
            key: Clé de cache
            response: Réponse à cacher
            ttl: Durée de vie en secondes (optionnel)
        """
        async with self._lock:
            # Calculer l'expiration
            ttl = ttl or self.default_ttl
            expiry = time.time() + ttl
            
            # Ajouter au cache
            self.cache[key] = (response, expiry)
            self.cache.move_to_end(key)
            
            # Limiter la taille
            if len(self.cache) > self.max_size:
                # Supprimer le plus ancien
                self.cache.popitem(last=False)
            
            # Mettre à jour les stats utilisateur
            user_id = response.results[0].user_id if response.results else None
            if user_id:
                self._update_user_stats(user_id, "sets")
    
    async def clear_user_cache(self, user_id: int) -> int:
        """
        Vide le cache pour un utilisateur spécifique.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            int: Nombre d'entrées supprimées
        """
        async with self._lock:
            # Trouver toutes les clés de l'utilisateur
            keys_to_remove = []
            for key, (response, _) in self.cache.items():
                if response.results and response.results[0].user_id == user_id:
                    keys_to_remove.append(key)
            
            # Supprimer les entrées
            for key in keys_to_remove:
                del self.cache[key]
            
            # Réinitialiser les stats utilisateur
            if user_id in self.user_stats:
                del self.user_stats[user_id]
            
            return len(keys_to_remove)
    
    async def clear_expired(self) -> int:
        """
        Supprime toutes les entrées expirées.
        
        Returns:
            int: Nombre d'entrées supprimées
        """
        async with self._lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key, (_, expiry) in self.cache.items():
                if current_time > expiry:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
            
            return len(keys_to_remove)
    
    def get_hit_rate(self) -> float:
        """
        Calcule le taux de hit du cache.
        
        Returns:
            float: Taux de hit (0-1)
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache.
        
        Returns:
            Dict: Statistiques détaillées
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "user_count": len(self.user_stats)
        }
    
    def get_user_stats(self, user_id: int) -> Dict[str, int]:
        """
        Retourne les statistiques pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Statistiques utilisateur
        """
        return self.user_stats.get(user_id, {
            "hits": 0,
            "sets": 0
        })
    
    def _update_user_stats(self, user_id: int, stat: str):
        """Met à jour les statistiques utilisateur."""
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {
                "hits": 0,
                "sets": 0
            }
        
        if stat in self.user_stats[user_id]:
            self.user_stats[user_id][stat] += 1
    
    async def cleanup_task(self):
        """
        Tâche de nettoyage périodique du cache.
        À lancer en arrière-plan.
        """
        while True:
            try:
                # Attendre 5 minutes
                await asyncio.sleep(300)
                
                # Nettoyer les entrées expirées
                removed = await self.clear_expired()
                if removed > 0:
                    logger.info(f"Cache cleanup: {removed} entrées expirées supprimées")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur dans la tâche de nettoyage du cache: {e}")