"""
Cache LRU intelligent pour le Search Service.

Implémente un cache haute performance avec TTL, métriques
et éviction LRU pour optimiser les performances des recherches.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, OrderedDict, List
from threading import RLock
from dataclasses import dataclass
from enum import Enum

from ..config.settings import SearchServiceSettings, get_settings


logger = logging.getLogger(__name__)


class CacheEntryStatus(str, Enum):
    """Statuts des entrées de cache."""
    FRESH = "fresh"           # Entrée fraîche
    STALE = "stale"          # Entrée expirée
    WARMING = "warming"       # En cours de réchauffement
    INVALID = "invalid"       # Entrée invalide


@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    expires_at: datetime
    size_bytes: int
    status: CacheEntryStatus = CacheEntryStatus.FRESH
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée est expirée."""
        return datetime.utcnow() > self.expires_at
    
    def is_fresh(self) -> bool:
        """Vérifie si l'entrée est fraîche."""
        return not self.is_expired() and self.status == CacheEntryStatus.FRESH
    
    def touch(self) -> None:
        """Met à jour l'heure du dernier accès et le compteur."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def get_age_seconds(self) -> float:
        """Retourne l'âge de l'entrée en secondes."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def get_time_to_expiry_seconds(self) -> float:
        """Retourne le temps avant expiration en secondes."""
        return max(0, (self.expires_at - datetime.utcnow()).total_seconds())


class SearchCache:
    """
    Cache LRU intelligent pour les résultats de recherche.
    
    Fonctionnalités:
    - Cache LRU avec TTL configurable
    - Métriques détaillées de performance
    - Éviction intelligente basée sur la fréquence d'accès
    - Thread-safe pour usage concurrent
    - Nettoyage automatique des entrées expirées
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 300,
        cleanup_interval_seconds: int = 60,
        settings: Optional[SearchServiceSettings] = None
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.settings = settings or get_settings()
        
        # Stockage des entrées avec ordre LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()  # Thread-safe
        
        # Métriques
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.total_size_bytes = 0
        self.max_size_bytes = 100 * 1024 * 1024  # 100MB par défaut
        
        # Tâche de nettoyage automatique
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        logger.info(f"Cache initialisé: max_size={max_size}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de cache
            
        Returns:
            Optional[Any]: Valeur ou None si pas trouvée/expirée
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self.misses += 1
                logger.debug(f"Cache miss: {key}")
                return None
            
            # Vérification expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.misses += 1
                self.expirations += 1
                logger.debug(f"Cache expired: {key}")
                return None
            
            # Mise à jour accès et déplacement en fin (LRU)
            entry.touch()
            self._cache.move_to_end(key)
            
            self.hits += 1
            logger.debug(f"Cache hit: {key} (accès #{entry.access_count})")
            
            return entry.value
    
    def put(self, key: str, value: Any) -> None:
        """
        Met une valeur en cache.
        
        Args:
            key: Clé de cache
            value: Valeur à mettre en cache
        """
        with self._lock:
            # Calcul de la taille approximative
            size_bytes = self._estimate_size(value)
            
            # Vérification de la limite de taille
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Valeur trop grande pour le cache: {size_bytes} bytes")
                return
            
            # Suppression de l'ancienne entrée si elle existe
            if key in self._cache:
                self._remove_entry(key)
            
            # Éviction si nécessaire
            self._evict_if_needed(size_bytes)
            
            # Création de la nouvelle entrée
            now = datetime.utcnow()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                expires_at=now + timedelta(seconds=self.ttl_seconds),
                size_bytes=size_bytes
            )
            
            # Ajout au cache
            self._cache[key] = entry
            self.total_size_bytes += size_bytes
            
            logger.debug(f"Cache put: {key} ({size_bytes} bytes, TTL={self.ttl_seconds}s)")
    
    def delete(self, key: str) -> bool:
        """
        Supprime une entrée du cache.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            bool: True si supprimée, False si pas trouvée
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                logger.debug(f"Cache delete: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Vide complètement le cache."""
        with self._lock:
            self._cache.clear()
            self.total_size_bytes = 0
            logger.info("Cache vidé complètement")
    
    def exists(self, key: str) -> bool:
        """
        Vérifie si une clé existe et est valide.
        
        Args:
            key: Clé à vérifier
            
        Returns:
            bool: True si existe et valide
        """
        with self._lock:
            entry = self._cache.get(key)
            return entry is not None and entry.is_fresh()
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations d'une entrée.
        
        Args:
            key: Clé de l'entrée
            
        Returns:
            Optional[Dict]: Informations de l'entrée ou None
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            return {
                "key": entry.key,
                "created_at": entry.created_at.isoformat(),
                "last_accessed": entry.last_accessed.isoformat(),
                "access_count": entry.access_count,
                "expires_at": entry.expires_at.isoformat(),
                "size_bytes": entry.size_bytes,
                "status": entry.status.value,
                "age_seconds": entry.get_age_seconds(),
                "time_to_expiry_seconds": entry.get_time_to_expiry_seconds(),
                "is_expired": entry.is_expired()
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques détaillées du cache.
        
        Returns:
            Dict: Métriques du cache
        """
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            # Calcul des statistiques sur les entrées
            if self._cache:
                ages = [entry.get_age_seconds() for entry in self._cache.values()]
                access_counts = [entry.access_count for entry in self._cache.values()]
                sizes = [entry.size_bytes for entry in self._cache.values()]
                
                avg_age = sum(ages) / len(ages)
                avg_access_count = sum(access_counts) / len(access_counts)
                avg_size = sum(sizes) / len(sizes)
            else:
                avg_age = avg_access_count = avg_size = 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_size_bytes": self.total_size_bytes,
                "max_size_bytes": self.max_size_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "expirations": self.expirations,
                "ttl_seconds": self.ttl_seconds,
                "avg_entry_age_seconds": avg_age,
                "avg_access_count": avg_access_count,
                "avg_entry_size_bytes": avg_size,
                "cleanup_interval_seconds": self.cleanup_interval_seconds
            }
    
    def get_top_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les entrées les plus utilisées.
        
        Args:
            limit: Nombre max d'entrées à retourner
            
        Returns:
            List[Dict]: Liste des top entrées
        """
        with self._lock:
            # Tri par nombre d'accès décroissant
            sorted_entries = sorted(
                self._cache.values(),
                key=lambda e: e.access_count,
                reverse=True
            )
            
            return [
                {
                    "key": entry.key,
                    "access_count": entry.access_count,
                    "age_seconds": entry.get_age_seconds(),
                    "size_bytes": entry.size_bytes,
                    "status": entry.status.value
                }
                for entry in sorted_entries[:limit]
            ]
    
    def cleanup_expired(self) -> int:
        """
        Nettoie les entrées expirées.
        
        Returns:
            int: Nombre d'entrées supprimées
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
                self.expirations += 1
            
            if expired_keys:
                logger.debug(f"Nettoyage: {len(expired_keys)} entrées expirées supprimées")
            
            return len(expired_keys)
    
    def start_auto_cleanup(self) -> None:
        """Démarre le nettoyage automatique."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._auto_cleanup_loop())
            logger.info(f"Nettoyage automatique démarré (intervalle: {self.cleanup_interval_seconds}s)")
    
    async def stop_auto_cleanup(self) -> None:
        """Arrête le nettoyage automatique."""
        self._shutdown = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Nettoyage automatique arrêté")
    
    async def _auto_cleanup_loop(self) -> None:
        """Boucle de nettoyage automatique."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                expired_count = self.cleanup_expired()
                
                # Log seulement si on a supprimé des entrées
                if expired_count > 0:
                    logger.info(f"🧹 Nettoyage automatique: {expired_count} entrées expirées supprimées")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Erreur nettoyage automatique: {str(e)}")
    
    def _remove_entry(self, key: str) -> None:
        """
        Supprime une entrée du cache (méthode interne).
        
        Args:
            key: Clé à supprimer
        """
        entry = self._cache.pop(key, None)
        if entry:
            self.total_size_bytes -= entry.size_bytes
    
    def _evict_if_needed(self, new_entry_size: int) -> None:
        """
        Évince des entrées si nécessaire pour faire de la place.
        
        Args:
            new_entry_size: Taille de la nouvelle entrée
        """
        # Éviction par nombre d'entrées
        while len(self._cache) >= self.max_size:
            self._evict_lru_entry()
        
        # Éviction par taille mémoire
        while (self.total_size_bytes + new_entry_size) > self.max_size_bytes:
            if not self._evict_lru_entry():
                break  # Plus d'entrées à évincer
    
    def _evict_lru_entry(self) -> bool:
        """
        Évince l'entrée la moins récemment utilisée.
        
        Returns:
            bool: True si une entrée a été évincée
        """
        if not self._cache:
            return False
        
        # La première entrée est la moins récemment utilisée (LRU)
        lru_key = next(iter(self._cache))
        self._remove_entry(lru_key)
        self.evictions += 1
        
        logger.debug(f"Éviction LRU: {lru_key}")
        return True
    
    def _estimate_size(self, value: Any) -> int:
        """
        Estime la taille en bytes d'une valeur.
        
        Args:
            value: Valeur à mesurer
            
        Returns:
            int: Taille estimée en bytes
        """
        try:
            import sys
            
            # Estimation approximative
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return sys.getsizeof(value)
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in value.items()
                )
            else:
                # Fallback avec sys.getsizeof
                return sys.getsizeof(value)
                
        except Exception:
            # Si estimation impossible, utiliser une taille par défaut
            return 1024  # 1KB par défaut
    
    def reset_metrics(self) -> None:
        """Remet à zéro les métriques."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.expirations = 0
            logger.info("📊 Métriques du cache réinitialisées")
    
    def warmup(self, key_value_pairs: List[tuple]) -> None:
        """
        Préchauffe le cache avec des valeurs prédéfinies.
        
        Args:
            key_value_pairs: Liste de tuples (clé, valeur)
        """
        logger.info(f"🔥 Préchauffage du cache avec {len(key_value_pairs)} entrées...")
        
        for key, value in key_value_pairs:
            self.put(key, value)
        
        logger.info("✅ Préchauffage terminé")
    
    def __len__(self) -> int:
        """Retourne le nombre d'entrées dans le cache."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Vérifie si une clé existe dans le cache."""
        return self.exists(key)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear()


# === HELPER FUNCTIONS ===

def create_search_cache(settings: Optional[SearchServiceSettings] = None) -> SearchCache:
    """
    Factory pour créer un cache de recherche configuré.
    
    Args:
        settings: Configuration (optionnel)
        
    Returns:
        SearchCache: Cache configuré
    """
    settings = settings or get_settings()
    
    return SearchCache(
        max_size=settings.CACHE_MAX_SIZE,
        ttl_seconds=settings.CACHE_TTL_SECONDS,
        cleanup_interval_seconds=60,
        settings=settings
    )


async def benchmark_cache(cache: SearchCache, num_operations: int = 1000) -> Dict[str, float]:
    """
    Benchmark les performances du cache.
    
    Args:
        cache: Cache à tester
        num_operations: Nombre d'opérations à effectuer
        
    Returns:
        Dict: Résultats du benchmark
    """
    import time
    import random
    import string
    
    def random_string(length: int = 10) -> str:
        return ''.join(random.choices(string.ascii_letters, k=length))
    
    # Préparation des données de test
    test_data = [
        (f"key_{i}", {"data": random_string(100), "number": i})
        for i in range(num_operations)
    ]
    
    # Test d'écriture
    start_time = time.time()
    for key, value in test_data:
        cache.put(key, value)
    write_time = time.time() - start_time
    
    # Test de lecture
    start_time = time.time()
    for key, _ in test_data:
        cache.get(key)
    read_time = time.time() - start_time
    
    # Test de lecture avec misses
    start_time = time.time()
    for i in range(num_operations // 10):
        cache.get(f"missing_key_{i}")
    miss_time = time.time() - start_time
    
    # Calcul des métriques
    metrics = cache.get_metrics()
    
    return {
        "write_ops_per_second": num_operations / write_time,
        "read_ops_per_second": num_operations / read_time,
        "miss_ops_per_second": (num_operations // 10) / miss_time,
        "hit_rate": metrics["hit_rate"],
        "total_entries": metrics["size"],
        "total_size_mb": metrics["total_size_bytes"] / (1024 * 1024),
        "avg_entry_size_bytes": metrics["avg_entry_size_bytes"]
    }


class CacheWarmer:
    """
    Utilitaire pour préchauffer le cache avec des données pertinentes.
    """
    
    def __init__(self, cache: SearchCache):
        self.cache = cache
    
    async def warmup_common_queries(self, user_ids: List[int]) -> None:
        """
        Préchauffe le cache avec des requêtes communes.
        
        Args:
            user_ids: Liste des user_ids à préchauffer
        """
        logger.info(f"🔥 Préchauffage pour {len(user_ids)} utilisateurs...")
        
        # Requêtes communes à préchauffer
        common_patterns = [
            {"category": "restaurant", "period": "month"},
            {"category": "transport", "period": "week"},
            {"merchant": "AMAZON", "period": "month"},
            {"amount_range": "50-200", "period": "month"}
        ]
        
        for user_id in user_ids:
            for pattern in common_patterns:
                # Génération d'une clé de cache simulée
                cache_key = self._generate_warmup_key(user_id, pattern)
                
                # Données simulées pour le préchauffage
                mock_data = self._generate_mock_search_result(user_id, pattern)
                
                # Mise en cache
                self.cache.put(cache_key, mock_data)
        
        logger.info("✅ Préchauffage terminé")
    
    def _generate_warmup_key(self, user_id: int, pattern: Dict[str, str]) -> str:
        """Génère une clé de cache pour le préchauffage."""
        import hashlib
        import json
        
        key_data = {"user_id": user_id, **pattern}
        key_string = json.dumps(key_data, sort_keys=True)
        return f"warmup_{hashlib.md5(key_string.encode()).hexdigest()[:8]}"
    
    def _generate_mock_search_result(self, user_id: int, pattern: Dict[str, str]) -> Dict[str, Any]:
        """Génère un résultat de recherche simulé."""
        return {
            "results": [
                {
                    "transaction_id": f"tx_{user_id}_{i}",
                    "user_id": user_id,
                    "amount": -50.0 * (i + 1),
                    "category": pattern.get("category", "general"),
                    "merchant": pattern.get("merchant", "MERCHANT"),
                    "date": "2024-01-15"
                }
                for i in range(5)
            ],
            "total_hits": 5,
            "execution_time_ms": 45
        }


class CacheAnalyzer:
    """
    Analyseur de performance et d'efficacité du cache.
    """
    
    def __init__(self, cache: SearchCache):
        self.cache = cache
    
    def analyze_efficiency(self) -> Dict[str, Any]:
        """
        Analyse l'efficacité du cache.
        
        Returns:
            Dict: Analyse détaillée
        """
        metrics = self.cache.get_metrics()
        top_entries = self.cache.get_top_entries(20)
        
        # Calcul de l'efficacité
        hit_rate = metrics["hit_rate"]
        memory_efficiency = metrics["total_size_bytes"] / metrics["max_size_bytes"]
        
        # Classification de l'efficacité
        if hit_rate > 0.8:
            efficiency_level = "excellent"
        elif hit_rate > 0.6:
            efficiency_level = "good"
        elif hit_rate > 0.4:
            efficiency_level = "fair"
        else:
            efficiency_level = "poor"
        
        # Analyse des patterns d'accès
        access_distribution = self._analyze_access_patterns(top_entries)
        
        return {
            "efficiency_level": efficiency_level,
            "hit_rate": hit_rate,
            "memory_efficiency": memory_efficiency,
            "total_requests": metrics["hits"] + metrics["misses"],
            "eviction_rate": metrics["evictions"] / max(metrics["hits"] + metrics["misses"], 1),
            "expiration_rate": metrics["expirations"] / max(metrics["hits"] + metrics["misses"], 1),
            "access_distribution": access_distribution,
            "recommendations": self._generate_recommendations(metrics, access_distribution)
        }
    
    def _analyze_access_patterns(self, top_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse les patterns d'accès."""
        if not top_entries:
            return {"pattern": "no_data"}
        
        access_counts = [entry["access_count"] for entry in top_entries]
        
        # Calcul de la concentration (coefficient de Gini simplifié)
        sorted_counts = sorted(access_counts)
        n = len(sorted_counts)
        cumsum = sum((i + 1) * count for i, count in enumerate(sorted_counts))
        gini = (2 * cumsum) / (n * sum(sorted_counts)) - (n + 1) / n
        
        # Classification du pattern
        if gini > 0.7:
            pattern = "highly_concentrated"  # Quelques clés très utilisées
        elif gini > 0.4:
            pattern = "moderately_concentrated"
        else:
            pattern = "evenly_distributed"
        
        return {
            "pattern": pattern,
            "gini_coefficient": gini,
            "top_access_count": max(access_counts),
            "avg_access_count": sum(access_counts) / len(access_counts),
            "access_range": max(access_counts) - min(access_counts)
        }
    
    def _generate_recommendations(
        self, 
        metrics: Dict[str, Any], 
        access_distribution: Dict[str, Any]
    ) -> List[str]:
        """Génère des recommandations d'optimisation."""
        recommendations = []
        
        hit_rate = metrics["hit_rate"]
        memory_efficiency = metrics["total_size_bytes"] / metrics["max_size_bytes"]
        eviction_rate = metrics["evictions"] / max(metrics["hits"] + metrics["misses"], 1)
        
        # Recommandations basées sur le hit rate
        if hit_rate < 0.5:
            recommendations.append("Augmenter la taille du cache pour améliorer le hit rate")
            recommendations.append("Réviser la stratégie de clés de cache")
        
        # Recommandations basées sur la mémoire
        if memory_efficiency > 0.9:
            recommendations.append("Envisager d'augmenter la taille max du cache")
        elif memory_efficiency < 0.3:
            recommendations.append("Diminuer la taille max ou augmenter le TTL")
        
        # Recommandations basées sur les évictions
        if eviction_rate > 0.1:
            recommendations.append("Trop d'évictions - augmenter la taille du cache")
        
        # Recommandations basées sur les patterns d'accès
        pattern = access_distribution["pattern"]
        if pattern == "highly_concentrated":
            recommendations.append("Optimiser le cache pour les clés les plus utilisées")
        elif pattern == "evenly_distributed":
            recommendations.append("Cache bien équilibré - maintenir la stratégie actuelle")
        
        # TTL recommendations
        avg_age = metrics["avg_entry_age_seconds"]
        ttl = metrics["ttl_seconds"]
        if avg_age < ttl * 0.3:
            recommendations.append("Réduire le TTL pour libérer la mémoire plus rapidement")
        elif avg_age > ttl * 0.8:
            recommendations.append("Augmenter le TTL pour réduire les expirations")
        
        return recommendations[:5]  # Limiter à 5 recommandations
    
    def generate_report(self) -> str:
        """
        Génère un rapport détaillé du cache.
        
        Returns:
            str: Rapport formaté
        """
        metrics = self.cache.get_metrics()
        analysis = self.analyze_efficiency()
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          RAPPORT DE CACHE - SEARCH SERVICE                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 📊 MÉTRIQUES GÉNÉRALES                                                      ║
║   • Taille actuelle      : {metrics['size']:,} / {metrics['max_size']:,} entrées        ║
║   • Utilisation mémoire  : {metrics['total_size_bytes'] / (1024*1024):.1f} / {metrics['max_size_bytes'] / (1024*1024):.1f} MB  ║
║   • Hit Rate            : {metrics['hit_rate']:.1%}                                  ║
║   • Requêtes totales    : {metrics['hits'] + metrics['misses']:,}                    ║
║                                                                              ║
║ ⚡ PERFORMANCE                                                               ║
║   • Cache Hits          : {metrics['hits']:,}                                        ║
║   • Cache Misses        : {metrics['misses']:,}                                      ║
║   • Évictions           : {metrics['evictions']:,}                                   ║
║   • Expirations         : {metrics['expirations']:,}                                 ║
║                                                                              ║
║ 🔍 ANALYSE D'EFFICACITÉ                                                     ║
║   • Niveau              : {analysis['efficiency_level'].upper()}                     ║
║   • Pattern d'accès     : {analysis['access_distribution']['pattern']}               ║
║   • Coefficient Gini    : {analysis['access_distribution']['gini_coefficient']:.3f}  ║
║                                                                              ║
║ 💡 RECOMMANDATIONS                                                          ║
"""
        
        for i, rec in enumerate(analysis['recommendations'], 1):
            report += f"║   {i}. {rec:<67} ║\n"
        
        report += "╚══════════════════════════════════════════════════════════════════════════════╝"
        
        return report


# === CACHE STRATEGIES ===

class CacheKeyGenerator:
    """
    Générateur de clés de cache optimisées pour les recherches financières.
    """
    
    @staticmethod
    def generate_search_key(
        user_id: int,
        query_type: str,
        filters: Dict[str, Any],
        limit: int,
        offset: int
    ) -> str:
        """
        Génère une clé de cache pour une recherche.
        
        Args:
            user_id: ID utilisateur
            query_type: Type de requête
            filters: Filtres appliqués
            limit: Limite de résultats
            offset: Offset de pagination
            
        Returns:
            str: Clé de cache optimisée
        """
        import hashlib
        import json
        
        # Normalisation des filtres pour consistency
        normalized_filters = CacheKeyGenerator._normalize_filters(filters)
        
        key_components = {
            "user_id": user_id,
            "query_type": query_type,
            "filters": normalized_filters,
            "limit": limit,
            "offset": offset
        }
        
        # Génération de la clé
        key_string = json.dumps(key_components, sort_keys=True, separators=(',', ':'))
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        # Préfixe pour identification
        return f"search_{user_id}_{key_hash[:12]}"
    
    @staticmethod
    def generate_aggregation_key(
        user_id: int,
        aggregation_type: str,
        group_by: List[str],
        filters: Dict[str, Any] = None
    ) -> str:
        """
        Génère une clé de cache pour une agrégation.
        
        Args:
            user_id: ID utilisateur
            aggregation_type: Type d'agrégation
            group_by: Champs de groupement
            filters: Filtres appliqués
            
        Returns:
            str: Clé de cache pour agrégation
        """
        import hashlib
        import json
        
        key_components = {
            "user_id": user_id,
            "agg_type": aggregation_type,
            "group_by": sorted(group_by),
            "filters": CacheKeyGenerator._normalize_filters(filters or {})
        }
        
        key_string = json.dumps(key_components, sort_keys=True, separators=(',', ':'))
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"agg_{user_id}_{key_hash[:12]}"
    
    @staticmethod
    def _normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise les filtres pour une clé consistante."""
        normalized = {}
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Tri des listes pour consistency
                normalized[key] = sorted(value) if all(isinstance(x, (str, int, float)) for x in value) else value
            elif isinstance(value, dict):
                # Récursion pour les dictionnaires
                normalized[key] = CacheKeyGenerator._normalize_filters(value)
            else:
                normalized[key] = value
        
        return normalized