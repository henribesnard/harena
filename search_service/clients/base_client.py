"""
Classe de base pour tous les clients du Search Service.

Fournit une interface commune et des fonctionnalités partagées
pour tous les clients externes (Elasticsearch, Redis, etc.).
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

from ..config.settings import SearchServiceSettings, get_settings


logger = logging.getLogger(__name__)


class BaseClient(ABC):
    """
    Classe de base abstraite pour tous les clients externes.
    
    Fournit:
    - Interface commune pour connexion/déconnexion
    - Gestion des erreurs standardisée
    - Métriques de base
    - Configuration centralisée
    """
    
    def __init__(self, settings: Optional[SearchServiceSettings] = None):
        self.settings = settings or get_settings()
        self.connected = False
        self.connection_time = None
        self.last_error = None
        self.error_count = 0
        
        # Métriques de base
        self.request_count = 0
        self.total_request_time = 0.0
        self.last_request_time = None
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Établit la connexion avec le service externe.
        
        Raises:
            Exception: Si la connexion échoue
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Ferme la connexion avec le service externe."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Vérifie la santé de la connexion.
        
        Returns:
            bool: True si la connexion est saine
        """
        pass
    
    def _record_request(self, execution_time_ms: float) -> None:
        """
        Enregistre les métriques d'une requête.
        
        Args:
            execution_time_ms: Temps d'exécution en millisecondes
        """
        self.request_count += 1
        self.total_request_time += execution_time_ms
        self.last_request_time = datetime.utcnow()
    
    def _record_error(self, error: Exception) -> None:
        """
        Enregistre une erreur.
        
        Args:
            error: Exception survenue
        """
        self.error_count += 1
        self.last_error = {
            "error": str(error),
            "type": type(error).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.error(f"❌ Erreur dans {self.__class__.__name__}: {str(error)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du client.
        
        Returns:
            Dict: Métriques de performance
        """
        avg_request_time = (
            self.total_request_time / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        return {
            "client_name": self.__class__.__name__,
            "connected": self.connected,
            "connection_time": self.connection_time.isoformat() if self.connection_time else None,
            "request_count": self.request_count,
            "total_request_time_ms": self.total_request_time,
            "average_request_time_ms": avg_request_time,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
            "last_error": self.last_error
        }
    
    def is_connected(self) -> bool:
        """
        Vérifie si le client est connecté.
        
        Returns:
            bool: True si connecté
        """
        return self.connected
    
    def get_connection_age(self) -> Optional[timedelta]:
        """
        Récupère l'âge de la connexion.
        
        Returns:
            Optional[timedelta]: Âge de la connexion ou None
        """
        if not self.connection_time:
            return None
        return datetime.utcnow() - self.connection_time
    
    async def ensure_connected(self) -> None:
        """
        S'assure que le client est connecté.
        
        Reconnecte automatiquement si nécessaire.
        """
        if not self.connected:
            await self.connect()
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()


class RetryableClient(BaseClient):
    """
    Client avec capacité de retry automatique.
    
    Étend BaseClient avec des fonctionnalités de retry
    et de gestion des erreurs temporaires.
    """
    
    def __init__(self, settings: Optional[SearchServiceSettings] = None):
        super().__init__(settings)
        self.max_retries = getattr(settings, 'MAX_RETRIES', 3)
        self.retry_delay = getattr(settings, 'RETRY_DELAY', 1.0)
        self.backoff_factor = getattr(settings, 'BACKOFF_FACTOR', 2.0)
    
    async def execute_with_retry(
        self,
        operation,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Exécute une opération avec retry automatique.
        
        Args:
            operation: Fonction à exécuter
            *args: Arguments positionnels
            max_retries: Nombre max de tentatives (optionnel)
            **kwargs: Arguments nommés
            
        Returns:
            Any: Résultat de l'opération
            
        Raises:
            Exception: Si toutes les tentatives échouent
        """
        max_retries = max_retries or self.max_retries
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await operation(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                self._record_error(e)
                
                if attempt < max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    logger.warning(
                        f"⚠️ Tentative {attempt + 1}/{max_retries + 1} échouée pour {operation.__name__}: {str(e)}. "
                        f"Retry dans {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ Toutes les tentatives échouées pour {operation.__name__}")
        
        raise last_exception


class CachedClient(BaseClient):
    """
    Client avec cache intégré.
    
    Étend BaseClient avec des fonctionnalités de cache
    pour améliorer les performances.
    """
    
    def __init__(self, settings: Optional[SearchServiceSettings] = None):
        super().__init__(settings)
        self.cache = {}
        self.cache_ttl = getattr(settings, 'CACHE_TTL_SECONDS', 300)
        self.cache_max_size = getattr(settings, 'CACHE_MAX_SIZE', 1000)
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """
        Génère une clé de cache.
        
        Args:
            *args: Arguments positionnels
            **kwargs: Arguments nommés
            
        Returns:
            str: Clé de cache
        """
        import hashlib
        key_data = f"{args}_{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_item: Dict[str, Any]) -> bool:
        """
        Vérifie si un élément du cache est encore valide.
        
        Args:
            cached_item: Élément du cache
            
        Returns:
            bool: True si valide
        """
        expiry_time = cached_item.get("expiry_time")
        if not expiry_time:
            return False
        
        return datetime.utcnow() < expiry_time
    
    def _clean_cache(self) -> None:
        """Nettoie le cache des éléments expirés."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, item in self.cache.items()
            if item.get("expiry_time") and now >= item["expiry_time"]
        ]
        
        for key in expired_keys:
            del self.cache[key]
    
    def _evict_if_full(self) -> None:
        """Évince des éléments si le cache est plein."""
        if len(self.cache) >= self.cache_max_size:
            # Éviction LRU (Least Recently Used)
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].get("last_accessed", datetime.min)
            )
            del self.cache[oldest_key]
    
    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Récupère un élément du cache.
        
        Args:
            cache_key: Clé de cache
            
        Returns:
            Optional[Any]: Valeur mise en cache ou None
        """
        self._clean_cache()
        
        cached_item = self.cache.get(cache_key)
        if not cached_item:
            self.cache_misses += 1
            return None
        
        if not self._is_cache_valid(cached_item):
            del self.cache[cache_key]
            self.cache_misses += 1
            return None
        
        # Mise à jour du timestamp d'accès
        cached_item["last_accessed"] = datetime.utcnow()
        self.cache_hits += 1
        
        return cached_item["value"]
    
    def put_in_cache(self, cache_key: str, value: Any) -> None:
        """
        Met un élément en cache.
        
        Args:
            cache_key: Clé de cache
            value: Valeur à mettre en cache
        """
        self._evict_if_full()
        
        self.cache[cache_key] = {
            "value": value,
            "cached_at": datetime.utcnow(),
            "expiry_time": datetime.utcnow() + timedelta(seconds=self.cache_ttl),
            "last_accessed": datetime.utcnow()
        }
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du cache.
        
        Returns:
            Dict: Métriques du cache
        """
        hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0.0
        )
        
        return {
            "cache_size": len(self.cache),
            "cache_max_size": self.cache_max_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "cache_ttl_seconds": self.cache_ttl
        }
    
    def clear_cache(self) -> None:
        """Vide le cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("🧹 Cache vidé")


class HealthMonitoredClient(BaseClient):
    """
    Client avec monitoring de santé automatique.
    
    Étend BaseClient avec des vérifications de santé
    périodiques et des alertes.
    """
    
    def __init__(self, settings: Optional[SearchServiceSettings] = None):
        super().__init__(settings)
        self.health_check_interval = getattr(settings, 'HEALTH_CHECK_INTERVAL_SECONDS', 30)
        self.health_check_task = None
        self.health_status = True
        self.health_history = []
        self.max_health_history = 100
    
    async def start_health_monitoring(self) -> None:
        """Démarre le monitoring de santé automatique."""
        if self.health_check_task:
            return
        
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"🏥 Monitoring de santé démarré (intervalle: {self.health_check_interval}s)")
    
    async def stop_health_monitoring(self) -> None:
        """Arrête le monitoring de santé."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
            logger.info("🏥 Monitoring de santé arrêté")
    
    async def _health_check_loop(self) -> None:
        """Boucle de vérification de santé."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                health_result = await self.health_check()
                self._record_health_status(health_result)
                
                if not health_result and self.health_status:
                    logger.warning("⚠️ Dégradation de la santé détectée")
                elif health_result and not self.health_status:
                    logger.info("✅ Santé rétablie")
                
                self.health_status = health_result
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Erreur dans le monitoring de santé: {str(e)}")
                self._record_health_status(False)
    
    def _record_health_status(self, is_healthy: bool) -> None:
        """
        Enregistre le statut de santé.
        
        Args:
            is_healthy: True si sain
        """
        self.health_history.append({
            "timestamp": datetime.utcnow(),
            "healthy": is_healthy
        })
        
        # Limiter la taille de l'historique
        if len(self.health_history) > self.max_health_history:
            self.health_history.pop(0)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de santé.
        
        Returns:
            Dict: Métriques de santé
        """
        recent_checks = self.health_history[-20:] if self.health_history else []
        uptime_percentage = (
            sum(1 for check in recent_checks if check["healthy"]) / len(recent_checks)
            if recent_checks else 0.0
        )
        
        return {
            "current_health_status": self.health_status,
            "health_check_interval": self.health_check_interval,
            "health_checks_count": len(self.health_history),
            "recent_uptime_percentage": uptime_percentage,
            "monitoring_active": self.health_check_task is not None
        }
    
    async def disconnect(self) -> None:
        """Déconnexion avec arrêt du monitoring."""
        await self.stop_health_monitoring()
        await super().disconnect()