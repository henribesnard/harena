"""
Classe de base pour tous les clients de services externes du Search Service
Fournit les patterns communs pour la gestion des connexions, circuit breaker et monitoring
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, TypeVar, Generic, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)

# Forward references pour éviter les imports circulaires
if TYPE_CHECKING:
    from .elasticsearch_client import ElasticsearchClient

T = TypeVar('T')


class ClientStatus(str, Enum):
    """Statuts possibles d'un client de service"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(str, Enum):
    """États du circuit breaker"""
    CLOSED = "closed"        # Fonctionnement normal
    OPEN = "open"            # Circuit ouvert (trop d'erreurs)
    HALF_OPEN = "half_open"  # Test de récupération


@dataclass
class RetryConfig:
    """Configuration pour la logique de retry avec backoff exponentiel"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Configuration pour le circuit breaker de protection"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    enabled: bool = True


@dataclass
class HealthCheckConfig:
    """Configuration pour les vérifications de santé automatiques"""
    enabled: bool = True
    interval_seconds: float = 30.0
    timeout_seconds: float = 5.0
    endpoint: Optional[str] = None


class CircuitBreaker:
    """
    Circuit breaker pour protection contre les cascades d'erreurs
    Implémente le pattern Circuit Breaker avec 3 états
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.enabled = config.enabled
    
    def can_execute(self) -> bool:
        """Détermine si une requête peut être exécutée selon l'état du circuit"""
        if not self.enabled:
            return True
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Vérifier si le timeout de récupération est expiré
            if time.time() - self.last_failure_time > self.config.timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def on_success(self):
        """Appelé en cas de succès d'une requête"""
        if not self.enabled:
            return
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def on_failure(self):
        """Appelé en cas d'échec d'une requête"""
        if not self.enabled:
            return
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN


class BaseClient(ABC, Generic[T]):
    """
    Classe de base pour tous les clients de services externes du Search Service
    
    Responsabilités principales:
    - Gestion des connexions HTTP asynchrones avec aiohttp
    - Retry logic avec backoff exponentiel et jitter
    - Circuit breaker pour protection contre les erreurs en cascade
    - Health checks automatiques et monitoring
    - Métriques standardisées (latence, taux d'erreur, etc.)
    - Gestion SSL pour Bonsai Elasticsearch
    
    Cette classe est utilisée comme base pour:
    - ElasticsearchClient (Bonsai) : recherches lexicales
    - CacheClient (Redis) : mise en cache des résultats
    - Futurs clients externes
    """
    
    def __init__(
        self,
        base_url: str,
        service_name: str,
        timeout: float = 10.0,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        health_check_config: Optional[HealthCheckConfig] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.service_name = service_name
        self.timeout = timeout
        self.headers = headers or {}
        
        # Configuration avec valeurs par défaut optimisées pour Search Service
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3,
            base_delay=0.5,  # Démarrage rapide pour recherche
            max_delay=10.0,  # Limite adaptée aux requêtes de recherche
            exponential_base=2.0,
            jitter=True
        )
        
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig(
            failure_threshold=5,      # Tolérance pour services de recherche
            success_threshold=3,      # Récupération rapide
            timeout_seconds=30.0,     # Court pour search service
            enabled=True
        )
        
        self.health_check_config = health_check_config or HealthCheckConfig(
            enabled=True,
            interval_seconds=30.0,
            timeout_seconds=5.0
        )
        
        # État interne du client
        self.circuit_breaker = CircuitBreaker(self.circuit_breaker_config)
        self.status = ClientStatus.UNKNOWN
        self.last_health_check = 0
        self.last_error: Optional[str] = None
        
        # Métriques de performance
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.slow_request_count = 0  # > 1s pour recherche
        self.cache_hit_count = 0     # Pour clients avec cache
        
        # Session HTTP réutilisable (important pour performance)
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Initializing {service_name} client: {base_url}")
    
    async def __aenter__(self):
        """Gestionnaire de contexte asynchrone - entrée"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Gestionnaire de contexte asynchrone - sortie"""
        await self.close()
    
    async def start(self):
        """
        Démarre le client et initialise la session HTTP
        À surcharger par les clients spécialisés (ex: SSL pour Bonsai)
        """
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.headers
            )
            logger.info(f"{self.service_name} client started")
    
    async def close(self):
        """Ferme le client et nettoie les ressources"""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info(f"{self.service_name} client closed")
    
    @property
    def session(self) -> aiohttp.ClientSession:
        """Accès sécurisé à la session HTTP"""
        if self._session is None:
            raise RuntimeError(f"{self.service_name} client not started. Call start() first.")
        return self._session
    
    async def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str = "request"
    ) -> T:
        """
        Exécute une opération avec retry logic, circuit breaker et métriques
        
        Cette méthode est le cœur de la robustesse du Search Service:
        - Vérifie l'état du circuit breaker avant exécution
        - Applique la logique de retry avec backoff exponentiel
        - Enregistre les métriques de performance
        - Gère les transitions d'état du circuit breaker
        
        Args:
            operation: Fonction asynchrone à exécuter
            operation_name: Nom pour les logs et métriques
            
        Returns:
            Résultat de l'opération
            
        Raises:
            Exception: Si toutes les tentatives échouent ou circuit ouvert
        """
        # Vérifier le circuit breaker avant d'essayer
        if not self.circuit_breaker.can_execute():
            error_msg = f"{self.service_name} circuit breaker is OPEN"
            logger.warning(error_msg)
            raise Exception(error_msg)
        
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                start_time = time.time()
                
                # Exécuter l'opération
                result = await operation()
                
                # Enregistrer les métriques de succès
                response_time = time.time() - start_time
                self._record_success(response_time, operation_name)
                self.circuit_breaker.on_success()
                
                logger.debug(
                    f"{self.service_name} {operation_name} succeeded "
                    f"(attempt {attempt + 1}, {response_time:.3f}s)"
                )
                return result
                
            except Exception as e:
                last_exception = e
                response_time = time.time() - start_time
                self._record_error(str(e), operation_name, response_time)
                self.circuit_breaker.on_failure()
                
                logger.warning(
                    f"{self.service_name} {operation_name} failed "
                    f"(attempt {attempt + 1}/{self.retry_config.max_attempts}): {e}"
                )
                
                # Ne pas retry sur la dernière tentative
                if attempt == self.retry_config.max_attempts - 1:
                    break
                
                # Calculer le délai de retry avec backoff exponentiel
                delay = self._calculate_retry_delay(attempt)
                logger.debug(f"Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
        
        # Toutes les tentatives ont échoué
        logger.error(
            f"{self.service_name} {operation_name} failed after "
            f"{self.retry_config.max_attempts} attempts"
        )
        raise last_exception
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calcule le délai de retry avec backoff exponentiel et jitter
        
        Le jitter évite le thundering herd problem quand plusieurs
        clients retry en même temps
        """
        delay = self.retry_config.base_delay * (
            self.retry_config.exponential_base ** attempt
        )
        delay = min(delay, self.retry_config.max_delay)
        
        # Ajouter du jitter pour éviter la synchronisation
        if self.retry_config.jitter:
            import random
            jitter_factor = 0.5 + random.random() * 0.5  # Entre 0.5 et 1.0
            delay *= jitter_factor
        
        return delay
    
    def _record_success(self, response_time: float, operation_name: str):
        """Enregistre les métriques d'une requête réussie"""
        self.request_count += 1
        self.total_response_time += response_time
        
        # Compter les requêtes lentes (importantes pour search service)
        if response_time > 1.0:  # > 1s considéré comme lent pour recherche
            self.slow_request_count += 1
            logger.warning(
                f"Slow {self.service_name} {operation_name}: {response_time:.3f}s"
            )
        
        # Mettre à jour le statut global
        self.status = ClientStatus.HEALTHY
        self.last_error = None
    
    def _record_error(self, error_message: str, operation_name: str, response_time: float):
        """Enregistre les métriques d'une requête échouée"""
        self.request_count += 1
        self.error_count += 1
        self.last_error = error_message
        
        # Déterminer le statut basé sur le taux d'erreur
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        
        if error_rate > 0.5:  # Plus de 50% d'erreurs
            self.status = ClientStatus.UNHEALTHY
        elif error_rate > 0.1:  # Plus de 10% d'erreurs
            self.status = ClientStatus.DEGRADED
        else:
            self.status = ClientStatus.HEALTHY  # Erreur ponctuelle
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Effectue une vérification de santé complète du service
        
        Returns:
            Dictionnaire avec toutes les informations de santé
        """
        health_info = {
            "service_name": self.service_name,
            "status": self.status.value,
            "base_url": self.base_url,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "last_error": self.last_error,
            "metrics": self.get_metrics(),
            "timestamp": time.time()
        }
        
        # Vérification spécifique au service si configurée
        if self.health_check_config.enabled:
            try:
                service_health = await self._perform_health_check()
                health_info.update(service_health)
                self.last_health_check = time.time()
                
                # Mettre à jour le statut basé sur le health check
                if service_health.get("status") == "healthy":
                    self.status = ClientStatus.HEALTHY
                elif service_health.get("status") == "degraded":
                    self.status = ClientStatus.DEGRADED
                else:
                    self.status = ClientStatus.UNHEALTHY
                    
            except Exception as e:
                health_info["health_check_error"] = str(e)
                self.status = ClientStatus.UNHEALTHY
                logger.warning(f"{self.service_name} health check failed: {e}")
        
        return health_info
    
    @abstractmethod
    async def _perform_health_check(self) -> Dict[str, Any]:
        """
        Effectue une vérification de santé spécifique au service
        
        À implémenter par les classes dérivées (ElasticsearchClient, etc.)
        
        Returns:
            Dictionnaire avec les informations de santé spécifiques
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques détaillées du client
        
        Métriques importantes pour le monitoring du Search Service
        """
        # Calculs des métriques
        avg_response_time = 0.0
        successful_requests = self.request_count - self.error_count
        
        if successful_requests > 0:
            avg_response_time = self.total_response_time / successful_requests
        
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        slow_request_rate = self.slow_request_count / self.request_count if self.request_count > 0 else 0
        
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "slow_request_count": self.slow_request_count,
            "error_rate": round(error_rate, 3),
            "slow_request_rate": round(slow_request_rate, 3),
            "average_response_time_ms": round(avg_response_time * 1000, 2),
            "total_response_time_seconds": round(self.total_response_time, 2),
            "status": self.status.value,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_failure_count": self.circuit_breaker.failure_count,
            "last_health_check": self.last_health_check
        }
    
    def reset_metrics(self):
        """Remet à zéro toutes les métriques (utile pour monitoring)"""
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.slow_request_count = 0
        self.cache_hit_count = 0
        logger.info(f"{self.service_name} metrics reset")
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Teste la connectivité de base au service
        
        À implémenter par les classes dérivées pour tester:
        - Bonsai Elasticsearch: ping cluster
        - Redis Cache: ping server
        - Autres services: endpoint de health
        
        Returns:
            True si la connexion fonctionne
        """
        pass
    
    def get_circuit_breaker_info(self) -> Dict[str, Any]:
        """Retourne les informations détaillées du circuit breaker"""
        return {
            "state": self.circuit_breaker.state.value,
            "failure_count": self.circuit_breaker.failure_count,
            "success_count": self.circuit_breaker.success_count,
            "last_failure_time": self.circuit_breaker.last_failure_time,
            "enabled": self.circuit_breaker.enabled,
            "config": {
                "failure_threshold": self.circuit_breaker.config.failure_threshold,
                "success_threshold": self.circuit_breaker.config.success_threshold,
                "timeout_seconds": self.circuit_breaker.config.timeout_seconds
            }
        }
    
    def is_healthy(self) -> bool:
        """Détermine si le client est en état sain"""
        return (
            self.status == ClientStatus.HEALTHY and
            self.circuit_breaker.state == CircuitBreakerState.CLOSED
        )
    
    def is_available(self) -> bool:
        """Détermine si le client peut traiter des requêtes"""
        return (
            self.status in [ClientStatus.HEALTHY, ClientStatus.DEGRADED] and
            self.circuit_breaker.can_execute()
        )


# === HELPERS ET FACTORY ===

class ClientFactory:
    """Factory pour créer des clients avec configurations standardisées"""
    
    @staticmethod
    def create_elasticsearch_client(bonsai_url: str, **kwargs) -> "ElasticsearchClient":
        """Crée un client Elasticsearch optimisé pour Bonsai"""
        # Import local pour éviter la circularité
        from .elasticsearch_client import ElasticsearchClient
        
        # Configuration optimisée pour Bonsai Elasticsearch
        default_config = {
            "timeout": 10.0,  # Timeout généreux pour recherches complexes
            "retry_config": RetryConfig(
                max_attempts=3,
                base_delay=0.5,
                max_delay=5.0,
                exponential_base=1.5  # Moins agressif pour Elasticsearch
            ),
            "circuit_breaker_config": CircuitBreakerConfig(
                failure_threshold=3,   # Plus strict pour search critique
                success_threshold=2,   # Récupération rapide
                timeout_seconds=30.0
            )
        }
        
        # Merger avec les kwargs fournis
        config = {**default_config, **kwargs}
        
        return ElasticsearchClient(bonsai_url, **config)
    
    @staticmethod
    def create_cache_client(redis_url: str, **kwargs):
        """
        Crée un client Redis optimisé pour le cache
        
        Note: CacheClient sera implémenté dans une prochaine étape
        Pour l'instant, cette méthode lève une NotImplementedError
        """
        raise NotImplementedError(
            "CacheClient not yet implemented. "
            "Use create_elasticsearch_client for now."
        )


# === EXPORTS ===

__all__ = [
    # Classes principales
    "BaseClient",
    "CircuitBreaker",
    
    # Enums
    "ClientStatus",
    "CircuitBreakerState",
    
    # Configuration
    "RetryConfig",
    "CircuitBreakerConfig", 
    "HealthCheckConfig",
    
    # Factory
    "ClientFactory"
]