"""
Classe de base pour tous les clients de services externes.

Ce module fournit les patterns communs pour la gestion des connexions,
retry logic, circuit breaker, et monitoring.
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import aiohttp
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ClientStatus(Enum):
    """Statuts possibles d'un client."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuration pour les tentatives de retry."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Configuration pour le circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    enabled: bool = True


@dataclass
class HealthCheckConfig:
    """Configuration pour les vérifications de santé."""
    enabled: bool = True
    interval_seconds: float = 30.0
    timeout_seconds: float = 5.0
    endpoint: Optional[str] = None


class CircuitBreakerState(Enum):
    """États du circuit breaker."""
    CLOSED = "closed"    # Fonctionnement normal
    OPEN = "open"        # Circuit ouvert (erreurs)
    HALF_OPEN = "half_open"  # Test de récupération


class CircuitBreaker:
    """Implémentation d'un circuit breaker simple."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.enabled = config.enabled
    
    def can_execute(self) -> bool:
        """Détermine si une requête peut être exécutée."""
        if not self.enabled:
            return True
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Vérifier si le timeout est expiré
            if time.time() - self.last_failure_time > self.config.timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def on_success(self):
        """Appelé en cas de succès."""
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
        """Appelé en cas d'échec."""
        if not self.enabled:
            return
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN


class BaseClient(ABC, Generic[T]):
    """
    Classe de base pour tous les clients de services externes.
    
    Fournit:
    - Gestion des connexions HTTP
    - Retry logic avec backoff exponentiel
    - Circuit breaker
    - Health checks automatiques
    - Métriques et logging standardisés
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
        
        # Configuration par défaut
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.health_check_config = health_check_config or HealthCheckConfig()
        
        # État interne
        self.circuit_breaker = CircuitBreaker(self.circuit_breaker_config)
        self.status = ClientStatus.UNKNOWN
        self.last_health_check = 0
        self.last_error: Optional[str] = None
        
        # Métriques
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        
        # Session HTTP réutilisable
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Initializing {service_name} client: {base_url}")
    
    async def __aenter__(self):
        """Gestionnaire de contexte asynchrone - entrée."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Gestionnaire de contexte asynchrone - sortie."""
        await self.close()
    
    async def start(self):
        """Démarre le client et initialise la session HTTP."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.headers
            )
            logger.info(f"{self.service_name} client started")
    
    async def close(self):
        """Ferme le client et nettoie les ressources."""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info(f"{self.service_name} client closed")
    
    @property
    def session(self) -> aiohttp.ClientSession:
        """Accès à la session HTTP."""
        if self._session is None:
            raise RuntimeError(f"{self.service_name} client not started. Call start() first.")
        return self._session
    
    async def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str = "request"
    ) -> T:
        """
        Exécute une opération avec retry logic et circuit breaker.
        
        Args:
            operation: Fonction asynchrone à exécuter
            operation_name: Nom de l'opération pour les logs
            
        Returns:
            Résultat de l'opération
            
        Raises:
            Exception: Si toutes les tentatives échouent
        """
        # Vérifier le circuit breaker
        if not self.circuit_breaker.can_execute():
            raise Exception(f"{self.service_name} circuit breaker is OPEN")
        
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                start_time = time.time()
                
                # Exécuter l'opération
                result = await operation()
                
                # Enregistrer les métriques de succès
                response_time = time.time() - start_time
                self._record_success(response_time)
                self.circuit_breaker.on_success()
                
                logger.debug(f"{self.service_name} {operation_name} succeeded (attempt {attempt + 1})")
                return result
                
            except Exception as e:
                last_exception = e
                self._record_error(str(e))
                self.circuit_breaker.on_failure()
                
                logger.warning(
                    f"{self.service_name} {operation_name} failed (attempt {attempt + 1}/{self.retry_config.max_attempts}): {e}"
                )
                
                # Ne pas retry sur la dernière tentative
                if attempt == self.retry_config.max_attempts - 1:
                    break
                
                # Calculer le délai de retry
                delay = self._calculate_retry_delay(attempt)
                await asyncio.sleep(delay)
        
        # Toutes les tentatives ont échoué
        logger.error(f"{self.service_name} {operation_name} failed after {self.retry_config.max_attempts} attempts")
        raise last_exception
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calcule le délai de retry avec backoff exponentiel."""
        delay = self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt)
        delay = min(delay, self.retry_config.max_delay)
        
        # Ajouter du jitter pour éviter le thundering herd
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def _record_success(self, response_time: float):
        """Enregistre une requête réussie."""
        self.request_count += 1
        self.total_response_time += response_time
        self.status = ClientStatus.HEALTHY
        self.last_error = None
    
    def _record_error(self, error_message: str):
        """Enregistre une erreur."""
        self.request_count += 1
        self.error_count += 1
        self.last_error = error_message
        
        # Déterminer le statut basé sur le taux d'erreur
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        if error_rate > 0.5:
            self.status = ClientStatus.UNHEALTHY
        elif error_rate > 0.1:
            self.status = ClientStatus.DEGRADED
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Effectue une vérification de santé du service.
        
        Returns:
            Dictionnaire avec les informations de santé
        """
        health_info = {
            "service_name": self.service_name,
            "status": self.status.value,
            "base_url": self.base_url,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "last_error": self.last_error,
            "metrics": self.get_metrics()
        }
        
        # Vérification spécifique au service si configurée
        if self.health_check_config.enabled:
            try:
                service_health = await self._perform_health_check()
                health_info.update(service_health)
                self.last_health_check = time.time()
            except Exception as e:
                health_info["health_check_error"] = str(e)
                logger.warning(f"{self.service_name} health check failed: {e}")
        
        return health_info
    
    @abstractmethod
    async def _perform_health_check(self) -> Dict[str, Any]:
        """
        Effectue une vérification de santé spécifique au service.
        
        À implémenter par les classes dérivées.
        
        Returns:
            Dictionnaire avec les informations de santé spécifiques
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques du client.
        
        Returns:
            Dictionnaire avec les métriques
        """
        avg_response_time = (
            self.total_response_time / (self.request_count - self.error_count)
            if self.request_count > self.error_count
            else 0
        )
        
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "average_response_time_ms": avg_response_time * 1000,
            "status": self.status.value,
            "circuit_breaker_state": self.circuit_breaker.state.value
        }
    
    def reset_metrics(self):
        """Remet à zéro les métriques."""
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        logger.info(f"{self.service_name} metrics reset")
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Teste la connectivité de base au service.
        
        À implémenter par les classes dérivées.
        
        Returns:
            True si la connexion fonctionne
        """
        pass