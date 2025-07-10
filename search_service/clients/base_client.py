"""
Client de base pour tous les clients externes du Search Service.

Cette classe fournit une base robuste pour tous les clients externes avec :
- Gestion d'erreurs et retry avec backoff exponentiel
- Circuit breaker pour prévenir les cascades de pannes
- Health monitoring et métriques
- Configuration centralisée et validation
- Logging structuré et observabilité

ARCHITECTURE:
- Patron Template Method pour extensibilité
- Configuration par dataclasses typées
- Gestion d'état du circuit breaker
- Métriques et monitoring intégrés
- Support async/await natif

USAGE:
    class MyClient(BaseClient):
        async def _perform_health_check(self) -> Dict[str, Any]:
            # Implémentation spécifique
            return {"status": "healthy"}
    
    client = MyClient("https://api.example.com", "my_service")
    await client.start()
    result = await client.execute_with_retry(my_operation, "operation_name")
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar, Generic
from datetime import datetime, timedelta

import aiohttp

# Configuration centralisée
from config_service.config import settings

# Types
T = TypeVar('T')

logger = logging.getLogger(__name__)

# ==================== ENUMS ====================

class CircuitBreakerState(str, Enum):
    """États du circuit breaker."""
    CLOSED = "closed"      # Fonctionnement normal
    OPEN = "open"          # Circuit ouvert, rejette les requêtes
    HALF_OPEN = "half_open"  # Test de récupération

# ==================== EXCEPTIONS ====================

class ClientError(Exception):
    """Exception de base pour les erreurs de client."""
    def __init__(self, message: str, operation: str = "", details: Optional[Dict] = None):
        super().__init__(message)
        self.operation = operation
        self.details = details or {}
        self.timestamp = datetime.utcnow()

class ConnectionError(ClientError):
    """Erreur de connexion au service externe."""
    pass

class TimeoutError(ClientError):
    """Erreur de timeout."""
    pass

class RetryExhaustedError(ClientError):
    """Toutes les tentatives de retry ont échoué."""
    pass

class CircuitBreakerOpenError(ClientError):
    """Circuit breaker ouvert, requête rejetée."""
    pass

# ==================== CONFIGURATION ====================

@dataclass
class RetryConfig:
    """Configuration des tentatives de retry."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True
    
    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be > 0")
        if self.backoff_factor <= 1:
            raise ValueError("backoff_factor must be > 1")

@dataclass
class CircuitBreakerConfig:
    """Configuration du circuit breaker."""
    failure_threshold: int = 5
    timeout_threshold: float = 10.0
    recovery_timeout: float = 60.0
    
    def __post_init__(self):
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.timeout_threshold <= 0:
            raise ValueError("timeout_threshold must be > 0")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")

@dataclass
class HealthCheckConfig:
    """Configuration des health checks."""
    enabled: bool = True
    interval_seconds: float = 30.0
    timeout_seconds: float = 5.0
    endpoint: str = "/health"
    
    def __post_init__(self):
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

# ==================== MÉTRIQUES ====================

@dataclass
class ClientMetrics:
    """Métriques du client."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retry_attempts: int = 0
    circuit_breaker_opens: int = 0
    avg_response_time: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    def record_success(self, response_time: float):
        """Enregistre une requête réussie."""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_success = datetime.utcnow()
        
        # Moyenne mobile simple
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * 0.9) + (response_time * 0.1)
    
    def record_failure(self):
        """Enregistre une requête échouée."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_failure = datetime.utcnow()
    
    def record_retry(self):
        """Enregistre une tentative de retry."""
        self.retry_attempts += 1
    
    def record_circuit_breaker_open(self):
        """Enregistre l'ouverture du circuit breaker."""
        self.circuit_breaker_opens += 1
    
    @property
    def success_rate(self) -> float:
        """Taux de succès."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        """Taux d'échec."""
        return 1.0 - self.success_rate

# ==================== CIRCUIT BREAKER ====================

class CircuitBreaker:
    """Circuit breaker pour protection contre les cascades de pannes."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        
    def can_execute(self) -> bool:
        """Vérifie si une requête peut être exécutée."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Vérifier si on peut passer en half-open
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time).total_seconds() > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        
        # HALF_OPEN : on peut tenter une requête
        return True
    
    def record_success(self):
        """Enregistre un succès."""
        self.failure_count = 0
        self.last_success_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker transitioning to CLOSED")
    
    def record_failure(self):
        """Enregistre un échec."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker transitioning to OPEN after {self.failure_count} failures")

# ==================== CLIENT DE BASE ====================

class BaseClient(ABC):
    """
    Client de base pour tous les services externes.
    
    Fournit:
    - Gestion robuste des erreurs avec retry et circuit breaker
    - Health monitoring automatique
    - Métriques et observabilité
    - Configuration centralisée
    """
    
    def __init__(
        self,
        base_url: str,
        service_name: str,
        timeout: float = 5.0,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        health_check_config: Optional[HealthCheckConfig] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.service_name = service_name
        self.timeout = timeout
        
        # Configuration avec valeurs par défaut
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.health_check_config = health_check_config or HealthCheckConfig()
        
        # État interne
        self.circuit_breaker = CircuitBreaker(self.circuit_breaker_config)
        self.metrics = ClientMetrics()
        self._session: Optional[aiohttp.ClientSession] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_health_check: Optional[Dict[str, Any]] = None
        
        # Headers par défaut
        self.headers = {
            "User-Agent": f"search-service/{getattr(settings, 'VERSION', '1.0.0')}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized {service_name} client: {base_url}")
    
    @property
    def session(self) -> aiohttp.ClientSession:
        """Session HTTP (lazy loading)."""
        if self._session is None:
            raise RuntimeError(f"{self.service_name} client not started. Call start() first.")
        return self._session
    
    async def start(self):
        """Démarre le client."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.headers
            )
            logger.info(f"{self.service_name} client started")
        
        # Démarrer le health monitoring
        if self.health_check_config.enabled:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self):
        """Arrête le client."""
        # Arrêter le health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Fermer la session
        if self._session:
            await self._session.close()
            self._session = None
            logger.info(f"{self.service_name} client stopped")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    # ==================== MÉTHODES PUBLIQUES ====================
    
    async def execute_with_retry(
        self,
        operation: Callable[[], Awaitable[T]],
        operation_name: str = "unknown"
    ) -> T:
        """
        Exécute une opération avec retry et circuit breaker.
        
        Args:
            operation: Fonction async à exécuter
            operation_name: Nom de l'opération pour les logs
            
        Returns:
            Résultat de l'opération
            
        Raises:
            CircuitBreakerOpenError: Si le circuit breaker est ouvert
            RetryExhaustedError: Si toutes les tentatives ont échoué
        """
        # Vérifier le circuit breaker
        if not self.circuit_breaker.can_execute():
            self.metrics.record_circuit_breaker_open()
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open for {self.service_name}",
                operation_name
            )
        
        last_exception = None
        start_time = time.time()
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                result = await operation()
                
                # Succès
                response_time = time.time() - start_time
                self.metrics.record_success(response_time)
                self.circuit_breaker.record_success()
                
                if attempt > 0:
                    logger.info(
                        f"{self.service_name} {operation_name} succeeded on attempt {attempt + 1}"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                self.metrics.record_failure()
                
                # Ne pas retry sur certaines erreurs
                if self._is_non_retryable_error(e):
                    logger.error(
                        f"{self.service_name} {operation_name} failed with non-retryable error: {e}"
                    )
                    self.circuit_breaker.record_failure()
                    raise ClientError(str(e), operation_name, {"attempt": attempt + 1}) from e
                
                # Dernière tentative
                if attempt == self.retry_config.max_attempts - 1:
                    logger.error(
                        f"{self.service_name} {operation_name} failed after {attempt + 1} attempts: {e}"
                    )
                    self.circuit_breaker.record_failure()
                    break
                
                # Calculer le délai de retry
                delay = self._calculate_retry_delay(attempt)
                self.metrics.record_retry()
                
                logger.warning(
                    f"{self.service_name} {operation_name} failed on attempt {attempt + 1}, "
                    f"retrying in {delay:.1f}s: {e}"
                )
                
                await asyncio.sleep(delay)
        
        # Toutes les tentatives ont échoué
        raise RetryExhaustedError(
            f"All {self.retry_config.max_attempts} attempts failed for {operation_name}",
            operation_name,
            {"last_error": str(last_exception)}
        ) from last_exception
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du client."""
        return {
            "service_name": self.service_name,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": self.metrics.success_rate,
            "avg_response_time": self.metrics.avg_response_time,
            "retry_attempts": self.metrics.retry_attempts,
            "circuit_breaker_opens": self.metrics.circuit_breaker_opens,
            "last_success": self.metrics.last_success.isoformat() if self.metrics.last_success else None,
            "last_failure": self.metrics.last_failure.isoformat() if self.metrics.last_failure else None,
            "last_health_check": self._last_health_check
        }
    
    # ==================== MÉTHODES ABSTRAITES ====================
    
    @abstractmethod
    async def _perform_health_check(self) -> Dict[str, Any]:
        """
        Effectue un health check spécifique au service.
        
        Returns:
            Résultat du health check avec au minimum {"status": "healthy|unhealthy"}
        """
        pass
    
    # ==================== MÉTHODES PRIVÉES ====================
    
    def _is_non_retryable_error(self, error: Exception) -> bool:
        """Détermine si une erreur ne doit pas être retryée."""
        # Erreurs HTTP 4xx (client errors) ne doivent pas être retryées
        if isinstance(error, aiohttp.ClientResponseError):
            return 400 <= error.status < 500
        
        # Erreurs de validation ne doivent pas être retryées
        if isinstance(error, (ValueError, TypeError)):
            return True
        
        return False
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calcule le délai de retry avec backoff exponentiel."""
        delay = self.retry_config.base_delay * (self.retry_config.backoff_factor ** attempt)
        delay = min(delay, self.retry_config.max_delay)
        
        # Ajouter du jitter pour éviter le thundering herd
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    async def _health_check_loop(self):
        """Boucle de health check périodique."""
        while True:
            try:
                await asyncio.sleep(self.health_check_config.interval_seconds)
                
                # Effectuer le health check avec timeout
                health_check_start = time.time()
                
                try:
                    async with asyncio.timeout(self.health_check_config.timeout_seconds):
                        health_result = await self._perform_health_check()
                        health_check_time = time.time() - health_check_start
                        
                        self._last_health_check = {
                            **health_result,
                            "timestamp": datetime.utcnow().isoformat(),
                            "response_time": health_check_time
                        }
                        
                        if health_result.get("status") == "healthy":
                            logger.debug(f"{self.service_name} health check passed")
                        else:
                            logger.warning(f"{self.service_name} health check degraded: {health_result}")
                            
                except asyncio.TimeoutError:
                    logger.warning(f"{self.service_name} health check timeout")
                    self._last_health_check = {
                        "status": "unhealthy",
                        "error": "timeout",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
            except asyncio.CancelledError:
                logger.info(f"{self.service_name} health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"{self.service_name} health check error: {e}")
                self._last_health_check = {
                    "status": "unhealthy", 
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }