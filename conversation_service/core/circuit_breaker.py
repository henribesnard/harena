"""
Circuit Breaker Pattern pour protection search_service
Implémentation robuste avec états multiples et métriques détaillées
"""
import asyncio
import logging
import time
from typing import Any, Callable, Optional, Dict, List
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("conversation_service.circuit_breaker")


class CircuitBreakerState(str, Enum):
    """États du circuit breaker"""
    CLOSED = "closed"        # Fonctionnement normal
    OPEN = "open"           # Circuit ouvert - bloque les appels
    HALF_OPEN = "half_open" # Test de récupération


class CircuitBreakerError(Exception):
    """Erreur circuit breaker ouvert"""
    def __init__(self, message: str, state: str, last_failure_time: Optional[float] = None):
        super().__init__(message)
        self.state = state
        self.last_failure_time = last_failure_time


class CircuitBreakerMetrics:
    """Métriques détaillées du circuit breaker"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rejected_requests = 0  # Rejetées par circuit ouvert
        
        self.failure_rate = 0.0
        self.average_response_time = 0.0
        self.last_failure_time: Optional[datetime] = None
        self.state_changes: List[Dict[str, Any]] = []
        
        self.response_times: List[float] = []  # Fenêtre glissante
        self.max_response_times_window = 100
    
    def record_success(self, response_time: float):
        """Enregistre un succès"""
        self.total_requests += 1
        self.successful_requests += 1
        self._update_response_times(response_time)
        self._calculate_failure_rate()
    
    def record_failure(self, response_time: Optional[float] = None):
        """Enregistre un échec"""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if response_time:
            self._update_response_times(response_time)
        
        self._calculate_failure_rate()
    
    def record_rejection(self):
        """Enregistre une requête rejetée"""
        self.rejected_requests += 1
    
    def record_state_change(self, from_state: str, to_state: str, reason: str):
        """Enregistre un changement d'état"""
        self.state_changes.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason
        })
        
        # Limiter historique
        if len(self.state_changes) > 50:
            self.state_changes = self.state_changes[-25:]
    
    def _update_response_times(self, response_time: float):
        """Met à jour les temps de réponse"""
        self.response_times.append(response_time)
        
        if len(self.response_times) > self.max_response_times_window:
            self.response_times = self.response_times[-self.max_response_times_window:]
        
        if self.response_times:
            self.average_response_time = sum(self.response_times) / len(self.response_times)
    
    def _calculate_failure_rate(self):
        """Calcule le taux d'échec"""
        if self.total_requests > 0:
            self.failure_rate = self.failed_requests / self.total_requests
        else:
            self.failure_rate = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "failure_rate": self.failure_rate,
            "average_response_time_ms": self.average_response_time * 1000,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "recent_state_changes": self.state_changes[-5:] if self.state_changes else []
        }


class CircuitBreaker:
    """
    Circuit Breaker Pattern avec protection robuste
    
    Fonctionnalités:
    - États CLOSED/OPEN/HALF_OPEN
    - Seuils configurables échecs/temps
    - Fenêtre glissante pour calculs
    - Métriques détaillées
    - Recovery automatique
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,          # Échecs avant ouverture
        recovery_timeout: float = 60.0,      # Secondes avant test recovery
        success_threshold: int = 2,          # Succès pour refermer
        timeout_threshold: float = 30.0,     # Timeout considéré comme échec
        expected_exceptions: tuple = None,    # Exceptions déclenchant circuit
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout_threshold = timeout_threshold
        self.expected_exceptions = expected_exceptions or (Exception,)
        self.name = name
        
        # État interne
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state_changed_time = time.time()
        
        # Métriques
        self.metrics = CircuitBreakerMetrics()
        
        # Verrou pour thread safety
        self._lock = asyncio.Lock()
        
        logger.info(
            f"Circuit breaker '{name}' initialisé - "
            f"Seuil échecs: {failure_threshold}, "
            f"Timeout recovery: {recovery_timeout}s"
        )
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Appel protégé par circuit breaker
        
        Args:
            func: Fonction à appeler
            *args, **kwargs: Arguments pour la fonction
            
        Returns:
            Résultat de la fonction
            
        Raises:
            CircuitBreakerError: Si circuit ouvert
            Exception: Exceptions de la fonction originale
        """
        async with self._lock:
            await self._check_state()
        
        if self.state == CircuitBreakerState.OPEN:
            self.metrics.record_rejection()
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is OPEN",
                state=self.state.value,
                last_failure_time=self.last_failure_time
            )
        
        # Appel de la fonction
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            response_time = time.time() - start_time
            
            # Enregistrer succès
            async with self._lock:
                await self._on_success(response_time)
            
            return result
            
        except self.expected_exceptions as e:
            response_time = time.time() - start_time
            
            # Enregistrer échec
            async with self._lock:
                await self._on_failure(response_time, e)
            
            raise  # Re-raise l'exception originale
    
    async def _check_state(self):
        """Vérification et mise à jour d'état"""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.OPEN:
            # Tester si on peut passer en HALF_OPEN
            if self.last_failure_time and (current_time - self.last_failure_time) >= self.recovery_timeout:
                await self._transition_to_state(
                    CircuitBreakerState.HALF_OPEN,
                    f"Recovery timeout atteint après {self.recovery_timeout}s"
                )
    
    async def _on_success(self, response_time: float):
        """Traitement d'un succès"""
        self.metrics.record_success(response_time)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.success_threshold:
                await self._transition_to_state(
                    CircuitBreakerState.CLOSED,
                    f"Recovery réussie après {self.success_count} succès"
                )
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset compteur échecs sur succès
            if self.failure_count > 0:
                self.failure_count = 0
    
    async def _on_failure(self, response_time: float, exception: Exception):
        """Traitement d'un échec"""
        self.metrics.record_failure(response_time)
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        logger.warning(
            f"Circuit breaker '{self.name}' - Échec #{self.failure_count}: "
            f"{type(exception).__name__}: {str(exception)}"
        )
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Retour immédiat à OPEN sur échec en HALF_OPEN
            await self._transition_to_state(
                CircuitBreakerState.OPEN,
                f"Échec en recovery: {type(exception).__name__}"
            )
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Ouvrir circuit si seuil atteint
            if self.failure_count >= self.failure_threshold:
                await self._transition_to_state(
                    CircuitBreakerState.OPEN,
                    f"Seuil d'échecs atteint: {self.failure_count}/{self.failure_threshold}"
                )
    
    async def _transition_to_state(self, new_state: CircuitBreakerState, reason: str):
        """Transition vers un nouvel état"""
        old_state = self.state
        self.state = new_state
        self.state_changed_time = time.time()
        
        # Reset compteurs selon l'état
        if new_state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitBreakerState.HALF_OPEN:
            self.success_count = 0
        
        # Enregistrer transition
        self.metrics.record_state_change(old_state.value, new_state.value, reason)
        
        logger.info(
            f"Circuit breaker '{self.name}': {old_state.value} → {new_state.value} "
            f"(Raison: {reason})"
        )
    
    def is_open(self) -> bool:
        """Vérifie si le circuit est ouvert"""
        return self.state == CircuitBreakerState.OPEN
    
    def is_half_open(self) -> bool:
        """Vérifie si le circuit est en semi-ouverture"""
        return self.state == CircuitBreakerState.HALF_OPEN
    
    def is_closed(self) -> bool:
        """Vérifie si le circuit est fermé"""
        return self.state == CircuitBreakerState.CLOSED
    
    def get_state(self) -> str:
        """Retourne l'état actuel"""
        return self.state.value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne métriques complètes"""
        base_metrics = self.metrics.get_stats()
        
        return {
            **base_metrics,
            "circuit_info": {
                "name": self.name,
                "current_state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold,
                "recovery_timeout": self.recovery_timeout,
                "time_in_current_state": time.time() - self.state_changed_time
            }
        }
    
    async def force_open(self, reason: str = "Manual override"):
        """Force l'ouverture du circuit (pour tests)"""
        async with self._lock:
            await self._transition_to_state(CircuitBreakerState.OPEN, reason)
    
    async def force_close(self, reason: str = "Manual reset"):
        """Force la fermeture du circuit (pour tests)"""
        async with self._lock:
            await self._transition_to_state(CircuitBreakerState.CLOSED, reason)


class CircuitBreakerManager:
    """Gestionnaire de multiples circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        **kwargs
    ) -> CircuitBreaker:
        """Récupère ou crée un circuit breaker"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                **kwargs
            )
        
        return self._breakers[name]
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Métriques de tous les circuit breakers"""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Résumé santé de tous les circuits"""
        total_breakers = len(self._breakers)
        open_breakers = sum(1 for b in self._breakers.values() if b.is_open())
        half_open_breakers = sum(1 for b in self._breakers.values() if b.is_half_open())
        
        return {
            "total_circuit_breakers": total_breakers,
            "healthy_circuits": total_breakers - open_breakers - half_open_breakers,
            "open_circuits": open_breakers,
            "half_open_circuits": half_open_breakers,
            "overall_health": "healthy" if open_breakers == 0 else "degraded"
        }


# Instance globale pour réutilisation
circuit_breaker_manager = CircuitBreakerManager()