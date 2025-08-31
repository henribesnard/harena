"""
Gestionnaire de retry intelligent avec backoff adaptatif
Stratégies spécialisées par type d'erreur
"""
import asyncio
import logging
import random
import time
from typing import Any, Callable, Optional, Dict, List, Type, Union
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("conversation_service.retry_handler")


class BackoffStrategy(str, Enum):
    """Stratégies de backoff"""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


class RetryResult:
    """Résultat d'une tentative de retry"""
    
    def __init__(self):
        self.attempts = 0
        self.total_time = 0.0
        self.success = False
        self.last_exception: Optional[Exception] = None
        self.attempt_times: List[float] = []
        self.backoff_times: List[float] = []


class RetryConfig:
    """Configuration retry pour un type d'erreur"""
    
    def __init__(
        self,
        max_retries: int,
        backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True,
        retry_exceptions: tuple = (Exception,),
        stop_exceptions: tuple = (),
        timeout_per_attempt: Optional[float] = None
    ):
        self.max_retries = max_retries
        self.backoff_strategy = backoff_strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.retry_exceptions = retry_exceptions
        self.stop_exceptions = stop_exceptions
        self.timeout_per_attempt = timeout_per_attempt


class RetryHandler:
    """
    Gestionnaire de retry intelligent avec backoff adaptatif
    
    Fonctionnalités:
    - Stratégies multiples (fixed, exponential, jitter)
    - Configuration par type d'erreur
    - Timeout par tentative
    - Métriques détaillées
    - Circuit breaker integration
    """
    
    def __init__(self, default_config: Optional[RetryConfig] = None):
        self.default_config = default_config or RetryConfig(
            max_retries=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            base_delay=1.0,
            max_delay=30.0
        )
        
        # Configurations spécialisées par type d'erreur
        self.error_configs = self._load_error_configs()
        
        # Métriques
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_retries = 0
        self.retry_success_rate = 0.0
        
        logger.info("Retry handler initialisé avec configurations intelligentes")
    
    def _load_error_configs(self) -> Dict[str, RetryConfig]:
        """Configuration retry spécialisée par type d'erreur"""
        return {
            # Timeout - retry rapide
            "timeout": RetryConfig(
                max_retries=2,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                base_delay=1.0,
                max_delay=5.0,
                retry_exceptions=(asyncio.TimeoutError, TimeoutError)
            ),
            
            # Rate limiting - backoff agressif  
            "rate_limit": RetryConfig(
                max_retries=3,
                backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
                base_delay=2.0,
                max_delay=10.0,
                multiplier=2.5,
                retry_exceptions=(Exception,)  # À préciser selon l'exception réelle
            ),
            
            # Erreur serveur - retry modéré
            "server_error": RetryConfig(
                max_retries=2,
                backoff_strategy=BackoffStrategy.FIXED,
                base_delay=2.0,
                retry_exceptions=(Exception,)
            ),
            
            # Validation - un seul retry avec correction
            "validation_error": RetryConfig(
                max_retries=1,
                backoff_strategy=BackoffStrategy.FIXED,
                base_delay=0.5,
                retry_exceptions=(ValueError, TypeError)
            ),
            
            # Connexion réseau - retry patient
            "connection_error": RetryConfig(
                max_retries=3,
                backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
                base_delay=1.0,
                max_delay=15.0,
                retry_exceptions=(ConnectionError, OSError)
            ),
            
            # Erreur authentification - pas de retry
            "auth_error": RetryConfig(
                max_retries=0,
                stop_exceptions=(PermissionError,)
            )
        }
    
    async def execute(
        self,
        func: Callable,
        *args,
        config: Optional[RetryConfig] = None,
        error_type: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Exécute une fonction avec retry intelligent
        
        Args:
            func: Fonction à exécuter
            *args: Arguments positionnels
            config: Configuration retry spécifique
            error_type: Type d'erreur pour config automatique
            **kwargs: Arguments nommés
            
        Returns:
            Résultat de la fonction
            
        Raises:
            Exception: Dernière exception si tous les retries échouent
        """
        # Sélection configuration
        if config is None:
            config = self._get_config_for_error_type(error_type)
        
        result = RetryResult()
        last_exception = None
        
        self.total_operations += 1
        start_time = time.time()
        
        for attempt in range(config.max_retries + 1):  # +1 pour première tentative
            result.attempts = attempt + 1
            
            try:
                # Timeout par tentative si configuré
                if config.timeout_per_attempt:
                    if asyncio.iscoroutinefunction(func):
                        attempt_result = await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=config.timeout_per_attempt
                        )
                    else:
                        # Pour fonctions sync, on utilise un wrapper
                        attempt_result = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(None, func, *args, **kwargs),
                            timeout=config.timeout_per_attempt
                        )
                else:
                    if asyncio.iscoroutinefunction(func):
                        attempt_result = await func(*args, **kwargs)
                    else:
                        attempt_result = func(*args, **kwargs)
                
                # Succès !
                result.success = True
                result.total_time = time.time() - start_time
                
                # Métriques succès
                self.successful_operations += 1
                if attempt > 0:
                    self.total_retries += attempt
                    logger.info(f"Retry réussi après {attempt} tentatives")
                
                self._update_retry_success_rate()
                
                return attempt_result
                
            except config.stop_exceptions as e:
                # Exceptions qui arrêtent immédiatement les retries
                logger.warning(f"Retry arrêté par exception stop: {type(e).__name__}")
                result.last_exception = e
                break
                
            except config.retry_exceptions as e:
                last_exception = e
                result.last_exception = e
                
                attempt_time = time.time() - start_time
                result.attempt_times.append(attempt_time)
                
                logger.warning(
                    f"Tentative {attempt + 1}/{config.max_retries + 1} échouée: "
                    f"{type(e).__name__}: {str(e)}"
                )
                
                # Pas de backoff après la dernière tentative
                if attempt < config.max_retries:
                    backoff_time = self._calculate_backoff_delay(config, attempt)
                    result.backoff_times.append(backoff_time)
                    
                    logger.debug(f"Attente {backoff_time:.2f}s avant retry")
                    await asyncio.sleep(backoff_time)
                
            except Exception as e:
                # Exception non prévue - arrêt immédiat
                logger.error(f"Exception inattendue, arrêt retry: {type(e).__name__}: {str(e)}")
                result.last_exception = e
                break
        
        # Tous les retries ont échoué
        result.total_time = time.time() - start_time
        self.failed_operations += 1
        self.total_retries += result.attempts - 1
        self._update_retry_success_rate()
        
        logger.error(
            f"Retry échoué après {result.attempts} tentatives "
            f"en {result.total_time:.2f}s"
        )
        
        # Re-raise la dernière exception
        if last_exception:
            raise last_exception
        else:
            raise Exception("Retry failed without specific exception")
    
    def _get_config_for_error_type(self, error_type: Optional[str]) -> RetryConfig:
        """Sélectionne la configuration selon le type d'erreur"""
        if error_type and error_type in self.error_configs:
            return self.error_configs[error_type]
        return self.default_config
    
    def _calculate_backoff_delay(self, config: RetryConfig, attempt: int) -> float:
        """Calcule le délai de backoff selon la stratégie"""
        if config.backoff_strategy == BackoffStrategy.FIXED:
            delay = config.base_delay
        
        elif config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = config.base_delay * (attempt + 1)
        
        elif config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = config.base_delay * (config.multiplier ** attempt)
        
        elif config.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            base_delay = config.base_delay * (config.multiplier ** attempt)
            # Jitter ±25%
            jitter = random.uniform(-0.25, 0.25) * base_delay
            delay = base_delay + jitter
        
        else:
            delay = config.base_delay
        
        # Limiter le délai maximum
        delay = min(delay, config.max_delay)
        
        # Assurer délai minimum positif
        delay = max(delay, 0.1)
        
        return delay
    
    def _update_retry_success_rate(self):
        """Met à jour le taux de succès des retries"""
        if self.total_operations > 0:
            self.retry_success_rate = self.successful_operations / self.total_operations
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques retry"""
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.retry_success_rate,
            "total_retries": self.total_retries,
            "average_retries_per_operation": (
                self.total_retries / self.total_operations if self.total_operations > 0 else 0
            ),
            "configured_error_types": list(self.error_configs.keys())
        }
    
    def add_error_config(self, error_type: str, config: RetryConfig):
        """Ajoute une configuration pour un type d'erreur"""
        self.error_configs[error_type] = config
        logger.info(f"Configuration retry ajoutée pour '{error_type}'")
    
    def reset_metrics(self):
        """Remet à zéro les métriques"""
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_retries = 0
        self.retry_success_rate = 0.0
        logger.info("Métriques retry remises à zéro")


class SearchServiceRetryHandler(RetryHandler):
    """Retry handler spécialisé pour search_service"""
    
    def __init__(self):
        super().__init__()
        
        # Configurations spécialisées search_service
        self.error_configs.update({
            "search_timeout": RetryConfig(
                max_retries=2,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                base_delay=1.0,
                max_delay=5.0,
                timeout_per_attempt=10.0
            ),
            
            "search_validation": RetryConfig(
                max_retries=1,
                backoff_strategy=BackoffStrategy.FIXED,
                base_delay=0.5,
                # Auto-correction sera appliquée dans SearchServiceClient
            ),
            
            "search_rate_limit": RetryConfig(
                max_retries=3,
                backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
                base_delay=2.0,
                max_delay=12.0,
                multiplier=2.0
            ),
            
            "search_server_error": RetryConfig(
                max_retries=2,
                backoff_strategy=BackoffStrategy.FIXED,
                base_delay=3.0
            ),
            
            "search_unavailable": RetryConfig(
                max_retries=1,
                backoff_strategy=BackoffStrategy.FIXED,
                base_delay=1.0
                # Fallback sera géré dans SearchExecutor
            )
        })
        
        logger.info("Search service retry handler configuré")
    
    async def execute_search_request(
        self,
        search_func: Callable,
        query: Dict[str, Any],
        error_context: Optional[str] = None
    ) -> Any:
        """
        Exécute une requête search avec retry intelligent
        
        Args:
            search_func: Fonction de recherche
            query: Requête search_service
            error_context: Contexte pour sélection config
            
        Returns:
            Résultats de recherche
        """
        # Détection automatique du type d'erreur basé sur le contexte
        error_type = self._detect_error_type_from_context(error_context)
        
        try:
            return await self.execute(
                search_func,
                query,
                error_type=error_type
            )
        except Exception as e:
            # Log spécialisé pour search_service
            logger.error(
                f"Search service retry final failure - "
                f"Query: {query.get('user_id', 'unknown')}, "
                f"Error: {type(e).__name__}: {str(e)}"
            )
            raise
    
    def _detect_error_type_from_context(self, error_context: Optional[str]) -> Optional[str]:
        """Détecte le type d'erreur basé sur le contexte"""
        if not error_context:
            return None
        
        context_lower = error_context.lower()
        
        if "timeout" in context_lower:
            return "search_timeout"
        elif "rate" in context_lower or "429" in context_lower:
            return "search_rate_limit"
        elif "validation" in context_lower or "422" in context_lower:
            return "search_validation"
        elif "unavailable" in context_lower or "503" in context_lower:
            return "search_unavailable"
        elif "server" in context_lower or "500" in context_lower:
            return "search_server_error"
        
        return None