"""
Classe base optimisée pour tous les agents conversation service
"""
import logging
import hashlib
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

# Configuration du logger
logger = logging.getLogger("conversation_service.agents")


class AgentStatus(str, Enum):
    """Statuts possibles des agents"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ExecutionResult:
    """Résultat d'exécution agent avec métadonnées"""
    
    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: Optional[str] = None,
        processing_time_ms: int = 0,
        cache_hit: bool = False,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.processing_time_ms = processing_time_ms
        self.cache_hit = cache_hit
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)


@dataclass
class AgentMetrics:
    """Métriques détaillées agent"""
    executions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    total_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    max_processing_time_ms: float = 0.0
    total_confidence: float = 0.0
    successful_executions: int = 0
    
    def add_execution(
        self, 
        processing_time_ms: float, 
        success: bool, 
        cache_hit: bool = False, 
        confidence: float = 0.0
    ) -> None:
        """Ajoute une exécution aux métriques"""
        self.executions += 1
        self.total_processing_time_ms += processing_time_ms
        
        if processing_time_ms < self.min_processing_time_ms:
            self.min_processing_time_ms = processing_time_ms
        if processing_time_ms > self.max_processing_time_ms:
            self.max_processing_time_ms = processing_time_ms
        
        if success:
            self.successful_executions += 1
            self.total_confidence += confidence
        else:
            self.errors += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_avg_processing_time(self) -> float:
        """Temps de traitement moyen"""
        return self.total_processing_time_ms / max(self.executions, 1)
    
    def get_success_rate(self) -> float:
        """Taux de succès"""
        return (self.successful_executions / max(self.executions, 1)) * 100
    
    def get_cache_hit_rate(self) -> float:
        """Taux de hit cache"""
        total_cache_ops = self.cache_hits + self.cache_misses
        return (self.cache_hits / max(total_cache_ops, 1)) * 100
    
    def get_avg_confidence(self) -> float:
        """Confiance moyenne"""
        return self.total_confidence / max(self.successful_executions, 1)


class BaseAgent(ABC):
    """
    Classe de base optimisée pour tous les agents avec gestion avancée
    
    Features:
    - Gestion d'état et statut
    - Métriques détaillées
    - Cache intelligent
    - Circuit breaker
    - Retry automatique
    - Validation inputs/outputs
    """
    
    def __init__(
        self,
        name: str,
        deepseek_client: Any,
        cache_manager: Any,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        enable_circuit_breaker: bool = True
    ):
        self.name = name
        self.deepseek_client = deepseek_client
        self.cache_manager = cache_manager
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        # État agent
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.now(timezone.utc)
        self.last_execution = None
        
        # Métriques
        self.metrics = AgentMetrics()
        
        # Circuit breaker
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = 5  # Erreurs consécutives
        self.circuit_breaker_reset_time = 60  # secondes
        self.consecutive_errors = 0
        self.circuit_opened_at = None
        
        # Configuration cache
        self.cache_enabled = bool(cache_manager)
        self.cache_ttl_default = 300  # 5 minutes
        
        self.status = AgentStatus.READY
        logger.info(f"Agent {self.name} initialisé avec circuit breaker: {enable_circuit_breaker}")
    
    async def execute_safe(
        self, 
        input_data: Union[str, Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None
    ) -> ExecutionResult:
        """
        Exécution sécurisée avec gestion d'erreurs, cache et circuit breaker
        
        Args:
            input_data: Données d'entrée
            context: Contexte optionnel
            use_cache: Utiliser le cache
            cache_ttl: TTL cache spécifique
            
        Returns:
            ExecutionResult: Résultat avec métadonnées
        """
        start_time = datetime.now(timezone.utc)
        execution_id = self._generate_execution_id(input_data)
        
        try:
            # Vérification circuit breaker
            if not await self._check_circuit_breaker():
                return ExecutionResult(
                    success=False,
                    error="Circuit breaker ouvert",
                    processing_time_ms=0
                )
            
            # Validation inputs
            validation_error = self._validate_inputs(input_data, context)
            if validation_error:
                return ExecutionResult(
                    success=False,
                    error=f"Validation input: {validation_error}",
                    processing_time_ms=0
                )
            
            # Tentative récupération cache
            cache_result = None
            if use_cache and self.cache_enabled:
                cache_result = await self._try_get_cache(input_data, context)
                if cache_result:
                    processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                    self.metrics.add_execution(processing_time, True, cache_hit=True)
                    
                    return ExecutionResult(
                        success=True,
                        data=cache_result,
                        processing_time_ms=processing_time,
                        cache_hit=True,
                        confidence=cache_result.get("confidence", 0.0) if isinstance(cache_result, dict) else 0.0
                    )
            
            # Exécution avec retry
            result = await self._execute_with_retry(input_data, context, execution_id)
            
            # Validation output
            if result.success:
                output_error = self._validate_output(result.data)
                if output_error:
                    result.success = False
                    result.error = f"Validation output: {output_error}"
            
            # Sauvegarde cache si succès
            if result.success and use_cache and self.cache_enabled:
                await self._try_set_cache(input_data, result.data, cache_ttl or self.cache_ttl_default)
            
            # Mise à jour métriques
            processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            result.processing_time_ms = processing_time
            
            self.metrics.add_execution(
                processing_time, 
                result.success, 
                cache_hit=False, 
                confidence=result.confidence
            )
            
            # Gestion circuit breaker
            if result.success:
                self.consecutive_errors = 0
                if self.circuit_opened_at:
                    logger.info(f"Agent {self.name} circuit breaker fermé après succès")
                    self.circuit_opened_at = None
            else:
                self.consecutive_errors += 1
                if self.enable_circuit_breaker and self.consecutive_errors >= self.circuit_breaker_threshold:
                    self.circuit_opened_at = datetime.now(timezone.utc)
                    logger.warning(f"Agent {self.name} circuit breaker ouvert après {self.consecutive_errors} erreurs")
            
            self.last_execution = datetime.now(timezone.utc)
            
            return result
            
        except Exception as e:
            processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            logger.error(f"Agent {self.name} erreur critique: {str(e)}", exc_info=True)
            
            self.metrics.add_execution(processing_time, False)
            self.consecutive_errors += 1
            
            return ExecutionResult(
                success=False,
                error=f"Erreur critique: {str(e)}",
                processing_time_ms=processing_time
            )
    
    async def _execute_with_retry(
        self, 
        input_data: Any, 
        context: Optional[Dict[str, Any]], 
        execution_id: str
    ) -> ExecutionResult:
        """Exécution avec retry automatique"""
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Agent {self.name} exécution tentative {attempt + 1}/{self.max_retries + 1}")
                
                # Timeout pour l'exécution
                result = await asyncio.wait_for(
                    self.execute(input_data, context),
                    timeout=self.timeout_seconds
                )
                
                # Conversion en ExecutionResult si nécessaire
                if not isinstance(result, ExecutionResult):
                    # Assuming execute() returns the data directly
                    confidence = 0.0
                    if hasattr(result, 'confidence'):
                        confidence = result.confidence
                    elif isinstance(result, dict) and 'confidence' in result:
                        confidence = result['confidence']
                    
                    result = ExecutionResult(
                        success=True,
                        data=result,
                        confidence=confidence
                    )
                
                if result.success:
                    if attempt > 0:
                        logger.info(f"Agent {self.name} succès après {attempt + 1} tentatives")
                    return result
                else:
                    last_error = result.error
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2 ** attempt, 10))  # Backoff exponentiel
                        
            except asyncio.TimeoutError:
                last_error = f"Timeout après {self.timeout_seconds}s"
                logger.warning(f"Agent {self.name} timeout tentative {attempt + 1}")
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Agent {self.name} erreur tentative {attempt + 1}: {str(e)}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 10))
        
        return ExecutionResult(
            success=False,
            error=f"Échec après {self.max_retries + 1} tentatives: {last_error}"
        )
    
    async def _check_circuit_breaker(self) -> bool:
        """Vérification état circuit breaker"""
        if not self.enable_circuit_breaker or not self.circuit_opened_at:
            return True
        
        # Vérification délai reset
        time_since_opened = (datetime.now(timezone.utc) - self.circuit_opened_at).total_seconds()
        if time_since_opened >= self.circuit_breaker_reset_time:
            logger.info(f"Agent {self.name} circuit breaker reset après {time_since_opened}s")
            self.circuit_opened_at = None
            self.consecutive_errors = 0
            return True
        
        return False
    
    def _validate_inputs(self, input_data: Any, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Validation des inputs (à surcharger par les agents)"""
        if input_data is None:
            return "Input data ne peut pas être None"
        
        if isinstance(input_data, str) and len(input_data.strip()) == 0:
            return "Input string ne peut pas être vide"
        
        if isinstance(input_data, dict) and len(input_data) == 0:
            return "Input dict ne peut pas être vide"
        
        return None
    
    def _validate_output(self, output_data: Any) -> Optional[str]:
        """Validation des outputs (à surcharger par les agents)"""
        if output_data is None:
            return "Output ne peut pas être None"
        
        return None
    
    def _generate_execution_id(self, input_data: Any) -> str:
        """Génération ID unique pour l'exécution"""
        timestamp = datetime.now(timezone.utc).isoformat()
        input_str = str(input_data)
        combined = f"{self.name}:{timestamp}:{input_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def _generate_cache_key(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> str:
        """Génération clé cache optimisée"""
        # Sérialisation déterministe pour cache cohérent
        input_str = self._serialize_for_cache(input_data)
        context_str = self._serialize_for_cache(context) if context else ""
        
        cache_input = f"{self.name}:{input_str}:{context_str}"
        return hashlib.sha256(cache_input.encode()).hexdigest()
    
    def _serialize_for_cache(self, data: Any) -> str:
        """Sérialisation déterministe pour cache"""
        if isinstance(data, dict):
            # Tri des clés pour déterminisme
            sorted_items = sorted(data.items())
            return str(sorted_items)
        elif isinstance(data, list):
            return str(sorted(data) if all(isinstance(x, (str, int, float)) for x in data) else data)
        else:
            return str(data)
    
    async def _try_get_cache(self, input_data: Any, context: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Tentative récupération cache avec gestion d'erreurs"""
        try:
            if not self.cache_manager:
                return None
            
            cache_key = self._generate_cache_key(input_data, context)
            return await self.cache_manager.get_semantic_cache(cache_key, cache_type=self.name.lower())
            
        except Exception as e:
            logger.debug(f"Agent {self.name} erreur cache get: {str(e)}")
            return None
    
    async def _try_set_cache(self, input_data: Any, output_data: Any, ttl: int) -> bool:
        """Tentative sauvegarde cache avec gestion d'erreurs"""
        try:
            if not self.cache_manager:
                return False
            
            cache_key = self._generate_cache_key(input_data)
            return await self.cache_manager.set_semantic_cache(
                cache_key, 
                output_data, 
                ttl=ttl, 
                cache_type=self.name.lower()
            )
            
        except Exception as e:
            logger.debug(f"Agent {self.name} erreur cache set: {str(e)}")
            return False
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Métriques détaillées pour monitoring"""
        uptime_seconds = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        
        return {
            "agent_name": self.name,
            "status": self.status.value,
            "uptime_seconds": uptime_seconds,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            
            # Métriques exécution
            "total_executions": self.metrics.executions,
            "successful_executions": self.metrics.successful_executions,
            "failed_executions": self.metrics.errors,
            "success_rate_percent": self.metrics.get_success_rate(),
            
            # Métriques performance
            "avg_processing_time_ms": self.metrics.get_avg_processing_time(),
            "min_processing_time_ms": self.metrics.min_processing_time_ms if self.metrics.min_processing_time_ms != float('inf') else 0,
            "max_processing_time_ms": self.metrics.max_processing_time_ms,
            "total_processing_time_ms": self.metrics.total_processing_time_ms,
            
            # Métriques cache
            "cache_enabled": self.cache_enabled,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate_percent": self.metrics.get_cache_hit_rate(),
            
            # Métriques qualité
            "avg_confidence": self.metrics.get_avg_confidence(),
            
            # Circuit breaker
            "circuit_breaker_enabled": self.enable_circuit_breaker,
            "consecutive_errors": self.consecutive_errors,
            "circuit_opened": bool(self.circuit_opened_at),
            "circuit_opened_at": self.circuit_opened_at.isoformat() if self.circuit_opened_at else None
        }
    
    def reset_metrics(self) -> None:
        """Reset métriques agent"""
        self.metrics = AgentMetrics()
        self.consecutive_errors = 0
        self.circuit_opened_at = None
        logger.info(f"Agent {self.name} métriques réinitialisées")
    
    def set_status(self, status: AgentStatus, reason: Optional[str] = None) -> None:
        """Mise à jour statut agent"""
        old_status = self.status
        self.status = status
        
        log_message = f"Agent {self.name} status: {old_status.value} -> {status.value}"
        if reason:
            log_message += f" ({reason})"
        
        logger.info(log_message)
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification santé agent"""
        is_healthy = (
            self.status in [AgentStatus.READY, AgentStatus.PROCESSING] and
            not (self.enable_circuit_breaker and self.circuit_opened_at) and
            self.metrics.get_success_rate() > 50.0  # Seuil configurable
        )
        
        return {
            "healthy": is_healthy,
            "status": self.status.value,
            "success_rate": self.metrics.get_success_rate(),
            "circuit_breaker_open": bool(self.circuit_opened_at),
            "last_execution": self.last_execution.isoformat() if self.last_execution else None
        }
    
    @abstractmethod
    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Méthode d'exécution principale à implémenter par chaque agent
        
        Args:
            input_data: Données d'entrée
            context: Contexte optionnel
            
        Returns:
            Résultat de l'exécution (format libre selon l'agent)
        """
        pass
    
    def __str__(self) -> str:
        return f"Agent({self.name}, status={self.status.value})"
    
    def __repr__(self) -> str:
        metrics = self.get_detailed_metrics()
        return (
            f"Agent(name={self.name}, status={self.status.value}, "
            f"executions={metrics['total_executions']}, "
            f"success_rate={metrics['success_rate_percent']:.1f}%, "
            f"cache_rate={metrics['cache_hit_rate_percent']:.1f}%)"
        )