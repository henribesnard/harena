"""
Classe base pour tous les agents conversation service
"""
import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime, timezone

# Configuration du logger
logger = logging.getLogger("conversation_service.agents")

class BaseAgent(ABC):
    """Classe de base pour tous les agents DeepSeek"""
    
    def __init__(
        self,
        name: str,
        deepseek_client: Any,
        cache_manager: Any
    ):
        self.name = name
        self.deepseek_client = deepseek_client
        self.cache_manager = cache_manager
        self.metrics: Dict[str, Any] = {
            "executions": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_processing_time": 0.0
        }
        
        logger.info(f"Agent {self.name} initialisé")
    
    def _generate_cache_key(self, input_data: str) -> str:
        """Génération clé cache consistente"""
        # Hash MD5 de l'input pour clé cache stable
        cache_input = f"{self.name}:{input_data}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _update_metrics(self, event: str, start_time: datetime) -> None:
        """Mise à jour métriques agent"""
        try:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            self.metrics["executions"] += 1
            self.metrics["total_processing_time"] += execution_time
            
            if event == "cache_hit":
                self.metrics["cache_hits"] += 1
            elif event.endswith("_error"):
                self.metrics["errors"] += 1
                
            # Log métriques importantes
            if event.endswith("_error"):
                logger.error(f"Agent {self.name} error: {event}")
            elif execution_time > 5000:  # > 5 secondes
                logger.warning(f"Agent {self.name} slow execution: {execution_time}ms")
                
        except Exception as e:
            logger.error(f"Erreur mise à jour métriques agent {self.name}: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Récupération métriques agent"""
        try:
            total_executions = max(self.metrics["executions"], 1)
            
            return {
                "agent_name": self.name,
                "total_executions": self.metrics["executions"],
                "cache_hits": self.metrics["cache_hits"],
                "cache_hit_rate": self.metrics["cache_hits"] / total_executions,
                "errors": self.metrics["errors"],
                "error_rate": self.metrics["errors"] / total_executions,
                "avg_processing_time_ms": self.metrics["total_processing_time"] / total_executions,
                "total_processing_time_ms": self.metrics["total_processing_time"]
            }
            
        except Exception as e:
            logger.error(f"Erreur récupération métriques agent {self.name}: {str(e)}")
            return {"agent_name": self.name, "error": str(e)}
    
    def reset_metrics(self) -> None:
        """Reset métriques agent"""
        self.metrics = {
            "executions": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_processing_time": 0.0
        }
        
        logger.info(f"Agent {self.name} métriques réinitialisées")
    
    @abstractmethod
    async def execute(self, input_data: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Méthode d'exécution à implémenter par chaque agent"""
        pass
    
    def __str__(self) -> str:
        return f"Agent({self.name})"
    
    def __repr__(self) -> str:
        metrics = self.get_metrics()
        return f"Agent(name={self.name}, executions={metrics['total_executions']}, cache_rate={metrics['cache_hit_rate']:.2f})"