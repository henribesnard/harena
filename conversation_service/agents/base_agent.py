"""
Base Agent pour la nouvelle architecture v2.0
Agent de base simplifié sans AutoGen
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Agent de base pour tous les agents LLM"""
    
    def __init__(self, name: str):
        self.name = name
        self.created_at = datetime.now()
        self.logger = logging.getLogger(f"agent.{name}")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Traite une requête d'entrée"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut de l'agent"""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "status": "active"
        }