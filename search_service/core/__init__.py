"""
🔧 Module Core - Moteur de recherche et logique métier
"""

# === IMPORTS MOTEUR ===
from .lexical_engine import LexicalEngine

# === IMPORTS EXECUTEUR ===
from .query_executor import QueryExecutor

# === IMPORTS PROCESSEUR ===
from .result_processor import ResultProcessor

# === IMPORTS OPTIMISEUR ===
from .performance_optimizer import PerformanceOptimizer

# === CLASSE GESTIONNAIRE ===
class CoreManager:
    """Gestionnaire simplifié pour le core"""
    def __init__(self):
        self.lexical_engine = LexicalEngine()
        self.query_executor = QueryExecutor()
        self.result_processor = ResultProcessor()
        self.performance_optimizer = PerformanceOptimizer()

# === EXPORTS ===
__all__ = [
    # Gestionnaire
    "CoreManager",
    
    # Composants principaux
    "LexicalEngine",
    "QueryExecutor", 
    "ResultProcessor",
    "PerformanceOptimizer"
]