"""
Search Service Core Package

Ce package contient le cœur du moteur de recherche lexicale optimisé pour Elasticsearch.
Architecture spécialisée sans IA, focus sur la performance pure.

Components:
- query_executor: Construction et exécution des requêtes Elasticsearch
- result_processor: Traitement et formatage des résultats
- performance_optimizer: Optimisations et cache
- lexical_engine: Orchestrateur principal (utilise tous les composants)
"""

from .query_executor import QueryExecutor
from .result_processor import ResultProcessor
from .performance_optimizer import PerformanceOptimizer
from .lexical_engine import LexicalEngine

__all__ = [
    "QueryExecutor",
    "ResultProcessor", 
    "PerformanceOptimizer",
    "LexicalEngine"
]