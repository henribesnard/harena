"""
Moteurs de recherche et composants core.

Ce module expose tous les composants principaux
du service de recherche hybride.
"""

from .query_processor import QueryProcessor, QueryAnalysis, QueryValidator
from .embeddings import EmbeddingService, EmbeddingManager, EmbeddingConfig
from .lexical_engine import LexicalSearchEngine, LexicalSearchConfig, LexicalSearchResult
from .semantic_engine import SemanticSearchEngine, SemanticSearchConfig, SemanticSearchResult
from .search_engine import HybridSearchEngine, HybridSearchConfig, HybridSearchResult, FusionStrategy

__all__ = [
    # Query Processing
    "QueryProcessor",
    "QueryAnalysis", 
    "QueryValidator",
    
    # Embeddings
    "EmbeddingService",
    "EmbeddingManager",
    "EmbeddingConfig",
    
    # Lexical Search
    "LexicalSearchEngine",
    "LexicalSearchConfig",
    "LexicalSearchResult",
    
    # Semantic Search
    "SemanticSearchEngine", 
    "SemanticSearchConfig",
    "SemanticSearchResult",
    
    # Hybrid Search
    "HybridSearchEngine",
    "HybridSearchConfig",
    "HybridSearchResult",
    "FusionStrategy"
]