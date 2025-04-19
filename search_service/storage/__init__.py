"""
Package de stockage pour le service de recherche.

Ce package fournit les interfaces pour interagir avec les différents
systèmes de stockage (BM25, Whoosh, Qdrant).
"""

from search_service.storage.bm25_engine import get_bm25f_engine
from search_service.storage.whoosh_engine import get_whoosh_engine
from search_service.storage.unified_engine import get_unified_engine, SearchEngineType

# Import conditionnel de Qdrant
try:
    from search_service.storage.qdrant import get_qdrant_client
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

__all__ = [
    'get_bm25f_engine',
    'get_whoosh_engine',
    'get_unified_engine',
    'SearchEngineType'
]

if QDRANT_AVAILABLE:
    __all__.append('get_qdrant_client')