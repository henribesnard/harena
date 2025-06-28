"""
🔍 SEARCH SERVICE HARENA

Service de recherche hybride pour les transactions financières.
Combine recherche lexicale (Elasticsearch) et sémantique (Qdrant).

Architecture:
- Recherche lexicale: Bonsai Elasticsearch
- Recherche sémantique: Qdrant avec embeddings OpenAI
- Fusion des résultats avec pondération intelligente
- Cache LRU pour performances optimales

Usage:
    from search_service import create_search_app
    
    app = create_search_app()

Responsabilités:
✅ LECTURE UNIQUEMENT - Recherche dans les données existantes
❌ PAS D'ÉCRITURE - Enrichment géré par enrichment_service
"""

__version__ = "1.0.0"
__author__ = "Harena Finance Team"

# Imports principaux pour faciliter l'utilisation
from search_service.main import create_search_app
from search_service.core.search_engine import HybridSearchEngine
from search_service.models.requests import SearchRequest, AdvancedSearchRequest
from search_service.models.responses import SearchResponse, SearchResultItem

__all__ = [
    "create_search_app",
    "HybridSearchEngine", 
    "SearchRequest",
    "AdvancedSearchRequest",
    "SearchResponse",
    "SearchResultItem"
]