"""
🔍 Search Service - Module Principal
===================================

Point d'entrée du Search Service spécialisé en recherche lexicale pure Elasticsearch.
Architecture hybride séparant Search Service (lexical) et Conversation Service (IA).

Exports principaux:
- Modèles et contrats
- Configuration
- Version et métadonnées
"""

from .models import (
    # Contrats principaux
    SearchServiceQuery, SearchServiceResponse, ContractValidator,
    # Modèles requêtes/réponses
    SimpleLexicalSearchRequest, BaseResponse, ResponseFactory,
    # Filtres
    FilterBuilder, CompositeFilter,
    # Elasticsearch
    ElasticsearchQuery, ElasticsearchQueryBuilder
)

# Métadonnées module
__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Search Service spécialisé recherche lexicale Elasticsearch"

# Exports principaux
__all__ = [
    "SearchServiceQuery", "SearchServiceResponse", "ContractValidator",
    "SimpleLexicalSearchRequest", "BaseResponse", "ResponseFactory",
    "FilterBuilder", "CompositeFilter",
    "ElasticsearchQuery", "ElasticsearchQueryBuilder",
    "__version__"
]