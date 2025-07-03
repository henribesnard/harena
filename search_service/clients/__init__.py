"""
Clients pour le service de recherche.

Ce module expose les clients pour Elasticsearch et Qdrant
ainsi que la classe de base commune.
"""

from .elasticsearch_client import ElasticsearchClient
from .qdrant_client import QdrantClient
from .base_client import BaseClient

__all__ = [
    "ElasticsearchClient",
    "QdrantClient", 
    "BaseClient"
]