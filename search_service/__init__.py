"""
Service de recherche hybride pour Harena.

Ce service combine recherche lexicale (Elasticsearch) et sémantique (Qdrant)
avec reranking (Cohere) pour optimiser la pertinence des résultats.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]