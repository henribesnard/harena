# storage/__init__.py
"""
Package de stockage pour le service de recherche.

Ce package fournit les interfaces pour interagir avec les différents
systèmes de stockage (Elasticsearch, Qdrant, Redis).
"""

from search_service.storage.elasticsearch import get_es_client, init_elasticsearch
from search_service.storage.qdrant import get_qdrant_client, init_qdrant
from search_service.storage.cache import get_cache, set_cache, invalidate_cache

__all__ = [
    'get_es_client',
    'init_elasticsearch',
    'get_qdrant_client',
    'init_qdrant',
    'get_cache',
    'set_cache',
    'invalidate_cache'
]

# utils/__init__.py
"""
Package d'utilitaires pour le service de recherche.

Ce package fournit des fonctionnalités communes utilisées
par les différents composants du service de recherche.
"""

from search_service.utils.timing import timer, get_current_request_timings, reset_request_timings
from search_service.utils.metrics import record_search_metrics, get_search_metrics, calculate_metrics_summary

__all__ = [
    'timer',
    'get_current_request_timings',
    'reset_request_timings',
    'record_search_metrics',
    'get_search_metrics',
    'calculate_metrics_summary'
]