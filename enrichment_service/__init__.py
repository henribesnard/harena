"""
Service d'enrichissement et d'indexation pour Harena - Elasticsearch uniquement.

Ce service est responsable de la structuration des données financières
et de leur indexation dans Elasticsearch pour des recherches lexicales optimisées.
"""

__version__ = "2.0.0-elasticsearch"
__all__ = ["__version__"]

# ============================================================
# enrichment_service/api/__init__.py

"""
Module API pour le service d'enrichissement Elasticsearch.

Contient les routes et endpoints REST pour l'enrichissement et l'indexation
des transactions dans Elasticsearch.
"""

__all__ = []

# ============================================================
# enrichment_service/core/__init__.py

"""
Module core contenant la logique métier de l'enrichissement.

Inclut le traitement des transactions et la structuration des données
pour l'indexation Elasticsearch.
"""

__all__ = []

# ============================================================
# enrichment_service/storage/__init__.py

"""
Module de stockage pour Elasticsearch.

Fournit l'interface pour interagir avec Elasticsearch
pour l'indexation et la gestion des documents.
"""

__all__ = []