"""
Search Service - Service de recherche simplifié
==============================================

Service de recherche unifié avec un seul endpoint pour toutes les requêtes.
Architecture simplifiée pour preuve de concept rapide.

Modules principaux:
- api.routes: Endpoints FastAPI
- core.search_engine: Moteur de recherche unifié
- core.query_builder: Construction des requêtes Elasticsearch
- models: Modèles de requête et réponse
- config: Configuration du service
"""

__version__ = "1.0.0"
__title__ = "Search Service"
__description__ = "Service de recherche simplifié pour les transactions"