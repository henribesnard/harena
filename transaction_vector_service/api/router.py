# transaction_vector_service/api/router.py
"""
Routeur central pour l'API du Service de Vecteurs de Transactions.

Ce module rassemble et organise tous les points de terminaison (endpoints) de l'API
dans une structure de routeur cohérente, qui est ensuite montée sur l'application FastAPI principale.
"""

from fastapi import APIRouter

from .endpoints.transactions import router as transactions_router

# Créer le routeur principal de l'API
api_router = APIRouter()

# Inclure le routeur des transactions, qui contient tous les endpoints actuellement implémentés
api_router.include_router(
    transactions_router,
    prefix="/transactions",
    tags=["transactions"]
)

# Remarque : L'endpoint de santé (/health) est défini directement dans main.py,
# et non par ce système de routeur.

# Les routeurs d'endpoints futurs seraient inclus ici une fois implémentés, par exemple :
# - Endpoints pour les commerçants (merchants)
# - Endpoints pour les insights
# - Endpoints pour les catégories