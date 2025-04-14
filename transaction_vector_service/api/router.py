# transaction_vector_service/api/router.py
"""
Central router for the Transaction Vector Service API.

This module collects and organizes all API endpoints into a cohesive router structure,
which is then mounted to the main FastAPI application.
"""

from fastapi import APIRouter

from .endpoints.transactions import router as transactions_router

# Create the main API router
api_router = APIRouter()

# Include sub-routers for specific resources
api_router.include_router(
    transactions_router,
    prefix="/transactions",
    tags=["transactions"]
)

# Add additional endpoint routers as needed, for example:
# api_router.include_router(
#     merchants_router,
#     prefix="/merchants",
#     tags=["merchants"]
# )

# api_router.include_router(
#     insights_router,
#     prefix="/insights",
#     tags=["insights"]
# )