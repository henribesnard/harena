"""
Modèles pour les réponses de l'API.

Ce module définit les modèles standards pour les réponses
de l'API, incluant les réponses d'erreur et de succès.
"""

from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Generic, TypeVar
from enum import Enum


class ErrorCode(str, Enum):
    """Codes d'erreur standardisés pour l'API."""
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RESOURCE_CONFLICT = "resource_conflict"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SERVICE_UNAVAILABLE = "service_unavailable"
    EXTERNAL_API_ERROR = "external_api_error"
    DATABASE_ERROR = "database_error"
    INTERNAL_SERVER_ERROR = "internal_server_error"


T = TypeVar('T')


class APIResponse(BaseModel, Generic[T]):
    """Réponse standard de l'API pour les requêtes réussies."""
    success: bool = True
    data: Optional[T] = None
    message: Optional[str] = None


class ErrorDetail(BaseModel):
    """Détails d'une erreur."""
    loc: Optional[List[str]] = None
    msg: str
    type: str


class ErrorResponse(BaseModel):
    """Réponse standard de l'API pour les erreurs."""
    success: bool = False
    error: ErrorCode
    message: str
    details: Optional[List[ErrorDetail]] = None
    data: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Réponse paginée."""
    success: bool = True
    data: List[T]
    page: int
    page_size: int
    total: int
    has_more: bool 