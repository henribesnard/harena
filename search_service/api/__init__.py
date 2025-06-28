"""
API pour le service de recherche.

Ce module expose les routes et dépendances pour l'API REST
du service de recherche hybride.
"""

from .routes import router
from .dependencies import (
    get_current_user,
    get_admin_user,
    validate_search_request,
    rate_limit,
    validate_user_access,
    validate_search_params,
    check_service_availability,
    SearchPermissions,
    require_search_permission,
    require_admin_permission,
    require_metrics_permission
)

__all__ = [
    "router",
    "get_current_user",
    "get_admin_user", 
    "validate_search_request",
    "rate_limit",
    "validate_user_access",
    "validate_search_params",
    "check_service_availability",
    "SearchPermissions",
    "require_search_permission",
    "require_admin_permission",
    "require_metrics_permission"
]