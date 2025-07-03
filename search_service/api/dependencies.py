"""
Dépendances FastAPI pour le service de recherche.

Ce module définit les dépendances réutilisables pour l'authentification,
la validation et la limitation de taux.
"""
import time
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from sqlalchemy.orm import Session

# Imports pour l'authentification JWT
from db_service.session import get_db
from db_service.models.user import User
from user_service.services.users import get_user_by_id
from config_service.config import settings
from user_service.core.security import ALGORITHM

logger = logging.getLogger(__name__)

# Configuration de sécurité
security = HTTPBearer(auto_error=False)

# Rate limiting simple en mémoire (pour production, utiliser Redis)
rate_limit_storage: Dict[str, Dict[str, Any]] = {}


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Récupère l'utilisateur actuel à partir du token d'authentification JWT.
    
    Utilise la même logique que user_service pour une cohérence d'authentification.
    """
    # Si pas de token, mode développement avec utilisateur par défaut
    if not credentials:
        logger.warning("Aucun token fourni - Mode développement")
        return {
            "id": 34,
            "username": "test_user",
            "is_superuser": False,
            "permissions": ["search:read"],
            "is_active": True
        }
    
    token = credentials.credentials
    
    # Exception pour tokens invalides
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication token",
        headers={"WWW-Authenticate": "Bearer"}
    )
    
    try:
        # Décoder le token JWT
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[ALGORITHM]
        )
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
        # Récupérer l'utilisateur depuis la base de données
        user = get_user_by_id(db, user_id=int(user_id))
        if user is None:
            raise credentials_exception
            
        # Vérifier que l'utilisateur est actif
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        # Retourner les informations utilisateur au format attendu
        return {
            "id": user.id,
            "username": user.email,  # Utiliser l'email comme username
            "is_superuser": getattr(user, 'is_superuser', False),
            "permissions": ["search:read"],  # Permissions de base pour la recherche
            "is_active": user.is_active,
            "email": user.email,
            "first_name": getattr(user, 'first_name', None),
            "last_name": getattr(user, 'last_name', None)
        }
        
    except JWTError as e:
        logger.error(f"Erreur JWT: {e}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'authentification: {e}")
        raise credentials_exception


async def get_current_user_fallback(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Version de fallback sans base de données pour les tests.
    
    À utiliser si la base de données n'est pas disponible.
    """
    if not credentials:
        return {
            "id": 34,
            "username": "test_user",
            "is_superuser": False,
            "permissions": ["search:read"]
        }
    
    token = credentials.credentials
    
    try:
        # Décoder le token JWT sans vérifier la base de données
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[ALGORITHM]
        )
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
            
        return {
            "id": int(user_id),
            "username": f"user_{user_id}",
            "is_superuser": False,
            "permissions": ["search:read"],
            "is_active": True
        }
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Vérifie que l'utilisateur actuel a les droits d'administration.
    
    Args:
        current_user: Utilisateur actuel
        
    Returns:
        Utilisateur avec droits admin
        
    Raises:
        HTTPException: Si l'utilisateur n'est pas admin
    """
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required"
        )
    
    return current_user


async def validate_search_request(request: Request) -> None:
    """
    Valide les paramètres de requête de recherche.
    
    Args:
        request: Requête FastAPI
        
    Raises:
        HTTPException: Si la requête est invalide
    """
    # Validation de base des paramètres de recherche
    query_params = dict(request.query_params)
    
    # Vérifier la longueur de la requête
    if "query" in query_params:
        query = query_params["query"]
        if len(query) > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query too long (max 500 characters)"
            )
        
        if len(query.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
    
    # Vérifier les limites de pagination
    if "limit" in query_params:
        try:
            limit = int(query_params["limit"])
            if limit > 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Limit cannot exceed 100"
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid limit parameter"
            )
    
    # Vérifier les seuils de similarité
    if "similarity_threshold" in query_params:
        try:
            threshold = float(query_params["similarity_threshold"])
            if not 0.1 <= threshold <= 0.95:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Similarity threshold must be between 0.1 and 0.95"
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid similarity threshold"
            )


async def rate_limit(request: Request) -> None:
    """
    Limitation de taux pour les requêtes de recherche.
    
    Implémentation simple en mémoire. En production, utiliser Redis.
    
    Args:
        request: Requête FastAPI
        
    Raises:
        HTTPException: Si la limite est dépassée
    """
    client_ip = request.client.host
    current_time = time.time()
    
    # Nettoyage des anciens enregistrements (plus de 1 minute)
    cutoff_time = current_time - 60
    for ip in list(rate_limit_storage.keys()):
        if rate_limit_storage[ip]["last_request"] < cutoff_time:
            del rate_limit_storage[ip]
    
    # Vérifier/mettre à jour pour l'IP actuelle
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = {
            "count": 1,
            "last_request": current_time,
            "window_start": current_time
        }
    else:
        # Réinitialiser si nouvelle fenêtre (1 minute)
        if current_time - rate_limit_storage[client_ip]["window_start"] > 60:
            rate_limit_storage[client_ip] = {
                "count": 1,
                "last_request": current_time,
                "window_start": current_time
            }
        else:
            rate_limit_storage[client_ip]["count"] += 1
            rate_limit_storage[client_ip]["last_request"] = current_time
    
    # Vérifier la limite (60 requêtes par minute)
    if rate_limit_storage[client_ip]["count"] > 60:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 60 requests per minute."
        )


async def validate_user_access(
    user_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> int:
    """
    Valide que l'utilisateur peut accéder aux données demandées.
    
    Args:
        user_id: ID de l'utilisateur cible
        current_user: Utilisateur actuel
        
    Returns:
        user_id validé
        
    Raises:
        HTTPException: Si l'accès n'est pas autorisé
    """
    # L'utilisateur peut accéder à ses propres données
    if user_id == current_user.get("id"):
        return user_id
    
    # Les admins peuvent accéder aux données de tous les utilisateurs
    if current_user.get("is_superuser", False):
        return user_id
    
    # Vérifier les permissions spéciales (ex: support client)
    permissions = current_user.get("permissions", [])
    if "search:all_users" in permissions:
        return user_id
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Cannot access other user's data"
    )


async def validate_search_params(
    query: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    similarity_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Valide et normalise les paramètres de recherche.
    
    Args:
        query: Terme de recherche
        limit: Limite de résultats
        offset: Décalage de pagination
        similarity_threshold: Seuil de similarité
        
    Returns:
        Paramètres validés et normalisés
        
    Raises:
        HTTPException: Si les paramètres sont invalides
    """
    validated_params = {}
    
    # Validation de la requête
    if query is not None:
        query = query.strip()
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        if len(query) > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query too long (max 500 characters)"
            )
        
        # Vérifier les caractères dangereux
        dangerous_chars = ['<', '>', '&', '"', "'", ';']
        if any(char in query for char in dangerous_chars):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query contains invalid characters"
            )
        
        validated_params["query"] = query
    
    # Validation de la limite
    if limit is not None:
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 100"
            )
        validated_params["limit"] = limit
    
    # Validation de l'offset
    if offset is not None:
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Offset cannot be negative"
            )
        if offset > 10000:  # Limite raisonnable pour éviter les problèmes de performance
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Offset too large (max 10000)"
            )
        validated_params["offset"] = offset
    
    # Validation du seuil de similarité
    if similarity_threshold is not None:
        if not 0.1 <= similarity_threshold <= 0.95:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Similarity threshold must be between 0.1 and 0.95"
            )
        validated_params["similarity_threshold"] = similarity_threshold
    
    return validated_params


async def check_service_availability(request: Request) -> None:
    """
    Vérifie que les services nécessaires sont disponibles.
    
    Args:
        request: Requête FastAPI
        
    Raises:
        HTTPException: Si les services ne sont pas disponibles
    """
    # Importer ici pour éviter les dépendances circulaires
    from search_service.api.routes import elasticsearch_client, qdrant_client, embedding_manager
    
    path = request.url.path
    
    # Vérifier selon le type de endpoint
    if "/lexical" in path and not elasticsearch_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lexical search service unavailable"
        )
    
    if "/semantic" in path and not qdrant_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search service unavailable"
        )
    
    if "/semantic" in path and not embedding_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable"
        )
    
    if "/recommendations" in path and not qdrant_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service unavailable"
        )


class SearchPermissions:
    """Gestionnaire de permissions pour la recherche."""
    
    @staticmethod
    def can_search(user: Dict[str, Any]) -> bool:
        """Vérifie si l'utilisateur peut effectuer des recherches."""
        permissions = user.get("permissions", [])
        return "search:read" in permissions or user.get("is_superuser", False)
    
    @staticmethod
    def can_access_admin_endpoints(user: Dict[str, Any]) -> bool:
        """Vérifie si l'utilisateur peut accéder aux endpoints d'admin."""
        return user.get("is_superuser", False) or "search:admin" in user.get("permissions", [])
    
    @staticmethod
    def can_view_metrics(user: Dict[str, Any]) -> bool:
        """Vérifie si l'utilisateur peut voir les métriques."""
        permissions = user.get("permissions", [])
        return ("metrics:read" in permissions or 
                user.get("is_superuser", False) or
                "search:admin" in permissions)
    
    @staticmethod
    def can_search_for_user(current_user: Dict[str, Any], target_user_id: int) -> bool:
        """Vérifie si l'utilisateur peut rechercher pour un autre utilisateur."""
        # Propres données
        if current_user.get("id") == target_user_id:
            return True
        
        # Admin
        if current_user.get("is_superuser", False):
            return True
        
        # Permission spéciale
        permissions = current_user.get("permissions", [])
        return "search:all_users" in permissions


async def require_permission(permission: str):
    """
    Factory pour créer des dépendances de permission.
    
    Args:
        permission: Permission requise
        
    Returns:
        Fonction de dépendance
    """
    async def permission_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        permissions = current_user.get("permissions", [])
        
        if permission not in permissions and not current_user.get("is_superuser", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
        
        return current_user
    
    return permission_dependency


# Dépendances pré-configurées
require_search_permission = require_permission("search:read")
require_admin_permission = require_permission("search:admin")
require_metrics_permission = require_permission("metrics:read")