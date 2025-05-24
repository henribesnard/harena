"""
Sécurité et authentification pour le service d'enrichissement.

Ce module fournit les fonctions d'authentification et d'autorisation
pour les endpoints du service d'enrichissement, en s'appuyant sur
le système d'authentification centralisé.
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from datetime import datetime

from db_service.session import get_db
from db_service.models.user import User
from user_service.services.users import get_user_by_id
from enrichment_service.core.config import enrichment_settings
from enrichment_service.core.logging import get_contextual_logger

logger = logging.getLogger(__name__)

# Configuration OAuth2 pour l'enrichissement service
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{enrichment_settings.api_v1_str}/auth/login",
    auto_error=True
)

# Algorithme JWT utilisé (cohérent avec user_service)
ALGORITHM = "HS256"

class EnrichmentPermissions:
    """Classe pour gérer les permissions spécifiques à l'enrichissement."""
    
    # Permissions de base
    READ_ENRICHED_DATA = "enrichment:read"
    WRITE_ENRICHED_DATA = "enrichment:write"
    DELETE_ENRICHED_DATA = "enrichment:delete"
    
    # Permissions administratives
    ADMIN_ENRICHMENT = "enrichment:admin"
    MANAGE_PATTERNS = "enrichment:patterns:manage"
    MANAGE_INSIGHTS = "enrichment:insights:manage"
    MANAGE_SUMMARIES = "enrichment:summaries:manage"
    
    # Permissions de tâches
    TRIGGER_ENRICHMENT = "enrichment:trigger"
    VIEW_TASKS = "enrichment:tasks:view"
    MANAGE_TASKS = "enrichment:tasks:manage"
    
    # Permissions d'export
    EXPORT_DATA = "enrichment:export"
    BULK_OPERATIONS = "enrichment:bulk"

async def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Récupère l'utilisateur actuel à partir du token JWT.
    
    Args:
        db: Session de base de données
        token: Token JWT fourni dans l'en-tête Authorization
        
    Returns:
        User: Utilisateur authentifié
        
    Raises:
        HTTPException: Si le token est invalide ou l'utilisateur n'existe pas
    """
    ctx_logger = get_contextual_logger(__name__, enrichment_type="auth")
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Décoder le token JWT
        payload = jwt.decode(
            token, 
            enrichment_settings.secret_key, 
            algorithms=[ALGORITHM]
        )
        user_id: Optional[str] = payload.get("sub")
        
        if user_id is None:
            ctx_logger.warning("Token JWT sans subject (sub)")
            raise credentials_exception
            
        # Convertir en entier
        try:
            user_id_int = int(user_id)
        except ValueError:
            ctx_logger.warning(f"ID utilisateur invalide dans le token: {user_id}")
            raise credentials_exception
            
    except JWTError as e:
        ctx_logger.warning(f"Erreur de décodage JWT: {str(e)}")
        raise credentials_exception
    
    # Récupérer l'utilisateur depuis la base de données
    user = get_user_by_id(db, user_id=user_id_int)
    if user is None:
        ctx_logger.warning(f"Utilisateur non trouvé: {user_id_int}")
        raise credentials_exception
        
    ctx_logger.debug(f"Utilisateur authentifié: {user.email} (ID: {user.id})")
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Vérifie que l'utilisateur actuel est actif.
    
    Args:
        current_user: Utilisateur récupéré par get_current_user
        
    Returns:
        User: Utilisateur actif
        
    Raises:
        HTTPException: Si l'utilisateur est inactif
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Inactive user"
        )
    return current_user

async def get_current_superuser(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Vérifie que l'utilisateur actuel est un super-utilisateur.
    
    Args:
        current_user: Utilisateur actif
        
    Returns:
        User: Super-utilisateur
        
    Raises:
        HTTPException: Si l'utilisateur n'est pas un super-utilisateur
    """
    if not getattr(current_user, 'is_superuser', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user

def check_user_owns_data(user_id: int, data_user_id: int) -> bool:
    """
    Vérifie que l'utilisateur possède les données demandées.
    
    Args:
        user_id: ID de l'utilisateur authentifié
        data_user_id: ID de l'utilisateur propriétaire des données
        
    Returns:
        bool: True si l'utilisateur possède les données
    """
    return user_id == data_user_id

def require_user_owns_data(
    current_user: User,
    data_user_id: int,
    resource_name: str = "resource"
) -> None:
    """
    Vérifie que l'utilisateur possède les données et lève une exception sinon.
    
    Args:
        current_user: Utilisateur authentifié
        data_user_id: ID de l'utilisateur propriétaire des données
        resource_name: Nom de la ressource pour le message d'erreur
        
    Raises:
        HTTPException: Si l'utilisateur ne possède pas les données
    """
    if not check_user_owns_data(current_user.id, data_user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You don't have permission to access this {resource_name}"
        )

def check_permission(user: User, permission: str) -> bool:
    """
    Vérifie si un utilisateur a une permission spécifique.
    
    Args:
        user: Utilisateur à vérifier
        permission: Permission à vérifier
        
    Returns:
        bool: True si l'utilisateur a la permission
    """
    # Pour l'instant, système simplifié basé sur le statut superuser
    # TODO: Implémenter un système de permissions plus granulaire
    
    # Les super-utilisateurs ont toutes les permissions
    if getattr(user, 'is_superuser', False):
        return True
    
    # Permissions de base pour tous les utilisateurs actifs
    basic_permissions = [
        EnrichmentPermissions.READ_ENRICHED_DATA,
        EnrichmentPermissions.VIEW_TASKS
    ]
    
    if permission in basic_permissions:
        return user.is_active
    
    # Autres permissions nécessitent des droits spéciaux
    return False

def require_permission(permission: str):
    """
    Décorateur de dépendance pour exiger une permission spécifique.
    
    Args:
        permission: Permission requise
        
    Returns:
        Fonction de dépendance FastAPI
    """
    def permission_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if not check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
        return current_user
    
    return permission_checker

class EnrichmentRateLimit:
    """Gestionnaire de limitation de taux pour l'enrichissement."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialise le gestionnaire de limitation de taux.
        
        Args:
            max_requests: Nombre maximum de requêtes par fenêtre
            window_seconds: Durée de la fenêtre en secondes
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[int, List[float]] = {}  # user_id -> timestamps
    
    def is_allowed(self, user_id: int) -> bool:
        """
        Vérifie si une requête est autorisée pour cet utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si la requête est autorisée
        """
        import time
        
        now = time.time()
        
        # Nettoyer les anciennes requêtes
        if user_id in self.requests:
            self.requests[user_id] = [
                timestamp for timestamp in self.requests[user_id]
                if now - timestamp < self.window_seconds
            ]
        else:
            self.requests[user_id] = []
        
        # Vérifier la limite
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        # Enregistrer cette requête
        self.requests[user_id].append(now)
        return True
    
    def get_remaining_requests(self, user_id: int) -> int:
        """
        Retourne le nombre de requêtes restantes pour cet utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            int: Nombre de requêtes restantes
        """
        if user_id not in self.requests:
            return self.max_requests
        
        return max(0, self.max_requests - len(self.requests[user_id]))

# Instance globale de limitation de taux
enrichment_rate_limiter = EnrichmentRateLimit(
    max_requests=enrichment_settings.rate_limit_requests,
    window_seconds=enrichment_settings.rate_limit_period
)

def apply_rate_limit(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Applique la limitation de taux pour les endpoints d'enrichissement.
    
    Args:
        current_user: Utilisateur authentifié
        
    Returns:
        User: Utilisateur si la limite n'est pas atteinte
        
    Raises:
        HTTPException: Si la limite de taux est atteinte
    """
    if not enrichment_settings.rate_limit_enabled:
        return current_user
    
    # Les super-utilisateurs ne sont pas limités
    if getattr(current_user, 'is_superuser', False):
        return current_user
    
    if not enrichment_rate_limiter.is_allowed(current_user.id):
        remaining = enrichment_rate_limiter.get_remaining_requests(current_user.id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. {remaining} requests remaining.",
            headers={"Retry-After": str(enrichment_settings.rate_limit_period)}
        )
    
    return current_user

def get_user_context(user: User) -> Dict[str, Any]:
    """
    Récupère le contexte utilisateur pour les logs et l'audit.
    
    Args:
        user: Utilisateur authentifié
        
    Returns:
        Dict: Contexte utilisateur
    """
    return {
        "user_id": user.id,
        "user_email": user.email,
        "is_superuser": getattr(user, 'is_superuser', False),
        "is_active": user.is_active
    }

def log_user_action(
    user: User,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
):
    """
    Journalise une action utilisateur pour l'audit.
    
    Args:
        user: Utilisateur qui effectue l'action
        action: Type d'action (create, read, update, delete)
        resource_type: Type de ressource (transaction, pattern, insight, etc.)
        resource_id: ID de la ressource concernée
        details: Détails supplémentaires
    """
    ctx_logger = get_contextual_logger(
        __name__,
        user_id=user.id,
        enrichment_type="audit"
    )
    
    log_entry = {
        "user_id": user.id,
        "user_email": user.email,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {}
    }
    
    ctx_logger.info(f"User action: {action} on {resource_type}", extra=log_entry)

class EnrichmentSecurity:
    """Classe utilitaire pour les opérations de sécurité de l'enrichissement."""
    
    @staticmethod
    def sanitize_user_input(text: str, max_length: int = 1000) -> str:
        """
        Nettoie et limite la taille des entrées utilisateur.
        
        Args:
            text: Texte à nettoyer
            max_length: Longueur maximale autorisée
            
        Returns:
            str: Texte nettoyé
        """
        if not text:
            return ""
        
        # Nettoyer les caractères dangereux
        cleaned = text.strip()
        
        # Limiter la longueur
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
        
        return cleaned
    
    @staticmethod
    def validate_user_tags(tags: List[str]) -> List[str]:
        """
        Valide et nettoie les tags utilisateur.
        
        Args:
            tags: Liste des tags à valider
            
        Returns:
            List[str]: Tags validés
        """
        if not tags:
            return []
        
        validated_tags = []
        for tag in tags[:10]:  # Maximum 10 tags
            cleaned_tag = EnrichmentSecurity.sanitize_user_input(tag, 50)
            if cleaned_tag and len(cleaned_tag) >= 2:
                validated_tags.append(cleaned_tag.lower())
        
        return list(set(validated_tags))  # Supprimer les doublons
    
    @staticmethod
    def is_safe_collection_name(collection_name: str) -> bool:
        """
        Vérifie si un nom de collection est sûr.
        
        Args:
            collection_name: Nom de la collection
            
        Returns:
            bool: True si le nom est sûr
        """
        allowed_collections = [
            "enriched_transactions",
            "financial_patterns", 
            "financial_insights",
            "financial_summaries",
            "enriched_accounts"
        ]
        
        return collection_name in allowed_collections

# Dépendances FastAPI couramment utilisées
def get_current_user_with_rate_limit(
    current_user: User = Depends(apply_rate_limit)
) -> User:
    """Dépendance combinée pour l'authentification et la limitation de taux."""
    return current_user

def get_admin_user(
    current_user: User = Depends(get_current_superuser)
) -> User:
    """Dépendance pour les endpoints d'administration."""
    return current_user

def get_enrichment_write_user(
    current_user: User = Depends(require_permission(EnrichmentPermissions.WRITE_ENRICHED_DATA))
) -> User:
    """Dépendance pour les endpoints d'écriture d'enrichissement."""
    return current_user

def get_enrichment_admin_user(
    current_user: User = Depends(require_permission(EnrichmentPermissions.ADMIN_ENRICHMENT))
) -> User:
    """Dépendance pour les endpoints d'administration d'enrichissement."""
    return current_user