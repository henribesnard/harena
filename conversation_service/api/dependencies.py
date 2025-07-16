from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging
import time

from ..config.settings import settings

logger = logging.getLogger(__name__)

# Sécurité API (optionnel pour MVP)
security = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[dict]:
    """
    Validation optionnelle du token d'authentification
    Pour le MVP, cette fonction peut être désactivée
    """
    # TODO: Implémenter validation JWT si nécessaire
    # Pour le MVP, on retourne None (pas d'auth)
    return None

async def validate_request_size(request: Request):
    """Valide la taille de la requête"""
    
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length:
            content_length = int(content_length)
            # Limite à 1MB
            if content_length > 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail="Request too large"
                )

async def rate_limit_check(request: Request):
    """
    Vérification simple du rate limiting
    Pour le MVP, cette fonction peut être désactivée
    """
    # TODO: Implémenter rate limiting avec Redis si nécessaire
    # Pour le MVP, on skip cette vérification
    pass

class RequestTimer:
    """Context manager pour mesurer le temps de traitement"""
    
    def __init__(self, request: Request):
        self.request = request
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"Request {self.request.method} {self.request.url.path} - {duration:.3f}s")

async def get_request_timer(request: Request) -> RequestTimer:
    """Dépendance pour mesurer le temps de traitement"""
    return RequestTimer(request)

# Validation des headers requis
async def validate_headers(request: Request):
    """Valide les headers requis"""
    
    # Content-Type pour POST
    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            raise HTTPException(
                status_code=415,
                detail="Content-Type must be application/json"
            )
    
    # User-Agent optionnel mais recommandé
    user_agent = request.headers.get("user-agent", "")
    if not user_agent:
        logger.warning(f"Request without User-Agent from {request.client.host}")

# Dépendance combinée pour les endpoints principaux
async def common_dependencies(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
    _: None = Depends(validate_request_size),
    __: None = Depends(rate_limit_check),
    ___: None = Depends(validate_headers)
):
    """
    Dépendances communes pour tous les endpoints
    
    Returns:
        dict: Informations de contexte de la requête
    """
    
    return {
        "request": request,
        "user": current_user,
        "client_ip": request.client.host,
        "user_agent": request.headers.get("user-agent", ""),
        "timestamp": time.time()
    }

# Dépendance spécifique pour les endpoints de conversation
async def conversation_dependencies(
    context: dict = Depends(common_dependencies)
):
    """
    Dépendances spécifiques aux endpoints de conversation
    
    Returns:
        dict: Contexte enrichi pour la conversation
    """
    
    # Vérifications additionnelles pour les conversations
    request = context["request"]
    
    # Log de la requête de conversation
    logger.info(f"Conversation request from {context['client_ip']}")
    
    return context

# Dépendance pour les endpoints système (health, metrics, config)
async def system_dependencies(
    request: Request,
    _: None = Depends(validate_request_size)
):
    """
    Dépendances allégées pour les endpoints système
    
    Returns:
        dict: Contexte minimal pour les endpoints système
    """
    
    return {
        "request": request,
        "client_ip": request.client.host,
        "timestamp": time.time()
    }