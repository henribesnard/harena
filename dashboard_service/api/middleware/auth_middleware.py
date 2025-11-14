"""
Lightweight JWT authentication middleware shared with the front applications.
Inspired by the budget_profiling_service version but simplified for dashboard use cases.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from jose import jwt, JWTError, ExpiredSignatureError
from fastapi import Request
from fastapi.security.utils import get_authorization_scheme_param
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config_service.config import settings

logger = logging.getLogger("dashboard_service.auth")


class JWTAuthMiddleware(BaseHTTPMiddleware):
  """
  Minimal middleware that validates Bearer tokens issued by user_service.
  Skips validation for health/docs endpoints to keep the developer experience smooth.
  """

  PUBLIC_PATHS: Set[str] = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
  }

  def __init__(self, app, enabled: bool = True):
    super().__init__(app)
    self.enabled = enabled and bool(getattr(settings, "SECRET_KEY", None))
    self.algorithm = getattr(settings, "JWT_ALGORITHM", "HS256")
    self.secret_key = getattr(settings, "SECRET_KEY", "")
    if not self.enabled:
      logger.warning("JWTAuthMiddleware désactivé (SECRET_KEY manquant).")

  def _is_public_path(self, path: str) -> bool:
    if path in self.PUBLIC_PATHS:
      return True
    # Autoriser les fichiers statiques de la doc (swagger-ui assets)
    return path.startswith("/static/swagger-ui") or path.startswith("/static/redoc")

  async def dispatch(self, request: Request, call_next):
    if request.method == "OPTIONS":
      return await call_next(request)

    if not self.enabled or self._is_public_path(request.url.path):
      return await call_next(request)

    auth_header: Optional[str] = request.headers.get("Authorization")
    scheme, token = get_authorization_scheme_param(auth_header)

    if not token or scheme.lower() != "bearer":
      return JSONResponse({"detail": "Authorization header missing or invalid"}, status_code=401)

    try:
      payload: Dict[str, Any] = jwt.decode(
        token,
        self.secret_key,
        algorithms=[self.algorithm],
        options={"verify_aud": False},
      )
      request.state.user = {
        "id": payload.get("sub") or payload.get("user_id"),
        "scopes": payload.get("scopes", []),
        "raw": payload,
      }
    except ExpiredSignatureError:
      logger.warning("Token expiré détecté.")
      return JSONResponse({"detail": "Token expiré"}, status_code=401)
    except JWTError as exc:
      logger.error("Token invalide: %s", exc)
      return JSONResponse({"detail": "Token invalide"}, status_code=401)

    response = await call_next(request)
    response.headers["X-Service"] = "dashboard_service"
    return response
