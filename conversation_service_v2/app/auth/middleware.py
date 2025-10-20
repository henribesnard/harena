"""Authentication and authorization middleware."""

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime
from typing import Optional
import redis
import os

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "votre-secret-key-ultra-securisee-256-bits")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

security = HTTPBearer()

# Redis client for token blacklist
try:
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        # Use REDIS_URL which includes password
        redis_client = redis.from_url(
            redis_url,
            decode_responses=True
        )
    else:
        # Fallback to individual parameters
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD"),
            db=0,
            decode_responses=True
        )
except Exception as e:
    print(f"Warning: Could not connect to Redis: {e}")
    redis_client = None


class AuthMiddleware:
    """Middleware for authentication and authorization."""

    async def verify_token(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> dict:
        """
        Verify the validity of the JWT token.

        Args:
            credentials: HTTP authorization credentials

        Returns:
            dict: Decoded JWT payload

        Raises:
            HTTPException: 401 if token is invalid/expired
        """
        token = credentials.credentials

        try:
            # Check if token is revoked (blacklist Redis)
            if redis_client and redis_client.exists(f"blacklist:{token}"):
                raise HTTPException(
                    status_code=401,
                    detail="Token révoqué"
                )

            # Decode and validate token
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            # Verify expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise HTTPException(
                    status_code=401,
                    detail="Token expiré"
                )

            return payload

        except JWTError as e:
            raise HTTPException(
                status_code=401,
                detail=f"Token invalide: {str(e)}"
            )

    async def verify_user_access(
        self,
        user_id: int,
        token_payload: dict
    ) -> None:
        """
        Verify that the user_id in the token matches the one in the URL.

        Args:
            user_id: User ID in the URL
            token_payload: Decoded JWT content

        Raises:
            HTTPException: 403 if user_id doesn't match
        """
        # Accept both 'user_id' and 'sub' (JWT standard) formats
        token_user_id = token_payload.get("user_id") or token_payload.get("sub")

        # Convert to int if it's a string
        if token_user_id and isinstance(token_user_id, str):
            try:
                token_user_id = int(token_user_id)
            except ValueError:
                raise HTTPException(
                    status_code=401,
                    detail="user_id dans le token doit être un nombre"
                )

        if token_user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail=f"Accès refusé: user_id du token ({token_user_id}) "
                       f"ne correspond pas à l'URL ({user_id})"
            )

    async def get_current_user_id(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> int:
        """
        Get the current user ID from the JWT token.

        Args:
            credentials: HTTP authorization credentials

        Returns:
            int: User ID from token

        Raises:
            HTTPException: If token is invalid or user_id is missing
        """
        payload = await self.verify_token(credentials)
        # Accept both 'user_id' and 'sub' (JWT standard) formats
        user_id = payload.get("user_id") or payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="user_id manquant dans le token"
            )

        # Convert to int if it's a string
        if isinstance(user_id, str):
            try:
                user_id = int(user_id)
            except ValueError:
                raise HTTPException(
                    status_code=401,
                    detail="user_id dans le token doit être un nombre"
                )

        return user_id
