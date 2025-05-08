from datetime import datetime, timedelta
from typing import Any, Union, Optional
import logging

from jose import jwt
from passlib.context import CryptContext

from config_service.config import settings

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vérification de la disponibilité de bcrypt
try:
    import bcrypt
    logger.info(f"bcrypt version: {bcrypt.__version__}")
except ImportError:
    logger.error("bcrypt module not found. Please install it with 'pip install bcrypt'")
except Exception as e:
    logger.error(f"Error importing bcrypt: {str(e)}")

# Configuration de passlib avec des schemes de repli
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)

ALGORITHM = "HS256"


def create_access_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        return False


def get_password_hash(password: str) -> str:
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Error hashing password: {str(e)}")
        # Utiliser un algorithme de repli en cas d'erreur avec bcrypt
        import hashlib
        logger.warning("Using fallback SHA-256 hashing - THIS IS NOT RECOMMENDED FOR PRODUCTION")
        return hashlib.sha256(password.encode()).hexdigest()