from datetime import datetime, timedelta, timezone
from typing import Any, Union, Optional, List
import logging

from jose import jwt
from passlib.context import CryptContext

from config_service.config import settings

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vérification de la disponibilité de bcrypt
bcrypt_version = "unknown"
try:
    import bcrypt

    # Préférence pour bcrypt.__version__ avec repli sur __about__
    bcrypt_version = getattr(bcrypt, "__version__", None)
    if not bcrypt_version:
        about = getattr(bcrypt, "__about__", {})
        bcrypt_version = getattr(about, "__version__", getattr(about, "get", lambda *a: "unknown")("__version__"))

    logger.info(f"bcrypt version: {bcrypt_version}")
except ImportError:
    logger.error("bcrypt module not found. Please install it with 'pip install bcrypt'")
except Exception as e:
    logger.error(f"Error importing bcrypt: {str(e)}")

# Configuration de passlib avec gestion d'erreur
try:
    pwd_context = CryptContext(
        schemes=["bcrypt"],
        deprecated="auto",
    )
except Exception as e:
    logger.warning(f"Error initializing bcrypt context: {e}")
    # Utiliser un contexte de repli si bcrypt échoue
    pwd_context = CryptContext(
        schemes=["pbkdf2_sha256"],
        deprecated="auto",
    )
    logger.warning("Using pbkdf2_sha256 as fallback password scheme")

ALGORITHM = "HS256"


def create_access_token(
    subject: Union[str, Any],
    permissions: Optional[List[str]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "permissions": permissions or ["chat:write"],
    }
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
        raise  # Il vaut mieux lever l'exception pour le hachage
