import os
from sys import exit

# Ensure required environment variables for settings
os.environ.setdefault("SECRET_KEY", "a" * 32 + "b" * 32)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")

from user_service.core.security import create_access_token
from conversation_service.api.middleware.auth_middleware import JWTValidator


def main() -> int:
    token = create_access_token(subject=1)
    validator = JWTValidator()
    result = validator.validate_token(token)
    if result.success:
        print("COMPATIBILITÉ TOTALE")
        return 0
    return 1


if __name__ == "__main__":
    try:
        exit(main())
    except Exception:
        exit(1)
