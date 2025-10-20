"""Generate a test JWT token for API testing."""

from jose import jwt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-ultra-securisee-256-bits")
ALGORITHM = "HS256"


def generate_token(user_id: int, email: str = None, expire_hours: int = 24):
    """
    Generate a JWT token for testing.

    Args:
        user_id: User ID to include in the token
        email: Optional email address
        expire_hours: Token expiration time in hours (default: 24)

    Returns:
        str: JWT token
    """
    if email is None:
        email = f"user{user_id}@example.com"

    # Create payload
    payload = {
        "user_id": user_id,
        "email": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=expire_hours)
    }

    # Generate token
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return token


def verify_token(token: str):
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token to verify

    Returns:
        dict: Decoded payload
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception as e:
        print(f"Error verifying token: {e}")
        return None


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("JWT TOKEN GENERATOR - Conversation Service V2")
    print("=" * 60 + "\n")

    # Get user ID from command line or use default
    if len(sys.argv) > 1:
        try:
            user_id = int(sys.argv[1])
        except ValueError:
            print("Error: user_id must be an integer")
            sys.exit(1)
    else:
        user_id = 12345

    # Get email from command line or use default
    if len(sys.argv) > 2:
        email = sys.argv[2]
    else:
        email = f"user{user_id}@example.com"

    # Generate token
    print(f"Generating token for:")
    print(f"  User ID: {user_id}")
    print(f"  Email:   {email}")
    print(f"  Expires: 24 hours")
    print()

    token = generate_token(user_id, email)

    print("Generated JWT Token:")
    print("-" * 60)
    print(token)
    print("-" * 60)
    print()

    # Verify token
    print("Verifying token...")
    payload = verify_token(token)

    if payload:
        print("✓ Token is valid!")
        print(f"\nPayload:")
        print(f"  user_id: {payload.get('user_id')}")
        print(f"  email:   {payload.get('email')}")
        print(f"  issued:  {datetime.fromtimestamp(payload.get('iat'))}")
        print(f"  expires: {datetime.fromtimestamp(payload.get('exp'))}")
    else:
        print("✗ Token is invalid!")

    print()
    print("=" * 60)
    print("Usage with cURL:")
    print("=" * 60)
    print(f"""
curl -X POST http://localhost:3003/api/v2/conversation/{user_id} \\
  -H "Authorization: Bearer {token}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "query": "Combien j'ai dépensé en restaurants ce mois-ci ?"
  }}'
""")
    print("=" * 60)
    print()
