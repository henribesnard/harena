"""
Dependencies for metric_service API
"""
from .auth import get_current_user_id, get_optional_user_id

__all__ = [
    "get_current_user_id",
    "get_optional_user_id"
]
