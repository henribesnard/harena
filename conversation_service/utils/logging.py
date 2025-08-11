import logging
from typing import Optional

logger = logging.getLogger(__name__)


def log_unauthorized_access(
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    reason: str = ""
) -> None:
    """Log unauthorized access attempts for audit purposes.

    Args:
        user_id: Identifier of the user performing the request.
        conversation_id: Targeted conversation identifier.
        reason: Explanation of why access was denied.
    """
    logger.warning(
        "Unauthorized access: user_id=%s conversation_id=%s reason=%s",
        user_id,
        conversation_id,
        reason,
    )
