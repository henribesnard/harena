from functools import wraps
from typing import Any, Callable
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

def validate_search_request(func: Callable) -> Callable:
    """
    Décorateur pour valider automatiquement les search_request
    """
    @wraps(func)
    async def wrapper(self, search_request, *args, **kwargs):
        # Validation automatique
        if isinstance(search_request, dict):
            logger.warning(f"⚠️ Received dict instead of SearchServiceQuery object")
            search_request = self._convert_dict_to_search_request(search_request)
            logger.info(f"✅ Successfully converted dict to SearchServiceQuery")
        elif not hasattr(search_request, 'query_metadata'):
            raise ValueError(f"Invalid search_request: {type(search_request)}")
        
        # Validation de sécurité
        if not search_request.query_metadata.user_id:
            raise ValueError("user_id is required for security")
        
        return await func(self, search_request, *args, **kwargs)
    return wrapper