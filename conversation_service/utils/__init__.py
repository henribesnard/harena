"""
Utilities package for Conversation Service MVP.

This package provides utility functions and classes for validation,
metrics collection, caching, and other common functionality used
throughout the conversation service.

Modules:
    - validators: Contract and data validation utilities
    - metrics: Performance monitoring and metrics collection
    - cache: Caching utilities and implementations
    - helpers: Common helper functions

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP
"""

# Import only essential utilities to avoid circular imports
try:
    from .validators import ContractValidator
    VALIDATORS_AVAILABLE = True
except ImportError:
    VALIDATORS_AVAILABLE = False

try:
    from .logging import log_unauthorized_access
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False

__all__ = [
    "ContractValidator" if VALIDATORS_AVAILABLE else "# ContractValidator not available",
    "log_unauthorized_access" if LOGGING_AVAILABLE else "# log_unauthorized_access not available",
]

def check_dependencies():
    """Check if all utility dependencies are available."""
    missing_deps = []
    
    if not VALIDATORS_AVAILABLE:
        missing_deps.append("validators module")
    if not LOGGING_AVAILABLE:
        missing_deps.append("logging module")
    
    if missing_deps:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Some utilities not available: {', '.join(missing_deps)}")
    
    return len(missing_deps) == 0