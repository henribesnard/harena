# transaction_vector_service/config/logging_config.py
"""
Logging configuration for the Transaction Vector Service.

This module configures structured logging with different detail levels
and JSON formatting to facilitate log analysis.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from .settings import settings


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format.
    
    This makes logs easier to parse by log aggregation tools.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Add extra attributes from record
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", 
                          "filename", "funcName", "id", "levelname", "levelno", 
                          "lineno", "module", "msecs", "message", "msg", 
                          "name", "pathname", "process", "processName", 
                          "relativeCreated", "stack_info", "thread", "threadName"]:
                log_data[key] = value
                
        return json.dumps(log_data)


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """
    Configure and set up logging for the application.
    
    Args:
        log_level: Optional override for the log level from settings
        
    Returns:
        Logger instance for the application
    """
    # Set the log level from settings if not overridden
    level = log_level or settings.LOG_LEVEL
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with JSON formatter for structured logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)
    
    # Create the application-specific logger
    logger = logging.getLogger("transaction_vector_service")
    logger.setLevel(numeric_level)
    
    logger.info(f"Logging configured with level: {level}")
    return logger


# Module-level function to inject contextual information into log records
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name, properly configured.
    
    Args:
        name: The name of the logger, typically __name__
        
    Returns:
        A configured Logger instance
    """
    return logging.getLogger(name)