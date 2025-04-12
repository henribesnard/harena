"""
Logging utility functions.

This module provides utilities for configuring and using logging
throughout the application.
"""

import logging
import json
import sys
import os
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Union


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format.
    
    This formatter is useful for structured logging that can be
    easily parsed by log management systems.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage()
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra attributes from record
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", 
                          "filename", "funcName", "id", "levelname", "levelno", 
                          "lineno", "module", "msecs", "message", "msg", 
                          "name", "pathname", "process", "processName", 
                          "relativeCreated", "stack_info", "thread", "threadName"]:
                log_data[key] = value
        
        return json.dumps(log_data)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    json_format: bool = True
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        console: Whether to output logs to console
        json_format: Whether to format logs as JSON
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Add file handler if specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if specified
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def log_exception(logger: logging.Logger, exception: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log an exception with context information.
    
    Args:
        logger: Logger to use
        exception: Exception to log
        context: Additional context information
    """
    if context is None:
        context = {}
    
    # Extract exception details
    exc_type = type(exception).__name__
    exc_message = str(exception)
    exc_traceback = traceback.format_exc()
    
    # Prepare log message
    log_context = {
        "exception_type": exc_type,
        "exception_message": exc_message,
        "traceback": exc_traceback,
        **context
    }
    
    # Log the exception
    logger.error(f"Exception: {exc_type}: {exc_message}", extra=log_context)


def log_api_request(
    logger: logging.Logger,
    method: str,
    url: str,
    status_code: Optional[int] = None,
    duration_ms: Optional[float] = None,
    request_id: Optional[str] = None,
    user_id: Optional[Union[str, int]] = None,
    request_data: Optional[Dict[str, Any]] = None,
    response_data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> None:
    """
    Log an API request with detailed information.
    
    Args:
        logger: Logger to use
        method: HTTP method
        url: Request URL
        status_code: Response status code
        duration_ms: Request duration in milliseconds
        request_id: Request ID for tracing
        user_id: User ID associated with the request
        request_data: Request data (will be sanitized)
        response_data: Response data (will be sanitized)
        error: Error message if any
    """
    # Sanitize request and response data
    safe_request_data = None
    if request_data:
        # Remove sensitive fields
        safe_request_data = {}
        for key, value in request_data.items():
            if key.lower() in ["password", "secret", "token", "key", "authorization"]:
                safe_request_data[key] = "***REDACTED***"
            else:
                safe_request_data[key] = value
    
    # Prepare log context
    context = {
        "http_method": method,
        "url": url,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "request_id": request_id,
        "user_id": user_id
    }
    
    if safe_request_data:
        context["request_data"] = safe_request_data
    
    if response_data and status_code and status_code < 300:
        # Only include response data for successful requests
        # And limit the size to avoid huge log entries
        context["response_size"] = len(str(response_data))
    
    if error:
        context["error"] = error
    
    # Log the request
    log_message = f"{method} {url}"
    if status_code:
        log_message += f" - {status_code}"
    if duration_ms:
        log_message += f" - {duration_ms:.2f}ms"
    if error:
        log_message += f" - Error: {error}"
    
    # Choose log level based on status code
    if status_code is None or status_code < 400:
        logger.info(log_message, extra=context)
    elif status_code < 500:
        logger.warning(log_message, extra=context)
    else:
        logger.error(log_message, extra=context)


def sanitize_log_data(data: Dict[str, Any], sensitive_fields: List[str] = None) -> Dict[str, Any]:
    """
    Sanitize data for logging by removing sensitive information.
    
    Args:
        data: Data to sanitize
        sensitive_fields: List of sensitive field names to redact
        
    Returns:
        Sanitized data
    """
    if data is None:
        return {}
    
    # Default sensitive fields
    if sensitive_fields is None:
        sensitive_fields = [
            "password", "secret", "token", "key", "authorization",
            "credit_card", "card_number", "cvv", "ssn", "social_security",
            "access_token", "refresh_token"
        ]
    
    # Create a copy of the data
    sanitized = {}
    
    # Sanitize each field
    for key, value in data.items():
        # Check if current key contains any sensitive field name
        is_sensitive = any(field.lower() in key.lower() for field in sensitive_fields)
        
        if is_sensitive:
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized[key] = sanitize_log_data(value, sensitive_fields)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # Recursively sanitize lists of dictionaries
            sanitized[key] = [sanitize_log_data(item, sensitive_fields) for item in value]
        else:
            sanitized[key] = value
    
    return sanitized


class LoggingContext:
    """
    Context manager for temporarily changing log level.
    
    This is useful for debugging specific parts of code.
    """
    
    def __init__(self, logger: logging.Logger, level: int):
        """
        Initialize the context manager.
        
        Args:
            logger: Logger to modify
            level: Logging level to set temporarily
        """
        self.logger = logger
        self.level = level
        self.old_level = logger.level
    
    def __enter__(self):
        """Enter the context manager, setting temporary log level."""
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, restoring original log level."""
        self.logger.setLevel(self.old_level)


class BulkLogHandler:
    """
    Handler for collecting and logging multiple messages in bulk.
    
    This is useful for high-volume log events where individual logs
    would be too numerous.
    """
    
    def __init__(self, logger: logging.Logger, max_logs: int = 100, level: str = "INFO"):
        """
        Initialize the bulk log handler.
        
        Args:
            logger: Logger to use
            max_logs: Maximum number of logs to collect before flushing
            level: Default log level
        """
        self.logger = logger
        self.max_logs = max_logs
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.logs = []
    
    def add(self, message: str, level: Optional[str] = None, **kwargs):
        """
        Add a log message to the collection.
        
        Args:
            message: Log message
            level: Log level override
            **kwargs: Additional context data
        """
        log_level = self.level
        if level:
            log_level = getattr(logging, level.upper(), self.level)
        
        self.logs.append((log_level, message, kwargs))
        
        if len(self.logs) >= self.max_logs:
            self.flush()
    
    def flush(self):
        """Flush all collected logs to the logger."""
        for level, message, kwargs in self.logs:
            self.logger.log(level, message, extra=kwargs)
        
        self.logs = []
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and flush logs."""
        self.flush()
