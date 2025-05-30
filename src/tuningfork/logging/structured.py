"""Structured logging implementation for TuningFork.

This module provides structured logging capabilities with context management,
correlation IDs, and consistent formatting across the system.

Classes:
    StructuredLogger: Main structured logging interface
    LogContext: Context manager for log correlation
    ContextFilter: Filter for adding context to log records

Example:
    >>> logger = StructuredLogger("database.connector")
    >>> with logger.context(database_id="prod_db", operation="analyze"):
    ...     logger.info("Starting analysis", table_count=150)
    ...     logger.warning("Large table detected", table="user_events", size_gb=45.2)
"""

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Union
from datetime import datetime

import structlog

from ..core.exceptions import TuningForkException
from ..core.utils import ValidationUtils


class LogContext:
    """Thread-local context for log correlation and metadata.
    
    This class manages context information that should be included
    in all log messages within a specific execution context.
    
    Example:
        >>> context = LogContext()
        >>> context.set("request_id", "req_123")
        >>> context.set("user_id", "user_456")
        >>> print(context.get_all())
        {'request_id': 'req_123', 'user_id': 'user_456'}
    """
    
    def __init__(self) -> None:
        """Initialize log context with thread-local storage."""
        self._local = threading.local()
    
    def set(self, key: str, value: Any) -> None:
        """Set context value.
        
        Args:
            key: Context key
            value: Context value
        """
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        self._local.context[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get context value.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Context value or default
        """
        if not hasattr(self._local, 'context'):
            return default
        return self._local.context.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all context values.
        
        Returns:
            Dictionary of all context values
        """
        if not hasattr(self._local, 'context'):
            return {}
        return self._local.context.copy()
    
    def clear(self) -> None:
        """Clear all context values."""
        if hasattr(self._local, 'context'):
            self._local.context.clear()
    
    def update(self, context: Dict[str, Any]) -> None:
        """Update context with multiple values.
        
        Args:
            context: Dictionary of context values to add
        """
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        self._local.context.update(context)


class ContextFilter(logging.Filter):
    """Logging filter that adds context information to log records.
    
    This filter automatically adds thread-local context information
    to all log records that pass through it.
    """
    
    def __init__(self, context: LogContext) -> None:
        """Initialize context filter.
        
        Args:
            context: LogContext instance to use for context data
        """
        super().__init__()
        self._context = context
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True (always allow record through)
        """
        # Add context information to record
        context_data = self._context.get_all()
        for key, value in context_data.items():
            # Ensure we don't override existing record attributes
            if not hasattr(record, key):
                setattr(record, key, value)
        
        # Add standard TuningFork metadata
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = self._context.get('correlation_id', 'unknown')
        
        if not hasattr(record, 'component'):
            record.component = getattr(record, 'name', record.name)
        
        if not hasattr(record, 'timestamp_iso'):
            record.timestamp_iso = datetime.fromtimestamp(record.created).isoformat()
        
        return True


class StructuredLogger:
    """Structured logger with context management and correlation.
    
    This class provides structured logging capabilities with automatic
    context correlation, performance tracking, and consistent formatting.
    
    Attributes:
        name: Logger name
        level: Current log level
        context: Thread-local context manager
        
    Example:
        >>> logger = StructuredLogger("database.analyzer")
        >>> logger.set_level("INFO")
        >>> with logger.context(operation="analyze", database_id="prod_db"):
        ...     logger.info("Analysis started", table_count=100)
        ...     logger.error("Analysis failed", error="timeout", duration_ms=30000)
    """
    
    def __init__(
        self,
        name: str,
        *,
        level: str = "INFO",
        enable_correlation: bool = True,
        auto_correlation: bool = True,
    ) -> None:
        """Initialize structured logger.
        
        Args:
            name: Logger name (typically module name)
            level: Initial log level
            enable_correlation: Whether to enable correlation IDs
            auto_correlation: Whether to auto-generate correlation IDs
        """
        self.name = name
        self._enable_correlation = enable_correlation
        self._auto_correlation = auto_correlation
        
        # Set up structlog logger
        self._logger = structlog.get_logger(name)
        
        # Set up context management
        self._context = LogContext()
        
        # Set up standard Python logger for integration
        self._stdlib_logger = logging.getLogger(name)
        self._stdlib_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Add context filter
        context_filter = ContextFilter(self._context)
        self._stdlib_logger.addFilter(context_filter)
        
        # Initialize correlation if enabled
        if self._enable_correlation and self._auto_correlation:
            self._ensure_correlation_id()
    
    def _ensure_correlation_id(self) -> str:
        """Ensure correlation ID exists in context.
        
        Returns:
            Current or newly generated correlation ID
        """
        correlation_id = self._context.get('correlation_id')
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
            self._context.set('correlation_id', correlation_id)
        return correlation_id
    
    def _prepare_event_dict(self, **kwargs: Any) -> Dict[str, Any]:
        """Prepare event dictionary with context and metadata.
        
        Args:
            **kwargs: Event data
            
        Returns:
            Prepared event dictionary
        """
        event_dict = {
            'timestamp': time.time(),
            'logger': self.name,
        }
        
        # Add context information
        event_dict.update(self._context.get_all())
        
        # Add correlation ID if enabled
        if self._enable_correlation:
            event_dict['correlation_id'] = self._ensure_correlation_id()
        
        # Add event-specific data
        event_dict.update(kwargs)
        
        return event_dict
    
    @contextmanager
    def context(self, **context_data: Any) -> Generator[None, None, None]:
        """Context manager for adding temporary context data.
        
        Args:
            **context_data: Context data to add temporarily
            
        Yields:
            None
            
        Example:
            >>> with logger.context(user_id="123", operation="login"):
            ...     logger.info("User login attempt")
            ...     # All logs in this block will include user_id and operation
        """
        # Save current context
        old_context = self._context.get_all()
        
        try:
            # Add new context
            self._context.update(context_data)
            yield
        finally:
            # Restore old context
            self._context.clear()
            self._context.update(old_context)
    
    def bind(self, **context_data: Any) -> 'StructuredLogger':
        """Create new logger instance with bound context.
        
        Args:
            **context_data: Context data to bind
            
        Returns:
            New logger instance with bound context
            
        Example:
            >>> db_logger = logger.bind(database_id="prod_db", component="analyzer")
            >>> db_logger.info("Processing started")  # Will include bound context
        """
        # Create new logger instance
        bound_logger = StructuredLogger(
            self.name,
            level=self.get_level(),
            enable_correlation=self._enable_correlation,
            auto_correlation=False,  # Don't auto-generate for bound loggers
        )
        
        # Copy current context and add bound data
        current_context = self._context.get_all()
        current_context.update(context_data)
        bound_logger._context.update(current_context)
        
        return bound_logger
    
    def set_level(self, level: str) -> None:
        """Set logging level.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if not ValidationUtils.validate_identifier(level):
            raise TuningForkException(
                f"Invalid log level: {level}",
                code="INVALID_LOG_LEVEL"
            )
        
        log_level = getattr(logging, level.upper(), None)
        if log_level is None:
            raise TuningForkException(
                f"Unknown log level: {level}",
                code="UNKNOWN_LOG_LEVEL"
            )
        
        self._stdlib_logger.setLevel(log_level)
    
    def get_level(self) -> str:
        """Get current logging level.
        
        Returns:
            Current log level name
        """
        level_num = self._stdlib_logger.getEffectiveLevel()
        return logging.getLevelName(level_num)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message.
        
        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        event_dict = self._prepare_event_dict(message=message, level="debug", **kwargs)
        self._logger.debug(message, **event_dict)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message.
        
        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        event_dict = self._prepare_event_dict(message=message, level="info", **kwargs)
        self._logger.info(message, **event_dict)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message.
        
        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        event_dict = self._prepare_event_dict(message=message, level="warning", **kwargs)
        self._logger.warning(message, **event_dict)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message.
        
        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        event_dict = self._prepare_event_dict(message=message, level="error", **kwargs)
        self._logger.error(message, **event_dict)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message.
        
        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        event_dict = self._prepare_event_dict(message=message, level="critical", **kwargs)
        self._logger.critical(message, **event_dict)
    
    def exception(self, message: str, exc_info: bool = True, **kwargs: Any) -> None:
        """Log exception with traceback.
        
        Args:
            message: Log message
            exc_info: Whether to include exception info
            **kwargs: Additional structured data
        """
        event_dict = self._prepare_event_dict(
            message=message, 
            level="error", 
            **kwargs  # Removed exc_info from here
        )
        self._logger.error(message, exc_info=exc_info, **event_dict)
    
    def log_operation_start(
        self,
        operation: str,
        **context: Any
    ) -> Dict[str, Any]:
        """Log operation start with timing context.
        
        Args:
            operation: Operation name
            **context: Operation context
            
        Returns:
            Operation context for completion logging
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        operation_context = {
            'operation_id': operation_id,
            'operation': operation,
            'start_time': start_time,
            **context
        }
        
        self.info(
            "Operation started",
            **operation_context
        )
        
        return operation_context
    
    def log_operation_success(
        self,
        operation_context: Dict[str, Any],
        **results: Any
    ) -> None:
        """Log successful operation completion.
        
        Args:
            operation_context: Context from log_operation_start
            **results: Operation results
        """
        end_time = time.time()
        duration_ms = (end_time - operation_context['start_time']) * 1000
        
        self.info(
            "Operation completed successfully",
            duration_ms=duration_ms,
            **operation_context,
            **results
        )
    
    def log_operation_failure(
        self,
        operation_context: Dict[str, Any],
        error: Exception,
        **error_context: Any
    ) -> None:
        """Log operation failure.
        
        Args:
            operation_context: Context from log_operation_start
            error: Exception that occurred
            **error_context: Additional error context
        """
        end_time = time.time()
        duration_ms = (end_time - operation_context['start_time']) * 1000
        
        self.error(
            "Operation failed",
            duration_ms=duration_ms,
            error=str(error),
            error_type=type(error).__name__,
            **operation_context,
            **error_context
        )
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for this logger.
        
        Args:
            correlation_id: Correlation ID to use
        """
        if not self._enable_correlation:
            return
        
        self._context.set('correlation_id', correlation_id)
    
    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID.
        
        Returns:
            Current correlation ID or None
        """
        if not self._enable_correlation:
            return None
        
        return self._context.get('correlation_id')
    
    def clear_context(self) -> None:
        """Clear all context data."""
        self._context.clear()
        
        # Restore correlation ID if auto-generation is enabled
        if self._enable_correlation and self._auto_correlation:
            self._ensure_correlation_id()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context data.
        
        Returns:
            Dictionary of current context data
        """
        return self._context.get_all()
    
    def __repr__(self) -> str:
        """Return string representation of logger."""
        return (
            f"StructuredLogger("
            f"name={self.name!r}, "
            f"level={self.get_level()!r}, "
            f"correlation={self._enable_correlation})"
        )