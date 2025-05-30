"""TuningFork structured logging framework.

This package provides comprehensive logging capabilities for TuningFork
including structured logging, performance monitoring, and audit trails.

Classes:
    StructuredLogger: Main structured logging interface
    PerformanceLogger: Performance monitoring and timing
    AuditLogger: Change tracking and audit trails
    LoggerFactory: Logger creation and configuration
    LogFormatter: Custom log formatters
    LogHandler: Custom log handlers

Example:
    >>> from tuningfork.logging import get_logger, PerformanceLogger
    >>> logger = get_logger(__name__)
    >>> logger.info("Operation started", operation="database_analysis")
    >>> 
    >>> perf_logger = PerformanceLogger("optimization")
    >>> with perf_logger.measure("query_execution"):
    ...     # Query execution code
    ...     pass
"""

from .factory import LoggerFactory, get_logger, configure_logging
from .formatters import JSONFormatter, TextFormatter, get_formatter
from .handlers import FileHandler, RotatingFileHandler, ConsoleHandler
from .performance import PerformanceLogger, TimingContext
from .audit import AuditLogger, AuditEvent
from .structured import StructuredLogger, LogContext

__all__ = [
    # Factory and configuration
    "LoggerFactory",
    "get_logger", 
    "configure_logging",
    
    # Formatters
    "JSONFormatter",
    "TextFormatter",
    "get_formatter",
    
    # Handlers
    "FileHandler",
    "RotatingFileHandler", 
    "ConsoleHandler",
    
    # Performance logging
    "PerformanceLogger",
    "TimingContext",
    
    # Audit logging
    "AuditLogger",
    "AuditEvent",
    
    # Structured logging
    "StructuredLogger",
    "LogContext",
]