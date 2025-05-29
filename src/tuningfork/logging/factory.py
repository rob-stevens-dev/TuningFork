"""Logger factory and configuration for TuningFork.

This module provides centralized logger creation and configuration
management for the TuningFork logging system.

Classes:
    LoggerFactory: Main logger factory and configuration manager
    LoggerConfig: Configuration for logger instances

Functions:
    get_logger: Convenience function for getting loggers
    configure_logging: Configure logging system globally

Example:
    >>> from tuningfork.logging import get_logger, configure_logging
    >>> configure_logging(level="INFO", format="json")
    >>> logger = get_logger(__name__)
    >>> logger.info("Application started", version="1.0.0")
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import structlog

from .structured import StructuredLogger
from .performance import PerformanceLogger
from .audit import AuditLogger
from .formatters import JSONFormatter, TextFormatter, get_formatter
from .handlers import ConsoleHandler, FileHandler, RotatingFileHandler
from ..config.models import LoggingConfig
from ..core.exceptions import TuningForkException, ValidationError


@dataclass
class LoggerConfig:
    """Configuration for logger instances.
    
    Attributes:
        level: Log level
        format: Log format (json, text)
        console_output: Enable console output
        file_output: Enable file output
        file_path: Log file path
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files to keep
        structured: Enable structured logging
        performance_tracking: Enable performance tracking
        audit_logging: Enable audit logging
        correlation_ids: Enable correlation ID tracking
    """
    level: str = "INFO"
    format: str = "json"
    console_output: bool = True
    file_output: bool = False
    file_path: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    structured: bool = True
    performance_tracking: bool = False
    audit_logging: bool = False
    correlation_ids: bool = True


class LoggerFactory:
    """Factory for creating and configuring TuningFork loggers.
    
    This class provides centralized logger creation and configuration
    management with support for structured logging, performance monitoring,
    and audit trails.
    
    Attributes:
        config: Default logger configuration
        initialized: Whether factory has been initialized
        loggers: Cache of created loggers
        
    Example:
        >>> factory = LoggerFactory()
        >>> factory.configure_from_config(logging_config)
        >>> logger = factory.get_logger("database.connector")
        >>> perf_logger = factory.get_performance_logger("operations")
    """
    
    def __init__(self, config: Optional[LoggerConfig] = None) -> None:
        """Initialize logger factory.
        
        Args:
            config: Default logger configuration
        """
        self.config = config or LoggerConfig()
        self.initialized = False
        self._loggers: Dict[str, StructuredLogger] = {}
        self._performance_loggers: Dict[str, PerformanceLogger] = {}
        self._audit_loggers: Dict[str, AuditLogger] = {}
        
        # Structlog configuration state
        self._structlog_configured = False
    
    def configure_from_config(self, logging_config: LoggingConfig) -> None:
        """Configure factory from LoggingConfig instance.
        
        Args:
            logging_config: TuningFork logging configuration
        """
        self.config = LoggerConfig(
            level=logging_config.level,
            format=logging_config.format,
            console_output=logging_config.console_output,
            file_output=logging_config.file_path is not None,
            file_path=str(logging_config.file_path) if logging_config.file_path else None,
            max_file_size=logging_config.max_file_size,
            backup_count=logging_config.backup_count,
            structured=logging_config.structured,
        )
        
        self._configure_logging_system()
    
    def configure_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Configure factory from dictionary.
        
        Args:
            config_dict: Dictionary containing logging configuration
        """
        # Validate and convert config dictionary
        valid_keys = {
            'level', 'format', 'console_output', 'file_output', 'file_path',
            'max_file_size', 'backup_count', 'structured', 'performance_tracking',
            'audit_logging', 'correlation_ids'
        }
        
        # Filter out invalid keys
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        # Update configuration
        for key, value in filtered_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self._configure_logging_system()
    
    def _configure_logging_system(self) -> None:
        """Configure the underlying logging system."""
        if self.initialized:
            return
        
        # Configure Python standard logging
        self._configure_stdlib_logging()
        
        # Configure structlog
        self._configure_structlog()
        
        self.initialized = True
    
    def _configure_stdlib_logging(self) -> None:
        """Configure Python standard library logging."""
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.upper(), logging.INFO))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.config.console_output:
            console_handler = ConsoleHandler()
            console_handler.setLevel(getattr(logging, self.config.level.upper(), logging.INFO))
            console_handler.setFormatter(get_formatter(self.config.format))
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.config.file_output and self.config.file_path:
            file_path = Path(self.config.file_path)
            
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler
            file_handler = RotatingFileHandler(
                filename=str(file_path),
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.level.upper(), logging.INFO))
            file_handler.setFormatter(get_formatter(self.config.format))
            root_logger.addHandler(file_handler)
        
        # Prevent duplicate log messages
        root_logger.propagate = False
    
    def _configure_structlog(self) -> None:
        """Configure structlog for structured logging."""
        if self._structlog_configured:
            return
        
        # Configure processors
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        # Add JSON processor for structured output
        if self.config.format.lower() == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )
        
        self._structlog_configured = True
    
    def get_logger(
        self, 
        name: str,
        *,
        level: Optional[str] = None,
        enable_correlation: Optional[bool] = None,
    ) -> StructuredLogger:
        """Get or create a structured logger.
        
        Args:
            name: Logger name (typically module name)
            level: Override default log level
            enable_correlation: Override correlation ID setting
            
        Returns:
            StructuredLogger instance
        """
        # Ensure logging system is configured
        if not self.initialized:
            self._configure_logging_system()
        
        # Check cache first
        cache_key = f"{name}_{level}_{enable_correlation}"
        if cache_key in self._loggers:
            return self._loggers[cache_key]
        
        # Create new logger
        logger = StructuredLogger(
            name=name,
            level=level or self.config.level,
            enable_correlation=enable_correlation if enable_correlation is not None else self.config.correlation_ids,
        )
        
        # Cache logger
        self._loggers[cache_key] = logger
        
        return logger
    
    def get_performance_logger(
        self,
        name: str,
        *,
        auto_log: Optional[bool] = None,
        track_metrics: bool = True,
    ) -> PerformanceLogger:
        """Get or create a performance logger.
        
        Args:
            name: Logger name
            auto_log: Whether to automatically log timing results
            track_metrics: Whether to track aggregated metrics
            
        Returns:
            PerformanceLogger instance
        """
        # Ensure logging system is configured
        if not self.initialized:
            self._configure_logging_system()
        
        # Check cache first
        cache_key = f"{name}_{auto_log}_{track_metrics}"
        if cache_key in self._performance_loggers:
            return self._performance_loggers[cache_key]
        
        # Create underlying structured logger
        structured_logger = self.get_logger(f"perf.{name}")
        
        # Create performance logger
        perf_logger = PerformanceLogger(
            name=name,
            auto_log=auto_log if auto_log is not None else True,
            track_metrics=track_metrics,
            logger=structured_logger,
        )
        
        # Cache logger
        self._performance_loggers[cache_key] = perf_logger
        
        return perf_logger
    
    def get_audit_logger(
        self,
        name: str,
        *,
        compliance_mode: bool = False,
        retain_in_memory: bool = True,
        default_retention_days: int = 2555,
    ) -> AuditLogger:
        """Get or create an audit logger.
        
        Args:
            name: Logger name
            compliance_mode: Enable compliance-specific features
            retain_in_memory: Whether to retain events in memory
            default_retention_days: Default retention period in days
            
        Returns:
            AuditLogger instance
        """
        # Ensure logging system is configured
        if not self.initialized:
            self._configure_logging_system()
        
        # Check cache first
        cache_key = f"{name}_{compliance_mode}_{retain_in_memory}_{default_retention_days}"
        if cache_key in self._audit_loggers:
            return self._audit_loggers[cache_key]
        
        # Create underlying structured logger
        structured_logger = self.get_logger(f"audit.{name}")
        
        # Create audit logger
        audit_logger = AuditLogger(
            name=name,
            compliance_mode=compliance_mode,
            retain_in_memory=retain_in_memory,
            default_retention_days=default_retention_days,
            logger=structured_logger,
        )
        
        # Cache logger
        self._audit_loggers[cache_key] = audit_logger
        
        return audit_logger
    
    def set_level(self, level: str, logger_name: Optional[str] = None) -> None:
        """Set log level for specific logger or all loggers.
        
        Args:
            level: New log level
            logger_name: Specific logger name, or None for all loggers
        """
        # Validate level
        if not hasattr(logging, level.upper()):
            raise ValidationError(f"Invalid log level: {level}")
        
        if logger_name:
            # Set level for specific logger
            if logger_name in self._loggers:
                self._loggers[logger_name].set_level(level)
            
            # Also set stdlib logger level
            stdlib_logger = logging.getLogger(logger_name)
            stdlib_logger.setLevel(getattr(logging, level.upper()))
        else:
            # Set level for all loggers
            self.config.level = level
            
            # Update all cached loggers
            for logger in self._loggers.values():
                logger.set_level(level)
            
            # Update root logger
            logging.getLogger().setLevel(getattr(logging, level.upper()))
    
    def add_handler(self, handler: logging.Handler, logger_name: Optional[str] = None) -> None:
        """Add handler to specific logger or all loggers.
        
        Args:
            handler: Logging handler to add
            logger_name: Specific logger name, or None for root logger
        """
        if logger_name:
            stdlib_logger = logging.getLogger(logger_name)
            stdlib_logger.addHandler(handler)
        else:
            logging.getLogger().addHandler(handler)
    
    def remove_handler(self, handler: logging.Handler, logger_name: Optional[str] = None) -> None:
        """Remove handler from specific logger or all loggers.
        
        Args:
            handler: Logging handler to remove
            logger_name: Specific logger name, or None for root logger
        """
        if logger_name:
            stdlib_logger = logging.getLogger(logger_name)
            stdlib_logger.removeHandler(handler)
        else:
            logging.getLogger().removeHandler(handler)
    
    def get_logger_info(self) -> Dict[str, Any]:
        """Get information about configured loggers.
        
        Returns:
            Dictionary containing logger information
        """
        return {
            "config": {
                "level": self.config.level,
                "format": self.config.format,
                "console_output": self.config.console_output,
                "file_output": self.config.file_output,
                "file_path": self.config.file_path,
                "structured": self.config.structured,
            },
            "initialized": self.initialized,
            "loggers": {
                "structured": list(self._loggers.keys()),
                "performance": list(self._performance_loggers.keys()),
                "audit": list(self._audit_loggers.keys()),
            },
            "handlers": [
                {
                    "type": type(handler).__name__,
                    "level": handler.level,
                    "formatter": type(handler.formatter).__name__ if handler.formatter else None,
                }
                for handler in logging.getLogger().handlers
            ],
        }
    
    def shutdown(self) -> None:
        """Shutdown logging system and clean up resources."""
        # Clear logger caches
        self._loggers.clear()
        self._performance_loggers.clear()
        self._audit_loggers.clear()
        
        # Shutdown standard library logging
        logging.shutdown()
        
        self.initialized = False
        self._structlog_configured = False
    
    def __repr__(self) -> str:
        """Return string representation of logger factory."""
        return (
            f"LoggerFactory("
            f"level={self.config.level!r}, "
            f"format={self.config.format!r}, "
            f"initialized={self.initialized})"
        )


# Global logger factory instance
_global_factory = LoggerFactory()


def configure_logging(
    *,
    level: str = "INFO",
    format: str = "json",
    console_output: bool = True,
    file_output: bool = False,
    file_path: Optional[str] = None,
    structured: bool = True,
    **kwargs: Any
) -> None:
    """Configure TuningFork logging system globally.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format (json, text)
        console_output: Enable console output
        file_output: Enable file output
        file_path: Log file path (required if file_output=True)
        structured: Enable structured logging
        **kwargs: Additional configuration options
        
    Example:
        >>> configure_logging(
        ...     level="DEBUG",
        ...     format="json",
        ...     file_output=True,
        ...     file_path="/var/log/tuningfork.log"
        ... )
    """
    config_dict = {
        "level": level,
        "format": format,
        "console_output": console_output,
        "file_output": file_output,
        "file_path": file_path,
        "structured": structured,
        **kwargs
    }
    
    _global_factory.configure_from_dict(config_dict)


def get_logger(
    name: str,
    *,
    level: Optional[str] = None,
    enable_correlation: Optional[bool] = None,
) -> StructuredLogger:
    """Get or create a structured logger using global factory.
    
    Args:
        name: Logger name (typically __name__)
        level: Override default log level
        enable_correlation: Override correlation ID setting
        
    Returns:
        StructuredLogger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started", version="1.0.0")
    """
    return _global_factory.get_logger(
        name=name,
        level=level,
        enable_correlation=enable_correlation,
    )


def get_performance_logger(
    name: str,
    *,
    auto_log: Optional[bool] = None,
    track_metrics: bool = True,
) -> PerformanceLogger:
    """Get or create a performance logger using global factory.
    
    Args:
        name: Logger name
        auto_log: Whether to automatically log timing results
        track_metrics: Whether to track aggregated metrics
        
    Returns:
        PerformanceLogger instance
        
    Example:
        >>> perf_logger = get_performance_logger("database.operations")
        >>> with perf_logger.measure("query_execution"):
        ...     result = database.execute_query("SELECT * FROM users")
    """
    return _global_factory.get_performance_logger(
        name=name,
        auto_log=auto_log,
        track_metrics=track_metrics,
    )


def get_audit_logger(
    name: str,
    *,
    compliance_mode: bool = False,
    retain_in_memory: bool = True,
) -> AuditLogger:
    """Get or create an audit logger using global factory.
    
    Args:
        name: Logger name
        compliance_mode: Enable compliance-specific features
        retain_in_memory: Whether to retain events in memory
        
    Returns:
        AuditLogger instance
        
    Example:
        >>> audit = get_audit_logger("system.configuration")
        >>> audit.log_change(
        ...     action="update_config",
        ...     resource="database.timeout",
        ...     actor="admin@company.com",
        ...     old_value=30,
        ...     new_value=60
        ... )
    """
    return _global_factory.get_audit_logger(
        name=name,
        compliance_mode=compliance_mode,
        retain_in_memory=retain_in_memory,
    )


def get_factory() -> LoggerFactory:
    """Get the global logger factory instance.
    
    Returns:
        Global LoggerFactory instance
    """
    return _global_factory


def shutdown_logging() -> None:
    """Shutdown the global logging system and clean up resources."""
    _global_factory.shutdown()