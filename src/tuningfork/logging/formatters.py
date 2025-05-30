"""Log formatters for TuningFork logging system.

This module provides various log formatters for different output formats
including JSON, text, and custom formats for different use cases.

Classes:
    JSONFormatter: JSON format for structured logging
    TextFormatter: Human-readable text format
    CompactFormatter: Compact text format for production
    DevelopmentFormatter: Verbose format for development

Example:
    >>> formatter = JSONFormatter()
    >>> handler.setFormatter(formatter)
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured log output.
    
    This formatter outputs log records as JSON objects with consistent
    field names and structure for easy parsing by log aggregation systems.
    
    Example:
        {
            "timestamp": "2023-12-07T10:30:45.123456Z",
            "level": "INFO",
            "logger": "tuningfork.database.connector",
            "message": "Database connection established",
            "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
            "database_id": "prod_db",
            "connection_time_ms": 245.7
        }
    """
    
    def __init__(
        self,
        *,
        timestamp_format: str = "iso",
        include_level_name: bool = True,
        include_logger_name: bool = True,
        include_module: bool = False,
        include_function: bool = False,
        include_line_number: bool = False,
        exclude_fields: Optional[list] = None,
    ) -> None:
        """Initialize JSON formatter.
        
        Args:
            timestamp_format: Timestamp format ("iso", "unix", "custom")
            include_level_name: Include log level name
            include_logger_name: Include logger name
            include_module: Include module name
            include_function: Include function name
            include_line_number: Include line number
            exclude_fields: List of fields to exclude from output
        """
        super().__init__()
        self.timestamp_format = timestamp_format
        self.include_level_name = include_level_name
        self.include_logger_name = include_logger_name
        self.include_module = include_module
        self.include_function = include_function
        self.include_line_number = include_line_number
        self.exclude_fields = set(exclude_fields or [])
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        # Start with basic log data
        log_data = {
            "message": record.getMessage(),
        }
        
        # Add timestamp
        if "timestamp" not in self.exclude_fields:
            if self.timestamp_format == "iso":
                log_data["timestamp"] = datetime.fromtimestamp(record.created).isoformat()
            elif self.timestamp_format == "unix":
                log_data["timestamp"] = record.created
            else:
                log_data["timestamp"] = self.formatTime(record)
        
        # Add log level
        if self.include_level_name and "level" not in self.exclude_fields:
            log_data["level"] = record.levelname
        
        # Add logger name
        if self.include_logger_name and "logger" not in self.exclude_fields:
            log_data["logger"] = record.name
        
        # Add module information
        if self.include_module and "module" not in self.exclude_fields:
            log_data["module"] = record.module
        
        # Add function information
        if self.include_function and "function" not in self.exclude_fields:
            log_data["function"] = record.funcName
        
        # Add line number
        if self.include_line_number and "line" not in self.exclude_fields:
            log_data["line"] = record.lineno
        
        # Add exception information if present
        if record.exc_info and "exception" not in self.exclude_fields:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None,
            }
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            # Skip standard logging fields
            if key in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'exc_info', 'exc_text',
                'stack_info'
            }:
                continue
            
            # Skip excluded fields
            if key in self.exclude_fields:
                continue
            
            # Add field if it's serializable
            try:
                json.dumps(value, default=str)
                extra_fields[key] = value
            except (TypeError, ValueError):
                # If value is not JSON serializable, convert to string
                extra_fields[key] = str(value)
        
        # Merge extra fields
        log_data.update(extra_fields)
        
        # Convert to JSON
        try:
            return json.dumps(log_data, default=str, separators=(',', ':'))
        except (TypeError, ValueError) as e:
            # Fallback to basic message if JSON serialization fails
            return json.dumps({
                "message": record.getMessage(),
                "level": record.levelname,
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "error": f"JSON serialization failed: {e}"
            })


class TextFormatter(logging.Formatter):
    """Human-readable text formatter.
    
    This formatter produces human-readable log output with configurable
    format and color support for development and debugging.
    
    Example:
        2023-12-07 10:30:45.123 [INFO] tuningfork.database.connector: Database connection established (correlation_id=550e8400, database_id=prod_db)
    """
    
    def __init__(
        self,
        *,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_extras: bool = True,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S.%f",
        colors: bool = False,
        max_line_length: Optional[int] = None,
    ) -> None:
        """Initialize text formatter.
        
        Args:
            include_timestamp: Include timestamp in output
            include_level: Include log level in output
            include_logger: Include logger name in output
            include_extras: Include extra fields in output
            timestamp_format: Timestamp format string
            colors: Enable colored output
            max_line_length: Maximum line length (truncate if longer)
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_extras = include_extras
        self.timestamp_format = timestamp_format
        self.colors = colors
        self.max_line_length = max_line_length
        
        # Color codes for different log levels
        self.color_codes = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m',     # Reset
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as human-readable text.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        parts = []
        
        # Add timestamp
        if self.include_timestamp:
            if self.timestamp_format == "%Y-%m-%d %H:%M:%S.%f":
                # Custom microsecond formatting
                dt = datetime.fromtimestamp(record.created)
                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond:06d}"[:3]
            else:
                timestamp = datetime.fromtimestamp(record.created).strftime(self.timestamp_format)
            parts.append(timestamp)
        
        # Add log level with optional colors
        if self.include_level:
            level = record.levelname
            if self.colors and level in self.color_codes:
                level = f"{self.color_codes[level]}[{level}]{self.color_codes['RESET']}"
            else:
                level = f"[{level}]"
            parts.append(level)
        
        # Add logger name
        if self.include_logger:
            parts.append(f"{record.name}:")
        
        # Add main message
        message = record.getMessage()
        parts.append(message)
        
        # Add extra fields
        if self.include_extras:
            extras = []
            for key, value in record.__dict__.items():
                # Skip standard logging fields
                if key in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'message', 'exc_info', 'exc_text',
                    'stack_info'
                }:
                    continue
                
                # Format value
                if isinstance(value, str):
                    extras.append(f"{key}={value}")
                else:
                    extras.append(f"{key}={value!r}")
            
            if extras:
                extras_str = "(" + ", ".join(extras) + ")"
                parts.append(extras_str)
        
        # Join parts
        formatted = " ".join(parts)
        
        # Add exception information
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        
        # Truncate if necessary
        if self.max_line_length and len(formatted) > self.max_line_length:
            formatted = formatted[:self.max_line_length - 3] + "..."
        
        return formatted


class CompactFormatter(logging.Formatter):
    """Compact formatter for production environments.
    
    This formatter produces compact log output optimized for
    production environments where log volume is a concern.
    
    Example:
        10:30:45 INFO conn: DB connected (id=prod_db, t=245ms)
    """
    
    def __init__(self, *, include_microseconds: bool = False) -> None:
        """Initialize compact formatter.
        
        Args:
            include_microseconds: Include microseconds in timestamp
        """
        super().__init__()
        self.include_microseconds = include_microseconds
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record in compact format.
        
        Args:
            record: Log record to format
            
        Returns:
            Compact formatted log string
        """
        # Compact timestamp
        dt = datetime.fromtimestamp(record.created)
        if self.include_microseconds:
            timestamp = dt.strftime("%H:%M:%S.%f")[:12]  # Include 3 digits of microseconds
        else:
            timestamp = dt.strftime("%H:%M:%S")
        
        # Compact level
        level = record.levelname[:4]  # First 4 characters
        
        # Compact logger name (last component only)
        logger_parts = record.name.split('.')
        logger = logger_parts[-1] if logger_parts else record.name
        
        # Main message
        message = record.getMessage()
        
        # Build compact log line
        parts = [timestamp, level, f"{logger}:", message]
        
        # Add important extra fields only
        important_fields = ['correlation_id', 'database_id', 'operation', 'duration_ms', 'error']
        extras = []
        
        for field in important_fields:
            if hasattr(record, field):
                value = getattr(record, field)
                if field == 'correlation_id' and value:
                    # Shorten correlation ID
                    extras.append(f"id={value[:8]}")
                elif field == 'duration_ms' and value is not None:
                    extras.append(f"t={value:.0f}ms")
                elif value:
                    extras.append(f"{field}={value}")
        
        if extras:
            parts.append(f"({', '.join(extras)})")
        
        return " ".join(parts)


class DevelopmentFormatter(logging.Formatter):
    """Verbose formatter for development environments.
    
    This formatter produces detailed output optimized for development
    and debugging with full context information and colors.
    """
    
    def __init__(self) -> None:
        """Initialize development formatter."""
        super().__init__()
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green  
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m\033[1m',  # Bold Magenta
            'RESET': '\033[0m',     # Reset
            'BOLD': '\033[1m',      # Bold
            'DIM': '\033[2m',       # Dim
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for development.
        
        Args:
            record: Log record to format
            
        Returns:
            Development-formatted log string
        """
        # Colorized timestamp
        dt = datetime.fromtimestamp(record.created)
        timestamp = f"{self.colors['DIM']}{dt.strftime('%H:%M:%S.%f')[:-3]}{self.colors['RESET']}"
        
        # Colorized level
        level_color = self.colors.get(record.levelname, '')
        level = f"{level_color}[{record.levelname:8s}]{self.colors['RESET']}"
        
        # Logger name with hierarchy
        logger_parts = record.name.split('.')
        if len(logger_parts) > 3:
            # Show first, last two parts for long names
            logger_display = f"{logger_parts[0]}...{'.'.join(logger_parts[-2:])}"
        else:
            logger_display = record.name
        
        logger = f"{self.colors['DIM']}{logger_display:30s}{self.colors['RESET']}"
        
        # Main message
        message = f"{self.colors['BOLD']}{record.getMessage()}{self.colors['RESET']}"
        
        # Context information
        context_parts = []
        
        # File and line info
        if hasattr(record, 'pathname') and hasattr(record, 'lineno'):
            file_info = f"{record.filename}:{record.lineno}"
            context_parts.append(f"📁 {file_info}")
        
        # Function name
        if hasattr(record, 'funcName') and record.funcName != '<module>':
            context_parts.append(f"🔧 {record.funcName}()")
        
        # Thread info for multi-threaded applications
        if hasattr(record, 'threadName') and record.threadName != 'MainThread':
            context_parts.append(f"🧵 {record.threadName}")
        
        # Extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'exc_info', 'exc_text',
                'stack_info'
            }:
                extra_fields[key] = value
        
        if extra_fields:
            extras_str = " ".join(f"{k}={v!r}" for k, v in extra_fields.items())
            context_parts.append(f"📊 {extras_str}")
        
        # Build final message
        main_line = f"{timestamp} {level} {logger} {message}"
        
        if context_parts:
            context_line = f"{self.colors['DIM']}     {'  '.join(context_parts)}{self.colors['RESET']}"
            result = f"{main_line}\n{context_line}"
        else:
            result = main_line
        
        # Add exception information with formatting
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            # Indent exception lines
            exc_lines = exc_text.split('\n')
            formatted_exc = '\n'.join(f"{self.colors['DIM']}     {line}{self.colors['RESET']}" for line in exc_lines)
            result += f"\n{formatted_exc}"
        
        return result


def get_formatter(format_type: str, **kwargs: Any) -> logging.Formatter:
    """Get formatter instance by type.
    
    Args:
        format_type: Formatter type ('json', 'text', 'compact', 'development')
        **kwargs: Additional formatter arguments
        
    Returns:
        Logging formatter instance
        
    Raises:
        ValueError: If format_type is not supported
    """
    format_type = format_type.lower()
    
    if format_type == 'json':
        return JSONFormatter(**kwargs)
    elif format_type == 'text':
        return TextFormatter(**kwargs)
    elif format_type == 'compact':
        return CompactFormatter(**kwargs)
    elif format_type == 'development' or format_type == 'dev':
        return DevelopmentFormatter(**kwargs)
    else:
        raise ValueError(f"Unsupported formatter type: {format_type}")


# Convenience formatters with predefined configurations
DEFAULT_JSON_FORMATTER = JSONFormatter()
DEFAULT_TEXT_FORMATTER = TextFormatter()
COMPACT_FORMATTER = CompactFormatter()
DEVELOPMENT_FORMATTER = DevelopmentFormatter()
COLORED_TEXT_FORMATTER = TextFormatter(colors=True)
PRODUCTION_JSON_FORMATTER = JSONFormatter(
    include_module=False,
    include_function=False,
    include_line_number=False,
    exclude_fields=['thread', 'threadName', 'process', 'processName']
)