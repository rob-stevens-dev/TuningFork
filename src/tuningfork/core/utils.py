"""Utility functions for TuningFork operations.

This module provides common utility functions used throughout the TuningFork
system including validation, conversion, timing, and data manipulation utilities.

Functions:
    validate_identifier: Validate identifier strings
    sanitize_sql_identifier: Sanitize SQL identifiers
    format_bytes: Format byte counts into human-readable strings
    format_duration: Format durations into human-readable strings
    retry_with_backoff: Retry operations with exponential backoff
    measure_time: Context manager for measuring execution time
    deep_merge: Deep merge dictionaries
    flatten_dict: Flatten nested dictionaries
    get_nested_value: Get value from nested dictionary

Example:
    >>> with measure_time() as timer:
    ...     # Some operation
    ...     pass
    >>> print(f"Operation took {timer.duration:.2f}s")
"""

import asyncio
import functools
import hashlib
import re
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union
from pathlib import Path
import random

import structlog

from .exceptions import TuningForkException, ValidationError

logger = structlog.get_logger(__name__)

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ValidationUtils:
    """Utility class for validation operations."""
    
    # Common regex patterns
    IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    SQL_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    EMAIL_PATTERN = re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )
    URL_PATTERN = re.compile(
        r"^https?://(?:[-\w.])+(?::[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$"
    )
    VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+)?$")
    
    @classmethod
    def validate_identifier(cls, identifier: str, *, allow_empty: bool = False) -> bool:
        """Validate identifier string.
        
        Args:
            identifier: String to validate as identifier
            allow_empty: Whether to allow empty strings
            
        Returns:
            True if identifier is valid
            
        Example:
            >>> ValidationUtils.validate_identifier("my_var_123")
            True
            >>> ValidationUtils.validate_identifier("123_invalid")
            False
        """
        if not identifier:
            return allow_empty
        
        return bool(cls.IDENTIFIER_PATTERN.match(identifier))
    
    @classmethod
    def validate_sql_identifier(cls, identifier: str) -> bool:
        """Validate SQL identifier string.
        
        Args:
            identifier: String to validate as SQL identifier
            
        Returns:
            True if SQL identifier is valid
        """
        if not identifier:
            return False
        
        # Check basic pattern
        if not cls.SQL_IDENTIFIER_PATTERN.match(identifier):
            return False
        
        # Check against SQL reserved words (basic set)
        reserved_words = {
            "select", "insert", "update", "delete", "from", "where", "join",
            "inner", "outer", "left", "right", "on", "group", "order", "by",
            "having", "union", "all", "distinct", "as", "and", "or", "not",
            "in", "exists", "null", "true", "false", "table", "index", "view",
            "database", "schema", "primary", "foreign", "key", "constraint",
            "create", "alter", "drop", "truncate", "grant", "revoke"
        }
        
        return identifier.lower() not in reserved_words
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email address.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if email is valid
        """
        if not email:
            return False
        
        return bool(cls.EMAIL_PATTERN.match(email))
    
    @classmethod
    def validate_url(cls, url: str) -> bool:
        """Validate URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid
        """
        if not url:
            return False
        
        return bool(cls.URL_PATTERN.match(url))
    
    @classmethod
    def validate_version(cls, version: str) -> bool:
        """Validate semantic version string.
        
        Args:
            version: Version string to validate
            
        Returns:
            True if version is valid
        """
        if not version:
            return False
        
        return bool(cls.VERSION_PATTERN.match(version))
    
    @classmethod
    def validate_port(cls, port: Union[int, str]) -> bool:
        """Validate network port number.
        
        Args:
            port: Port number to validate
            
        Returns:
            True if port is valid
        """
        try:
            port_int = int(port)
            return 1 <= port_int <= 65535
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def validate_file_path(cls, path: Union[str, Path]) -> bool:
        """Validate file path exists and is readable.
        
        Args:
            path: File path to validate
            
        Returns:
            True if file path is valid
        """
        try:
            path_obj = Path(path)
            return path_obj.exists() and path_obj.is_file()
        except (TypeError, OSError):
            return False
    
    @classmethod
    def validate_directory_path(cls, path: Union[str, Path]) -> bool:
        """Validate directory path exists and is accessible.
        
        Args:
            path: Directory path to validate
            
        Returns:
            True if directory path is valid
        """
        try:
            path_obj = Path(path)
            return path_obj.exists() and path_obj.is_dir()
        except (TypeError, OSError):
            return False


class StringUtils:
    """Utility class for string operations."""
    
    @staticmethod
    def sanitize_sql_identifier(identifier: str) -> str:
        """Sanitize string for use as SQL identifier.
        
        Args:
            identifier: String to sanitize
            
        Returns:
            Sanitized SQL identifier
            
        Example:
            >>> StringUtils.sanitize_sql_identifier("my-table name!")
            'my_table_name'
        """
        if not identifier:
            return ""
        
        # Replace non-alphanumeric characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", identifier)
        
        # Ensure it starts with letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = f"_{sanitized}"
        
        # Remove consecutive underscores
        sanitized = re.sub(r"_{2,}", "_", sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        
        return sanitized or "identifier"
    
    @staticmethod
    def camel_to_snake(text: str) -> str:
        """Convert camelCase to snake_case.
        
        Args:
            text: CamelCase string
            
        Returns:
            snake_case string
            
        Example:
            >>> StringUtils.camel_to_snake("myVariableName")
            'my_variable_name'
        """
        # Insert underscore before uppercase letters
        result = re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()
        return result
    
    @staticmethod
    def snake_to_camel(text: str, *, capitalize_first: bool = False) -> str:
        """Convert snake_case to camelCase.
        
        Args:
            text: snake_case string
            capitalize_first: Whether to capitalize first letter
            
        Returns:
            camelCase string
            
        Example:
            >>> StringUtils.snake_to_camel("my_variable_name")
            'myVariableName'
        """
        components = text.split("_")
        if not components:
            return ""
        
        if capitalize_first:
            return "".join(word.capitalize() for word in components)
        else:
            return components[0] + "".join(word.capitalize() for word in components[1:])
    
    @staticmethod
    def truncate_string(text: str, max_length: int, *, suffix: str = "...") -> str:
        """Truncate string to maximum length.
        
        Args:
            text: String to truncate
            max_length: Maximum length including suffix
            suffix: Suffix to add when truncating
            
        Returns:
            Truncated string
            
        Example:
            >>> StringUtils.truncate_string("This is a long string", 10)
            'This is...'
        """
        if len(text) <= max_length:
            return text
        
        if len(suffix) >= max_length:
            return suffix[:max_length]
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def generate_random_string(length: int, *, charset: str = "abcdefghijklmnopqrstuvwxyz0123456789") -> str:
        """Generate random string of specified length.
        
        Args:
            length: Length of string to generate
            charset: Character set to use
            
        Returns:
            Random string
            
        Example:
            >>> len(StringUtils.generate_random_string(10))
            10
        """
        return "".join(random.choices(charset, k=length))
    
    @staticmethod
    def compute_hash(text: str, *, algorithm: str = "sha256") -> str:
        """Compute hash of string.
        
        Args:
            text: String to hash
            algorithm: Hash algorithm to use
            
        Returns:
            Hex digest of hash
            
        Example:
            >>> hash_value = StringUtils.compute_hash("hello world")
            >>> len(hash_value)
            64
        """
        hasher = hashlib.new(algorithm)
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()


class FormatUtils:
    """Utility class for formatting operations."""
    
    @staticmethod
    def format_bytes(bytes_count: Union[int, float], *, decimal_places: int = 2) -> str:
        """Format byte count into human-readable string.
        
        Args:
            bytes_count: Number of bytes
            decimal_places: Number of decimal places
            
        Returns:
            Formatted byte string
            
        Example:
            >>> FormatUtils.format_bytes(1536)
            '1.50 KB'
            >>> FormatUtils.format_bytes(1048576)
            '1.00 MB'
        """
        if bytes_count == 0:
            return "0 B"
        
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        unit_index = 0
        size = float(bytes_count)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        return f"{size:.{decimal_places}f} {units[unit_index]}"
    
    @staticmethod
    def format_duration(seconds: Union[int, float], *, precision: str = "auto") -> str:
        """Format duration into human-readable string.
        
        Args:
            seconds: Duration in seconds
            precision: Precision level ('auto', 'seconds', 'milliseconds', 'microseconds')
            
        Returns:
            Formatted duration string
            
        Example:
            >>> FormatUtils.format_duration(3661)
            '1h 1m 1s'
            >>> FormatUtils.format_duration(0.001)
            '1.00ms'
        """
        if seconds == 0:
            return "0s"
        
        abs_seconds = abs(seconds)
        sign = "-" if seconds < 0 else ""
        
        # Auto-detect precision
        if precision == "auto":
            if abs_seconds >= 1:
                precision = "seconds"
            elif abs_seconds >= 0.001:
                precision = "milliseconds"
            else:
                precision = "microseconds"
        
        if precision == "microseconds":
            microseconds = abs_seconds * 1_000_000
            return f"{sign}{microseconds:.2f}μs"
        elif precision == "milliseconds":
            milliseconds = abs_seconds * 1000
            return f"{sign}{milliseconds:.2f}ms"
        else:
            # Format as hours, minutes, seconds
            hours = int(abs_seconds // 3600)
            minutes = int((abs_seconds % 3600) // 60)
            secs = abs_seconds % 60
            
            parts = []
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            if secs > 0 or not parts:
                if secs == int(secs):
                    parts.append(f"{int(secs)}s")
                else:
                    parts.append(f"{secs:.2f}s")
            
            return sign + " ".join(parts)
    
    @staticmethod
    def format_percentage(value: float, *, decimal_places: int = 1) -> str:
        """Format percentage value.
        
        Args:
            value: Percentage value (0-100)
            decimal_places: Number of decimal places
            
        Returns:
            Formatted percentage string
            
        Example:
            >>> FormatUtils.format_percentage(85.7)
            '85.7%'
        """
        return f"{value:.{decimal_places}f}%"
    
    @staticmethod
    def format_number(number: Union[int, float], *, thousands_separator: str = ",") -> str:
        """Format number with thousands separator.
        
        Args:
            number: Number to format
            thousands_separator: Separator character
            
        Returns:
            Formatted number string
            
        Example:
            >>> FormatUtils.format_number(1234567)
            '1,234,567'
        """
        return f"{number:,}".replace(",", thousands_separator)


class TimerContext:
    """Context manager for measuring execution time."""
    
    def __init__(self) -> None:
        """Initialize timer context."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds.
        
        Returns:
            Duration in seconds or None if not completed
        """
        if self.start_time is None:
            return None
        
        end_time = self.end_time or time.perf_counter()
        return end_time - self.start_time
    
    def __enter__(self) -> "TimerContext":
        """Enter context and start timer."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and stop timer."""
        self.end_time = time.perf_counter()


@contextmanager
def measure_time() -> Generator[TimerContext, None, None]:
    """Context manager for measuring execution time.
    
    Yields:
        TimerContext instance for accessing duration
        
    Example:
        >>> with measure_time() as timer:
        ...     time.sleep(0.1)
        >>> print(f"Duration: {timer.duration:.3f}s")
    """
    timer = TimerContext()
    with timer:
        yield timer


async def retry_with_backoff(
    operation: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    **kwargs: Any,
) -> T:
    """Retry operation with exponential backoff.
    
    Args:
        operation: Async operation to retry
        *args: Positional arguments for operation
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter
        **kwargs: Keyword arguments for operation
        
    Returns:
        Result of successful operation
        
    Raises:
        TuningForkException: If all retries are exhausted
        
    Example:
        >>> async def unreliable_operation():
        ...     # Might fail randomly
        ...     pass
        >>> result = await retry_with_backoff(unreliable_operation, max_retries=5)
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(operation):
                return await operation(*args, **kwargs)
            else:
                return operation(*args, **kwargs)
                
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries:
                break
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            
            # Add jitter if requested
            if jitter:
                delay *= (0.5 + random.random() * 0.5)
            
            logger.warning(
                "Operation failed, retrying",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
                error=str(e),
            )
            
            await asyncio.sleep(delay)
    
    # All retries exhausted
    raise TuningForkException(
        f"Operation failed after {max_retries} retries",
        code="MAX_RETRIES_EXCEEDED",
        context={
            "max_retries": max_retries,
            "operation": getattr(operation, "__name__", str(operation)),
        },
        cause=last_exception,
    )


class DictUtils:
    """Utility class for dictionary operations."""
    
    @staticmethod
    def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary (takes precedence)
            
        Returns:
            Merged dictionary
            
        Example:
            >>> d1 = {"a": {"b": 1, "c": 2}}
            >>> d2 = {"a": {"c": 3, "d": 4}}
            >>> DictUtils.deep_merge(d1, d2)
            {'a': {'b': 1, 'c': 3, 'd': 4}}
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = DictUtils.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def flatten_dict(
        nested_dict: Dict[str, Any],
        *,
        separator: str = ".",
        prefix: str = "",
    ) -> Dict[str, Any]:
        """Flatten nested dictionary.
        
        Args:
            nested_dict: Dictionary to flatten
            separator: Separator for nested keys
            prefix: Prefix for keys
            
        Returns:
            Flattened dictionary
            
        Example:
            >>> nested = {"a": {"b": {"c": 1}}, "d": 2}
            >>> DictUtils.flatten_dict(nested)
            {'a.b.c': 1, 'd': 2}
        """
        flattened = {}
        
        for key, value in nested_dict.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(
                    DictUtils.flatten_dict(
                        value,
                        separator=separator,
                        prefix=new_key,
                    )
                )
            else:
                flattened[new_key] = value
        
        return flattened
    
    @staticmethod
    def unflatten_dict(
        flattened_dict: Dict[str, Any],
        *,
        separator: str = ".",
    ) -> Dict[str, Any]:
        """Unflatten dictionary.
        
        Args:
            flattened_dict: Flattened dictionary
            separator: Separator used in keys
            
        Returns:
            Nested dictionary
            
        Example:
            >>> flattened = {'a.b.c': 1, 'd': 2}
            >>> DictUtils.unflatten_dict(flattened)
            {'a': {'b': {'c': 1}}, 'd': 2}
        """
        result = {}
        
        for key, value in flattened_dict.items():
            keys = key.split(separator)
            current = result
            
            for i, k in enumerate(keys[:-1]):
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        return result
    
    @staticmethod
    def get_nested_value(
        nested_dict: Dict[str, Any],
        key_path: str,
        *,
        separator: str = ".",
        default: Any = None,
    ) -> Any:
        """Get value from nested dictionary using dot notation.
        
        Args:
            nested_dict: Dictionary to search
            key_path: Dot-separated key path
            separator: Separator character
            default: Default value if key not found
            
        Returns:
            Value at key path or default
            
        Example:
            >>> nested = {"a": {"b": {"c": 1}}}
            >>> DictUtils.get_nested_value(nested, "a.b.c")
            1
        """
        keys = key_path.split(separator)
        current = nested_dict
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    @staticmethod
    def set_nested_value(
        nested_dict: Dict[str, Any],
        key_path: str,
        value: Any,
        *,
        separator: str = ".",
    ) -> None:
        """Set value in nested dictionary using dot notation.
        
        Args:
            nested_dict: Dictionary to modify
            key_path: Dot-separated key path
            value: Value to set
            separator: Separator character
            
        Example:
            >>> nested = {}
            >>> DictUtils.set_nested_value(nested, "a.b.c", 1)
            >>> nested
            {'a': {'b': {'c': 1}}}
        """
        keys = key_path.split(separator)
        current = nested_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value


class ListUtils:
    """Utility class for list operations."""
    
    @staticmethod
    def chunk_list(items: List[T], chunk_size: int) -> Generator[List[T], None, None]:
        """Split list into chunks of specified size.
        
        Args:
            items: List to chunk
            chunk_size: Size of each chunk
            
        Yields:
            Chunks of the original list
            
        Example:
            >>> list(ListUtils.chunk_list([1, 2, 3, 4, 5], 2))
            [[1, 2], [3, 4], [5]]
        """
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]
    
    @staticmethod
    def deduplicate_list(items: List[T], *, preserve_order: bool = True) -> List[T]:
        """Remove duplicates from list.
        
        Args:
            items: List with potential duplicates
            preserve_order: Whether to preserve original order
            
        Returns:
            List without duplicates
            
        Example:
            >>> ListUtils.deduplicate_list([1, 2, 2, 3, 1])
            [1, 2, 3]
        """
        if preserve_order:
            seen = set()
            result = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        else:
            return list(set(items))
    
    @staticmethod
    def flatten_list(nested_list: List[Any]) -> List[Any]:
        """Flatten nested list structure.
        
        Args:
            nested_list: Nested list structure
            
        Returns:
            Flattened list
            
        Example:
            >>> ListUtils.flatten_list([[1, 2], [3, [4, 5]], 6])
            [1, 2, 3, 4, 5, 6]
        """
        result = []
        
        for item in nested_list:
            if isinstance(item, list):
                result.extend(ListUtils.flatten_list(item))
            else:
                result.append(item)
        
        return result


def safe_cast(value: Any, target_type: type, *, default: Any = None) -> Any:
    """Safely cast value to target type.
    
    Args:
        value: Value to cast
        target_type: Target type
        default: Default value if casting fails
        
    Returns:
        Cast value or default
        
    Example:
        >>> safe_cast("123", int)
        123
        >>> safe_cast("invalid", int, default=0)
        0
    """
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return default


def require_not_none(value: Optional[T], message: str = "Value cannot be None") -> T:
    """Require value to not be None.
    
    Args:
        value: Value to check
        message: Error message if None
        
    Returns:
        Value if not None
        
    Raises:
        ValidationError: If value is None
        
    Example:
        >>> require_not_none("hello")
        'hello'
        >>> require_not_none(None)
        ValidationError: Value cannot be None
    """
    if value is None:
        raise ValidationError(message, code="VALUE_REQUIRED")
    return value


def coalesce(*values: Optional[T]) -> Optional[T]:
    """Return first non-None value.
    
    Args:
        *values: Values to check
        
    Returns:
        First non-None value or None
        
    Example:
        >>> coalesce(None, "", "hello")
        ''
        >>> coalesce(None, None, "default")
        'default'
    """
    for value in values:
        if value is not None:
            return value
    return None