"""TuningFork exception hierarchy.

This module defines a comprehensive exception hierarchy for TuningFork operations,
providing structured error handling with context and error codes for better
debugging and monitoring.

Classes:
    TuningForkException: Base exception for all TuningFork operations
    ConfigurationError: Configuration related errors
    ConnectionError: Database connection errors
    AnalysisError: Database analysis errors
    OptimizationError: Database optimization errors
    PluginError: Plugin system errors
    ValidationError: Data validation errors

Example:
    >>> try:
    ...     connector.connect()
    ... except DatabaseConnectionError as e:
    ...     logger.error("Connection failed", error_code=e.code, context=e.context)
"""

from typing import Any, Dict, Optional


class TuningForkException(Exception):
    """Base exception for all TuningFork operations.
    
    This base class provides structured error handling with error codes,
    context information, and optional cause tracking for better debugging
    and monitoring in production environments.
    
    Attributes:
        code: Unique error code for categorization
        context: Additional context information about the error
        cause: Original exception that caused this error (if any)
    
    Example:
        >>> raise TuningForkException(
        ...     "Operation failed",
        ...     code="OPERATION_FAILED",
        ...     context={"operation": "analysis", "database_id": "db001"}
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize TuningFork exception.
        
        Args:
            message: Human-readable error description
            code: Unique error code for categorization (defaults to class name)
            context: Additional context information
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.code: str = code or self.__class__.__name__
        self.context: Dict[str, Any] = context or {}
        self.cause: Optional[Exception] = cause
        
    def __str__(self) -> str:
        """Return formatted error message with code."""
        return f"{self.code}: {super().__str__()}"
        
    def __repr__(self) -> str:
        """Return detailed representation of the exception."""
        return (
            f"{self.__class__.__name__}("
            f"message={super().__str__()!r}, "
            f"code={self.code!r}, "
            f"context={self.context!r}, "
            f"cause={self.cause!r})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            "error_type": self.__class__.__name__,
            "message": str(super().__str__()),
            "code": self.code,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
        }


class ConfigurationError(TuningForkException):
    """Configuration related errors.
    
    Raised when configuration is invalid, missing, or cannot be processed.
    This includes validation errors, missing required settings, and
    configuration format issues.
    """
    pass


class ValidationError(ConfigurationError):
    """Data validation errors.
    
    Raised when input data fails validation rules, including type mismatches,
    value constraints, and format requirements.
    """
    pass


class MigrationError(ConfigurationError):
    """Configuration migration errors.
    
    Raised when configuration cannot be migrated between versions or
    when migration process encounters errors.
    """
    pass


class ConnectionError(TuningForkException):
    """Database connection related errors.
    
    Base class for all database connection issues including authentication,
    network connectivity, and connection pool problems.
    """
    pass


class DatabaseConnectionError(ConnectionError):
    """Database connection establishment errors.
    
    Raised when unable to establish connection to database server.
    """
    pass


class AuthenticationError(ConnectionError):
    """Database authentication errors.
    
    Raised when database authentication fails due to invalid credentials
    or insufficient permissions.
    """
    pass


class NetworkError(ConnectionError):
    """Network connectivity errors.
    
    Raised when network-related issues prevent database connection.
    """
    pass


class ConnectionPoolError(ConnectionError):
    """Connection pool management errors.
    
    Raised when connection pool operations fail, including pool exhaustion
    and connection lifecycle issues.
    """
    pass


class AnalysisError(TuningForkException):
    """Database analysis related errors.
    
    Base class for errors during database analysis operations including
    metadata extraction and performance analysis.
    """
    pass


class MetadataError(AnalysisError):
    """Database metadata extraction errors.
    
    Raised when unable to extract required metadata from database.
    """
    pass


class PerformanceError(AnalysisError):
    """Performance analysis errors.
    
    Raised when performance analysis operations fail or produce invalid results.
    """
    pass


class QueryError(AnalysisError):
    """SQL query execution errors.
    
    Raised when SQL queries fail during analysis operations.
    """
    pass


class OptimizationError(TuningForkException):
    """Database optimization related errors.
    
    Base class for errors during optimization operations including
    recommendation generation and change application.
    """
    pass


class RecommendationError(OptimizationError):
    """Optimization recommendation errors.
    
    Raised when unable to generate valid optimization recommendations.
    """
    pass


class ApplicationError(OptimizationError):
    """Optimization application errors.
    
    Raised when optimization changes cannot be applied to database.
    """
    pass


class BackupError(OptimizationError):
    """Backup operation errors.
    
    Raised when backup operations fail before applying optimizations.
    """
    pass


class RollbackError(OptimizationError):
    """Rollback operation errors.
    
    Raised when unable to rollback failed optimization changes.
    """
    pass


class PluginError(TuningForkException):
    """Plugin system related errors.
    
    Base class for plugin loading, initialization, and execution errors.
    """
    pass


class PluginLoadError(PluginError):
    """Plugin loading errors.
    
    Raised when plugin cannot be loaded due to missing dependencies
    or invalid plugin structure.
    """
    pass


class PluginInitializationError(PluginError):
    """Plugin initialization errors.
    
    Raised when plugin initialization fails due to configuration
    or dependency issues.
    """
    pass


class PluginExecutionError(PluginError):
    """Plugin execution errors.
    
    Raised when plugin execution fails during runtime operations.
    """
    pass


class SecurityError(TuningForkException):
    """Security related errors.
    
    Raised when security violations or authentication issues occur.
    """
    pass


class PermissionError(SecurityError):
    """Permission related errors.
    
    Raised when operations are attempted without sufficient permissions.
    """
    pass


class TimeoutError(TuningForkException):
    """Operation timeout errors.
    
    Raised when operations exceed configured timeout limits.
    """
    pass


class ResourceError(TuningForkException):
    """System resource related errors.
    
    Raised when system resources are insufficient or unavailable.
    """
    pass


class MemoryError(ResourceError):
    """Memory allocation errors.
    
    Raised when insufficient memory is available for operations.
    """
    pass


class DiskSpaceError(ResourceError):
    """Disk space errors.
    
    Raised when insufficient disk space is available.
    """
    pass


# Error code constants for common scenarios
class ErrorCodes:
    """Common error codes for TuningFork exceptions."""
    
    # Configuration errors
    CONFIG_NOT_FOUND = "CONFIG_NOT_FOUND"
    CONFIG_INVALID = "CONFIG_INVALID"
    CONFIG_VALIDATION_FAILED = "CONFIG_VALIDATION_FAILED"
    CONFIG_MIGRATION_FAILED = "CONFIG_MIGRATION_FAILED"
    
    # Connection errors
    CONNECTION_TIMEOUT = "CONNECTION_TIMEOUT"
    CONNECTION_REFUSED = "CONNECTION_REFUSED"
    AUTH_FAILED = "AUTH_FAILED"
    NETWORK_UNREACHABLE = "NETWORK_UNREACHABLE"
    POOL_EXHAUSTED = "POOL_EXHAUSTED"
    
    # Analysis errors
    METADATA_EXTRACTION_FAILED = "METADATA_EXTRACTION_FAILED"
    PERFORMANCE_ANALYSIS_FAILED = "PERFORMANCE_ANALYSIS_FAILED"
    QUERY_EXECUTION_FAILED = "QUERY_EXECUTION_FAILED"
    
    # Optimization errors
    RECOMMENDATION_GENERATION_FAILED = "RECOMMENDATION_GENERATION_FAILED"
    OPTIMIZATION_APPLICATION_FAILED = "OPTIMIZATION_APPLICATION_FAILED"
    BACKUP_CREATION_FAILED = "BACKUP_CREATION_FAILED"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    
    # Plugin errors
    PLUGIN_NOT_FOUND = "PLUGIN_NOT_FOUND"
    PLUGIN_LOAD_FAILED = "PLUGIN_LOAD_FAILED"
    PLUGIN_INIT_FAILED = "PLUGIN_INIT_FAILED"
    PLUGIN_EXECUTION_FAILED = "PLUGIN_EXECUTION_FAILED"
    
    # Security errors
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    
    # Resource errors
    OPERATION_TIMEOUT = "OPERATION_TIMEOUT"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    INSUFFICIENT_DISK_SPACE = "INSUFFICIENT_DISK_SPACE"


def create_error_from_exception(
    exc: Exception,
    message: Optional[str] = None,
    code: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> TuningForkException:
    """Create TuningFork exception from generic exception.
    
    This utility function converts generic exceptions into TuningFork
    exceptions with appropriate error codes and context.
    
    Args:
        exc: Original exception to convert
        message: Override message (uses original if not provided)
        code: Error code to assign
        context: Additional context information
        
    Returns:
        Appropriate TuningFork exception type
        
    Example:
        >>> try:
        ...     database.connect()
        ... except ConnectionRefusedError as e:
        ...     raise create_error_from_exception(
        ...         e,
        ...         code=ErrorCodes.CONNECTION_REFUSED,
        ...         context={"host": "localhost", "port": 5432}
        ...     )
    """
    error_message = message or str(exc)
    error_context = context or {}
    
    # Map common exception types to TuningFork exceptions
    exception_mapping = {
        ConnectionRefusedError: DatabaseConnectionError,
        TimeoutError: TimeoutError,
        PermissionError: PermissionError,
        MemoryError: MemoryError,
        FileNotFoundError: ConfigurationError,
        ValueError: ValidationError,
        TypeError: ValidationError,
    }
    
    # Use specific TuningFork exception if mapping exists
    exception_class = exception_mapping.get(type(exc), TuningForkException)
    
    return exception_class(
        error_message,
        code=code,
        context=error_context,
        cause=exc,
    )