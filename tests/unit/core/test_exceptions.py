"""Unit tests for TuningFork exception hierarchy.

This module tests the exception classes and error handling utilities
to ensure proper error reporting and context management.
"""

import pytest
from typing import Any, Dict

from tuningfork.core.exceptions import (
    TuningForkException,
    ConfigurationError,
    ValidationError,
    MigrationError,
    ConnectionError,
    DatabaseConnectionError,
    AuthenticationError,
    NetworkError,
    ConnectionPoolError,
    AnalysisError,
    MetadataError,
    PerformanceError,
    QueryError,
    OptimizationError,
    RecommendationError,
    ApplicationError,
    BackupError,
    RollbackError,
    PluginError,
    PluginLoadError,
    PluginInitializationError,
    PluginExecutionError,
    SecurityError,
    PermissionError,
    TimeoutError,
    ResourceError,
    MemoryError,
    DiskSpaceError,
    ErrorCodes,
    create_error_from_exception,
)


class TestTuningForkException:
    """Test base TuningFork exception class."""
    
    def test_basic_exception_creation(self):
        """Test basic exception creation with message only."""
        exc = TuningForkException("Test error message")
        
        assert str(exc) == "TuningForkException: Test error message"
        assert exc.code == "TuningForkException"
        assert exc.context == {}
        assert exc.cause is None
    
    def test_exception_with_custom_code(self):
        """Test exception creation with custom error code."""
        exc = TuningForkException(
            "Test error", 
            code="CUSTOM_ERROR"
        )
        
        assert exc.code == "CUSTOM_ERROR"
        assert str(exc) == "CUSTOM_ERROR: Test error"
    
    def test_exception_with_context(self):
        """Test exception creation with context information."""
        context = {"database_id": "test_db", "operation": "connect"}
        exc = TuningForkException(
            "Test error",
            context=context
        )
        
        assert exc.context == context
    
    def test_exception_with_cause(self):
        """Test exception creation with cause."""
        original_error = ValueError("Original error")
        exc = TuningForkException(
            "Wrapped error",
            cause=original_error
        )
        
        assert exc.cause == original_error
    
    def test_exception_to_dict(self):
        """Test exception serialization to dictionary."""
        context = {"test_key": "test_value"}
        original_error = ValueError("Original")
        
        exc = TuningForkException(
            "Test message",
            code="TEST_CODE",
            context=context,
            cause=original_error
        )
        
        result = exc.to_dict()
        
        assert result["error_type"] == "TuningForkException"
        assert result["message"] == "Test message"
        assert result["code"] == "TEST_CODE"
        assert result["context"] == context
        assert result["cause"] == "Original"
    
    def test_exception_repr(self):
        """Test exception string representation."""
        exc = TuningForkException(
            "Test message",
            code="TEST_CODE",
            context={"key": "value"}
        )
        
        repr_str = repr(exc)
        
        assert "TuningForkException" in repr_str
        assert "Test message" in repr_str
        assert "TEST_CODE" in repr_str
        assert "{'key': 'value'}" in repr_str


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""
    
    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from TuningForkException."""
        exc = ConfigurationError("Config error")
        
        assert isinstance(exc, TuningForkException)
        assert isinstance(exc, ConfigurationError)
        assert exc.code == "ConfigurationError"
    
    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from ConfigurationError."""
        exc = ValidationError("Validation failed")
        
        assert isinstance(exc, TuningForkException)
        assert isinstance(exc, ConfigurationError)
        assert isinstance(exc, ValidationError)
        assert exc.code == "ValidationError"
    
    def test_connection_error_inheritance(self):
        """Test connection error hierarchy."""
        db_exc = DatabaseConnectionError("DB connection failed")
        auth_exc = AuthenticationError("Auth failed")
        network_exc = NetworkError("Network error")
        pool_exc = ConnectionPoolError("Pool error")
        
        # All should inherit from ConnectionError and TuningForkException
        for exc in [db_exc, auth_exc, network_exc, pool_exc]:
            assert isinstance(exc, TuningForkException)
            assert isinstance(exc, ConnectionError)
    
    def test_analysis_error_inheritance(self):
        """Test analysis error hierarchy."""
        metadata_exc = MetadataError("Metadata error")
        perf_exc = PerformanceError("Performance error")
        query_exc = QueryError("Query error")
        
        for exc in [metadata_exc, perf_exc, query_exc]:
            assert isinstance(exc, TuningForkException)
            assert isinstance(exc, AnalysisError)
    
    def test_optimization_error_inheritance(self):
        """Test optimization error hierarchy."""
        rec_exc = RecommendationError("Recommendation error")
        app_exc = ApplicationError("Application error")
        backup_exc = BackupError("Backup error")
        rollback_exc = RollbackError("Rollback error")
        
        for exc in [rec_exc, app_exc, backup_exc, rollback_exc]:
            assert isinstance(exc, TuningForkException)
            assert isinstance(exc, OptimizationError)
    
    def test_plugin_error_inheritance(self):
        """Test plugin error hierarchy."""
        load_exc = PluginLoadError("Load error")
        init_exc = PluginInitializationError("Init error")
        exec_exc = PluginExecutionError("Execution error")
        
        for exc in [load_exc, init_exc, exec_exc]:
            assert isinstance(exc, TuningForkException)
            assert isinstance(exc, PluginError)
    
    def test_resource_error_inheritance(self):
        """Test resource error hierarchy."""
        mem_exc = MemoryError("Memory error")
        disk_exc = DiskSpaceError("Disk error")
        
        for exc in [mem_exc, disk_exc]:
            assert isinstance(exc, TuningForkException)
            assert isinstance(exc, ResourceError)


class TestErrorCodes:
    """Test error code constants."""
    
    def test_error_codes_exist(self):
        """Test that all error code constants exist."""
        # Configuration error codes
        assert hasattr(ErrorCodes, "CONFIG_NOT_FOUND")
        assert hasattr(ErrorCodes, "CONFIG_INVALID")
        assert hasattr(ErrorCodes, "CONFIG_VALIDATION_FAILED")
        
        # Connection error codes
        assert hasattr(ErrorCodes, "CONNECTION_TIMEOUT")
        assert hasattr(ErrorCodes, "CONNECTION_REFUSED")
        assert hasattr(ErrorCodes, "AUTH_FAILED")
        
        # Analysis error codes
        assert hasattr(ErrorCodes, "METADATA_EXTRACTION_FAILED")
        assert hasattr(ErrorCodes, "PERFORMANCE_ANALYSIS_FAILED")
        
        # Plugin error codes
        assert hasattr(ErrorCodes, "PLUGIN_NOT_FOUND")
        assert hasattr(ErrorCodes, "PLUGIN_LOAD_FAILED")
    
    def test_error_codes_are_strings(self):
        """Test that error codes are string constants."""
        assert isinstance(ErrorCodes.CONFIG_NOT_FOUND, str)
        assert isinstance(ErrorCodes.CONNECTION_TIMEOUT, str)
        assert isinstance(ErrorCodes.PLUGIN_LOAD_FAILED, str)
    
    def test_error_codes_are_uppercase(self):
        """Test that error codes follow uppercase convention."""
        assert ErrorCodes.CONFIG_NOT_FOUND.isupper()
        assert ErrorCodes.CONNECTION_TIMEOUT.isupper()
        assert ErrorCodes.PLUGIN_LOAD_FAILED.isupper()


class TestCreateErrorFromException:
    """Test error creation utility function."""
    
    def test_create_from_connection_refused_error(self):
        """Test creating TuningFork error from ConnectionRefusedError."""
        original = ConnectionRefusedError("Connection refused")
        context = {"host": "localhost", "port": 5432}
        
        result = create_error_from_exception(
            original,
            code=ErrorCodes.CONNECTION_REFUSED,
            context=context
        )
        
        assert isinstance(result, DatabaseConnectionError)
        assert result.code == ErrorCodes.CONNECTION_REFUSED
        assert result.context == context
        assert result.cause == original
        assert str(original) in str(result)
    
    def test_create_from_timeout_error(self):
        """Test creating TuningFork error from TimeoutError."""
        import asyncio
        original = asyncio.TimeoutError("Operation timed out")
        
        result = create_error_from_exception(
            original,
            code=ErrorCodes.OPERATION_TIMEOUT
        )
        
        assert isinstance(result, TuningForkException)  # Note: maps to base TuningForkException, not our TimeoutError
        assert result.code == ErrorCodes.OPERATION_TIMEOUT
        assert result.cause == original
    
    def test_create_from_permission_error(self):
        """Test creating TuningFork error from PermissionError."""
        original = PermissionError("Access denied")
        
        result = create_error_from_exception(
            original,
            code=ErrorCodes.INSUFFICIENT_PERMISSIONS
        )
        
        assert isinstance(result, PermissionError)
        assert result.code == ErrorCodes.INSUFFICIENT_PERMISSIONS
        assert result.cause == original
    
    def test_create_from_value_error(self):
        """Test creating TuningFork error from ValueError."""
        original = ValueError("Invalid value")
        
        result = create_error_from_exception(
            original,
            message="Custom validation message"
        )
        
        assert isinstance(result, ValidationError)
        assert "Custom validation message" in str(result)
        assert result.cause == original
    
    def test_create_from_unknown_exception(self):
        """Test creating TuningFork error from unknown exception type."""
        original = RuntimeError("Unknown error")
        
        result = create_error_from_exception(
            original,
            code="UNKNOWN_ERROR"
        )
        
        assert isinstance(result, TuningForkException)
        assert result.code == "UNKNOWN_ERROR"
        assert result.cause == original
    
    def test_create_with_custom_message(self):
        """Test creating error with custom message override."""
        original = ValueError("Original message")
        custom_message = "Custom error message"
        
        result = create_error_from_exception(
            original,
            message=custom_message
        )
        
        assert custom_message in str(result)
        assert result.cause == original
    
    def test_create_with_context(self):
        """Test creating error with additional context."""
        original = ConnectionError("Connection failed")
        context = {
            "database_id": "test_db",
            "retry_count": 3,
            "last_attempt": "2023-01-01T00:00:00"
        }
        
        result = create_error_from_exception(
            original,
            context=context
        )
        
        assert result.context == context
        assert result.cause == original


class TestExceptionUsagePatterns:
    """Test common exception usage patterns."""
    
    def test_exception_chaining(self):
        """Test exception chaining with cause."""
        try:
            # Simulate nested exception scenario
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise ConfigurationError(
                    "Configuration parsing failed",
                    code="CONFIG_PARSE_ERROR",
                    context={"file": "config.yaml"},
                    cause=e
                ) from e
        except ConfigurationError as exc:
            assert exc.cause.__class__ == ValueError
            assert "Inner error" in str(exc.cause)
            assert exc.code == "CONFIG_PARSE_ERROR"
            assert exc.context["file"] == "config.yaml"
    
    def test_exception_in_context_manager(self):
        """Test exception handling in context managers."""
        class TestResource:
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type:
                    # Transform exception in context manager
                    raise ResourceError(
                        "Resource cleanup failed",
                        code="CLEANUP_FAILED",
                        cause=exc_val
                    ) from exc_val
        
        with pytest.raises(ResourceError) as exc_info:
            with TestResource():
                raise ValueError("Something went wrong")
        
        assert exc_info.value.code == "CLEANUP_FAILED"
        assert isinstance(exc_info.value.cause, ValueError)
    
    def test_exception_logging_context(self):
        """Test exception with logging context."""
        exc = DatabaseConnectionError(
            "Failed to connect to database",
            code=ErrorCodes.CONNECTION_TIMEOUT,
            context={
                "host": "db.example.com",
                "port": 5432,
                "timeout": 30,
                "retry_count": 3,
                "database": "production"
            }
        )
        
        # Test that exception contains sufficient context for logging
        exc_dict = exc.to_dict()
        
        assert "host" in exc_dict["context"]
        assert "port" in exc_dict["context"]
        assert "timeout" in exc_dict["context"]
        assert "retry_count" in exc_dict["context"]
        assert exc_dict["error_type"] == "DatabaseConnectionError"
        assert exc_dict["code"] == ErrorCodes.CONNECTION_TIMEOUT


class TestExceptionValidation:
    """Test exception validation and edge cases."""
    
    def test_empty_message_handling(self):
        """Test handling of empty error messages."""
        exc = TuningForkException("")
        
        assert str(exc) == "TuningForkException: "
        assert exc.to_dict()["message"] == ""
    
    def test_none_context_handling(self):
        """Test handling of None context."""
        exc = TuningForkException("Test", context=None)
        
        assert exc.context == {}
    
    def test_none_cause_handling(self):
        """Test handling of None cause."""
        exc = TuningForkException("Test", cause=None)
        
        assert exc.cause is None
        assert exc.to_dict()["cause"] is None
    
    def test_unicode_message_handling(self):
        """Test handling of unicode messages."""
        unicode_message = "测试错误消息 🚨"
        exc = TuningForkException(unicode_message)
        
        assert unicode_message in str(exc)
        assert exc.to_dict()["message"] == unicode_message
    
    def test_complex_context_serialization(self):
        """Test serialization of complex context objects."""
        from datetime import datetime
        
        complex_context = {
            "timestamp": datetime.now(),
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "none_value": None
        }
        
        exc = TuningForkException("Test", context=complex_context)
        
        # Should not raise exception during serialization
        result = exc.to_dict()
        assert "context" in result
        assert result["context"]["nested"]["key"] == "value"
        assert result["context"]["list"] == [1, 2, 3]
        assert result["context"]["none_value"] is None


@pytest.mark.parametrize("exception_class,expected_code", [
    (ConfigurationError, "ConfigurationError"),
    (ValidationError, "ValidationError"),
    (DatabaseConnectionError, "DatabaseConnectionError"),
    (AuthenticationError, "AuthenticationError"),
    (MetadataError, "MetadataError"),
    (PluginLoadError, "PluginLoadError"),
    (TimeoutError, "TimeoutError"),
    (MemoryError, "MemoryError"),
])
def test_exception_default_codes(exception_class, expected_code):
    """Test that exceptions have correct default error codes."""
    exc = exception_class("Test message")
    assert exc.code == expected_code


@pytest.mark.parametrize("exception_class", [
    TuningForkException,
    ConfigurationError,
    ValidationError,
    ConnectionError,
    DatabaseConnectionError,
    AnalysisError,
    OptimizationError,
    PluginError,
    SecurityError,
    ResourceError,
])
def test_exception_inheritance_from_base(exception_class):
    """Test that all exceptions inherit from TuningForkException."""
    exc = exception_class("Test message")
    assert isinstance(exc, TuningForkException)
    assert isinstance(exc, Exception)