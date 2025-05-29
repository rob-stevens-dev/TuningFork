"""Tests for structured logging module."""

import json
import time
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pytest
import structlog

from tuningfork.logging.structured import (
    StructuredLogger,
    LogContext,
    ContextFilter,
)


class TestLogContext:
    """Test cases for LogContext class."""
    
    def test_context_initialization(self):
        """Test LogContext initializes correctly."""
        context = LogContext()
        assert context.get_all() == {}
    
    def test_set_and_get_context_value(self):
        """Test setting and getting context values."""
        context = LogContext()
        
        context.set("key1", "value1")
        context.set("key2", 42)
        
        assert context.get("key1") == "value1"
        assert context.get("key2") == 42
        assert context.get("nonexistent") is None
        assert context.get("nonexistent", "default") == "default"
    
    def test_get_all_context(self):
        """Test getting all context values."""
        context = LogContext()
        
        context.set("user_id", "123")
        context.set("operation", "login")
        context.set("timestamp", 1234567890)
        
        all_context = context.get_all()
        
        assert all_context == {
            "user_id": "123",
            "operation": "login", 
            "timestamp": 1234567890
        }
    
    def test_update_context(self):
        """Test updating context with multiple values."""
        context = LogContext()
        
        context.set("existing", "value")
        
        context.update({
            "new_key1": "new_value1",
            "new_key2": "new_value2",
            "existing": "updated_value"
        })
        
        assert context.get("existing") == "updated_value"
        assert context.get("new_key1") == "new_value1"
        assert context.get("new_key2") == "new_value2"
    
    def test_clear_context(self):
        """Test clearing all context values."""
        context = LogContext()
        
        context.set("key1", "value1")
        context.set("key2", "value2")
        
        assert len(context.get_all()) == 2
        
        context.clear()
        
        assert context.get_all() == {}
    
    def test_thread_isolation(self):
        """Test that context is isolated between threads."""
        import threading
        
        context = LogContext()
        results = {}
        
        def thread_func(thread_id):
            context.set("thread_id", thread_id)
            context.set("data", f"data_{thread_id}")
            time.sleep(0.1)  # Allow other threads to run
            results[thread_id] = context.get_all()
        
        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_func, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Each thread should have its own context
        assert results[0]["thread_id"] == 0
        assert results[1]["thread_id"] == 1
        assert results[2]["thread_id"] == 2
        
        assert results[0]["data"] == "data_0"
        assert results[1]["data"] == "data_1"
        assert results[2]["data"] == "data_2"


class TestContextFilter:
    """Test cases for ContextFilter class."""
    
    def test_filter_adds_context_to_record(self):
        """Test that filter adds context to log records."""
        import logging
        
        context = LogContext()
        context.set("user_id", "12345")
        context.set("request_id", "req_abc")
        
        filter_obj = ContextFilter(context)
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Apply filter
        result = filter_obj.filter(record)
        
        assert result is True
        assert hasattr(record, 'user_id')
        assert hasattr(record, 'request_id')
        assert record.user_id == "12345"
        assert record.request_id == "req_abc"
    
    def test_filter_adds_standard_metadata(self):
        """Test that filter adds standard TuningFork metadata."""
        import logging
        
        context = LogContext()
        context.set("correlation_id", "test_correlation")
        
        filter_obj = ContextFilter(context)
        
        record = logging.LogRecord(
            name="tuningfork.test",
            level=logging.INFO,
            pathname="test.py", 
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        
        assert hasattr(record, 'correlation_id')
        assert hasattr(record, 'component')
        assert hasattr(record, 'timestamp_iso')
        
        assert record.correlation_id == "test_correlation"
        assert record.component == "tuningfork.test"
        assert isinstance(record.timestamp_iso, str)
    
    def test_filter_doesnt_override_existing_attributes(self):
        """Test that filter doesn't override existing record attributes."""
        import logging
        
        context = LogContext()
        context.set("name", "context_name")  # This should not override record.name
        
        filter_obj = ContextFilter(context)
        
        record = logging.LogRecord(
            name="original_logger_name", 
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        
        # Record name should remain unchanged
        assert record.name == "original_logger_name"


class TestStructuredLogger:
    """Test cases for StructuredLogger class."""
    
    def test_logger_initialization(self):
        """Test StructuredLogger initializes correctly."""
        logger = StructuredLogger("test.logger")
        
        assert logger.name == "test.logger"
        assert logger.get_level() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert logger._context is not None
        assert logger._enable_correlation is True
    
    def test_logger_initialization_with_options(self):
        """Test StructuredLogger initialization with custom options."""
        logger = StructuredLogger(
            "test.logger",
            level="DEBUG",
            enable_correlation=False,
            auto_correlation=False
        )
        
        assert logger.name == "test.logger"
        assert logger.get_level() == "DEBUG"
        assert logger._enable_correlation is False
        assert logger._auto_correlation is False
    
    def test_set_and_get_level(self):
        """Test setting and getting log levels."""
        logger = StructuredLogger("test.logger")
        
        logger.set_level("DEBUG")
        assert logger.get_level() == "DEBUG"
        
        logger.set_level("ERROR")
        assert logger.get_level() == "ERROR"
    
    def test_invalid_log_level_raises_exception(self):
        """Test that invalid log level raises exception."""
        from tuningfork.core.exceptions import TuningForkException
        
        logger = StructuredLogger("test.logger")
        
        with pytest.raises(TuningForkException) as exc_info:
            logger.set_level("INVALID_LEVEL")
        
        assert exc_info.value.code == "UNKNOWN_LOG_LEVEL"
    
    @patch('structlog.get_logger')
    def test_logging_methods_call_structlog(self, mock_get_logger):
        """Test that logging methods call structlog correctly."""
        mock_structlog_logger = Mock()
        mock_get_logger.return_value = mock_structlog_logger
        
        logger = StructuredLogger("test.logger")
        
        # Test each logging method
        logger.debug("Debug message", extra_field="debug_value")
        logger.info("Info message", extra_field="info_value")
        logger.warning("Warning message", extra_field="warning_value")
        logger.error("Error message", extra_field="error_value")
        logger.critical("Critical message", extra_field="critical_value")
        
        # Verify structlog methods were called
        mock_structlog_logger.debug.assert_called_once()
        mock_structlog_logger.info.assert_called_once()
        mock_structlog_logger.warning.assert_called_once()
        mock_structlog_logger.error.assert_called_once()
        mock_structlog_logger.critical.assert_called_once()
    
    def test_context_manager(self):
        """Test logger context manager functionality."""
        logger = StructuredLogger("test.logger")
        
        # Set some initial context
        logger._context.set("initial", "value")
        
        with logger.context(temp_key="temp_value", operation="test_op"):
            # Inside context, both initial and temporary values should be present
            context = logger._context.get_all()
            assert "initial" in context
            assert "temp_key" in context
            assert "operation" in context
            assert context["temp_key"] == "temp_value"
            assert context["operation"] == "test_op"
        
        # Outside context, only initial value should remain
        context = logger._context.get_all()
        assert "initial" in context
        assert "temp_key" not in context
        assert "operation" not in context
    
    def test_bind_creates_new_logger_with_context(self):
        """Test that bind creates new logger with bound context."""
        logger = StructuredLogger("test.logger")
        logger._context.set("original", "value")
        
        bound_logger = logger.bind(database_id="db_123", operation="query")
        
        # Original logger should be unchanged
        original_context = logger._context.get_all()
        assert "database_id" not in original_context
        assert "operation" not in original_context
        
        # Bound logger should have all context
        bound_context = bound_logger._context.get_all()
        assert bound_context["original"] == "value"
        assert bound_context["database_id"] == "db_123"
        assert bound_context["operation"] == "query"
        
        # Should be different logger instances
        assert logger is not bound_logger
        assert logger.name == bound_logger.name
    
    def test_correlation_id_generation(self):
        """Test automatic correlation ID generation."""
        logger = StructuredLogger("test.logger", enable_correlation=True, auto_correlation=True)
        
        correlation_id = logger.get_correlation_id()
        
        assert correlation_id is not None
        assert isinstance(correlation_id, str)
        # Should be a valid UUID format
        uuid.UUID(correlation_id)  # This will raise if not valid UUID
    
    def test_correlation_id_setting(self):
        """Test manual correlation ID setting."""
        logger = StructuredLogger("test.logger", enable_correlation=True)
        
        test_correlation_id = "test-correlation-123"
        logger.set_correlation_id(test_correlation_id)
        
        assert logger.get_correlation_id() == test_correlation_id
    
    def test_correlation_disabled(self):
        """Test correlation ID when disabled."""
        logger = StructuredLogger("test.logger", enable_correlation=False)
        
        assert logger.get_correlation_id() is None
        
        # Setting correlation ID should have no effect
        logger.set_correlation_id("test-id")
        assert logger.get_correlation_id() is None
    
    def test_operation_logging_success(self):
        """Test operation logging for successful operations."""
        logger = StructuredLogger("test.logger") 
        
        # Start operation
        operation_context = logger.log_operation_start("test_operation", param1="value1")
        
        assert "operation_id" in operation_context
        assert "operation" in operation_context
        assert "start_time" in operation_context
        assert operation_context["operation"] == "test_operation"
        assert operation_context["param1"] == "value1"
        
        # Complete operation successfully
        logger.log_operation_success(operation_context, result="success", items_processed=100)
        
        # Should not raise any exceptions
    
    def test_operation_logging_failure(self):
        """Test operation logging for failed operations."""
        logger = StructuredLogger("test.logger")
        
        # Start operation
        operation_context = logger.log_operation_start("test_operation")
        
        # Simulate failure
        test_error = Exception("Test error")
        logger.log_operation_failure(operation_context, test_error, additional_info="test")
        
        # Should not raise any exceptions
    
    def test_clear_context(self):
        """Test clearing logger context."""
        logger = StructuredLogger("test.logger", enable_correlation=True, auto_correlation=True)
        
        # Add some context
        logger._context.set("key1", "value1")
        logger._context.set("key2", "value2")
        
        # Get initial correlation ID
        initial_correlation = logger.get_correlation_id()
        
        # Clear context
        logger.clear_context()
        
        # Context should be empty except for correlation ID (if auto-generated)
        context = logger.get_context()
        assert "key1" not in context
        assert "key2" not in context
        
        # Correlation ID should be regenerated if auto-correlation is enabled
        new_correlation = logger.get_correlation_id()
        assert new_correlation is not None
        assert new_correlation != initial_correlation
    
    def test_exception_logging(self):
        """Test exception logging with traceback."""
        logger = StructuredLogger("test.logger")
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            # This should not raise any exceptions
            logger.exception("An error occurred", operation="test")
    
    def test_logger_repr(self):
        """Test logger string representation."""
        logger = StructuredLogger("test.logger", level="DEBUG", enable_correlation=True)
        
        repr_str = repr(logger)
        
        assert "StructuredLogger" in repr_str
        assert "test.logger" in repr_str
        assert "DEBUG" in repr_str
        assert "correlation=True" in repr_str