"""Simple import test to verify logging module can be imported."""

import pytest


def test_import_logging_module():
    """Test that logging module can be imported without errors."""
    try:
        from tuningfork.logging import (
            get_logger,
            configure_logging,
            StructuredLogger,
            PerformanceLogger,
            AuditLogger
        )
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.fail(f"Failed to import logging module: {e}")


def test_import_factory():
    """Test that factory module can be imported."""
    try:
        from tuningfork.logging.factory import LoggerFactory
        assert LoggerFactory is not None
    except ImportError as e:
        pytest.fail(f"Failed to import factory: {e}")


def test_import_structured():
    """Test that structured module can be imported."""
    try:
        from tuningfork.logging.structured import StructuredLogger
        assert StructuredLogger is not None
    except ImportError as e:
        pytest.fail(f"Failed to import structured: {e}")


def test_create_simple_logger():
    """Test creating a simple logger."""
    from tuningfork.logging import get_logger
    
    logger = get_logger("test.simple")
    assert logger is not None
    assert logger.name == "test.simple"


def test_basic_logging():
    """Test basic logging functionality."""
    from tuningfork.logging import get_logger
    
    logger = get_logger("test.basic")
    
    # These should not raise any exceptions
    logger.info("Test info message")
    logger.debug("Test debug message")
    logger.warning("Test warning message")
    logger.error("Test error message")