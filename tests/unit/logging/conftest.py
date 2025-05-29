"""Logging-specific test configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from tuningfork.logging.factory import LoggerFactory
from tuningfork.config.models import LoggingConfig


@pytest.fixture
def temp_log_file():
    """Create temporary log file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def sample_logging_config(temp_log_file):
    """Create sample logging configuration."""
    return LoggingConfig(
        level="INFO",
        format="json",
        file_path=temp_log_file,
        console_output=True,
        structured=True,
        max_file_size=1048576,  # 1MB
        backup_count=3
    )


@pytest.fixture
def logger_factory():
    """Create clean logger factory for testing."""
    factory = LoggerFactory()
    yield factory
    # Cleanup after test
    factory.shutdown()


@pytest.fixture
def mock_handler():
    """Create mock logging handler."""
    handler = Mock()
    handler.level = 20  # INFO level
    handler.flush = Mock()
    handler.close = Mock()
    handler.emit = Mock()
    return handler


@pytest.fixture(autouse=True)
def cleanup_global_logging():
    """Clean up global logging state after each test."""
    yield
    
    # Reset global factory
    from tuningfork.logging.factory import _global_factory
    _global_factory.shutdown()
    
    # Clear any handlers from root logger
    import logging
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)  # Reset to default