"""Pytest configuration and shared fixtures.

This module provides pytest configuration and shared fixtures for all tests
in the TuningFork test suite.
"""

import asyncio
import pytest
import tempfile
import structlog
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

# Configure test logging to suppress noise during tests
structlog.configure(
    processors=[structlog.testing.LogCapture()],
    wrapper_class=structlog.testing.ReturnLogger,
    logger_factory=structlog.testing.ReturnLoggerFactory(),
    cache_logger_on_first_use=True,
)

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_logger():
    """Mock structured logger for testing."""
    logger = MagicMock()
    logger.bind.return_value = logger
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def sample_config_data() -> dict:
    """Sample configuration data for testing."""
    return {
        "app_name": "TuningFork",
        "version": "1.0.0-test",
        "environment": "testing",
        "debug": True,
        "security": {
            "secret_key": "test-secret-key-for-testing-only-12345",
            "api_key_length": 32,
            "session_timeout": 3600,
            "require_ssl": False,  # Disabled for testing
        },
        "logging": {
            "level": "DEBUG",
            "format": "json",
            "console_output": True,
            "structured": True,
        },
        "performance": {
            "max_concurrent_operations": 5,
            "operation_timeout": 30,
            "cache_ttl": 300,
        },
        "databases": {},
        "plugins": {},
    }


@pytest.fixture
def sample_database_config_data() -> dict:
    """Sample database configuration data for testing."""
    return {
        "id": "test_db",
        "platform": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "test_database",
        "credentials": {
            "username": "test_user",
            "password": "test_password",
        },
        "ssl_config": {
            "enabled": False,  # Disabled for testing
        },
        "pool_config": {
            "min_size": 1,
            "max_size": 5,
        },
        "connection_timeout": 30,
        "query_timeout": 60,
    }


@pytest.fixture
def config_file(temp_dir: Path, sample_config_data: dict) -> Path:
    """Create temporary configuration file."""
    import yaml
    
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config_data, f)
    return config_path


@pytest.fixture
def database_config_file(temp_dir: Path, sample_database_config_data: dict) -> Path:
    """Create temporary database configuration file."""
    import yaml
    
    config_path = temp_dir / "test_db_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_database_config_data, f)
    return config_path


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slower, real dependencies)"
    )
    config.addinivalue_line(
        "markers", "system: marks tests as system tests (slowest, full system)"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (benchmark code)"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as pytest-benchmark tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that take > 1 second"
    )
    config.addinivalue_line(
        "markers", "database: marks tests requiring database connection"
    )
    config.addinivalue_line(
        "markers", "network: marks tests requiring network access"
    )


# Auto-mark tests based on their location
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Get test file path relative to tests directory
        test_path = Path(item.fspath).relative_to(Path(config.rootdir) / "tests")
        
        # Auto-mark based on directory structure
        if test_path.parts[0] == "unit":
            item.add_marker(pytest.mark.unit)
        elif test_path.parts[0] == "integration":
            item.add_marker(pytest.mark.integration)
        elif test_path.parts[0] == "system":
            item.add_marker(pytest.mark.system)
        elif test_path.parts[0] == "performance":
            item.add_marker(pytest.mark.performance)
        
        # Mark database tests
        if "database" in str(test_path):
            item.add_marker(pytest.mark.database)
        
        # Mark network tests
        if "network" in str(test_path):
            item.add_marker(pytest.mark.network)
        
        # Mark slow tests (can be overridden with explicit marker)
        if not any(mark.name == "slow" for mark in item.iter_markers()):
            # Add slow marker for integration and system tests by default
            if any(mark.name in ["integration", "system"] for mark in item.iter_markers()):
                item.add_marker(pytest.mark.slow)


# Clean up between tests
@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Clean up singleton instances between tests."""
    yield
    
    # Clean up any singleton components
    from tuningfork.core.base import SingletonComponent
    SingletonComponent.clear_instances()


# Performance test configuration
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "min_rounds": 5,
        "max_time": 1.0,
        "warmup": True,
        "warmup_iterations": 1,
    }


# Mock async functions for testing
@pytest.fixture
def mock_async_function():
    """Mock async function for testing."""
    async def mock_func(*args, **kwargs):
        return {"result": "success", "args": args, "kwargs": kwargs}
    
    return mock_func


@pytest.fixture
def mock_failing_async_function():
    """Mock failing async function for testing."""
    async def mock_func(*args, **kwargs):
        raise RuntimeError("Mock async function failed")
    
    return mock_func


# Test data generators
@pytest.fixture
def generate_test_data():
    """Generator for test data."""
    def _generate(count: int = 10, **kwargs):
        """Generate test data items."""
        defaults = {
            "name": "test_item",
            "value": 42,
            "active": True,
        }
        defaults.update(kwargs)
        
        return [
            {**defaults, "id": i, "name": f"{defaults['name']}_{i}"}
            for i in range(count)
        ]
    
    return _generate


# Error simulation helpers
@pytest.fixture
def error_simulator():
    """Helper for simulating various error conditions."""
    class ErrorSimulator:
        @staticmethod
        def connection_error():
            return ConnectionError("Simulated connection error")
        
        @staticmethod
        def timeout_error():
            return TimeoutError("Simulated timeout error")
        
        @staticmethod
        def permission_error():
            return PermissionError("Simulated permission error")
        
        @staticmethod
        def value_error():
            return ValueError("Simulated value error")
        
        @staticmethod
        def runtime_error():
            return RuntimeError("Simulated runtime error")
    
    return ErrorSimulator()


# Test environment helpers
@pytest.fixture
def test_environment():
    """Helper for test environment setup."""
    class TestEnvironment:
        def __init__(self):
            self.temp_files = []
            self.temp_dirs = []
        
        def create_temp_file(self, content: str, suffix: str = ".txt") -> Path:
            """Create temporary file with content."""
            import tempfile
            fd, path = tempfile.mkstemp(suffix=suffix)
            try:
                with open(path, 'w') as f:
                    f.write(content)
                self.temp_files.append(Path(path))
                return Path(path)
            finally:
                import os
                os.close(fd)
        
        def create_temp_dir(self) -> Path:
            """Create temporary directory."""
            import tempfile
            path = Path(tempfile.mkdtemp())
            self.temp_dirs.append(path)
            return path
        
        def cleanup(self):
            """Clean up temporary files and directories."""
            import shutil
            for file_path in self.temp_files:
                try:
                    file_path.unlink()
                except FileNotFoundError:
                    pass
            
            for dir_path in self.temp_dirs:
                try:
                    shutil.rmtree(dir_path)
                except FileNotFoundError:
                    pass
            
            self.temp_files.clear()
            self.temp_dirs.clear()
    
    env = TestEnvironment()
    yield env
    env.cleanup()