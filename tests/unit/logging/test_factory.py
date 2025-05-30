"""Tests for logging factory module."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tuningfork.logging.factory import (
    LoggerFactory,
    LoggerConfig,
    get_logger,
    configure_logging,
    get_factory,
)
from tuningfork.logging.structured import StructuredLogger
from tuningfork.logging.performance import PerformanceLogger
from tuningfork.logging.audit import AuditLogger
from tuningfork.config.models import LoggingConfig


class TestLoggerConfig:
    """Test cases for LoggerConfig class."""
    
    def test_logger_config_defaults(self):
        """Test LoggerConfig default values."""
        config = LoggerConfig()
        
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.console_output is True
        assert config.file_output is False
        assert config.file_path is None
        assert config.max_file_size == 10485760  # 10MB
        assert config.backup_count == 5
        assert config.structured is True
        assert config.performance_tracking is False
        assert config.audit_logging is False
        assert config.correlation_ids is True
    
    def test_logger_config_custom_values(self):
        """Test LoggerConfig with custom values."""
        config = LoggerConfig(
            level="DEBUG",
            format="text",
            console_output=False,
            file_output=True,
            file_path="/var/log/test.log",
            max_file_size=5242880,  # 5MB
            backup_count=3,
            structured=False,
            performance_tracking=True,
            audit_logging=True,
            correlation_ids=False
        )
        
        assert config.level == "DEBUG"
        assert config.format == "text"
        assert config.console_output is False
        assert config.file_output is True
        assert config.file_path == "/var/log/test.log"
        assert config.max_file_size == 5242880
        assert config.backup_count == 3
        assert config.structured is False
        assert config.performance_tracking is True
        assert config.audit_logging is True
        assert config.correlation_ids is False


class TestLoggerFactory:
    """Test cases for LoggerFactory class."""
    
    def test_factory_initialization(self):
        """Test LoggerFactory initializes correctly."""
        factory = LoggerFactory()
        
        assert isinstance(factory.config, LoggerConfig)
        assert factory.initialized is False
        assert len(factory._loggers) == 0
        assert len(factory._performance_loggers) == 0
        assert len(factory._audit_loggers) == 0
    
    def test_factory_with_custom_config(self):
        """Test LoggerFactory with custom config."""
        custom_config = LoggerConfig(level="DEBUG", format="text")
        factory = LoggerFactory(custom_config)
        
        assert factory.config.level == "DEBUG"
        assert factory.config.format == "text"
    
    def test_configure_from_dict(self):
        """Test configuring factory from dictionary."""
        factory = LoggerFactory()
        
        config_dict = {
            "level": "DEBUG",
            "format": "text",
            "console_output": False,
            "file_output": True,
            "file_path": "/tmp/test.log",
            "invalid_key": "should_be_ignored"  # Invalid keys should be filtered
        }
        
        factory.configure_from_dict(config_dict)
        
        assert factory.config.level == "DEBUG"
        assert factory.config.format == "text"
        assert factory.config.console_output is False
        assert factory.config.file_output is True
        assert factory.config.file_path == "/tmp/test.log"
        assert factory.initialized is True
    
    def test_configure_from_logging_config(self):
        """Test configuring factory from LoggingConfig."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            logging_config = LoggingConfig(
                level="WARNING",
                format="json",
                file_path=temp_path,
                max_file_size=5242880,
                backup_count=3,
                console_output=False,
                structured=True
            )
            
            factory = LoggerFactory()
            factory.configure_from_config(logging_config)
            
            assert factory.config.level == "WARNING"
            assert factory.config.format == "json"
            assert factory.config.file_output is True
            assert factory.config.file_path == str(temp_path)
            assert factory.config.max_file_size == 5242880
            assert factory.config.backup_count == 3
            assert factory.config.console_output is False
            assert factory.config.structured is True
            assert factory.initialized is True
        
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
    
    def test_get_logger_caching(self):
        """Test logger caching in factory."""
        factory = LoggerFactory()
        
        # Get logger twice with same parameters
        logger1 = factory.get_logger("test.logger")
        logger2 = factory.get_logger("test.logger")
        
        # Should return same instance
        assert logger1 is logger2
        assert isinstance(logger1, StructuredLogger)
        assert logger1.name == "test.logger"
    
    def test_get_logger_different_params(self):
        """Test getting loggers with different parameters."""
        factory = LoggerFactory()
        
        # Get loggers with different parameters - they should be different instances
        logger1 = factory.get_logger("test.logger.info", level="INFO")
        logger2 = factory.get_logger("test.logger.debug", level="DEBUG")
        
        # Should return different instances due to different cache keys
        assert logger1 is not logger2
        # Both should have their specified levels
        print(factory._loggers.keys())
        assert logger1.get_level() == "INFO"
        assert logger2.get_level() == "DEBUG"
    
    def test_get_performance_logger(self):
        """Test getting performance logger."""
        factory = LoggerFactory()
        
        perf_logger = factory.get_performance_logger("perf_test")
        
        assert isinstance(perf_logger, PerformanceLogger)
        assert perf_logger.name == "perf_test"
    
    def test_get_performance_logger_caching(self):
        """Test performance logger caching."""
        factory = LoggerFactory()
        
        # Get same performance logger twice
        logger1 = factory.get_performance_logger("perf_test")
        logger2 = factory.get_performance_logger("perf_test")
        
        # Should return same instance
        assert logger1 is logger2
    
    def test_get_audit_logger(self):
        """Test getting audit logger."""
        factory = LoggerFactory()
        
        audit_logger = factory.get_audit_logger("audit_test")
        
        assert isinstance(audit_logger, AuditLogger)
        assert audit_logger.name == "audit_test"
    
    def test_get_audit_logger_with_compliance(self):
        """Test getting audit logger with compliance mode."""
        factory = LoggerFactory()
        
        audit_logger = factory.get_audit_logger(
            "compliance_test",
            compliance_mode=True,
            default_retention_days=365
        )
        
        assert audit_logger.compliance_mode is True
        assert audit_logger.default_retention_days == 365
    
    def test_set_level_specific_logger(self):
        """Test setting level for specific logger."""
        factory = LoggerFactory()
        
        logger = factory.get_logger("test.logger", level="INFO")
        assert logger.get_level() == "INFO"
        
        # Change level for specific logger
        factory.set_level("DEBUG", "test.logger")
        assert logger.get_level() == "DEBUG"
    
    def test_set_level_all_loggers(self):
        """Test setting level for all loggers."""
        factory = LoggerFactory()
        
        logger1 = factory.get_logger("logger1", level="INFO")
        logger2 = factory.get_logger("logger2", level="WARNING")
        
        # Change level for all loggers
        factory.set_level("ERROR")
        
        assert factory.config.level == "ERROR"
        assert logger1.get_level() == "ERROR"
        assert logger2.get_level() == "ERROR"
    
    def test_invalid_log_level_raises_exception(self):
        """Test setting invalid log level raises exception."""
        from tuningfork.core.exceptions import ValidationError
        
        factory = LoggerFactory()
        
        with pytest.raises(ValidationError):
            factory.set_level("INVALID_LEVEL")
    
    def test_add_handler(self):
        """Test adding handler to logger."""
        factory = LoggerFactory()
        
        # Create a mock handler
        mock_handler = Mock(spec=logging.Handler)
        
        # Add handler to specific logger
        factory.add_handler(mock_handler, "test.logger")
        
        # Verify handler was added to stdlib logger
        test_logger = logging.getLogger("test.logger")
        assert mock_handler in test_logger.handlers
    
    def test_remove_handler(self):
        """Test removing handler from logger."""
        factory = LoggerFactory()
        
        # Create and add mock handler
        mock_handler = Mock(spec=logging.Handler)
        factory.add_handler(mock_handler, "test.logger")
        
        # Remove handler
        factory.remove_handler(mock_handler, "test.logger")
        
        # Verify handler was removed
        test_logger = logging.getLogger("test.logger")
        assert mock_handler not in test_logger.handlers
    
    def test_get_logger_info(self):
        """Test getting logger information."""
        factory = LoggerFactory()
        
        # Create some loggers
        factory.get_logger("test1")
        factory.get_performance_logger("perf1")
        factory.get_audit_logger("audit1")
        
        info = factory.get_logger_info()
        
        assert isinstance(info, dict)
        assert "config" in info
        assert "initialized" in info
        assert "loggers" in info
        assert "handlers" in info
        
        assert len(info["loggers"]["structured"]) >= 1
        assert len(info["loggers"]["performance"]) >= 1
        assert len(info["loggers"]["audit"]) >= 1
        
        assert info["config"]["level"] == factory.config.level
        assert info["config"]["format"] == factory.config.format
    
    def test_shutdown(self):
        """Test factory shutdown."""
        factory = LoggerFactory()
        
        # Create some loggers
        factory.get_logger("test1")
        factory.get_performance_logger("perf1")
        
        assert factory.initialized is True
        assert len(factory._loggers) > 0
        assert len(factory._performance_loggers) > 0
        
        # Shutdown factory
        factory.shutdown()
        
        assert factory.initialized is False
        assert len(factory._loggers) == 0
        assert len(factory._performance_loggers) == 0
    
    def test_factory_repr(self):
        """Test factory string representation."""
        config = LoggerConfig(level="DEBUG", format="text")
        factory = LoggerFactory(config)
        
        repr_str = repr(factory)
        
        assert "LoggerFactory" in repr_str
        assert "DEBUG" in repr_str
        assert "text" in repr_str

    # ChatGPT suggestions to improve coverage.
    def test_add_remove_handler_root_logger(self):
        """Test adding and removing a handler on the root logger."""
        factory = LoggerFactory()
        mock_handler = Mock(spec=logging.Handler)

        # Add handler to root logger
        factory.add_handler(mock_handler)
        assert mock_handler in logging.getLogger().handlers

        # Remove handler from root logger
        factory.remove_handler(mock_handler)
        assert mock_handler not in logging.getLogger().handlers

    def test_logger_info_handlers_detail(self):
        """Test get_logger_info() includes detailed handler metadata."""
        factory = LoggerFactory()
        factory.configure_from_dict({
            "level": "INFO",
            "console_output": True,
        })
        factory.get_logger("info.logger")

        info = factory.get_logger_info()

        assert "handlers" in info
        for handler in info["handlers"]:
            assert "type" in handler
            assert "level" in handler
            assert "formatter" in handler

    def test_logger_factory_repr_format(self):
        """Test string representation of LoggerFactory reflects config."""
        config = LoggerConfig(level="DEBUG", format="text")
        factory = LoggerFactory(config)

        repr_str = repr(factory)

        assert "LoggerFactory" in repr_str
        assert "DEBUG" in repr_str
        assert "text" in repr_str

    def test_get_audit_logger_custom_params(self):
        """Test get_audit_logger() with custom retention parameters."""
        factory = LoggerFactory()
        audit_logger = factory.get_audit_logger(
            "audit.custom",
            retain_in_memory=False,
            default_retention_days=999
        )

        assert audit_logger.default_retention_days == 999
        # Retain_in_memory is likely not public; don't access private attrs

    def test_shutdown_structlog_flag(self):
        """Test that shutdown resets internal factory state."""
        factory = LoggerFactory()
        factory.get_logger("test.shutdown")

        assert factory.initialized is True

        factory.shutdown()

        # These checks are more robust than inspecting internal flags
        assert factory.initialized is False
        assert factory._loggers == {}
        assert factory._performance_loggers == {}
        assert factory._audit_loggers == {}

    def test_configure_from_dict_ignores_invalid_keys(self):
        """Test configure_from_dict() ignores unknown keys safely."""
        factory = LoggerFactory()
        factory.configure_from_dict({
            "level": "WARNING",
            "unknown_key": "should_be_ignored"
        })

        assert factory.config.level == "WARNING"
        assert not hasattr(factory.config, "unknown_key")

    def test_get_logger_disable_correlation(self):
        """Test get_logger() with enable_correlation set to False."""
        factory = LoggerFactory()
        logger = factory.get_logger("correlation.test", enable_correlation=False)

        assert logger._enable_correlation is False

    def test_set_level_nonexistent_logger_name(self):
        """Test that set_level() for unknown logger name does not raise errors."""
        factory = LoggerFactory()
        factory.set_level("ERROR", logger_name="nonexistent.logger")

    def test_set_level_all_loggers_respects_new_level(self):
        """Test set_level() applies new level to all existing loggers."""
        factory = LoggerFactory()
        logger1 = factory.get_logger("test1", level="INFO")
        logger2 = factory.get_logger("test2", level="DEBUG")

        factory.set_level("CRITICAL")

        assert logger1.get_level() == "CRITICAL"
        assert logger2.get_level() == "CRITICAL"

    def test_get_logger_different_correlation_flag(self):
        """Test logger cache behavior with different correlation flags."""
        factory = LoggerFactory()

        logger1 = factory.get_logger("correlation.test", enable_correlation=True)
        logger2 = factory.get_logger("correlation.test", enable_correlation=False)

        assert logger1 is not logger2

    def test_performance_logger_shares_structured_logger(self):
        """Test performance logger uses structured logger internally."""
        factory = LoggerFactory()
        perf_logger = factory.get_performance_logger("perf.shared")

        assert isinstance(perf_logger.logger, StructuredLogger)
        assert perf_logger.logger.name == "perf.perf.shared"

    def test_file_output_creates_directory(self, tmp_path):
        """Test file output path is created automatically if it doesn't exist."""
        log_path = tmp_path / "logs" / "tuningfork.log"
        config = {
            "file_output": True,
            "file_path": str(log_path),
            "console_output": False,
        }

        factory = LoggerFactory()
        factory.configure_from_dict(config)
        factory.get_logger("filetest")

        assert log_path.parent.exists()

class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def teardown_method(self):
        """Clean up after each test."""
        # Reset global factory state
        from tuningfork.logging.factory import _global_factory
        _global_factory.shutdown()
    
    def test_configure_logging(self):
        """Test global configure_logging function."""
        configure_logging(
            level="DEBUG",
            format="text",
            console_output=True,
            file_output=False
        )
        
        factory = get_factory()
        assert factory.config.level == "DEBUG"
        assert factory.config.format == "text"
        assert factory.config.console_output is True
        assert factory.config.file_output is False
        assert factory.initialized is True
    
    def test_get_logger_global(self):
        """Test global get_logger function."""
        logger = get_logger("global.test")
        
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "global.test"
    
    def test_get_logger_with_params(self):
        """Test global get_logger with parameters."""
        logger = get_logger("param.test", level="ERROR", enable_correlation=False)
        
        assert logger.get_level() == "ERROR"
        assert logger._enable_correlation is False
    
    def test_get_performance_logger_global(self):
        """Test global get_performance_logger function."""
        from tuningfork.logging.factory import get_performance_logger
        
        perf_logger = get_performance_logger("global.perf")
        
        assert isinstance(perf_logger, PerformanceLogger)
        assert perf_logger.name == "global.perf"
    
    def test_get_audit_logger_global(self):
        """Test global get_audit_logger function."""
        from tuningfork.logging.factory import get_audit_logger
        
        audit_logger = get_audit_logger("global.audit")
        
        assert isinstance(audit_logger, AuditLogger)
        assert audit_logger.name == "global.audit"
    
    def test_get_factory_function(self):
        """Test get_factory function."""
        factory1 = get_factory()
        factory2 = get_factory()
        
        # Should return same instance
        assert factory1 is factory2
        assert isinstance(factory1, LoggerFactory)
    
    def test_shutdown_logging_global(self):
        """Test global shutdown_logging function."""
        from tuningfork.logging.factory import shutdown_logging
        
        # Create some loggers
        get_logger("test1")
        factory = get_factory()
        
        assert factory.initialized is True
        
        # Shutdown logging
        shutdown_logging()
        
        assert factory.initialized is False

    # ChatGPT suggested tests:
    def test_configure_logging_sets_global_factory_config(self):
        """Test module-level configure_logging() sets global factory state."""
        configure_logging(
            level="WARNING",
            format="text",
            console_output=False
        )
        factory = get_factory()

        assert factory.config.level == "WARNING"
        assert factory.config.format == "text"
        assert factory.config.console_output is False

    def test_shutdown_logging_resets_global_factory(self):
        """Test module-level shutdown_logging() resets logger factory."""
        factory = get_factory()
        factory.get_logger("reset.test")

        assert factory.initialized is True

        from tuningfork.logging.factory import shutdown_logging
        shutdown_logging()

        assert factory.initialized is False
        assert factory._loggers == {}


@pytest.mark.integration
class TestLoggerFactoryIntegration:
    """Integration tests for LoggerFactory."""
    
    def test_integration_with_config_models(self):
        """Test integration with TuningFork config models."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Create LoggingConfig
            logging_config = LoggingConfig(
                level="INFO",
                format="json",
                file_path=temp_path,
                console_output=True,
                structured=True
            )
            
            # Configure factory
            factory = LoggerFactory()
            factory.configure_from_config(logging_config)
            
            # Create logger and log messages
            logger = factory.get_logger("file.test")
            logger.info("Test file logging", test_param="value")
            logger.warning("Test warning message")
            
            # Force flush to ensure messages are written
            for handler in logging.getLogger().handlers:
                handler.flush()
            
            # Verify file exists and has content
            assert temp_path.exists()
            content = temp_path.read_text()
            assert "Test file logging" in content
            assert "Test warning message" in content
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_multiple_logger_types_integration(self):
        """Test using multiple logger types together."""
        factory = LoggerFactory()
        
        # Create different types of loggers
        structured_logger = factory.get_logger("multi.structured")
        perf_logger = factory.get_performance_logger("multi.performance")
        audit_logger = factory.get_audit_logger("multi.audit")
        
        # Use all loggers
        structured_logger.info("Structured log message")
        
        with perf_logger.measure("test_operation"):
            import time
            time.sleep(0.01)
        
        audit_logger.log_event(
            "test_event",
            actor="test_user",
            resource="test_resource",
            action="test_action"
        )
        
        # All should work without conflicts
        assert len(factory._loggers) >= 1
        assert len(factory._performance_loggers) >= 1
        assert len(factory._audit_loggers) >= 1


@pytest.mark.performance
class TestLoggerFactoryPerformance:
    """Performance tests for LoggerFactory."""
    
    def test_logger_creation_performance(self, benchmark):
        """Test performance of logger creation."""
        factory = LoggerFactory()
        
        def create_loggers():
            for i in range(10):
                factory.get_logger(f"perf.test.{i}")
        
        benchmark(create_loggers)
        
        # Should have created 10 loggers
        assert len(factory._loggers) >= 10
    
    def test_logger_caching_performance(self, benchmark):
        """Test performance of logger caching."""
        factory = LoggerFactory()
        
        # Pre-create logger
        factory.get_logger("cached.logger")
        
        def get_cached_logger():
            return factory.get_logger("cached.logger")
        
        # Should be very fast due to caching
        result = benchmark(get_cached_logger)
        assert result is not None
    
    def test_configuration_performance(self, benchmark):
        """Test configuration change performance."""
        factory = LoggerFactory()
        
        config_changes = [
            {"level": "DEBUG", "format": "json"},
            {"level": "INFO", "format": "text"},
            {"level": "WARNING", "format": "json"},
            {"level": "ERROR", "format": "text"},
        ]
        
        def apply_configurations():
            for config in config_changes:
                factory.configure_from_dict(config)
        
        benchmark(apply_configurations)
        
        # Should end with last configuration
        assert factory.config.level == "ERROR"
        assert factory.config.format == "text"