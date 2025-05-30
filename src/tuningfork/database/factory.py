# src/tuningfork/database/factory.py
"""Database connector factory for TuningFork Phase 2."""

from typing import Dict, Any, Optional
from tuningfork.core.protocols import DatabaseConnector
from tuningfork.config.models import DatabaseConfig
from tuningfork.logging import get_logger
from tuningfork.core.exceptions import (
    TuningForkException,
    ErrorCodes
)
from tuningfork.database.registry import get_global_registry


class DatabaseConnectorFactory:
    """Factory for creating database connectors with validation and error handling.
    
    Provides a high-level interface for creating database connectors with
    automatic registration discovery, configuration validation, and
    comprehensive error handling using Phase 1 infrastructure.
    """
    
    def __init__(self):
        self.logger = get_logger("database.factory")
        self._registry = get_global_registry()
    
    def create_connector(self, config: DatabaseConfig) -> DatabaseConnector:
        """Create a database connector from configuration.
        
        Args:
            config: Database configuration
            
        Returns:
            Initialized database connector
            
        Raises:
            TuningForkException: If connector creation fails
        """
        self.logger.info(
            "Creating database connector",
            platform=config.platform,
            database_id=config.id,
            host=config.host,
            port=config.port
        )
        
        try:
            # Validate configuration
            self._validate_configuration(config)
            
            # Create connector using registry
            connector = self._registry.create_connector(config)
            
            self.logger.info(
                "Database connector created successfully",
                platform=config.platform,
                database_id=config.id,
                connector_class=connector.__class__.__name__
            )
            
            return connector
            
        except Exception as e:
            self.logger.error(
                "Failed to create database connector",
                platform=config.platform,
                database_id=config.id,
                error=str(e)
            )
            raise
    
    def create_connector_from_dict(self, config_dict: Dict[str, Any]) -> DatabaseConnector:
        """Create a database connector from configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Initialized database connector
            
        Raises:
            TuningForkException: If configuration invalid or connector creation fails
        """
        try:
            # Convert dictionary to configuration object
            config = DatabaseConfig(**config_dict)
            return self.create_connector(config)
            
        except Exception as e:
            raise TuningForkException(
                f"Failed to create connector from configuration dictionary: {e}",
                code=ErrorCodes.CONFIG_INVALID,
                context={"config_dict": config_dict}
            ) from e
    
    def get_supported_platforms(self) -> Dict[str, Dict[str, str]]:
        """Get information about supported database platforms.
        
        Returns:
            Dictionary mapping platform names to metadata
        """
        return self._registry.list_connectors()
    
    def is_platform_supported(self, platform: str) -> bool:
        """Check if a database platform is supported.
        
        Args:
            platform: Database platform identifier
            
        Returns:
            True if platform is supported, False otherwise
        """
        return self._registry.is_platform_supported(platform)
    
    def validate_configuration(self, config: DatabaseConfig) -> bool:
        """Validate database configuration without creating connector.
        
        Args:
            config: Database configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            TuningForkException: If configuration is invalid
        """
        try:
            self._validate_configuration(config)
            return True
        except Exception:
            return False
    
    def _validate_configuration(self, config: DatabaseConfig) -> None:
        """Validate database configuration.
        
        Args:
            config: Configuration to validate
            
        Raises:
            TuningForkException: If configuration is invalid
        """
        # Check if platform is supported
        if not self._registry.is_platform_supported(config.platform):
            available_platforms = self._registry.get_available_platforms()
            raise TuningForkException(
                f"Unsupported database platform: {config.platform}",
                code=ErrorCodes.CONFIG_INVALID,
                context={
                    "platform": config.platform,
                    "available_platforms": available_platforms
                }
            )
        
        # Validate required fields
        if not config.id:
            raise TuningForkException(
                "Database configuration missing required field: id",
                code=ErrorCodes.CONFIG_VALIDATION_FAILED,
                context={"field": "id"}
            )
        
        if not config.host and config.platform != "sqlite":
            raise TuningForkException(
                "Database configuration missing required field: host",
                code=ErrorCodes.CONFIG_VALIDATION_FAILED,
                context={"field": "host", "platform": config.platform}
            )
        
        if not config.database:
            raise TuningForkException(
                "Database configuration missing required field: database",
                code=ErrorCodes.CONFIG_VALIDATION_FAILED,
                context={"field": "database"}
            )
        
        # Validate port range
        if config.port < 1 or config.port > 65535:
            raise TuningForkException(
                f"Invalid port number: {config.port}",
                code=ErrorCodes.CONFIG_INVALID,
                context={"port": config.port, "valid_range": "1-65535"}
            )
        
        self.logger.debug(
            "Database configuration validated successfully",
            platform=config.platform,
            database_id=config.id
        )


# Convenience functions for common usage patterns
def create_connector(config: DatabaseConfig) -> DatabaseConnector:
    """Create a database connector using the default factory.
    
    Args:
        config: Database configuration
        
    Returns:
        Initialized database connector
    """
    factory = DatabaseConnectorFactory()
    return factory.create_connector(config)


def create_connector_from_dict(config_dict: Dict[str, Any]) -> DatabaseConnector:
    """Create a database connector from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Initialized database connector
    """
    factory = DatabaseConnectorFactory()
    return factory.create_connector_from_dict(config_dict)


def get_supported_platforms() -> Dict[str, Dict[str, str]]:
    """Get information about supported database platforms.
    
    Returns:
        Dictionary mapping platform names to metadata
    """
    factory = DatabaseConnectorFactory()
    return factory.get_supported_platforms()


def is_platform_supported(platform: str) -> bool:
    """Check if a database platform is supported.
    
    Args:
        platform: Database platform identifier
        
    Returns:
        True if platform is supported, False otherwise
    """
    factory = DatabaseConnectorFactory()
    return factory.is_platform_supported(platform)