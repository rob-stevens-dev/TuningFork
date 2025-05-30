# src/tuningfork/database/registry.py
"""Database connector registry for TuningFork Phase 2."""

from typing import Dict, Type, Optional, List
from tuningfork.core import AsyncComponent
from tuningfork.core.protocols import DatabaseConnector
from tuningfork.config.models import DatabaseConfig
from tuningfork.logging import get_logger
from tuningfork.core.exceptions import (
    TuningForkException,
    ErrorCodes
)


class DatabaseConnectorRegistry:
    """Registry for database connector types with discovery and validation.
    
    Provides centralized management of database connector types, allowing
    registration, discovery, and creation of connector instances. Integrates
    with Phase 1 logging and error handling infrastructure.
    """
    
    def __init__(self):
        self.logger = get_logger("database.registry")
        self._connectors: Dict[str, Type[DatabaseConnector]] = {}
        self._metadata: Dict[str, Dict[str, str]] = {}
    
    def register_connector(
        self, 
        platform: str, 
        connector_class: Type[DatabaseConnector],
        description: Optional[str] = None,
        version: Optional[str] = None
    ) -> None:
        """Register a database connector class.
        
        Args:
            platform: Database platform identifier (e.g., 'postgresql', 'mysql')
            connector_class: Connector class implementing DatabaseConnector protocol
            description: Optional description of the connector
            version: Optional version of the connector
            
        Raises:
            TuningForkException: If connector is invalid or platform already registered
        """
        # Validate connector class
        if not issubclass(connector_class, AsyncComponent):
            raise TuningForkException(
                f"Connector class {connector_class.__name__} must extend AsyncComponent",
                code=ErrorCodes.CONFIG_INVALID,
                context={"platform": platform, "class": connector_class.__name__}
            )
        
        # Check if platform already registered
        if platform in self._connectors:
            self.logger.warning(
                "Overriding existing connector registration",
                platform=platform,
                existing_class=self._connectors[platform].__name__,
                new_class=connector_class.__name__
            )
        
        # Register connector
        self._connectors[platform] = connector_class
        self._metadata[platform] = {
            "class_name": connector_class.__name__,
            "description": description or f"{platform.title()} database connector",
            "version": version or getattr(connector_class, 'version', 'unknown'),
            "platform": platform
        }
        
        self.logger.info(
            "Database connector registered",
            platform=platform,
            class_name=connector_class.__name__,
            version=self._metadata[platform]["version"]
        )
    
    def get_connector_class(self, platform: str) -> Type[DatabaseConnector]:
        """Get connector class for a platform.
        
        Args:
            platform: Database platform identifier
            
        Returns:
            Connector class for the platform
            
        Raises:
            TuningForkException: If platform not registered
        """
        if platform not in self._connectors:
            raise TuningForkException(
                f"No connector registered for platform: {platform}",
                code=ErrorCodes.CONFIG_NOT_FOUND,
                context={
                    "platform": platform,
                    "available_platforms": list(self._connectors.keys())
                }
            )
        
        return self._connectors[platform]
    
    def create_connector(self, config: DatabaseConfig) -> DatabaseConnector:
        """Create connector instance for database configuration.
        
        Args:
            config: Database configuration
            
        Returns:
            Initialized connector instance
            
        Raises:
            TuningForkException: If platform not supported or creation fails
        """
        platform = config.platform
        
        try:
            connector_class = self.get_connector_class(platform)
            connector = connector_class(config)
            
            self.logger.info(
                "Database connector created",
                platform=platform,
                database_id=config.id,
                host=config.host,
                port=config.port
            )
            
            return connector
            
        except Exception as e:
            raise TuningForkException(
                f"Failed to create connector for platform {platform}: {e}",
                code=ErrorCodes.NETWORK_UNREACHABLE,
                context={
                    "platform": platform,
                    "database_id": config.id,
                    "host": config.host,
                    "port": config.port
                }
            ) from e
    
    def get_available_platforms(self) -> List[str]:
        """Get list of available database platforms.
        
        Returns:
            List of registered platform identifiers
        """
        return list(self._connectors.keys())
    
    def get_connector_metadata(self, platform: str) -> Dict[str, str]:
        """Get metadata for a registered connector.
        
        Args:
            platform: Database platform identifier
            
        Returns:
            Connector metadata dictionary
            
        Raises:
            TuningForkException: If platform not registered
        """
        if platform not in self._metadata:
            raise TuningForkException(
                f"No metadata available for platform: {platform}",
                code=ErrorCodes.CONFIG_NOT_FOUND,
                context={"platform": platform}
            )
        
        return self._metadata[platform].copy()
    
    def is_platform_supported(self, platform: str) -> bool:
        """Check if a platform is supported.
        
        Args:
            platform: Database platform identifier
            
        Returns:
            True if platform is supported, False otherwise
        """
        return platform in self._connectors
    
    def list_connectors(self) -> Dict[str, Dict[str, str]]:
        """Get metadata for all registered connectors.
        
        Returns:
            Dictionary mapping platform names to metadata dictionaries
        """
        return {
            platform: metadata.copy() 
            for platform, metadata in self._metadata.items()
        }
    
    def unregister_connector(self, platform: str) -> None:
        """Unregister a database connector.
        
        Args:
            platform: Database platform identifier
            
        Raises:
            TuningForkException: If platform not registered
        """
        if platform not in self._connectors:
            raise TuningForkException(
                f"Cannot unregister unknown platform: {platform}",
                code=ErrorCodes.CONFIG_NOT_FOUND,
                context={"platform": platform}
            )
        
        del self._connectors[platform]
        del self._metadata[platform]
        
        self.logger.info(
            "Database connector unregistered",
            platform=platform
        )
    
    def clear_registry(self) -> None:
        """Clear all registered connectors."""
        platforms = list(self._connectors.keys())
        self._connectors.clear()
        self._metadata.clear()
        
        self.logger.info(
            "Registry cleared",
            unregistered_platforms=platforms
        )


# Global registry instance
_global_registry: Optional[DatabaseConnectorRegistry] = None


def get_global_registry() -> DatabaseConnectorRegistry:
    """Get the global database connector registry.
    
    Returns:
        Global registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = DatabaseConnectorRegistry()
    return _global_registry


def register_connector(
    platform: str, 
    connector_class: Type[DatabaseConnector],
    description: Optional[str] = None,
    version: Optional[str] = None
) -> None:
    """Register a connector with the global registry.
    
    Args:
        platform: Database platform identifier
        connector_class: Connector class implementing DatabaseConnector protocol
        description: Optional description of the connector
        version: Optional version of the connector
    """
    registry = get_global_registry()
    registry.register_connector(platform, connector_class, description, version)


def create_connector(config: DatabaseConfig) -> DatabaseConnector:
    """Create connector instance using the global registry.
    
    Args:
        config: Database configuration
        
    Returns:
        Initialized connector instance
    """
    registry = get_global_registry()
    return registry.create_connector(config)


def get_available_platforms() -> List[str]:
    """Get available platforms from the global registry.
    
    Returns:
        List of supported platform identifiers
    """
    registry = get_global_registry()
    return registry.get_available_platforms()