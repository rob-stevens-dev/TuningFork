"""Plugin system for TuningFork extensibility.

This module provides a comprehensive plugin system that allows TuningFork
to be extended with additional functionality including custom database
connectors, heuristics engines, and optimization strategies.

Classes:
    PluginMetadata: Metadata information for plugins
    BasePlugin: Base class for all plugins
    PluginRegistry: Registry for managing plugin instances
    PluginLoader: Utility for loading plugins from various sources
    PluginManager: High-level plugin management interface

Example:
    >>> manager = PluginManager()
    >>> plugin = manager.load_plugin("path/to/plugin.py")
    >>> await manager.initialize_plugin(plugin.id)
    >>> connector = manager.get_plugin_by_type("database_connector")[0]
"""

import importlib.util
import inspect
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import structlog

from .base import AsyncComponent, ComponentRegistry
from .exceptions import (
    PluginError,
    PluginExecutionError,
    PluginInitializationError,
    PluginLoadError,
    ValidationError,
)
from .protocols import Plugin

logger = structlog.get_logger(__name__)

# Type variable for plugin types
P = TypeVar("P", bound="BasePlugin")


class PluginMetadata:
    """Metadata information for plugins.
    
    This class contains descriptive information about plugins including
    version, dependencies, and capabilities.
    
    Attributes:
        id: Unique plugin identifier
        name: Human-readable plugin name
        description: Detailed plugin description
        version: Plugin version string
        author: Plugin author information
        plugin_type: Type of plugin (connector, heuristics, etc.)
        dependencies: List of required plugin dependencies
        capabilities: List of plugin capabilities
        min_tuningfork_version: Minimum TuningFork version required
        
    Example:
        >>> metadata = PluginMetadata(
        ...     id="postgres_connector",
        ...     name="PostgreSQL Connector",
        ...     description="Database connector for PostgreSQL",
        ...     version="1.0.0",
        ...     plugin_type="database_connector"
        ... )
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        version: str,
        plugin_type: str,
        *,
        author: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        min_tuningfork_version: Optional[str] = None,
        homepage: Optional[str] = None,
        license: Optional[str] = None,
    ) -> None:
        """Initialize plugin metadata.
        
        Args:
            id: Unique plugin identifier
            name: Human-readable plugin name
            description: Detailed plugin description
            version: Plugin version string
            plugin_type: Type of plugin
            author: Plugin author information
            dependencies: List of required plugin dependencies
            capabilities: List of plugin capabilities
            min_tuningfork_version: Minimum TuningFork version required
            homepage: Plugin homepage URL
            license: Plugin license information
        """
        self.id = id
        self.name = name
        self.description = description
        self.version = version
        self.plugin_type = plugin_type
        self.author = author
        self.dependencies = dependencies or []
        self.capabilities = capabilities or []
        self.min_tuningfork_version = min_tuningfork_version
        self.homepage = homepage
        self.license = license
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary.
        
        Returns:
            Dictionary representation of metadata
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "plugin_type": self.plugin_type,
            "author": self.author,
            "dependencies": self.dependencies,
            "capabilities": self.capabilities,
            "min_tuningfork_version": self.min_tuningfork_version,
            "homepage": self.homepage,
            "license": self.license,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        """Create metadata from dictionary.
        
        Args:
            data: Dictionary containing metadata
            
        Returns:
            PluginMetadata instance
        """
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data["version"],
            plugin_type=data["plugin_type"],
            author=data.get("author"),
            dependencies=data.get("dependencies"),
            capabilities=data.get("capabilities"),
            min_tuningfork_version=data.get("min_tuningfork_version"),
            homepage=data.get("homepage"),
            license=data.get("license"),
        )
    
    def __repr__(self) -> str:
        """Return string representation of metadata."""
        return (
            f"PluginMetadata("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"version={self.version!r}, "
            f"type={self.plugin_type!r})"
        )


class BasePlugin(AsyncComponent[Dict[str, Any]], ABC):
    """Base class for all TuningFork plugins.
    
    This abstract base class provides the foundation for all plugins
    including lifecycle management, configuration, and metadata.
    
    Subclasses must implement the metadata property and plugin-specific
    functionality.
    
    Example:
        >>> class MyPlugin(BasePlugin):
        ...     @property
        ...     def metadata(self) -> PluginMetadata:
        ...         return PluginMetadata(
        ...             id="my_plugin",
        ...             name="My Plugin",
        ...             description="Example plugin",
        ...             version="1.0.0",
        ...             plugin_type="example"
        ...         )
        ...
        ...     async def _async_initialize(self) -> None:
        ...         # Plugin initialization logic
        ...         pass
    """
    
    component_name = "BasePlugin"
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize base plugin.
        
        Args:
            config: Plugin configuration dictionary
        """
        super().__init__(config)
        self._plugin_logger = structlog.get_logger(f"plugin.{self.metadata.id}")
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata.
        
        Returns:
            PluginMetadata instance describing this plugin
        """
        pass
    
    @property
    def id(self) -> str:
        """Get plugin ID from metadata.
        
        Returns:
            Plugin identifier
        """
        return self.metadata.id
    
    @property
    def name(self) -> str:
        """Get plugin name from metadata.
        
        Returns:
            Plugin name
        """
        return self.metadata.name
    
    @property
    def description(self) -> str:
        """Get plugin description from metadata.
        
        Returns:
            Plugin description
        """
        return self.metadata.description
    
    @property
    def version(self) -> str:
        """Get plugin version from metadata.
        
        Returns:
            Plugin version
        """
        return self.metadata.version
    
    @property
    def plugin_type(self) -> str:
        """Get plugin type from metadata.
        
        Returns:
            Plugin type
        """
        return self.metadata.plugin_type
    
    @property
    def dependencies(self) -> List[str]:
        """Get plugin dependencies from metadata.
        
        Returns:
            List of plugin dependencies
        """
        return self.metadata.dependencies
    
    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities from metadata.
        
        Returns:
            List of plugin capabilities
        """
        return self.metadata.capabilities
    
    def validate_config(self) -> bool:
        """Validate plugin configuration.
        
        Subclasses can override this method to implement
        plugin-specific configuration validation.
        
        Returns:
            True if configuration is valid
        """
        return super().validate_config()
    
    async def _async_initialize(self) -> None:
        """Initialize plugin resources.
        
        Subclasses should override this method to implement
        plugin-specific initialization logic.
        """
        self._plugin_logger.info(
            "Plugin initialized",
            plugin_id=self.id,
            plugin_type=self.plugin_type,
            version=self.version,
        )
    
    async def _async_cleanup(self) -> None:
        """Clean up plugin resources.
        
        Subclasses should override this method to implement
        plugin-specific cleanup logic.
        """
        self._plugin_logger.info(
            "Plugin cleaned up",
            plugin_id=self.id,
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get plugin health status.
        
        Returns:
            Dictionary containing plugin health information
        """
        status = super().get_health_status()
        status.update({
            "plugin_id": self.id,
            "plugin_type": self.plugin_type,
            "plugin_version": self.version,
            "capabilities": self.get_capabilities(),
        })
        return status
    
    def __repr__(self) -> str:
        """Return string representation of plugin."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.id!r}, "
            f"type={self.plugin_type!r}, "
            f"version={self.version!r})"
        )


class PluginRegistry:
    """Registry for managing plugin instances.
    
    This class provides centralized management of plugin instances
    with support for registration, lookup, and lifecycle coordination.
    
    Example:
        >>> registry = PluginRegistry()
        >>> registry.register(my_plugin)
        >>> plugin = registry.get("my_plugin")
        >>> connectors = registry.get_by_type("database_connector")
    """
    
    def __init__(self) -> None:
        """Initialize plugin registry."""
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugins_by_type: Dict[str, List[BasePlugin]] = {}
        self._logger = structlog.get_logger(self.__class__.__name__)
    
    def register(self, plugin: BasePlugin) -> None:
        """Register plugin with registry.
        
        Args:
            plugin: Plugin instance to register
            
        Raises:
            ValidationError: If plugin ID already exists
            PluginError: If plugin is invalid
        """
        plugin_id = plugin.id
        
        if plugin_id in self._plugins:
            raise ValidationError(
                f"Plugin already registered: {plugin_id}",
                code="PLUGIN_EXISTS",
                context={"plugin_id": plugin_id},
            )
        
        # Validate plugin
        if not isinstance(plugin, BasePlugin):
            raise PluginError(
                f"Invalid plugin type: {type(plugin)}",
                code="INVALID_PLUGIN_TYPE",
                context={"plugin_id": plugin_id},
            )
        
        # Register plugin
        self._plugins[plugin_id] = plugin
        
        # Add to type-based lookup
        plugin_type = plugin.plugin_type
        if plugin_type not in self._plugins_by_type:
            self._plugins_by_type[plugin_type] = []
        self._plugins_by_type[plugin_type].append(plugin)
        
        self._logger.info(
            "Plugin registered",
            plugin_id=plugin_id,
            plugin_type=plugin_type,
            plugin_version=plugin.version,
        )
    
    def unregister(self, plugin_id: str) -> None:
        """Unregister plugin from registry.
        
        Args:
            plugin_id: ID of plugin to unregister
            
        Raises:
            ValidationError: If plugin not found
        """
        if plugin_id not in self._plugins:
            raise ValidationError(
                f"Plugin not found: {plugin_id}",
                code="PLUGIN_NOT_FOUND",
                context={"plugin_id": plugin_id},
            )
        
        plugin = self._plugins[plugin_id]
        plugin_type = plugin.plugin_type
        
        # Remove from main registry
        del self._plugins[plugin_id]
        
        # Remove from type-based lookup
        if plugin_type in self._plugins_by_type:
            self._plugins_by_type[plugin_type].remove(plugin)
            if not self._plugins_by_type[plugin_type]:
                del self._plugins_by_type[plugin_type]
        
        self._logger.info("Plugin unregistered", plugin_id=plugin_id)
    
    def get(self, plugin_id: str) -> Optional[BasePlugin]:
        """Get plugin by ID.
        
        Args:
            plugin_id: ID of plugin to retrieve
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(plugin_id)
    
    def get_by_type(self, plugin_type: str) -> List[BasePlugin]:
        """Get all plugins of specified type.
        
        Args:
            plugin_type: Type of plugins to retrieve
            
        Returns:
            List of plugins of specified type
        """
        return self._plugins_by_type.get(plugin_type, []).copy()
    
    def get_all(self) -> Dict[str, BasePlugin]:
        """Get all registered plugins.
        
        Returns:
            Dictionary of plugin ID to plugin instance
        """
        return self._plugins.copy()
    
    def get_types(self) -> List[str]:
        """Get all registered plugin types.
        
        Returns:
            List of plugin types
        """
        return list(self._plugins_by_type.keys())
    
    def has_plugin(self, plugin_id: str) -> bool:
        """Check if plugin is registered.
        
        Args:
            plugin_id: ID of plugin to check
            
        Returns:
            True if plugin is registered
        """
        return plugin_id in self._plugins
    
    def has_type(self, plugin_type: str) -> bool:
        """Check if any plugins of type are registered.
        
        Args:
            plugin_type: Plugin type to check
            
        Returns:
            True if plugins of type are registered
        """
        return plugin_type in self._plugins_by_type
    
    def validate_dependencies(self, plugin: BasePlugin) -> List[str]:
        """Validate plugin dependencies.
        
        Args:
            plugin: Plugin to validate dependencies for
            
        Returns:
            List of missing dependencies
        """
        missing = []
        
        for dependency in plugin.dependencies:
            if not self.has_plugin(dependency):
                missing.append(dependency)
        
        return missing
    
    async def initialize_all(self) -> None:
        """Initialize all registered plugins in dependency order."""
        # Sort plugins by dependencies (topological sort)
        initialized = set()
        initialization_order = []
        
        def visit(plugin: BasePlugin) -> None:
            if plugin.id in initialized:
                return
            
            # Visit dependencies first
            for dep_id in plugin.dependencies:
                dep_plugin = self.get(dep_id)
                if dep_plugin:
                    visit(dep_plugin)
            
            initialized.add(plugin.id)
            initialization_order.append(plugin)
        
        # Visit all plugins
        for plugin in self._plugins.values():
            visit(plugin)
        
        # Initialize in order
        for plugin in initialization_order:
            try:
                await plugin.initialize()
                self._logger.info(
                    "Plugin initialized",
                    plugin_id=plugin.id,
                )
            except Exception as e:
                self._logger.error(
                    "Plugin initialization failed",
                    plugin_id=plugin.id,
                    error=str(e),
                )
                raise PluginInitializationError(
                    f"Failed to initialize plugin: {plugin.id}",
                    code="PLUGIN_INIT_FAILED",
                    context={"plugin_id": plugin.id},
                    cause=e,
                ) from e
    
    async def cleanup_all(self) -> None:
        """Clean up all registered plugins in reverse dependency order."""
        # Get cleanup order (reverse of initialization order)
        cleanup_order = list(reversed(self._plugins.values()))
        
        for plugin in cleanup_order:
            try:
                await plugin.cleanup()
                self._logger.info(
                    "Plugin cleaned up",
                    plugin_id=plugin.id,
                )
            except Exception as e:
                self._logger.error(
                    "Plugin cleanup failed",
                    plugin_id=plugin.id,
                    error=str(e),
                )
                # Don't raise during cleanup to avoid masking other errors
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all plugins.
        
        Returns:
            Dictionary with health status of all plugins
        """
        status = {}
        
        for plugin_id, plugin in self._plugins.items():
            status[plugin_id] = plugin.get_health_status()
        
        return status


class PluginLoader:
    """Utility for loading plugins from various sources.
    
    This class provides methods to load plugins from files, directories,
    and Python modules with proper error handling and validation.
    
    Example:
        >>> loader = PluginLoader()
        >>> plugin = loader.load_from_file("path/to/plugin.py")
        >>> plugins = loader.load_from_directory("plugins/")
    """
    
    def __init__(self) -> None:
        """Initialize plugin loader."""
        self._logger = structlog.get_logger(self.__class__.__name__)
    
    def load_from_file(self, file_path: Union[str, Path]) -> BasePlugin:
        """Load plugin from Python file.
        
        Args:
            file_path: Path to Python file containing plugin
            
        Returns:
            Loaded plugin instance
            
        Raises:
            PluginLoadError: If plugin cannot be loaded
        """
        path = Path(file_path)
        
        if not path.exists():
            raise PluginLoadError(
                f"Plugin file not found: {path}",
                code="FILE_NOT_FOUND",
                context={"file_path": str(path)},
            )
        
        if not path.suffix == ".py":
            raise PluginLoadError(
                f"Invalid plugin file extension: {path.suffix}",
                code="INVALID_FILE_TYPE",
                context={"file_path": str(path)},
            )
        
        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(
                    f"Cannot create module spec for: {path}",
                    code="SPEC_CREATION_FAILED",
                    context={"file_path": str(path)},
                )
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class in module
            plugin_class = self._find_plugin_class(module)
            if plugin_class is None:
                raise PluginLoadError(
                    f"No plugin class found in: {path}",
                    code="NO_PLUGIN_CLASS",
                    context={"file_path": str(path)},
                )
            
            # Create plugin instance with empty config
            plugin = plugin_class({})
            
            self._logger.info(
                "Plugin loaded from file",
                plugin_id=plugin.id,
                file_path=str(path),
            )
            
            return plugin
            
        except Exception as e:
            self._logger.error(
                "Failed to load plugin from file",
                file_path=str(path),
                error=str(e),
            )
            raise PluginLoadError(
                f"Failed to load plugin from: {path}",
                code="LOAD_FAILED",
                context={"file_path": str(path)},
                cause=e,
            ) from e
    
    def load_from_directory(self, directory_path: Union[str, Path]) -> List[BasePlugin]:
        """Load all plugins from directory.
        
        Args:
            directory_path: Path to directory containing plugin files
            
        Returns:
            List of loaded plugin instances
            
        Raises:
            PluginLoadError: If directory cannot be accessed
        """
        path = Path(directory_path)
        
        if not path.exists():
            raise PluginLoadError(
                f"Plugin directory not found: {path}",
                code="DIRECTORY_NOT_FOUND",
                context={"directory_path": str(path)},
            )
        
        if not path.is_dir():
            raise PluginLoadError(
                f"Path is not a directory: {path}",
                code="NOT_A_DIRECTORY",
                context={"directory_path": str(path)},
            )
        
        plugins = []
        
        # Load all .py files in directory
        for file_path in path.glob("*.py"):
            try:
                plugin = self.load_from_file(file_path)
                plugins.append(plugin)
            except PluginLoadError as e:
                self._logger.warning(
                    "Failed to load plugin file",
                    file_path=str(file_path),
                    error=str(e),
                )
                # Continue loading other plugins
        
        self._logger.info(
            "Plugins loaded from directory",
            directory_path=str(path),
            plugins_loaded=len(plugins),
        )
        
        return plugins
    
    def load_from_module(self, module_name: str) -> BasePlugin:
        """Load plugin from Python module.
        
        Args:
            module_name: Name of Python module containing plugin
            
        Returns:
            Loaded plugin instance
            
        Raises:
            PluginLoadError: If module cannot be loaded
        """
        try:
            module = importlib.import_module(module_name)
            
            # Find plugin class in module
            plugin_class = self._find_plugin_class(module)
            if plugin_class is None:
                raise PluginLoadError(
                    f"No plugin class found in module: {module_name}",
                    code="NO_PLUGIN_CLASS",
                    context={"module_name": module_name},
                )
            
            # Create plugin instance with empty config
            plugin = plugin_class({})
            
            self._logger.info(
                "Plugin loaded from module",
                plugin_id=plugin.id,
                module_name=module_name,
            )
            
            return plugin
            
        except ImportError as e:
            raise PluginLoadError(
                f"Failed to import module: {module_name}",
                code="IMPORT_FAILED",
                context={"module_name": module_name},
                cause=e,
            ) from e
        except Exception as e:
            raise PluginLoadError(
                f"Failed to load plugin from module: {module_name}",
                code="LOAD_FAILED",
                context={"module_name": module_name},
                cause=e,
            ) from e
    
    def _find_plugin_class(self, module: Any) -> Optional[Type[BasePlugin]]:
        """Find plugin class in module.
        
        Args:
            module: Python module to search
            
        Returns:
            Plugin class or None if not found
        """
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, BasePlugin)
                and obj is not BasePlugin
            ):
                return obj
        
        return None


class PluginManager:
    """High-level plugin management interface.
    
    This class provides a comprehensive interface for plugin management
    including loading, registration, initialization, and coordination.
    
    Example:
        >>> manager = PluginManager()
        >>> await manager.load_plugins_from_directory("plugins/")
        >>> await manager.initialize_all()
        >>> connector = manager.get_plugin_by_type("database_connector")[0]
    """
    
    def __init__(self, registry: Optional[PluginRegistry] = None) -> None:
        """Initialize plugin manager.
        
        Args:
            registry: Plugin registry to use (creates new if None)
        """
        self._registry = registry or PluginRegistry()
        self._loader = PluginLoader()
        self._logger = structlog.get_logger(self.__class__.__name__)
    
    @property
    def registry(self) -> PluginRegistry:
        """Get plugin registry.
        
        Returns:
            Plugin registry instance
        """
        return self._registry
    
    def load_plugin(self, source: Union[str, Path]) -> BasePlugin:
        """Load plugin from file or module.
        
        Args:
            source: Path to plugin file or module name
            
        Returns:
            Loaded plugin instance
        """
        if isinstance(source, (str, Path)) and Path(source).exists():
            return self._loader.load_from_file(source)
        else:
            return self._loader.load_from_module(str(source))
    
    def load_plugins_from_directory(self, directory: Union[str, Path]) -> List[BasePlugin]:
        """Load all plugins from directory.
        
        Args:
            directory: Path to directory containing plugins
            
        Returns:
            List of loaded plugin instances
        """
        return self._loader.load_from_directory(directory)
    
    def register_plugin(self, plugin: BasePlugin) -> None:
        """Register plugin with manager.
        
        Args:
            plugin: Plugin instance to register
        """
        self._registry.register(plugin)
    
    def unregister_plugin(self, plugin_id: str) -> None:
        """Unregister plugin from manager.
        
        Args:
            plugin_id: ID of plugin to unregister
        """
        self._registry.unregister(plugin_id)
    
    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """Get plugin by ID.
        
        Args:
            plugin_id: ID of plugin to retrieve
            
        Returns:
            Plugin instance or None if not found
        """
        return self._registry.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: str) -> List[BasePlugin]:
        """Get all plugins of specified type.
        
        Args:
            plugin_type: Type of plugins to retrieve
            
        Returns:
            List of plugins of specified type
        """
        return self._registry.get_by_type(plugin_type)
    
    def get_all_plugins(self) -> Dict[str, BasePlugin]:
        """Get all registered plugins.
        
        Returns:
            Dictionary of plugin ID to plugin instance
        """
        return self._registry.get_all()
    
    def get_plugin_types(self) -> List[str]:
        """Get all registered plugin types.
        
        Returns:
            List of plugin types
        """
        return self._registry.get_types()
    
    async def initialize_plugin(self, plugin_id: str) -> None:
        """Initialize specific plugin.
        
        Args:
            plugin_id: ID of plugin to initialize
            
        Raises:
            ValidationError: If plugin not found
            PluginInitializationError: If initialization fails
        """
        plugin = self.get_plugin(plugin_id)
        if plugin is None:
            raise ValidationError(
                f"Plugin not found: {plugin_id}",
                code="PLUGIN_NOT_FOUND",
                context={"plugin_id": plugin_id},
            )
        
        await plugin.initialize()
    
    async def initialize_all(self) -> None:
        """Initialize all registered plugins."""
        await self._registry.initialize_all()
    
    async def cleanup_plugin(self, plugin_id: str) -> None:
        """Clean up specific plugin.
        
        Args:
            plugin_id: ID of plugin to clean up
            
        Raises:
            ValidationError: If plugin not found
        """
        plugin = self.get_plugin(plugin_id)
        if plugin is None:
            raise ValidationError(
                f"Plugin not found: {plugin_id}",
                code="PLUGIN_NOT_FOUND",
                context={"plugin_id": plugin_id},
            )
        
        await plugin.cleanup()
    
    async def cleanup_all(self) -> None:
        """Clean up all registered plugins."""
        await self._registry.cleanup_all()
    
    def validate_plugin_dependencies(self, plugin_id: str) -> List[str]:
        """Validate dependencies for specific plugin.
        
        Args:
            plugin_id: ID of plugin to validate
            
        Returns:
            List of missing dependencies
            
        Raises:
            ValidationError: If plugin not found
        """
        plugin = self.get_plugin(plugin_id)
        if plugin is None:
            raise ValidationError(
                f"Plugin not found: {plugin_id}",
                code="PLUGIN_NOT_FOUND",
                context={"plugin_id": plugin_id},
            )
        
        return self._registry.validate_dependencies(plugin)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all plugins.
        
        Returns:
            Dictionary with health status of all plugins
        """
        return self._registry.get_health_status()
    
    async def reload_plugin(self, plugin_id: str, source: Union[str, Path]) -> None:
        """Reload plugin from source.
        
        Args:
            plugin_id: ID of plugin to reload
            source: Path to plugin file or module name
            
        Raises:
            ValidationError: If plugin not found
        """
        # Clean up existing plugin
        await self.cleanup_plugin(plugin_id)
        self.unregister_plugin(plugin_id)
        
        # Load and register new plugin
        plugin = self.load_plugin(source)
        self.register_plugin(plugin)
        
        # Initialize new plugin
        await self.initialize_plugin(plugin.id)
        
        self._logger.info(
            "Plugin reloaded",
            plugin_id=plugin_id,
            source=str(source),
        )


# Global plugin manager instance
_global_plugin_manager = PluginManager()


def get_global_plugin_manager() -> PluginManager:
    """Get the global plugin manager.
    
    Returns:
        Global PluginManager instance
    """
    return _global_plugin_manager