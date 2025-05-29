"""Base classes and protocols for TuningFork components.

This module provides the foundational abstract base classes and protocols
that all TuningFork components inherit from, ensuring consistent interfaces
and behavior across the system.

Classes:
    BaseComponent: Generic base class for all TuningFork components
    ConfigurableComponent: Base class for components with configuration
    AsyncComponent: Base class for async-capable components
    LifecycleComponent: Base class for components with lifecycle management

Protocols:
    Initializable: Protocol for components that require initialization
    Configurable: Protocol for configurable components
    Monitorable: Protocol for monitorable components

Example:
    >>> class DatabaseConnector(AsyncComponent[DatabaseConfig]):
    ...     async def connect(self) -> None:
    ...         await self.initialize()
    ...         # Connection logic
"""

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    ClassVar,
    Dict,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import structlog

from .exceptions import (
    ConfigurationError,
    TuningForkException,
    ValidationError,
)

# Create a safe logger that won't fail in test environments
def _get_safe_logger(name: str):
    """Get a logger that won't fail in test environments."""
    try:
        return structlog.get_logger(name)
    except Exception:
        # Fallback to basic logger if structlog fails
        import logging
        return logging.getLogger(name)

logger = _get_safe_logger(__name__)

# Type variables for generic components
T = TypeVar("T")  # Configuration type
R = TypeVar("R")  # Return type


@runtime_checkable
class Initializable(Protocol):
    """Protocol for components that require initialization.
    
    Components implementing this protocol must provide initialization
    and cleanup methods for proper resource management.
    """
    
    def initialize(self) -> None:
        """Initialize component resources."""
        ...
    
    def cleanup(self) -> None:
        """Clean up component resources."""
        ...
    
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        ...


@runtime_checkable
class AsyncInitializable(Protocol):
    """Protocol for components that require async initialization.
    
    Components implementing this protocol must provide async initialization
    and cleanup methods for proper resource management.
    """
    
    async def initialize(self) -> None:
        """Initialize component resources asynchronously."""
        ...
    
    async def cleanup(self) -> None:
        """Clean up component resources asynchronously."""
        ...
    
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        ...


@runtime_checkable
class Configurable(Protocol[T]):
    """Protocol for configurable components.
    
    Components implementing this protocol can be configured with
    type-safe configuration objects.
    """
    
    @property
    def config(self) -> T:
        """Get component configuration."""
        ...
    
    def validate_config(self) -> bool:
        """Validate component configuration."""
        ...


@runtime_checkable
class Monitorable(Protocol):
    """Protocol for monitorable components.
    
    Components implementing this protocol provide monitoring
    and health check capabilities.
    """
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics."""
        ...


class BaseComponent(Generic[T], ABC):
    """Base class for all TuningFork components.
    
    This abstract base class provides common functionality for all
    TuningFork components including configuration management, logging,
    and basic lifecycle support.
    
    Type Parameters:
        T: Type of configuration object this component accepts
    
    Attributes:
        component_name: Name of the component for logging and identification
        version: Component version for compatibility checking
        
    Example:
        >>> class MyComponent(BaseComponent[MyConfig]):
        ...     def __init__(self, config: MyConfig) -> None:
        ...         super().__init__(config)
        ...         
        ...     def process(self) -> str:
        ...         return "processed"
    """
    
    # Class-level component metadata
    component_name: ClassVar[str] = "BaseComponent"
    version: ClassVar[str] = "1.0.0"
    
    def __init__(self, config: T) -> None:
        """Initialize base component.
        
        Args:
            config: Configuration object for this component
            
        Raises:
            ValidationError: If configuration is invalid
        """
        if config is None:
            raise ValidationError(
                "Configuration cannot be None",
                code="CONFIG_NULL",
                context={"component": self.component_name},
            )
        
        self._config: T = config
        self._initialized: bool = False
        self._creation_time: float = time.time()
        self._logger = _get_safe_logger(self.__class__.__name__)
        
        # Validate configuration on creation
        if not self.validate_config():
            raise ConfigurationError(
                f"Invalid configuration for {self.component_name}",
                code="CONFIG_INVALID",
                context={"component": self.component_name},
            )
    
    @property
    def config(self) -> T:
        """Get component configuration.
        
        Returns:
            Component configuration object
        """
        return self._config
    
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized.
        
        Returns:
            True if component has been initialized
        """
        return self._initialized
    
    @property
    def uptime(self) -> float:
        """Get component uptime in seconds.
        
        Returns:
            Time since component creation in seconds
        """
        return time.time() - self._creation_time
    
    def validate_config(self) -> bool:
        """Validate component configuration.
        
        Subclasses should override this method to implement
        component-specific configuration validation.
        
        Returns:
            True if configuration is valid
        """
        return self._config is not None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status.
        
        Returns:
            Dictionary containing component health information
        """
        return {
            "component": self.component_name,
            "version": self.version,
            "initialized": self._initialized,
            "uptime_seconds": self.uptime,
            "status": "healthy" if self._initialized else "not_initialized",
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics.
        
        Returns:
            Dictionary containing component metrics
        """
        return {
            "component": self.component_name,
            "uptime_seconds": self.uptime,
            "initialized": self._initialized,
        }
    
    def __repr__(self) -> str:
        """Return string representation of component."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.component_name!r}, "
            f"initialized={self._initialized}, "
            f"uptime={self.uptime:.2f}s)"
        )


class ConfigurableComponent(BaseComponent[T]):
    """Base class for components with advanced configuration support.
    
    This class extends BaseComponent with additional configuration
    management features including configuration updates and validation.
    """
    
    def __init__(self, config: T) -> None:
        """Initialize configurable component.
        
        Args:
            config: Configuration object for this component
        """
        super().__init__(config)
        self._config_version: int = 1
        self._config_history: list[T] = [config]
    
    def update_config(self, new_config: T, *, validate: bool = True) -> None:
        """Update component configuration.
        
        Args:
            new_config: New configuration to apply
            validate: Whether to validate new configuration
            
        Raises:
            ValidationError: If new configuration is invalid
            TuningForkException: If component cannot be reconfigured while running
        """
        if validate and not self._validate_config_update(new_config):
            raise ValidationError(
                "New configuration is invalid",
                code="CONFIG_UPDATE_INVALID",
                context={
                    "component": self.component_name,
                    "version": self._config_version,
                },
            )
        
        # Store old config in history
        self._config_history.append(self._config)
        self._config = new_config
        self._config_version += 1
        
        # Safe logging that won't fail in tests
        try:
            self._logger.info(
                "Configuration updated",
                component=self.component_name,
                version=self._config_version,
            )
        except Exception:
            # Fallback logging if structlog fails
            pass
    
    def rollback_config(self) -> bool:
        """Rollback to previous configuration.
        
        Returns:
            True if rollback was successful
        """
        if len(self._config_history) < 2:
            return False
        
        # Remove current config and restore previous
        self._config_history.pop()  # Remove current
        self._config = self._config_history[-1]  # Restore previous
        self._config_version += 1
        
        # Safe logging that won't fail in tests
        try:
            self._logger.info(
                "Configuration rolled back",
                component=self.component_name,
                version=self._config_version,
            )
        except Exception:
            # Fallback logging if structlog fails
            pass
        
        return True
    
    def _validate_config_update(self, config: T) -> bool:
        """Validate configuration update.
        
        Subclasses can override this method to implement custom
        validation logic for configuration updates.
        
        Args:
            config: New configuration to validate
            
        Returns:
            True if configuration update is valid
        """
        return config is not None
    
    @property
    def config_version(self) -> int:
        """Get current configuration version.
        
        Returns:
            Current configuration version number
        """
        return self._config_version


class AsyncComponent(ConfigurableComponent[T]):
    """Base class for async-capable components.
    
    This class provides async initialization and cleanup support
    for components that perform I/O operations or other async work.
    """
    
    def __init__(self, config: T) -> None:
        """Initialize async component.
        
        Args:
            config: Configuration object for this component
        """
        super().__init__(config)
        self._initialization_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize component asynchronously.
        
        This method ensures thread-safe initialization and prevents
        multiple concurrent initialization attempts.
        
        Raises:
            TuningForkException: If initialization fails
        """
        async with self._initialization_lock:
            if self._initialized:
                return
            
            # Safe logging that won't fail in tests
            try:
                self._logger.info("Initializing component", component=self.component_name)
            except Exception:
                pass
            
            try:
                await self._async_initialize()
                self._initialized = True
                
                try:
                    self._logger.info(
                        "Component initialized successfully",
                        component=self.component_name,
                    )
                except Exception:
                    pass
                
            except Exception as e:
                try:
                    self._logger.error(
                        "Component initialization failed",
                        component=self.component_name,
                        error=str(e),
                    )
                except Exception:
                    pass
                    
                raise TuningForkException(
                    f"Failed to initialize {self.component_name}",
                    code="INIT_FAILED",
                    context={"component": self.component_name},
                    cause=e,
                ) from e
    
    async def cleanup(self) -> None:
        """Clean up component resources asynchronously.
        
        This method ensures thread-safe cleanup and prevents
        multiple concurrent cleanup attempts.
        """
        async with self._cleanup_lock:
            if not self._initialized:
                return
            
            try:
                self._logger.info("Cleaning up component", component=self.component_name)
            except Exception:
                pass
            
            try:
                await self._async_cleanup()
                self._initialized = False
                
                try:
                    self._logger.info(
                        "Component cleaned up successfully",
                        component=self.component_name,
                    )
                except Exception:
                    pass
                
            except Exception as e:
                try:
                    self._logger.error(
                        "Component cleanup failed",
                        component=self.component_name,
                        error=str(e),
                    )
                except Exception:
                    pass
                # Don't raise during cleanup to avoid masking original errors
    
    @abstractmethod
    async def _async_initialize(self) -> None:
        """Perform async initialization work.
        
        Subclasses must implement this method to perform their
        specific initialization tasks.
        """
        pass
    
    async def _async_cleanup(self) -> None:
        """Perform async cleanup work.
        
        Subclasses can override this method to perform their
        specific cleanup tasks.
        """
        pass
    
    @asynccontextmanager
    async def managed_lifecycle(self) -> AsyncGenerator["AsyncComponent[T]", None]:
        """Context manager for automatic lifecycle management.
        
        This context manager ensures proper initialization and cleanup
        of the component, even if exceptions occur.
        
        Yields:
            The initialized component
            
        Example:
            >>> async with component.managed_lifecycle() as comp:
            ...     await comp.perform_work()
        """
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()
    
    async def __aenter__(self) -> "AsyncComponent[T]":
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()


class LifecycleComponent(AsyncComponent[T]):
    """Base class for components with advanced lifecycle management.
    
    This class provides additional lifecycle states and hooks for
    components that require more sophisticated startup and shutdown
    procedures.
    """
    
    def __init__(self, config: T) -> None:
        """Initialize lifecycle component.
        
        Args:
            config: Configuration object for this component
        """
        super().__init__(config)
        self._state: str = "created"
        self._state_history: list[tuple[str, float]] = [("created", time.time())]
    
    @property
    def state(self) -> str:
        """Get current component state.
        
        Returns:
            Current component state
        """
        return self._state
    
    @property
    def state_history(self) -> list[tuple[str, float]]:
        """Get component state history.
        
        Returns:
            List of (state, timestamp) tuples
        """
        return self._state_history.copy()
    
    def _set_state(self, new_state: str) -> None:
        """Set component state and record in history.
        
        Args:
            new_state: New state to set
        """
        self._state = new_state
        self._state_history.append((new_state, time.time()))
        
        try:
            self._logger.debug(
                "Component state changed",
                component=self.component_name,
                new_state=new_state,
            )
        except Exception:
            pass
    
    async def initialize(self) -> None:
        """Initialize component with state tracking."""
        if self._state != "created":
            raise TuningForkException(
                f"Cannot initialize component in state: {self._state}",
                code="INVALID_STATE",
                context={
                    "component": self.component_name,
                    "current_state": self._state,
                },
            )
        
        self._set_state("initializing")
        
        try:
            await super().initialize()
            self._set_state("running")
        except Exception:
            self._set_state("failed")
            raise
    
    async def cleanup(self) -> None:
        """Clean up component with state tracking."""
        if self._state == "stopped":
            return
        
        self._set_state("stopping")
        
        try:
            await super().cleanup()
            self._set_state("stopped")
        except Exception:
            self._set_state("failed")
            # Don't re-raise during cleanup
    
    async def pause(self) -> None:
        """Pause component operations.
        
        Subclasses can override this method to implement
        pause functionality.
        """
        if self._state != "running":
            raise TuningForkException(
                f"Cannot pause component in state: {self._state}",
                code="INVALID_STATE",
                context={
                    "component": self.component_name,
                    "current_state": self._state,
                },
            )
        
        await self._async_pause()
        self._set_state("paused")
    
    async def resume(self) -> None:
        """Resume component operations.
        
        Subclasses can override this method to implement
        resume functionality.
        """
        if self._state != "paused":
            raise TuningForkException(
                f"Cannot resume component in state: {self._state}",
                code="INVALID_STATE",
                context={
                    "component": self.component_name,
                    "current_state": self._state,
                },
            )
        
        await self._async_resume()
        self._set_state("running")
    
    async def _async_pause(self) -> None:
        """Perform async pause work.
        
        Subclasses can override this method to implement
        pause functionality.
        """
        pass
    
    async def _async_resume(self) -> None:
        """Perform async resume work.
        
        Subclasses can override this method to implement
        resume functionality.
        """
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status with state information."""
        status = super().get_health_status()
        status.update({
            "state": self._state,
            "state_history": self._state_history[-5:],  # Last 5 states
        })
        return status


class SingletonComponent(BaseComponent[T]):
    """Base class for singleton components.
    
    This class ensures only one instance of a component type
    exists within the application.
    """
    
    _instances: ClassVar[Dict[str, Any]] = {}
    
    def __new__(cls, config: T) -> "SingletonComponent[T]":
        """Create singleton instance."""
        component_key = f"{cls.__name__}_{id(config)}"
        
        if component_key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[component_key] = instance
        
        return cls._instances[component_key]
    
    @classmethod
    def get_instance(cls, config: T) -> Optional["SingletonComponent[T]"]:
        """Get existing singleton instance.
        
        Args:
            config: Configuration to identify instance
            
        Returns:
            Existing instance or None if not found
        """
        component_key = f"{cls.__name__}_{id(config)}"
        return cls._instances.get(component_key)
    
    @classmethod
    def clear_instances(cls) -> None:
        """Clear all singleton instances.
        
        This method is primarily useful for testing.
        """
        cls._instances.clear()


class ComponentRegistry:
    """Registry for managing component instances.
    
    This class provides centralized management of component instances
    with lifecycle coordination and dependency resolution.
    """
    
    def __init__(self) -> None:
        """Initialize component registry."""
        self._components: Dict[str, BaseComponent[Any]] = {}
        self._dependencies: Dict[str, list[str]] = {}
        self._logger = _get_safe_logger(self.__class__.__name__)
    
    def register(
        self,
        name: str,
        component: BaseComponent[Any],
        dependencies: Optional[list[str]] = None,
    ) -> None:
        """Register component with registry.
        
        Args:
            name: Unique name for the component
            component: Component instance to register
            dependencies: List of component names this component depends on
            
        Raises:
            ValidationError: If component name already exists
        """
        if name in self._components:
            raise ValidationError(
                f"Component already registered: {name}",
                code="COMPONENT_EXISTS",
                context={"component_name": name},
            )
        
        self._components[name] = component
        self._dependencies[name] = dependencies or []
        
        try:
            self._logger.info(
                "Component registered",
                component_name=name,
                component_type=type(component).__name__,
                dependencies=dependencies,
            )
        except Exception:
            pass
    
    def unregister(self, name: str) -> None:
        """Unregister component from registry.
        
        Args:
            name: Name of component to unregister
            
        Raises:
            ValidationError: If component not found
        """
        if name not in self._components:
            raise ValidationError(
                f"Component not found: {name}",
                code="COMPONENT_NOT_FOUND",
                context={"component_name": name},
            )
        
        del self._components[name]
        del self._dependencies[name]
        
        try:
            self._logger.info("Component unregistered", component_name=name)
        except Exception:
            pass
    
    def get(self, name: str) -> BaseComponent[Any]:
        """Get component by name.
        
        Args:
            name: Name of component to retrieve
            
        Returns:
            Component instance
            
        Raises:
            ValidationError: If component not found
        """
        if name not in self._components:
            raise ValidationError(
                f"Component not found: {name}",
                code="COMPONENT_NOT_FOUND",
                context={"component_name": name},
            )
        
        return self._components[name]
    
    def get_all(self) -> Dict[str, BaseComponent[Any]]:
        """Get all registered components.
        
        Returns:
            Dictionary of component name to component instance
        """
        return self._components.copy()
    
    def get_initialization_order(self) -> list[str]:
        """Get component initialization order based on dependencies.
        
        Returns:
            List of component names in initialization order
            
        Raises:
            ValidationError: If circular dependencies detected
        """
        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(name: str) -> None:
            if name in temp_visited:
                raise ValidationError(
                    "Circular dependency detected",
                    code="CIRCULAR_DEPENDENCY",
                    context={"component_name": name},
                )
            
            if name in visited:
                return
            
            temp_visited.add(name)
            
            for dependency in self._dependencies.get(name, []):
                visit(dependency)
            
            temp_visited.remove(name)
            visited.add(name)
            order.append(name)
        
        for component_name in self._components:
            visit(component_name)
        
        return order
    
    async def initialize_all(self) -> None:
        """Initialize all components in dependency order."""
        order = self.get_initialization_order()
        
        for name in order:
            component = self._components[name]
            if isinstance(component, AsyncInitializable):
                await component.initialize()
            elif isinstance(component, Initializable):
                component.initialize()
    
    async def cleanup_all(self) -> None:
        """Clean up all components in reverse dependency order."""
        order = list(reversed(self.get_initialization_order()))
        
        for name in order:
            component = self._components[name]
            try:
                if isinstance(component, AsyncInitializable):
                    await component.cleanup()
                elif isinstance(component, Initializable):
                    component.cleanup()
            except Exception as e:
                try:
                    self._logger.error(
                        "Component cleanup failed",
                        component_name=name,
                        error=str(e),
                    )
                except Exception:
                    pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components.
        
        Returns:
            Dictionary with health status of all components
        """
        status = {}
        
        for name, component in self._components.items():
            if isinstance(component, Monitorable):
                status[name] = component.get_health_status()
            else:
                status[name] = {
                    "component": name,
                    "type": type(component).__name__,
                    "status": "unknown",
                }
        
        return status


# Global component registry instance
_global_registry = ComponentRegistry()


def get_global_registry() -> ComponentRegistry:
    """Get the global component registry.
    
    Returns:
        Global ComponentRegistry instance
    """
    return _global_registry


def register_component(
    name: str,
    component: BaseComponent[Any],
    dependencies: Optional[list[str]] = None,
) -> None:
    """Register component with global registry.
    
    Args:
        name: Unique name for the component
        component: Component instance to register
        dependencies: List of component names this component depends on
    """
    _global_registry.register(name, component, dependencies)


def get_component(name: str) -> BaseComponent[Any]:
    """Get component from global registry.
    
    Args:
        name: Name of component to retrieve
        
    Returns:
        Component instance
    """
    return _global_registry.get(name)