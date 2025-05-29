"""TuningFork core infrastructure.

This package provides the foundational components for the TuningFork system
including base classes, exception handling, plugin management, and utilities.

Modules:
    base: Base classes and component registry
    exceptions: Exception hierarchy
    plugins: Plugin system
    protocols: System protocols and interfaces
    utils: Utility functions

Classes:
    BaseComponent: Base class for all components
    AsyncComponent: Base class for async components
    LifecycleComponent: Base class with advanced lifecycle
    PluginManager: Plugin management interface
    ComponentRegistry: Component registry
    TuningForkException: Base exception class

Example:
    >>> from tuningfork.core import BaseComponent, PluginManager
    >>> from tuningfork.core.exceptions import ValidationError
    >>> from tuningfork.core.utils import measure_time
"""

from .base import (
    AsyncComponent,
    BaseComponent,
    ComponentRegistry,
    ConfigurableComponent,
    LifecycleComponent,
    SingletonComponent,
    get_component,
    get_global_registry,
    register_component,
)
from .exceptions import (
    AnalysisError,
    ApplicationError,
    AuthenticationError,
    BackupError,
    ConfigurationError,
    ConnectionError,
    ConnectionPoolError,
    DatabaseConnectionError,
    DiskSpaceError,
    ErrorCodes,
    MemoryError,
    MetadataError,
    MigrationError,
    NetworkError,
    OptimizationError,
    PerformanceError,
    PermissionError,
    PluginError,
    PluginExecutionError,
    PluginInitializationError,
    PluginLoadError,
    QueryError,
    RecommendationError,
    ResourceError,
    RollbackError,
    SecurityError,
    TimeoutError,
    TuningForkException,
    ValidationError,
    create_error_from_exception,
)
from .plugins import (
    BasePlugin,
    PluginLoader,
    PluginManager,
    PluginMetadata,
    PluginRegistry,
    get_global_plugin_manager,
)
from .protocols import (
    AnalysisReport,
    AsyncInitializable,
    AsyncResourceManager,
    CacheProvider,
    Cacheable,
    ConfigProvider,
    ConnectionProtocol,
    DatabaseConnector,
    DatabaseMetadata,
    Describable,
    EventEmitter,
    HeuristicsEngine,
    HealthChecker,
    Identifiable,
    Initializable,
    Issue,
    LoggerProvider,
    MLEngine,
    MetricsCollector,
    Monitorable,
    OptimizationEngine,
    PerformanceMetrics,
    Plugin,
    PluginManager as PluginManagerProtocol,
    Recommendation,
    Serializable,
    Validatable,
    Versionable,
)
from .utils import (
    DictUtils,
    FormatUtils,
    ListUtils,
    StringUtils,
    TimerContext,
    ValidationUtils,
    coalesce,
    measure_time,
    require_not_none,
    retry_with_backoff,
    safe_cast,
)

__version__ = "0.1.0"

__all__ = [
    # Base classes
    "BaseComponent",
    "AsyncComponent",
    "ConfigurableComponent",
    "LifecycleComponent",
    "SingletonComponent",
    "ComponentRegistry",
    "get_component",
    "get_global_registry",
    "register_component",
    
    # Exceptions
    "TuningForkException",
    "ConfigurationError",
    "ValidationError",
    "MigrationError",
    "ConnectionError",
    "DatabaseConnectionError",
    "AuthenticationError",
    "NetworkError",
    "ConnectionPoolError",
    "AnalysisError",
    "MetadataError",
    "PerformanceError",
    "QueryError",
    "OptimizationError",
    "RecommendationError",
    "ApplicationError",
    "BackupError",
    "RollbackError",
    "PluginError",
    "PluginLoadError",
    "PluginInitializationError",
    "PluginExecutionError",
    "SecurityError",
    "PermissionError",
    "TimeoutError",
    "ResourceError",
    "MemoryError",
    "DiskSpaceError",
    "ErrorCodes",
    "create_error_from_exception",
    
    # Plugin system
    "BasePlugin",
    "PluginMetadata",
    "PluginRegistry",
    "PluginLoader",
    "PluginManager",
    "get_global_plugin_manager",
    
    # Protocols
    "Identifiable",
    "Describable",
    "Versionable",
    "Serializable",
    "Validatable",
    "Cacheable",
    "Initializable",
    "AsyncInitializable",
    "Monitorable",
    "AsyncResourceManager",
    "ConnectionProtocol",
    "DatabaseMetadata",
    "PerformanceMetrics",
    "DatabaseConnector",
    "Issue",
    "Recommendation",
    "AnalysisReport",
    "HeuristicsEngine",
    "MLEngine",
    "OptimizationEngine",
    "ConfigProvider",
    "LoggerProvider",
    "CacheProvider",
    "EventEmitter",
    "MetricsCollector",
    "HealthChecker",
    "Plugin",
    "PluginManagerProtocol",
    
    # Utilities
    "ValidationUtils",
    "StringUtils",
    "FormatUtils",
    "DictUtils",
    "ListUtils",
    "TimerContext",
    "measure_time",
    "retry_with_backoff",
    "safe_cast",
    "require_not_none",
    "coalesce",
]