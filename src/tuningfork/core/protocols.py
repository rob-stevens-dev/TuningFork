"""Protocol definitions for TuningFork components.

This module defines the typing protocols that establish contracts between
different parts of the TuningFork system. These protocols ensure type safety
and define clear interfaces for component interaction.

Protocols:
    DatabaseConnector: Interface for database connection implementations
    HeuristicsEngine: Interface for rule-based analysis engines
    MLEngine: Interface for machine learning engines
    OptimizationEngine: Interface for optimization engines
    ConfigProvider: Interface for configuration providers
    LoggerProvider: Interface for logging providers

Example:
    >>> def analyze_database(connector: DatabaseConnector) -> AnalysisReport:
    ...     metadata = connector.get_metadata()
    ...     return analyze_metadata(metadata)
"""

from abc import abstractmethod
from typing import Any, AsyncContextManager, Dict, List, Optional, Protocol, TypeVar, runtime_checkable

# Type variables for protocol generics
T = TypeVar("T")
R = TypeVar("R")


@runtime_checkable
class Identifiable(Protocol):
    """Protocol for objects with unique identifiers."""
    
    @property
    def id(self) -> str:
        """Get unique identifier."""
        ...


@runtime_checkable
class Describable(Protocol):
    """Protocol for objects with descriptions."""
    
    @property
    def name(self) -> str:
        """Get human-readable name."""
        ...
    
    @property
    def description(self) -> str:
        """Get detailed description."""
        ...


@runtime_checkable
class Versionable(Protocol):
    """Protocol for versioned objects."""
    
    @property
    def version(self) -> str:
        """Get version string."""
        ...


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary representation."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """Create object from dictionary representation."""
        ...


@runtime_checkable
class Validatable(Protocol):
    """Protocol for objects that can be validated."""
    
    def validate(self) -> bool:
        """Validate object state."""
        ...
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        ...


@runtime_checkable
class Cacheable(Protocol):
    """Protocol for cacheable objects."""
    
    @property
    def cache_key(self) -> str:
        """Get cache key for this object."""
        ...
    
    @property
    def cache_ttl(self) -> Optional[int]:
        """Get cache TTL in seconds, None for no expiration."""
        ...


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


@runtime_checkable
class AsyncResourceManager(Protocol):
    """Protocol for async resource management."""
    
    async def acquire(self) -> Any:
        """Acquire resource."""
        ...
    
    async def release(self, resource: Any) -> None:
        """Release resource."""
        ...


@runtime_checkable
class ConnectionProtocol(Protocol):
    """Protocol for database connections."""
    
    async def connect(self) -> None:
        """Establish database connection."""
        ...
    
    async def disconnect(self) -> None:
        """Close database connection."""
        ...
    
    @property
    def is_connected(self) -> bool:
        """Check if connection is active."""
        ...
    
    async def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute SQL query and return results."""
        ...
    
    async def execute_script(self, script: str) -> None:
        """Execute SQL script."""
        ...


@runtime_checkable
class DatabaseMetadata(Protocol):
    """Protocol for database metadata."""
    
    @property
    def platform(self) -> str:
        """Database platform (postgresql, mysql, etc.)."""
        ...
    
    @property
    def version(self) -> str:
        """Database version string."""
        ...
    
    @property
    def tables(self) -> List[Dict[str, Any]]:
        """List of table metadata."""
        ...
    
    @property
    def indexes(self) -> List[Dict[str, Any]]:
        """List of index metadata."""
        ...
    
    @property
    def configuration(self) -> Dict[str, Any]:
        """Database configuration parameters."""
        ...


@runtime_checkable
class PerformanceMetrics(Protocol):
    """Protocol for performance metrics."""
    
    @property
    def timestamp(self) -> float:
        """Timestamp when metrics were collected."""
        ...
    
    @property
    def cpu_usage(self) -> float:
        """CPU usage percentage (0-100)."""
        ...
    
    @property
    def memory_usage(self) -> float:
        """Memory usage percentage (0-100)."""
        ...
    
    @property
    def query_stats(self) -> Dict[str, Any]:
        """Query execution statistics."""
        ...


@runtime_checkable
class DatabaseConnector(Protocol):
    """Protocol for database connector implementations.
    
    All database connectors must implement this protocol to ensure
    consistent behavior across different database platforms.
    """
    
    @property
    def platform(self) -> str:
        """Get database platform identifier."""
        ...
    
    @property
    def is_connected(self) -> bool:
        """Check if connector is connected to database."""
        ...
    
    async def connect(self) -> None:
        """Establish connection to database."""
        ...
    
    async def disconnect(self) -> None:
        """Close connection to database."""
        ...
    
    async def get_metadata(self) -> DatabaseMetadata:
        """Extract comprehensive database metadata."""
        ...
    
    async def get_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        ...
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute SQL query and return results."""
        ...
    
    async def test_connection(self) -> bool:
        """Test database connection health."""
        ...


@runtime_checkable
class Issue(Protocol):
    """Protocol for performance issues."""
    
    @property
    def id(self) -> str:
        """Get unique identifier."""
        ...
    
    @property
    def name(self) -> str:
        """Get human-readable name."""
        ...
    
    @property
    def description(self) -> str:
        """Get detailed description."""
        ...
    
    @property
    def severity(self) -> str:
        """Issue severity level."""
        ...
    
    @property
    def category(self) -> str:
        """Issue category."""
        ...
    
    @property
    def impact_score(self) -> float:
        """Numerical impact score (0-10)."""
        ...
    
    @property
    def confidence(self) -> float:
        """Confidence in issue detection (0-1)."""
        ...


@runtime_checkable
class Recommendation(Protocol):
    """Protocol for optimization recommendations."""
    
    @property
    def id(self) -> str:
        """Get unique identifier."""
        ...
    
    @property
    def name(self) -> str:
        """Get human-readable name."""
        ...
    
    @property
    def description(self) -> str:
        """Get detailed description."""
        ...
    
    @property
    def priority(self) -> str:
        """Recommendation priority level."""
        ...
    
    @property
    def action_type(self) -> str:
        """Type of action to perform."""
        ...
    
    @property
    def estimated_impact(self) -> float:
        """Estimated impact score (0-10)."""
        ...
    
    @property
    def effort_level(self) -> str:
        """Effort required to implement."""
        ...
    
    @property
    def risk_level(self) -> str:
        """Risk level of implementation."""
        ...


@runtime_checkable
class AnalysisReport(Protocol):
    """Protocol for analysis reports."""
    
    @property
    def id(self) -> str:
        """Get unique identifier."""
        ...
    
    @property
    def version(self) -> str:
        """Get version string."""
        ...
    
    @property
    def database_id(self) -> str:
        """Database identifier that was analyzed."""
        ...
    
    @property
    def timestamp(self) -> float:
        """When analysis was performed."""
        ...
    
    @property
    def issues(self) -> List[Issue]:
        """List of identified issues."""
        ...
    
    @property
    def recommendations(self) -> List[Recommendation]:
        """List of optimization recommendations."""
        ...
    
    @property
    def overall_score(self) -> float:
        """Overall database health score (0-10)."""
        ...
    
    @property
    def analysis_duration(self) -> float:
        """Time taken for analysis in seconds."""
        ...


@runtime_checkable
class HeuristicsEngine(Protocol):
    """Protocol for rule-based analysis engines.
    
    Heuristics engines apply rule-based logic to identify
    performance issues and optimization opportunities.
    """
    
    @property
    def platform(self) -> str:
        """Database platform this engine supports."""
        ...
    
    @property
    def rules(self) -> List[str]:
        """List of available rule identifiers."""
        ...
    
    async def analyze(
        self,
        metadata: DatabaseMetadata,
        metrics: PerformanceMetrics
    ) -> List[Issue]:
        """Analyze database using heuristic rules."""
        ...
    
    async def generate_recommendations(
        self,
        issues: List[Issue],
        metadata: DatabaseMetadata
    ) -> List[Recommendation]:
        """Generate recommendations for identified issues."""
        ...


@runtime_checkable
class MLEngine(Protocol):
    """Protocol for machine learning engines.
    
    ML engines use trained models to predict performance issues
    and recommend optimizations.
    """
    
    @property
    def model_version(self) -> str:
        """Version of the trained model."""
        ...
    
    @property
    def supported_platforms(self) -> List[str]:
        """List of supported database platforms."""
        ...
    
    async def predict_issues(
        self,
        metadata: DatabaseMetadata,
        metrics: PerformanceMetrics
    ) -> List[Issue]:
        """Predict potential issues using ML models."""
        ...
    
    async def recommend_configurations(
        self,
        metadata: DatabaseMetadata,
        target_workload: Dict[str, Any]
    ) -> List[Recommendation]:
        """Recommend optimal configurations using ML."""
        ...
    
    async def train_model(
        self,
        training_data: List[Dict[str, Any]]
    ) -> None:
        """Train or retrain the ML model."""
        ...


@runtime_checkable
class OptimizationEngine(Protocol):
    """Protocol for optimization engines.
    
    Optimization engines apply recommendations to databases
    and track the results.
    """
    
    async def apply_recommendation(
        self,
        connector: DatabaseConnector,
        recommendation: Recommendation
    ) -> Dict[str, Any]:
        """Apply optimization recommendation to database."""
        ...
    
    async def create_backup(
        self,
        connector: DatabaseConnector
    ) -> str:
        """Create backup before applying optimizations."""
        ...
    
    async def rollback_changes(
        self,
        connector: DatabaseConnector,
        backup_id: str
    ) -> None:
        """Rollback optimization changes using backup."""
        ...
    
    async def validate_changes(
        self,
        connector: DatabaseConnector,
        original_metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Validate optimization results."""
        ...


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for configuration providers.
    
    Configuration providers load and manage system configuration
    from various sources.
    """
    
    async def load_config(self, source: str) -> Dict[str, Any]:
        """Load configuration from source."""
        ...
    
    async def save_config(
        self,
        config: Dict[str, Any],
        destination: str
    ) -> None:
        """Save configuration to destination."""
        ...
    
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration data."""
        ...
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for validation."""
        ...


@runtime_checkable
class LoggerProvider(Protocol):
    """Protocol for logging providers.
    
    Logger providers create and configure loggers with
    appropriate formatters and handlers.
    """
    
    def create_logger(self, name: str) -> Any:
        """Create logger instance with given name."""
        ...
    
    def configure_logging(self, config: Dict[str, Any]) -> None:
        """Configure logging system with given configuration."""
        ...
    
    def get_log_level(self) -> str:
        """Get current log level."""
        ...
    
    def set_log_level(self, level: str) -> None:
        """Set log level."""
        ...


@runtime_checkable
class CacheProvider(Protocol):
    """Protocol for caching providers.
    
    Cache providers implement caching mechanisms for
    improving system performance.
    """
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set value in cache with optional TTL."""
        ...
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        ...
    
    async def clear(self) -> None:
        """Clear all cached values."""
        ...
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...


@runtime_checkable
class EventEmitter(Protocol):
    """Protocol for event emission.
    
    Event emitters publish events that other components
    can subscribe to for loose coupling.
    """
    
    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event with given type and data."""
        ...
    
    def subscribe(
        self,
        event_type: str,
        callback: Any
    ) -> str:
        """Subscribe to events of given type."""
        ...
    
    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events."""
        ...


@runtime_checkable
class MetricsCollector(Protocol):
    """Protocol for metrics collection.
    
    Metrics collectors gather system and application metrics
    for monitoring and analysis.
    """
    
    def record_counter(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record counter metric."""
        ...
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record gauge metric."""
        ...
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record histogram metric."""
        ...
    
    def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record timing metric."""
        ...


@runtime_checkable
class HealthChecker(Protocol):
    """Protocol for health checking.
    
    Health checkers verify system component health
    and availability.
    """
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        ...
    
    async def check_dependencies(self) -> Dict[str, Any]:
        """Check health of all dependencies."""
        ...
    
    def is_healthy(self) -> bool:
        """Quick health status check."""
        ...


@runtime_checkable
class Plugin(Protocol):
    """Protocol for plugin implementations.
    
    Plugins extend TuningFork functionality with
    additional features and capabilities.
    """
    
    @property
    def id(self) -> str:
        """Get unique identifier."""
        ...
    
    @property
    def name(self) -> str:
        """Get human-readable name."""
        ...
    
    @property
    def description(self) -> str:
        """Get detailed description."""
        ...
    
    @property
    def version(self) -> str:
        """Get version string."""
        ...
    
    @property
    def plugin_type(self) -> str:
        """Type of plugin (connector, heuristics, etc.)."""
        ...
    
    @property
    def dependencies(self) -> List[str]:
        """List of plugin dependencies."""
        ...
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        ...
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        ...
    
    def get_capabilities(self) -> List[str]:
        """Get list of plugin capabilities."""
        ...


@runtime_checkable
class PluginManager(Protocol):
    """Protocol for plugin management.
    
    Plugin managers handle plugin lifecycle including
    loading, initialization, and coordination.
    """
    
    def load_plugin(self, plugin_path: str) -> Plugin:
        """Load plugin from path."""
        ...
    
    def register_plugin(self, plugin: Plugin) -> None:
        """Register plugin with manager."""
        ...
    
    def unregister_plugin(self, plugin_id: str) -> None:
        """Unregister plugin from manager."""
        ...
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get plugin by ID."""
        ...
    
    def get_plugins_by_type(self, plugin_type: str) -> List[Plugin]:
        """Get all plugins of given type."""
        ...
    
    async def initialize_plugins(self) -> None:
        """Initialize all registered plugins."""
        ...
    
    async def cleanup_plugins(self) -> None:
        """Clean up all registered plugins."""
        ...