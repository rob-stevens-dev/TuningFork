# TuningFork - Phase 1: Foundation Layer

## Overview

TuningFork is an enterprise-grade database performance optimization platform that combines rule-based heuristics with machine learning to automatically identify and resolve database performance issues across multiple database platforms.

This repository contains **Phase 1: Foundation Layer** - the core infrastructure that all subsequent phases will build upon.

## Phase 1 Components

### 🏗️ Core Infrastructure (`tuningfork.core`)
- **Base Classes**: Foundation classes for all TuningFork components
- **Exception Hierarchy**: Comprehensive error handling with context and error codes
- **Plugin System**: Extensible architecture for adding new functionality
- **Protocols**: Type-safe interfaces defining component contracts
- **Utilities**: Common utility functions for validation, formatting, and data manipulation

### ⚙️ Configuration Management (`tuningfork.config`)
- **Pydantic Models**: Type-safe configuration with validation
- **Environment Support**: Environment variable resolution and multi-environment configs
- **Security**: Secure credential handling and validation
- **Migration**: Configuration version management and migration support

### 📝 Logging Framework (`tuningfork.logging`)
- **Structured Logging**: JSON-based structured logging with context
- **Multiple Outputs**: Console, file, and remote logging support
- **Performance Monitoring**: Built-in timing and metrics collection
- **Audit Trails**: Comprehensive change tracking and audit logging

## Key Features

### 🔒 Enterprise-Grade Security
- Secure credential storage with masked serialization
- SSL/TLS configuration validation
- Role-based access patterns
- Audit logging for all operations

### 🚀 High Performance
- Async-first architecture with proper resource management
- Connection pooling and resource optimization
- Caching support and performance monitoring
- Memory-efficient data structures

### 🔧 Extensible Design
- Plugin architecture for custom extensions
- Protocol-based interfaces for type safety
- Component registry with dependency management
- Hot-reloading and configuration updates

### 🧪 Comprehensive Testing
- 95%+ test coverage with unit, integration, and performance tests
- Property-based testing for configuration validation
- Benchmarking for performance-critical paths
- Mock-based testing for external dependencies

## Installation

### Requirements
- Python 3.11+
- Dependencies listed in `pyproject.toml`

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd tuningfork

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage
pytest --cov

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m performance    # Performance tests only
```

### Production Setup

```bash
# Install production dependencies
pip install -e .

# Basic usage
python -c "
from tuningfork.core import BaseComponent
from tuningfork.config.models import SystemConfig
print('TuningFork Phase 1 installed successfully!')
"
```

## Quick Start

### Basic Component Usage

```python
from tuningfork.core import AsyncComponent
from tuningfork.config.models import DatabaseConfig, CredentialConfig
from pydantic import SecretStr

# Create database configuration
db_config = DatabaseConfig(
    id="my_database",
    platform="postgresql",
    host="localhost",
    port=5432,
    database="myapp",
    credentials=CredentialConfig(
        username="app_user",
        password=SecretStr("secure_password")
    )
)

# Create custom component
class MyDatabaseComponent(AsyncComponent[DatabaseConfig]):
    component_name = "MyDatabaseComponent"
    version = "1.0.0"
    
    async def _async_initialize(self) -> None:
        print(f"Connecting to {self.config.connection_string}")
        # Your initialization logic here
    
    async def _async_cleanup(self) -> None:
        print("Cleaning up database connection")
        # Your cleanup logic here

# Use component with lifecycle management
async def main():
    component = MyDatabaseComponent(db_config)
    
    async with component:
        # Component is automatically initialized
        print(f"Component state: {component.get_health_status()}")
        # Your business logic here
    # Component is automatically cleaned up

# Run with asyncio
import asyncio
asyncio.run(main())
```

### Plugin System Usage

```python
from tuningfork.core.plugins import BasePlugin, PluginMetadata, PluginManager

# Create custom plugin
class MyPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="my_plugin",
            name="My Custom Plugin",
            description="Example plugin implementation",
            version="1.0.0",
            plugin_type="example",
            capabilities=["data_processing", "reporting"]
        )
    
    async def _async_initialize(self) -> None:
        print("Plugin initialized!")

# Use plugin manager
async def main():
    manager = PluginManager()
    
    # Register plugin
    plugin = MyPlugin({})
    manager.register_plugin(plugin)
    
    # Initialize all plugins
    await manager.initialize_all()
    
    # Get plugin by type
    plugins = manager.get_plugins_by_type("example")
    print(f"Found {len(plugins)} example plugins")

asyncio.run(main())
```

### Configuration Management

```python
from tuningfork.config.models import SystemConfig, SecurityConfig
from pydantic import SecretStr

# Create system configuration
config = SystemConfig(
    app_name="MyApp",
    version="1.0.0",
    environment="production",
    security=SecurityConfig(
        secret_key=SecretStr("very-secure-secret-key-here"),
        require_ssl=True,
        session_timeout=3600
    )
)

# Environment variable support
# Set environment variable: TUNINGFORK_SECRET_KEY=my-secret
config_with_env = SystemConfig(
    app_name="MyApp",
    version="1.0.0",
    environment="production",
    security=SecurityConfig(
        secret_key="${TUNINGFORK_SECRET_KEY}",  # Resolved from environment
        require_ssl=True
    )
)

# Validate and serialize (with secrets masked)
print(config.to_dict(mask_secrets=True))
```

## Architecture Overview

### Component Hierarchy

```
BaseComponent
├── ConfigurableComponent
│   └── AsyncComponent
│       └── LifecycleComponent
└── SingletonComponent
```

### Exception Hierarchy

```
TuningForkException
├── ConfigurationError
│   ├── ValidationError
│   └── MigrationError
├── ConnectionError
│   ├── DatabaseConnectionError
│   ├── AuthenticationError
│   ├── NetworkError
│   └── ConnectionPoolError
├── AnalysisError
│   ├── MetadataError
│   ├── PerformanceError
│   └── QueryError
├── OptimizationError
│   ├── RecommendationError
│   ├── ApplicationError
│   ├── BackupError
│   └── RollbackError
├── PluginError
│   ├── PluginLoadError
│   ├── PluginInitializationError
│   └── PluginExecutionError
├── SecurityError
│   └── PermissionError
├── ResourceError
│   ├── MemoryError
│   └── DiskSpaceError
└── TimeoutError
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/tuningfork --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m performance

# Run tests in parallel
pytest -n auto

# Run with benchmarking
pytest --benchmark-only
```

### Test Categories

- **Unit Tests** (`tests/unit/`): Fast, isolated tests of individual components
- **Integration Tests** (`tests/integration/`): Component interaction tests
- **Performance Tests** (`tests/performance/`): Benchmarking and performance validation
- **System Tests** (`tests/system/`): End-to-end workflow tests

### Writing Tests

```python
import pytest
from tuningfork.core import BaseComponent
from tuningfork.core.exceptions import ValidationError

class TestMyComponent:
    """Test my custom component."""
    
    def test_component_initialization(self):
        """Test component initializes correctly."""
        config = MyConfig(name="test")
        component = MyComponent(config)
        
        assert component.config == config
        assert not component.is_initialized
    
    @pytest.mark.asyncio
    async def test_component_lifecycle(self):
        """Test component lifecycle management."""
        config = MyConfig(name="test")
        component = MyAsyncComponent(config)
        
        async with component:
            assert component.is_initialized
            # Test component functionality
        
        assert not component.is_initialized
    
    @pytest.mark.performance
    def test_component_performance(self, benchmark):
        """Test component performance."""
        config = MyConfig(name="perf_test")
        
        def create_component():
            return MyComponent(config)
        
        result = benchmark(create_component)
        assert result is not None
```

## Code Quality Standards

### Type Hints (Required)
- 100% type hint coverage using Python 3.11+ features
- Use `typing` and `collections.abc` for modern type annotations
- Protocol-based interfaces for component contracts

```python
from typing import Dict, List, Optional, Protocol
from collections.abc import Callable, Awaitable

class MyProtocol(Protocol):
    def process(self, data: str) -> Dict[str, Any]: ...

async def my_function(
    items: List[str],
    processor: MyProtocol,
    *,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """Process items with type safety."""
    ...
```

### Error Handling (Required)
- Use TuningFork exception hierarchy with context
- Include error codes for categorization
- Provide meaningful error messages and context

```python
from tuningfork.core.exceptions import ValidationError, ErrorCodes

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration with proper error handling."""
    if "required_field" not in config:
        raise ValidationError(
            "Missing required configuration field",
            code=ErrorCodes.CONFIG_VALIDATION_FAILED,
            context={"missing_field": "required_field", "config": config}
        )
```

### Logging (Required)
- Use structured logging with context
- Include component and operation information
- Use appropriate log levels

```python
import structlog

logger = structlog.get_logger(__name__)

class MyComponent:
    def process_data(self, data_id: str) -> None:
        log = logger.bind(
            component=self.__class__.__name__,
            data_id=data_id,
            operation="process_data"
        )
        
        log.info("Starting data processing")
        
        try:
            # Processing logic
            log.info("Data processing completed", duration_ms=42)
        except Exception as e:
            log.error("Data processing failed", error=str(e))
            raise
```

### Documentation (Required)
- Google-style docstrings for all public APIs
- Type information in docstrings
- Usage examples for complex components

```python
def complex_function(
    data: List[Dict[str, Any]],
    *,
    filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
    batch_size: int = 100
) -> Iterator[List[Dict[str, Any]]]:
    """Process data in batches with optional filtering.
    
    This function processes large datasets by splitting them into batches
    and applying optional filtering logic.
    
    Args:
        data: List of data items to process
        filter_func: Optional function to filter items
        batch_size: Number of items per batch
        
    Yields:
        Batches of processed data items
        
    Raises:
        ValidationError: If batch_size is invalid
        
    Example:
        >>> data = [{"id": 1, "active": True}, {"id": 2, "active": False}]
        >>> for batch in complex_function(data, batch_size=1):
        ...     print(f"Processing batch of {len(batch)} items")
    """
```

## Performance Guidelines

### Async/Await Best Practices
- Use async/await for all I/O operations
- Implement proper resource management with context managers
- Use connection pooling for database operations

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

class DatabaseManager:
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Connection, None]:
        """Get database connection with automatic cleanup."""
        conn = await self.pool.acquire()
        try:
            yield conn
        finally:
            await self.pool.release(conn)
    
    async def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute query with connection management."""
        async with self.get_connection() as conn:
            return await conn.execute(query)
```

### Memory Management
- Use generators for large datasets
- Implement proper cleanup in finally blocks
- Monitor memory usage in performance tests

### Caching Strategy
- Implement caching for expensive operations
- Use TTL-based expiration
- Provide cache invalidation mechanisms

## Security Considerations

### Credential Management
- Use `SecretStr` for all sensitive data
- Implement credential masking in logs and serialization
- Support credential rotation

### SSL/TLS Configuration
- Validate SSL certificates in production
- Support different verification modes
- Provide secure defaults

### Input Validation
- Validate all user inputs
- Sanitize SQL identifiers
- Use parameterized queries

## Project Structure

```
tuningfork/
├── src/tuningfork/           # Main package
│   ├── __init__.py
│   ├── core/                 # Core infrastructure
│   │   ├── __init__.py
│   │   ├── base.py          # Base classes
│   │   ├── exceptions.py    # Exception hierarchy
│   │   ├── plugins.py       # Plugin system
│   │   ├── protocols.py     # System protocols
│   │   └── utils.py         # Utility functions
│   ├── config/              # Configuration management
│   │   ├── __init__.py
│   │   ├── models.py        # Pydantic models
│   │   ├── manager.py       # Configuration manager
│   │   └── validation.py    # Custom validators
│   └── logging/             # Logging framework
│       ├── __init__.py
│       ├── structured.py    # Structured logging
│       ├── factory.py       # Logger factory
│       └── formatters.py    # Log formatters
├── tests/                   # Test suite
│   ├── conftest.py         # Test configuration
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── performance/        # Performance tests
│   └── system/             # System tests
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── pyproject.toml          # Project configuration
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

## Contributing

### Development Workflow

1. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   pre-commit install
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Implement changes**:
   - Follow code quality standards
   - Add comprehensive tests
   - Update documentation

4. **Run quality checks**:
   ```bash
   # Format code
   black src tests
   isort src tests
   
   # Type checking
   mypy src
   
   # Linting
   flake8 src tests
   
   # Tests
   pytest --cov
   ```

5. **Submit pull request**:
   - Ensure all tests pass
   - Include test coverage report
   - Update documentation as needed

### Code Review Checklist

- [ ] All tests pass with 95%+ coverage
- [ ] Code follows style guidelines (black, isort, flake8)
- [ ] Type hints are comprehensive (mypy passes)
- [ ] Documentation is complete and accurate
- [ ] Error handling includes proper context
- [ ] Performance considerations addressed
- [ ] Security best practices followed

## Roadmap

### Phase 1 (Current): Foundation Layer ✅
- Core infrastructure and base classes
- Exception hierarchy and error handling
- Plugin system architecture
- Configuration management
- Logging framework

### Phase 2: Database Integration (Next)
- Database connector implementations
- Metadata extraction engines
- Connection pooling and management
- Platform-specific optimizations

### Phase 3: Analysis Engine
- Heuristics framework
- Rule-based analysis engines
- Performance metrics collection
- Issue detection algorithms

### Phase 4: Machine Learning
- ML model training pipelines
- Prediction engines
- Feature engineering
- Model evaluation and validation

### Phase 5: Optimization Engine
- Recommendation generation
- Change application framework
- Backup and rollback systems
- Impact measurement

### Phase 6: User Interface
- Web-based dashboard
- API endpoints
- Real-time monitoring
- Reporting and analytics

## License

MIT License - see LICENSE file for details.

## Support

For questions, issues, or contributions:

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Documentation**: Check the `docs/` directory for detailed documentation
- **Testing**: Run the test suite to verify your installation
- **Development**: Follow the contributing guidelines above

## Acknowledgments

Built with modern Python best practices and enterprise-grade requirements in mind. Designed for extensibility, maintainability, and production deployment.

---

**TuningFork Phase 1** - Building the foundation for intelligent database optimization.