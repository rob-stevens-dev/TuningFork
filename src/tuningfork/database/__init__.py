"""
TuningFork Database Layer - Phase 2

This module provides database connectivity and metadata extraction capabilities
for multiple database platforms, built on the Phase 1 foundation.

Key Features:
- Unified database connector interface
- Connection pooling and resource management
- Comprehensive metadata extraction
- Platform-specific optimizations
- Integration with Phase 1 infrastructure

Supported Platforms:
- PostgreSQL (asyncpg)
- MySQL/MariaDB (aiomysql)
- Microsoft SQL Server (aioodbc)
- Oracle Database (oracledb)
- SQLite (aiosqlite)
"""

from .models import (
    QueryResult,
    DatabaseMetadata,
    TableInfo,
    ColumnInfo,
    IndexInfo,
    ConstraintInfo,
    PerformanceStats,
    ConnectionInfo
)

from .base import BaseDatabaseConnector
from .pool import ConnectionPool
from .registry import DatabaseConnectorRegistry
from .factory import DatabaseConnectorFactory

# Platform-specific connectors
from .connectors import (
    PostgreSQLConnector,
    MySQLConnector,
    MSSQLConnector,
    OracleConnector,
    SQLiteConnector
)

# Metadata extractors
from .metadata import (
    DatabaseMetadataExtractor,
    PostgreSQLMetadataExtractor,
    MySQLMetadataExtractor,
    MSSQLMetadataExtractor,
    OracleMetadataExtractor,
    SQLiteMetadataExtractor
)

__version__ = "2.0.0"
__author__ = "TuningFork Development Team"

__all__ = [
    # Models
    "QueryResult",
    "DatabaseMetadata", 
    "TableInfo",
    "ColumnInfo",
    "IndexInfo",
    "ConstraintInfo",
    "PerformanceStats",
    "ConnectionInfo",
    
    # Core classes
    "BaseDatabaseConnector",
    "ConnectionPool",
    "DatabaseConnectorRegistry",
    "DatabaseConnectorFactory",
    
    # Connectors
    "PostgreSQLConnector",
    "MySQLConnector", 
    "MSSQLConnector",
    "OracleConnector",
    "SQLiteConnector",
    
    # Metadata extractors
    "DatabaseMetadataExtractor",
    "PostgreSQLMetadataExtractor",
    "MySQLMetadataExtractor",
    "MSSQLMetadataExtractor", 
    "OracleMetadataExtractor",
    "SQLiteMetadataExtractor"
]

# Global connector registry instance
_global_registry = DatabaseConnectorRegistry()

def get_connector_registry() -> DatabaseConnectorRegistry:
    """Get the global connector registry instance."""
    return _global_registry

def create_connector(config) -> BaseDatabaseConnector:
    """Create a database connector using the global factory."""
    factory = DatabaseConnectorFactory(_global_registry)
    return factory.create_connector(config)

def register_connector_type(platform: str, connector_class) -> None:
    """Register a connector type with the global registry."""
    _global_registry.register_connector_type(platform, connector_class)

def get_supported_platforms() -> list[str]:
    """Get list of supported database platforms."""
    return _global_registry.get_supported_platforms()

# Auto-register built-in connectors
def _register_builtin_connectors():
    """Register all built-in database connectors."""
    _global_registry.register_connector_type("postgresql", PostgreSQLConnector)
    _global_registry.register_connector_type("mysql", MySQLConnector)
    _global_registry.register_connector_type("mssql", MSSQLConnector)
    _global_registry.register_connector_type("oracle", OracleConnector)
    _global_registry.register_connector_type("sqlite", SQLiteConnector)

# Register connectors on module import
_register_builtin_connectors()