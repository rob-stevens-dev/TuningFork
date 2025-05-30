# tests/unit/connectors/test_connectors.py
"""Unit tests for database connectors using Phase 1 testing infrastructure."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from tuningfork.config.models import DatabaseConfig, CredentialConfig
from tuningfork.core.exceptions import (
    DatabaseConnectionError,
    QueryError,
    MetadataError,
    ErrorCodes
)
from tuningfork.database.models import QueryResult, DatabaseMetadata
from tuningfork.database.connectors.postgresql import PostgreSQLConnector
from tuningfork.database.connectors.mysql import MySQLConnector
from tuningfork.database.connectors.sqlite import SQLiteConnector
from tuningfork.database.registry import DatabaseConnectorRegistry
from tuningfork.database.factory import DatabaseConnectorFactory
from pydantic import SecretStr


@pytest.fixture
def postgres_config():
    """PostgreSQL test configuration."""
    return DatabaseConfig(
        id="test_postgres",
        platform="postgresql",
        host="localhost",
        port=5432,
        database="test_db",
        credentials=CredentialConfig(
            username="test_user",
            password=SecretStr("test_password")
        )
    )


@pytest.fixture
def mysql_config():
    """MySQL test configuration."""
    return DatabaseConfig(
        id="test_mysql",
        platform="mysql",
        host="localhost",
        port=3306,
        database="test_db",
        credentials=CredentialConfig(
            username="test_user",
            password=SecretStr("test_password")
        )
    )


@pytest.fixture
def sqlite_config():
    """SQLite test configuration."""
    return DatabaseConfig(
        id="test_sqlite",
        platform="sqlite",
        host="",
        port=0,
        database="test.db",
        credentials=CredentialConfig(
            username="",
            password=SecretStr("")
        )
    )


@pytest.fixture
def mock_asyncpg_pool():
    """Mock asyncpg connection pool."""
    pool = AsyncMock()
    connection = AsyncMock()
    
    # Configure connection mock
    connection.fetchval.return_value = "PostgreSQL 15.0"
    connection.fetch.return_value = []
    connection.fetchrow.return_value = {"count": 0}
    connection.execute.return_value = None
    
    # Configure pool mock
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    pool.close = AsyncMock()
    
    return pool


@pytest.fixture
def mock_aiomysql_pool():
    """Mock aiomysql connection pool."""
    pool = AsyncMock()
    connection = AsyncMock()
    cursor = AsyncMock()
    
    # Configure cursor mock
    cursor.execute = AsyncMock()
    cursor.fetchone.return_value = ("8.0.28",)
    cursor.fetchall.return_value = []
    cursor.description = [("column1",), ("column2",)]
    cursor.rowcount = 0
    
    # Configure connection mock
    connection.cursor.return_value.__aenter__ = AsyncMock(return_value=cursor)
    connection.cursor.return_value.__aexit__ = AsyncMock(return_value=None)
    
    # Configure pool mock
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    pool.close = AsyncMock()
    pool.wait_closed = AsyncMock()
    
    return pool


class TestPostgreSQLConnector:
    """Test suite for PostgreSQL connector."""
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, postgres_config, mock_asyncpg_pool):
        """Test successful PostgreSQL connector initialization."""
        connector = PostgreSQLConnector(postgres_config)
        
        with patch('asyncpg.create_pool', return_value=mock_asyncpg_pool):
            await connector.initialize()
            
            assert connector.is_initialized
            assert connector.is_connected
            assert connector.platform == "postgresql"
    
    @pytest.mark.asyncio
    async def test_initialization_auth_failure(self, postgres_config):
        """Test PostgreSQL authentication failure handling."""
        import asyncpg
        
        connector = PostgreSQLConnector(postgres_config)
        
        with patch('asyncpg.create_pool', side_effect=asyncpg.InvalidAuthorizationSpecificationError("auth failed")):
            with pytest.raises(DatabaseConnectionError) as exc_info:
                await connector.initialize()
            
            assert exc_info.value.code == ErrorCodes.AUTH_FAILED
            assert "auth failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_initialization_timeout(self, postgres_config):
        """Test PostgreSQL connection timeout handling."""
        import asyncpg
        
        connector = PostgreSQLConnector(postgres_config)
        
        with patch('asyncpg.create_pool', side_effect=asyncpg.ConnectionTimeoutError("timeout")):
            with pytest.raises(DatabaseConnectionError) as exc_info:
                await connector.initialize()
            
            assert exc_info.value.code == ErrorCodes.CONNECTION_TIMEOUT
    
    @pytest.mark.asyncio
    async def test_query_execution_simple(self, postgres_config, mock_asyncpg_pool):
        """Test simple PostgreSQL query execution."""
        connector = PostgreSQLConnector(postgres_config)
        
        # Configure mock response
        mock_connection = mock_asyncpg_pool.acquire.return_value.__aenter__.return_value
        mock_result = [{"id": 1, "name": "test"}]
        mock_connection.fetch.return_value = mock_result
        
        with patch('asyncpg.create_pool', return_value=mock_asyncpg_pool):
            await connector.initialize()
            
            result = await connector.execute_query("SELECT id, name FROM users")
            
            assert isinstance(result, QueryResult)
            assert result.row_count == 1
            assert result.rows[0]["id"] == 1
            assert result.rows[0]["name"] == "test"
    
    @pytest.mark.asyncio
    async def test_query_execution_with_parameters(self, postgres_config, mock_asyncpg_pool):
        """Test PostgreSQL parameterized query execution."""
        connector = PostgreSQLConnector(postgres_config)
        
        # Configure mock response
        mock_connection = mock_asyncpg_pool.acquire.return_value.__aenter__.return_value
        mock_result = [{"id": 1, "age": 25}]
        mock_connection.fetch.return_value = mock_result
        
        with patch('asyncpg.create_pool', return_value=mock_asyncpg_pool):
            await connector.initialize()
            
            result = await connector.execute_query(
                "SELECT id, age FROM users WHERE age > :min_age", 
                {"min_age": 25}
            )
            
            assert result.row_count == 1
            assert result.rows[0]["age"] == 25
    
    @pytest.mark.asyncio
    async def test_query_error_table_not_found(self, postgres_config, mock_asyncpg_pool):
        """Test PostgreSQL table not found error mapping."""
        import asyncpg
        
        connector = PostgreSQLConnector(postgres_config)
        
        # Configure mock to raise PostgreSQL error
        mock_connection = mock_asyncpg_pool.acquire.return_value.__aenter__.return_value
        postgres_error = asyncpg.UndefinedTableError("relation 'nonexistent' does not exist")
        postgres_error.sqlstate = "42P01"
        postgres_error.message = "relation 'nonexistent' does not exist"
        mock_connection.fetch.side_effect = postgres_error
        
        with patch('asyncpg.create_pool', return_value=mock_asyncpg_pool):
            await connector.initialize()
            
            with pytest.raises(QueryError) as exc_info:
                await connector.execute_query("SELECT * FROM nonexistent")
            
            assert exc_info.value.code == ErrorCodes.QUERY_EXECUTION_FAILED
    
    @pytest.mark.asyncio
    async def test_metadata_extraction(self, postgres_config, mock_asyncpg_pool):
        """Test PostgreSQL metadata extraction."""
        connector = PostgreSQLConnector(postgres_config)
        
        # Configure mock responses for metadata queries
        mock_connection = mock_asyncpg_pool.acquire.return_value.__aenter__.return_value
        mock_connection.fetch.side_effect = [
            [{"schema_name": "public"}],  # Schemas
            [{"schema_name": "public", "table_name": "users", "table_type": "table"}],  # Tables
            [{"column_name": "id", "data_type": "integer", "is_nullable": False, "default_value": None}],  # Columns
            [{"schema_name": "public", "table_name": "users", "index_name": "idx_users_id", "index_definition": "..."}],  # Indexes
            [{"name": "shared_buffers", "setting": "128MB"}]  # Configuration
        ]
        
        with patch('asyncpg.create_pool', return_value=mock_asyncpg_pool):
            await connector.initialize()
            
            metadata = await connector.get_metadata()
            
            assert isinstance(metadata, DatabaseMetadata)
            assert metadata.platform == "postgresql"
            assert len(metadata.schemas) == 1
            assert metadata.schemas[0] == "public"
            assert len(metadata.tables) == 1
            assert metadata.tables[0].table_name == "users"


class TestMySQLConnector:
    """Test suite for MySQL connector."""
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, mysql_config, mock_aiomysql_pool):
        """Test successful MySQL connector initialization."""
        connector = MySQLConnector(mysql_config)
        
        with patch('aiomysql.create_pool', return_value=mock_aiomysql_pool):
            await connector.initialize()
            
            assert connector.is_initialized
            assert connector.is_connected
            assert connector.platform == "mysql"
    
    @pytest.mark.asyncio
    async def test_initialization_auth_failure(self, mysql_config):
        """Test MySQL authentication failure handling."""
        import aiomysql
        
        connector = MySQLConnector(mysql_config)
        
        mysql_error = aiomysql.OperationalError(1045, "Access denied")
        with patch('aiomysql.create_pool', side_effect=mysql_error):
            with pytest.raises(DatabaseConnectionError) as exc_info:
                await connector.initialize()
            
            assert exc_info.value.code == ErrorCodes.AUTH_FAILED
    
    @pytest.mark.asyncio
    async def test_query_execution_select(self, mysql_config, mock_aiomysql_pool):
        """Test MySQL SELECT query execution."""
        connector = MySQLConnector(mysql_config)
        
        # Configure mock response
        mock_connection = mock_aiomysql_pool.acquire.return_value.__aenter__.return_value
        mock_cursor = mock_connection.cursor.return_value.__aenter__.return_value
        mock_cursor.fetchall.return_value = [(1, "test"), (2, "test2")]
        mock_cursor.description = [("id",), ("name",)]
        
        with patch('aiomysql.create_pool', return_value=mock_aiomysql_pool):
            await connector.initialize()
            
            result = await connector.execute_query("SELECT id, name FROM users")
            
            assert result.row_count == 2
            assert len(result.columns) == 2
            assert result.columns == ["id", "name"]


class TestSQLiteConnector:
    """Test suite for SQLite connector."""
    
    @pytest.mark.asyncio
    async def test_initialization_file_not_found(self, sqlite_config):
        """Test SQLite file not found error handling."""
        connector = SQLiteConnector(sqlite_config)
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            await connector.initialize()
        
        assert exc_info.value.code == ErrorCodes.CONFIG_NOT_FOUND
        assert "not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, sqlite_config):
        """Test successful SQLite connector initialization."""
        connector = SQLiteConnector(sqlite_config)
        
        # Mock file existence and connection
        with patch.object(connector._database_path, 'exists', return_value=True), \
             patch('aiosqlite.connect') as mock_connect:
            
            mock_connection = AsyncMock()
            mock_cursor = AsyncMock()
            mock_cursor.fetchone.return_value = ("3.39.0",)
            
            mock_connection.execute.return_value.__aenter__ = AsyncMock(return_value=mock_cursor)
            mock_connection.execute.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_connect.return_value = mock_connection
            
            await connector.initialize()
            
            assert connector.is_initialized
            assert connector.is_connected
            assert connector.platform == "sqlite"


class TestDatabaseConnectorRegistry:
    """Test suite for database connector registry."""
    
    def test_register_connector(self):
        """Test connector registration."""
        registry = DatabaseConnectorRegistry()
        
        registry.register_connector("postgresql", PostgreSQLConnector, "PostgreSQL connector", "2.0.0")
        
        assert registry.is_platform_supported("postgresql")
        assert "postgresql" in registry.get_available_platforms()
        
        connector_class = registry.get_connector_class("postgresql")
        assert connector_class == PostgreSQLConnector
    
    def test_register_invalid_connector(self):
        """Test registration of invalid connector class."""
        registry = DatabaseConnectorRegistry()
        
        class InvalidConnector:
            pass
        
        with pytest.raises(TuningForkException) as exc_info:
            registry.register_connector("invalid", InvalidConnector)
        
        assert exc_info.value.code == ErrorCodes.CONFIG_INVALID
    
    def test_create_connector(self, postgres_config):
        """Test connector creation through registry."""
        registry = DatabaseConnectorRegistry()
        registry.register_connector("postgresql", PostgreSQLConnector)
        
        connector = registry.create_connector(postgres_config)
        
        assert isinstance(connector, PostgreSQLConnector)
        assert connector.platform == "postgresql"
    
    def test_get_unsupported_platform(self):
        """Test getting connector for unsupported platform."""
        registry = DatabaseConnectorRegistry()
        
        with pytest.raises(TuningForkException) as exc_info:
            registry.get_connector_class("unsupported")
        
        assert exc_info.value.code == ErrorCodes.CONFIG_NOT_FOUND


class TestDatabaseConnectorFactory:
    """Test suite for database connector factory."""
    
    def test_create_connector_success(self, postgres_config):
        """Test successful connector creation via factory."""
        factory = DatabaseConnectorFactory()
        
        # Register connector first
        factory._registry.register_connector("postgresql", PostgreSQLConnector)
        
        connector = factory.create_connector(postgres_config)
        
        assert isinstance(connector, PostgreSQLConnector)
        assert connector.platform == "postgresql"
    
    def test_create_connector_unsupported_platform(self, postgres_config):
        """Test factory handling of unsupported platform."""
        factory = DatabaseConnectorFactory()
        
        # Don't register any connectors
        with pytest.raises(TuningForkException) as exc_info:
            factory.create_connector(postgres_config)
        
        assert exc_info.value.code == ErrorCodes.CONFIG_INVALID
    
    def test_validate_configuration_success(self, postgres_config):
        """Test configuration validation success."""
        factory = DatabaseConnectorFactory()
        factory._registry.register_connector("postgresql", PostgreSQLConnector)
        
        is_valid = factory.validate_configuration(postgres_config)
        assert is_valid
    
    def test_validate_configuration_missing_field(self):
        """Test configuration validation with missing required field."""
        factory = DatabaseConnectorFactory()
        factory._registry.register_connector("postgresql", PostgreSQLConnector)
        
        # Create config with missing database field
        invalid_config = DatabaseConfig(
            id="",  # Missing required field
            platform="postgresql",
            host="localhost",
            port=5432,
            database="test_db",
            credentials=CredentialConfig(username="test", password=SecretStr("test"))
        )
        
        is_valid = factory.validate_configuration(invalid_config)
        assert not is_valid
    
    def test_create_connector_from_dict(self):
        """Test creating connector from configuration dictionary."""
        factory = DatabaseConnectorFactory()
        factory._registry.register_connector("postgresql", PostgreSQLConnector)
        
        config_dict = {
            "id": "test_postgres",
            "platform": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "credentials": {
                "username": "test_user",
                "password": "test_password"
            }
        }
        
        connector = factory.create_connector_from_dict(config_dict)
        assert isinstance(connector, PostgreSQLConnector)
        assert connector.platform == "postgresql"