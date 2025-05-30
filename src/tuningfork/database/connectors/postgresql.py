# src/tuningfork/database/connectors/postgresql.py
"""PostgreSQL database connector implementation for TuningFork Phase 2."""

import asyncio
import asyncpg
from typing import Optional, Dict, Any, List
from datetime import datetime

from tuningfork.core import AsyncComponent
from tuningfork.core.protocols import DatabaseConnector
from tuningfork.config.models import DatabaseConfig
from tuningfork.logging import get_logger, get_performance_logger
from tuningfork.core.exceptions import (
    DatabaseConnectionError, 
    QueryError, 
    MetadataError,
    ErrorCodes
)
from tuningfork.database.models import (
    DatabaseMetadata, 
    QueryResult, 
    TableInfo, 
    ColumnInfo, 
    IndexInfo,
    PerformanceStats
)


class PostgreSQLConnector(AsyncComponent[DatabaseConfig]):
    """PostgreSQL database connector implementing DatabaseConnector protocol.
    
    Extends AsyncComponent and implements DatabaseConnector protocol using
    only the actual Phase 1 interfaces and properties.
    """
    
    component_name = "PostgreSQLConnector"
    version = "2.0.0"
    platform = "postgresql"
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self.logger = get_logger(f"connector.postgresql.{config.id}")
        self.perf_logger = get_performance_logger("connector.postgresql")
        
        # PostgreSQL-specific attributes
        self._connection_pool: Optional[asyncpg.Pool] = None
        self._server_version: Optional[str] = None
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to PostgreSQL database."""
        return self.is_initialized and self._connection_pool is not None
    
    async def _async_initialize(self) -> None:
        """Initialize PostgreSQL connection pool."""
        self.logger.info("Initializing PostgreSQL connector")
        
        try:
            # Create asyncpg connection pool with basic configuration
            self._connection_pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.credentials.username,
                password=self.config.credentials.password.get_secret_value(),
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            
            # Get server version
            async with self._connection_pool.acquire() as conn:
                version_result = await conn.fetchval('SELECT version()')
                self._server_version = self._parse_postgres_version(version_result)
            
            self.logger.info("PostgreSQL connector initialized successfully")
            
        except asyncpg.InvalidAuthorizationSpecificationError as e:
            raise DatabaseConnectionError(
                f"PostgreSQL authentication failed: {e}",
                code=ErrorCodes.AUTH_FAILED,
                context={
                    "host": self.config.host,
                    "port": self.config.port,
                    "database": self.config.database
                }
            ) from e
        except asyncpg.ConnectionTimeoutError as e:
            raise DatabaseConnectionError(
                f"PostgreSQL connection timeout: {e}",
                code=ErrorCodes.CONNECTION_TIMEOUT,
                context={
                    "host": self.config.host,
                    "port": self.config.port
                }
            ) from e
        except Exception as e:
            raise DatabaseConnectionError(
                f"Failed to initialize PostgreSQL connector: {e}",
                code=ErrorCodes.CONNECTION_REFUSED,
                context={
                    "host": self.config.host,
                    "port": self.config.port,
                    "database": self.config.database
                }
            ) from e
    
    async def _async_cleanup(self) -> None:
        """Cleanup PostgreSQL resources."""
        if self._connection_pool:
            await self._connection_pool.close()
            self._connection_pool = None
            self.logger.info("PostgreSQL connection pool closed")
    
    # DatabaseConnector protocol implementation
    async def connect(self) -> None:
        """Establish database connection."""
        await self.initialize()
    
    async def disconnect(self) -> None:
        """Close database connection."""
        await self.cleanup()
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute PostgreSQL query."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "PostgreSQL connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        # Measure execution time manually since timer properties aren't available inside context
        import time
        start_time = time.time()
        
        try:
            async with self._connection_pool.acquire() as conn:
                if params:
                    # Convert named parameters to positional for asyncpg
                    positional_params = list(params.values())
                    positional_query = self._convert_named_to_positional(query, params)
                    result = await conn.fetch(positional_query, *positional_params)
                else:
                    result = await conn.fetch(query)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Convert to standardized format
                query_result = QueryResult(
                    rows=[dict(row) for row in result],
                    row_count=len(result),
                    columns=[key for key in result[0].keys()] if result else [],
                    execution_time=execution_time,
                    warnings=[]
                )
                
                self.logger.info("Query executed successfully")
                return query_result
                
        except asyncpg.UndefinedTableError as e:
            self.logger.error("Table not found error")
            raise QueryError(
                f"Table not found: {e}",
                code=ErrorCodes.QUERY_EXECUTION_FAILED,
                context={"message": str(e)}
            ) from e
        except asyncpg.SyntaxOrAccessError as e:
            self.logger.error("SQL syntax error")
            raise QueryError(
                f"SQL syntax error: {e}",
                code=ErrorCodes.QUERY_EXECUTION_FAILED,
                context={"message": str(e)}
            ) from e
        except asyncpg.InsufficientPrivilegeError as e:
            self.logger.error("Insufficient privileges")
            raise QueryError(
                f"Insufficient privileges: {e}",
                code=ErrorCodes.INSUFFICIENT_PERMISSIONS,
                context={"message": str(e)}
            ) from e
        except Exception as e:
            self.logger.error("Query execution failed")
            raise
    
    async def execute_many(self, query: str, param_list: List[Dict[str, Any]]) -> List[QueryResult]:
        """Execute query with multiple parameter sets."""
        results = []
        for params in param_list:
            result = await self.execute_query(query, params)
            results.append(result)
        return results
    
    async def get_metadata(self) -> DatabaseMetadata:
        """Extract PostgreSQL metadata."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "PostgreSQL connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        # Measure extraction time manually
        import time
        start_time = time.time()
        
        self.logger.info("Starting PostgreSQL metadata extraction")
        
        try:
            async with self._connection_pool.acquire() as conn:
                # Extract schemas
                schemas = await self._extract_schemas(conn)
                
                # Extract tables
                tables = await self._extract_tables(conn)
                
                # Extract indexes
                indexes = await self._extract_indexes(conn)
                
                # Get basic configuration
                configuration = await self._get_configuration(conn)
                
                # Calculate extraction time
                extraction_time = time.time() - start_time
                
                metadata = DatabaseMetadata(
                    platform=self.platform,
                    version=self._server_version,
                    server_edition=None,
                    current_database=self.config.database,
                    extraction_timestamp=datetime.now(),
                    extraction_duration=extraction_time,
                    schemas=schemas,
                    tables=tables,
                    views=[],
                    indexes=indexes,
                    configuration=configuration,
                    extraction_warnings=[]
                )
                
                self.logger.info("PostgreSQL metadata extraction completed")
                return metadata
                
        except Exception as e:
            raise MetadataError(
                f"Failed to extract PostgreSQL metadata: {e}",
                code=ErrorCodes.METADATA_EXTRACTION_FAILED,
                context={"platform": "postgresql", "database": self.config.database}
            ) from e
    
    async def get_performance_stats(self) -> PerformanceStats:
        """Get PostgreSQL performance statistics."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "PostgreSQL connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        async with self._connection_pool.acquire() as conn:
            # Get basic connection stats
            connections_query = """
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections
                FROM pg_stat_activity
            """
            conn_result = await conn.fetchrow(connections_query)
            
            return PerformanceStats(
                timestamp=datetime.now(),
                connections_active=conn_result['active_connections'] if conn_result else 0,
                connections_total=conn_result['total_connections'] if conn_result else 0,
                queries_per_second=0.0,
                transactions_per_second=0.0,
                cache_hit_ratio=0.0
            )
    
    async def get_configuration(self) -> Dict[str, Any]:
        """Get PostgreSQL configuration."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "PostgreSQL connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        async with self._connection_pool.acquire() as conn:
            return await self._get_configuration(conn)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            "platform": self.platform,
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database,
            "connected": self.is_connected,
            "server_version": self._server_version
        }
    
    # Helper methods
    def _parse_postgres_version(self, version_string: str) -> str:
        """Parse PostgreSQL version string."""
        import re
        match = re.search(r'PostgreSQL (\d+\.\d+)', version_string)
        return match.group(1) if match else "unknown"
    
    def _convert_named_to_positional(self, query: str, params: Dict[str, Any]) -> str:
        """Convert named parameters to positional for asyncpg."""
        positional_query = query
        for i, key in enumerate(params.keys(), 1):
            positional_query = positional_query.replace(f':{key}', f'${i}')
        return positional_query
    
    async def _extract_schemas(self, conn) -> List[str]:
        """Extract schema names."""
        result = await conn.fetch("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            ORDER BY schema_name
        """)
        return [row['schema_name'] for row in result]
    
    async def _extract_tables(self, conn) -> List[TableInfo]:
        """Extract table information."""
        tables_query = """
            SELECT 
                schemaname as schema_name,
                tablename as table_name,
                'table' as table_type
            FROM pg_tables 
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
            ORDER BY schemaname, tablename
        """
        
        result = await conn.fetch(tables_query)
        tables = []
        
        for row in result:
            # Extract columns for this table
            columns = await self._extract_table_columns(
                conn, 
                row['schema_name'], 
                row['table_name']
            )
            
            table = TableInfo(
                schema_name=row['schema_name'],
                table_name=row['table_name'],
                table_type=row['table_type'],
                columns=columns,
                indexes=[]
            )
            tables.append(table)
        
        return tables
    
    async def _extract_table_columns(self, conn, schema_name: str, table_name: str) -> List[ColumnInfo]:
        """Extract column information for a table."""
        columns_query = """
            SELECT 
                column_name,
                data_type,
                is_nullable::boolean as is_nullable,
                column_default as default_value
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position
        """
        
        result = await conn.fetch(columns_query, schema_name, table_name)
        return [
            ColumnInfo(
                name=row['column_name'],
                data_type=row['data_type'],
                is_nullable=row['is_nullable'],
                default_value=row['default_value'],
                max_length=None,
                precision=None,
                scale=None
            )
            for row in result
        ]
    
    async def _extract_indexes(self, conn) -> List[IndexInfo]:
        """Extract index information."""
        indexes_query = """
            SELECT 
                schemaname as schema_name,
                tablename as table_name,
                indexname as index_name,
                indexdef as index_definition
            FROM pg_indexes
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
            ORDER BY schemaname, tablename, indexname
        """
        
        result = await conn.fetch(indexes_query)
        return [
            IndexInfo(
                schema_name=row['schema_name'],
                table_name=row['table_name'],
                index_name=row['index_name'],
                index_type='btree',
                is_unique=False,
                is_primary=False,
                columns=[]
            )
            for row in result
        ]
    
    async def _get_configuration(self, conn) -> Dict[str, Any]:
        """Get PostgreSQL configuration."""
        config_query = """
            SELECT name, setting
            FROM pg_settings
            WHERE name IN (
                'shared_buffers', 'work_mem', 'maintenance_work_mem',
                'max_connections', 'max_parallel_workers_per_gather'
            )
        """
        
        result = await conn.fetch(config_query)
        return {row['name']: row['setting'] for row in result}