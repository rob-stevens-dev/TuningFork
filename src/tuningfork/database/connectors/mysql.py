# src/tuningfork/database/connectors/mysql.py
"""MySQL/MariaDB database connector implementation for TuningFork Phase 2."""

import aiomysql
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


class MySQLConnector(AsyncComponent[DatabaseConfig]):
    """MySQL/MariaDB database connector implementing DatabaseConnector protocol.
    
    Extends AsyncComponent and implements DatabaseConnector protocol using
    only the actual Phase 1 interfaces and properties.
    """
    
    component_name = "MySQLConnector"
    version = "2.0.0"
    platform = "mysql"
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self.logger = get_logger(f"connector.mysql.{config.id}")
        self.perf_logger = get_performance_logger("connector.mysql")
        
        # MySQL-specific attributes
        self._connection_pool: Optional[aiomysql.Pool] = None
        self._server_version: Optional[str] = None
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to MySQL database."""
        return self.is_initialized and self._connection_pool is not None
    
    async def _async_initialize(self) -> None:
        """Initialize MySQL connection pool."""
        self.logger.info("Initializing MySQL connector")
        
        try:
            # Create aiomysql connection pool with basic configuration
            self._connection_pool = await aiomysql.create_pool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.database,
                user=self.config.credentials.username,
                password=self.config.credentials.password.get_secret_value(),
                charset='utf8mb4',
                minsize=1,
                maxsize=10,
                autocommit=True
            )
            
            # Get server information
            async with self._connection_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute('SELECT VERSION()')
                    result = await cursor.fetchone()
                    self._server_version = result[0] if result else "unknown"
            
            self.logger.info("MySQL connector initialized successfully")
            
        except aiomysql.OperationalError as e:
            error_code = e.args[0] if e.args else 0
            
            if error_code == 1045:  # Access denied
                raise DatabaseConnectionError(
                    f"MySQL authentication failed: {e}",
                    code=ErrorCodes.AUTH_FAILED,
                    context={
                        "host": self.config.host,
                        "port": self.config.port,
                        "database": self.config.database
                    }
                ) from e
            elif error_code == 2003:  # Can't connect to MySQL server
                raise DatabaseConnectionError(
                    f"MySQL connection refused: {e}",
                    code=ErrorCodes.CONNECTION_REFUSED,
                    context={
                        "host": self.config.host,
                        "port": self.config.port
                    }
                ) from e
            else:
                raise DatabaseConnectionError(
                    f"MySQL connection failed: {e}",
                    code=ErrorCodes.CONNECTION_REFUSED,
                    context={
                        "host": self.config.host,
                        "port": self.config.port
                    }
                ) from e
        except Exception as e:
            raise DatabaseConnectionError(
                f"Failed to initialize MySQL connector: {e}",
                code=ErrorCodes.NETWORK_UNREACHABLE,
                context={
                    "host": self.config.host,
                    "port": self.config.port,
                    "database": self.config.database
                }
            ) from e
    
    async def _async_cleanup(self) -> None:
        """Cleanup MySQL resources."""
        if self._connection_pool:
            self._connection_pool.close()
            await self._connection_pool.wait_closed()
            self._connection_pool = None
            self.logger.info("MySQL connection pool closed")
    
    # DatabaseConnector protocol implementation
    async def connect(self) -> None:
        """Establish database connection."""
        await self.initialize()
    
    async def disconnect(self) -> None:
        """Close database connection."""
        await self.cleanup()
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute MySQL query."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "MySQL connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        # Measure execution time manually
        import time
        start_time = time.time()
        
        try:
            async with self._connection_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    if params:
                        await cursor.execute(query, params)
                    else:
                        await cursor.execute(query)
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    
                    # Handle different query types
                    if query.strip().upper().startswith(('SELECT', 'SHOW', 'DESCRIBE')):
                        rows = await cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        query_result = QueryResult(
                            rows=[dict(zip(columns, row)) for row in rows] if rows else [],
                            row_count=len(rows) if rows else 0,
                            columns=columns,
                            execution_time=execution_time
                        )
                    else:
                        # INSERT, UPDATE, DELETE, etc.
                        query_result = QueryResult(
                            rows=[],
                            row_count=cursor.rowcount,
                            columns=[],
                            execution_time=execution_time
                        )
                    
                    self.logger.info("Query executed successfully")
                    return query_result
                    
        except aiomysql.Error as e:
            self.logger.error("Query execution failed")
            error_code = e.args[0] if e.args else 0
            
            if error_code == 1146:  # Table doesn't exist
                raise QueryError(
                    f"Table not found: {e}",
                    code=ErrorCodes.QUERY_EXECUTION_FAILED,
                    context={"mysql_error_code": error_code}
                ) from e
            elif error_code == 1064:  # Syntax error
                raise QueryError(
                    f"SQL syntax error: {e}",
                    code=ErrorCodes.QUERY_EXECUTION_FAILED,
                    context={"mysql_error_code": error_code}
                ) from e
            elif error_code == 1142:  # Access denied
                raise QueryError(
                    f"Access denied: {e}",
                    code=ErrorCodes.INSUFFICIENT_PERMISSIONS,
                    context={"mysql_error_code": error_code}
                ) from e
            else:
                raise QueryError(
                    f"MySQL query failed: {e}",
                    code=ErrorCodes.QUERY_EXECUTION_FAILED,
                    context={"mysql_error_code": error_code}
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
        """Extract MySQL metadata."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "MySQL connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        # Measure extraction time manually
        import time
        start_time = time.time()
        
        self.logger.info("Starting MySQL metadata extraction")
        
        try:
            async with self._connection_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Extract schemas
                    schemas = await self._extract_schemas(cursor)
                    
                    # Extract tables
                    tables = await self._extract_tables(cursor)
                    
                    # Extract indexes
                    indexes = await self._extract_indexes(cursor)
                    
                    # Get configuration
                    configuration = await self._get_configuration(cursor)
                    
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
                    
                    self.logger.info("MySQL metadata extraction completed")
                    return metadata
                    
        except Exception as e:
            raise MetadataError(
                f"Failed to extract MySQL metadata: {e}",
                code=ErrorCodes.METADATA_EXTRACTION_FAILED,
                context={"platform": "mysql", "database": self.config.database}
            ) from e
    
    async def get_performance_stats(self) -> PerformanceStats:
        """Get MySQL performance statistics."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "MySQL connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Get connection stats
                await cursor.execute("SHOW STATUS LIKE 'Threads_connected'")
                conn_result = await cursor.fetchone()
                active_connections = int(conn_result[1]) if conn_result else 0
                
                return PerformanceStats(
                    timestamp=datetime.now(),
                    connections_active=active_connections,
                    connections_total=active_connections,
                    queries_per_second=0.0,
                    transactions_per_second=0.0,
                    cache_hit_ratio=0.0
                )
    
    async def get_configuration(self) -> Dict[str, Any]:
        """Get MySQL configuration."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "MySQL connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                return await self._get_configuration(cursor)
    
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
    async def _extract_schemas(self, cursor) -> List[str]:
        """Extract schema names."""
        await cursor.execute("SHOW DATABASES")
        result = await cursor.fetchall()
        return [row[0] for row in result 
                if row[0] not in ('information_schema', 'performance_schema', 'mysql', 'sys')]
    
    async def _extract_tables(self, cursor) -> List[TableInfo]:
        """Extract table information."""
        tables_query = """
            SELECT 
                TABLE_SCHEMA as schema_name,
                TABLE_NAME as table_name,
                TABLE_TYPE as table_type
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = %s
            ORDER BY TABLE_NAME
        """
        
        await cursor.execute(tables_query, (self.config.database,))
        result = await cursor.fetchall()
        tables = []
        
        for row in result:
            # Extract columns for this table
            columns = await self._extract_table_columns(
                cursor,
                row[0],  # schema_name
                row[1]   # table_name
            )
            
            table = TableInfo(
                schema_name=row[0],
                table_name=row[1],
                table_type='table' if row[2] == 'BASE TABLE' else 'view',
                columns=columns,
                indexes=[]
            )
            tables.append(table)
        
        return tables
    
    async def _extract_table_columns(self, cursor, schema_name: str, table_name: str) -> List[ColumnInfo]:
        """Extract column information for a table."""
        columns_query = """
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
        """
        
        await cursor.execute(columns_query, (schema_name, table_name))
        result = await cursor.fetchall()
        return [
            ColumnInfo(
                name=row[0],
                data_type=row[1],
                is_nullable=row[2] == 'YES',
                default_value=row[3],
                max_length=None,
                precision=None,
                scale=None
            )
            for row in result
        ]
    
    async def _extract_indexes(self, cursor) -> List[IndexInfo]:
        """Extract index information."""
        indexes_query = """
            SELECT 
                TABLE_SCHEMA as schema_name,
                TABLE_NAME as table_name,
                INDEX_NAME as index_name,
                INDEX_TYPE as index_type,
                NON_UNIQUE
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = %s
            GROUP BY TABLE_SCHEMA, TABLE_NAME, INDEX_NAME, INDEX_TYPE, NON_UNIQUE
            ORDER BY TABLE_SCHEMA, TABLE_NAME, INDEX_NAME
        """
        
        await cursor.execute(indexes_query, (self.config.database,))
        result = await cursor.fetchall()
        return [
            IndexInfo(
                schema_name=row[0],
                table_name=row[1],
                index_name=row[2],
                index_type=row[3].lower(),
                is_unique=row[4] == 0,
                is_primary=row[2] == 'PRIMARY',
                columns=[]
            )
            for row in result
        ]
    
    async def _get_configuration(self, cursor) -> Dict[str, Any]:
        """Get MySQL configuration."""
        config_variables = [
            'innodb_buffer_pool_size', 'max_connections', 'table_open_cache',
            'query_cache_size', 'sort_buffer_size', 'join_buffer_size'
        ]
        
        config = {}
        for var in config_variables:
            await cursor.execute(f"SHOW VARIABLES LIKE '{var}'")
            result = await cursor.fetchone()
            if result:
                config[result[0]] = result[1]
        
        return config