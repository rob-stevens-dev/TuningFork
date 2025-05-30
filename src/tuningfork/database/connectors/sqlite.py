# src/tuningfork/database/connectors/sqlite.py
"""SQLite database connector implementation for TuningFork Phase 2."""

import aiosqlite
import sqlite3
from typing import Optional, Dict, Any, List
from pathlib import Path
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


class SQLiteConnector(AsyncComponent[DatabaseConfig]):
    """SQLite database connector implementing DatabaseConnector protocol.
    
    Extends AsyncComponent and implements DatabaseConnector protocol using
    only the actual Phase 1 interfaces and properties.
    """
    
    component_name = "SQLiteConnector"
    version = "2.0.0"
    platform = "sqlite"
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self.logger = get_logger(f"connector.sqlite.{config.id}")
        self.perf_logger = get_performance_logger("connector.sqlite")
        
        # SQLite-specific attributes
        self._connection: Optional[aiosqlite.Connection] = None
        self._database_path = Path(config.database)
        self._sqlite_version: Optional[str] = None
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to SQLite database."""
        return self.is_initialized and self._connection is not None
    
    async def _async_initialize(self) -> None:
        """Initialize SQLite connection."""
        self.logger.info("Initializing SQLite connector")
        
        try:
            # Check if database file exists
            if not self._database_path.exists():
                raise DatabaseConnectionError(
                    f"SQLite database file not found: {self._database_path}",
                    code=ErrorCodes.CONFIG_NOT_FOUND,
                    context={"database_path": str(self._database_path)}
                )
            
            # Create connection
            self._connection = await aiosqlite.connect(
                str(self._database_path),
                timeout=30
            )
            
            # Enable foreign keys and optimize settings
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.execute("PRAGMA synchronous = NORMAL")
            await self._connection.execute("PRAGMA cache_size = -2000")
            
            # Get SQLite version
            async with self._connection.execute("SELECT sqlite_version()") as cursor:
                result = await cursor.fetchone()
                self._sqlite_version = result[0] if result else "unknown"
            
            self.logger.info("SQLite connector initialized successfully")
            
        except sqlite3.DatabaseError as e:
            raise DatabaseConnectionError(
                f"SQLite database error: {e}",
                code=ErrorCodes.CONNECTION_REFUSED,
                context={"database_path": str(self._database_path)}
            ) from e
        except Exception as e:
            raise DatabaseConnectionError(
                f"Failed to initialize SQLite connector: {e}",
                code=ErrorCodes.CONNECTION_REFUSED,
                context={"database_path": str(self._database_path)}
            ) from e
    
    async def _async_cleanup(self) -> None:
        """Cleanup SQLite resources."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self.logger.info("SQLite connection closed")
    
    # DatabaseConnector protocol implementation
    async def connect(self) -> None:
        """Establish database connection."""
        await self.initialize()
    
    async def disconnect(self) -> None:
        """Close database connection."""
        await self.cleanup()
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute SQLite query."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "SQLite connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        # Measure execution time manually
        import time
        start_time = time.time()
        
        try:
            if params:
                cursor = await self._connection.execute(query, params)
            else:
                cursor = await self._connection.execute(query)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Handle different query types
            if query.strip().upper().startswith(('SELECT', 'PRAGMA')):
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
                await self._connection.commit()
                query_result = QueryResult(
                    rows=[],
                    row_count=cursor.rowcount,
                    columns=[],
                    execution_time=execution_time
                )
            
            await cursor.close()
            self.logger.info("Query executed successfully")
            return query_result
            
        except sqlite3.OperationalError as e:
            self.logger.error("Query execution failed")
            error_message = str(e).lower()
            
            if "no such table" in error_message:
                raise QueryError(
                    f"Table not found: {e}",
                    code=ErrorCodes.QUERY_EXECUTION_FAILED,
                    context={"sqlite_error": str(e)}
                ) from e
            elif "syntax error" in error_message:
                raise QueryError(
                    f"SQL syntax error: {e}",
                    code=ErrorCodes.QUERY_EXECUTION_FAILED,
                    context={"sqlite_error": str(e)}
                ) from e
            else:
                raise QueryError(
                    f"SQLite query failed: {e}",
                    code=ErrorCodes.QUERY_EXECUTION_FAILED,
                    context={"sqlite_error": str(e)}
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
        """Extract SQLite metadata."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "SQLite connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        # Measure extraction time manually
        import time
        start_time = time.time()
        
        self.logger.info("Starting SQLite metadata extraction")
        
        try:
            # Extract tables
            tables = await self._extract_tables()
            
            # Extract indexes
            indexes = await self._extract_indexes()
            
            # Get configuration
            configuration = await self._get_configuration()
            
            # Calculate extraction time
            extraction_time = time.time() - start_time
            
            metadata = DatabaseMetadata(
                platform=self.platform,
                version=self._sqlite_version,
                server_edition=None,
                current_database=str(self._database_path),
                extraction_timestamp=datetime.now(),
                extraction_duration=extraction_time,
                schemas=["main"],
                tables=tables,
                views=[],
                indexes=indexes,
                configuration=configuration,
                extraction_warnings=[]
            )
            
            self.logger.info("SQLite metadata extraction completed")
            return metadata
            
        except Exception as e:
            raise MetadataError(
                f"Failed to extract SQLite metadata: {e}",
                code=ErrorCodes.METADATA_EXTRACTION_FAILED,
                context={"platform": "sqlite", "database": str(self._database_path)}
            ) from e
    
    async def get_performance_stats(self) -> PerformanceStats:
        """Get SQLite performance statistics."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "SQLite connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        # SQLite is single-connection
        return PerformanceStats(
            timestamp=datetime.now(),
            connections_active=1,
            connections_total=1,
            queries_per_second=0.0,
            transactions_per_second=0.0,
            cache_hit_ratio=0.0
        )
    
    async def get_configuration(self) -> Dict[str, Any]:
        """Get SQLite configuration."""
        if not self.is_connected:
            raise DatabaseConnectionError(
                "SQLite connector not connected",
                code=ErrorCodes.CONNECTION_REFUSED
            )
        
        return await self._get_configuration()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            "platform": self.platform,
            "database_path": str(self._database_path),
            "connected": self.is_connected,
            "sqlite_version": self._sqlite_version
        }
    
    # Helper methods
    async def _extract_tables(self) -> List[TableInfo]:
        """Extract table information."""
        async with self._connection.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """) as cursor:
            table_rows = await cursor.fetchall()
        
        tables = []
        for table_row in table_rows:
            table_name = table_row[0]
            
            # Extract columns for this table
            columns = await self._extract_table_columns(table_name)
            
            table = TableInfo(
                schema_name="main",
                table_name=table_name,
                table_type="table",
                columns=columns,
                indexes=[]
            )
            tables.append(table)
        
        return tables
    
    async def _extract_table_columns(self, table_name: str) -> List[ColumnInfo]:
        """Extract column information for a table."""
        async with self._connection.execute(f"PRAGMA table_info({table_name})") as cursor:
            result = await cursor.fetchall()
        
        return [
            ColumnInfo(
                name=row[1],  # column name
                data_type=row[2],  # data type
                is_nullable=row[3] == 0,  # not null flag (0 = nullable)
                default_value=row[4],  # default value
                max_length=None,
                precision=None,
                scale=None,
                is_primary_key=row[5] == 1  # primary key flag
            )
            for row in result
        ]
    
    async def _extract_indexes(self) -> List[IndexInfo]:
        """Extract index information."""
        async with self._connection.execute("""
            SELECT name, tbl_name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """) as cursor:
            index_rows = await cursor.fetchall()
        
        indexes = []
        for index_row in index_rows:
            index_name = index_row[0]
            table_name = index_row[1]
            
            # Get index info
            async with self._connection.execute(f"PRAGMA index_info({index_name})") as cursor:
                index_info = await cursor.fetchall()
            
            columns = [row[2] for row in index_info]  # column names
            
            indexes.append(IndexInfo(
                schema_name="main",
                table_name=table_name,
                index_name=index_name,
                index_type="btree",
                is_unique=False,
                is_primary=False,
                columns=columns
            ))
        
        return indexes
    
    async def _get_configuration(self) -> Dict[str, Any]:
        """Get SQLite configuration via PRAGMA statements."""
        config = {}
        
        # Get important SQLite configuration values
        pragma_queries = [
            "cache_size", "foreign_keys", "synchronous", "journal_mode",
            "temp_store", "auto_vacuum", "page_size"
        ]
        
        for pragma in pragma_queries:
            try:
                async with self._connection.execute(f"PRAGMA {pragma}") as cursor:
                    result = await cursor.fetchone()
                    if result:
                        config[pragma] = result[0]
            except Exception:
                # Some PRAGMAs might not be available in all SQLite versions
                pass
        
        return config