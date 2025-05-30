from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, TypeVar

from tuningfork.config.models import DatabaseConfig
from tuningfork.core import AsyncComponent
from tuningfork.core.exceptions import DatabaseConnectionError, ErrorCodes
from tuningfork.database.models.query_result import QueryResult
from tuningfork.database.models.database_metadata import DatabaseMetadata
from tuningfork.logging import get_logger, get_performance_logger

T = TypeVar("T")


class BaseDatabaseConnector(AsyncComponent[DatabaseConfig], ABC):
    """
    Abstract base class for all database connectors.
    Provides lifecycle management, logging, and metadata caching.
    """

    platform: str = "unknown"
    required_version: str = "1.0"

    def __init__(self, config: DatabaseConfig) -> None:
        super().__init__(config)
        self.logger = get_logger(f"connector.{self.platform}.{config.id}")
        self.perf_logger = get_performance_logger(f"connector.{self.platform}")
        self._connection_pool: Any = None
        self._server_version: Optional[str] = None
        self._metadata_cache: Dict[str, Any] = {}
        self._cache_ttl: int = 300
        self._last_cache_update: Optional[float] = None

    async def connect(self) -> None:
        """Establish database connection via component initialization."""
        await self.initialize()

    async def disconnect(self) -> None:
        """Close database connection via component cleanup."""
        await self.cleanup()

    def is_connected(self) -> bool:
        """Return True if connector is initialized and has a connection pool."""
        return self.is_initialized and self._connection_pool is not None

    def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection details for monitoring and diagnostics."""
        return {
            "platform": self.platform,
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database,
            "connected": self.is_connected(),
            "server_version": self._server_version,
        }

    async def get_metadata(self) -> DatabaseMetadata:
        """Return cached metadata or extract fresh metadata if cache is stale."""
        if self._is_metadata_cache_valid():
            self.logger.debug("Returning cached metadata")
            return self._metadata_cache["data"]

        with self.perf_logger.measure("metadata_extraction", platform=self.platform) as timer:
            self.logger.info("Extracting database metadata")
            try:
                metadata = await self._extract_metadata_impl()
                self._metadata_cache = {
                    "data": metadata,
                    "timestamp": time.time(),
                    "ttl": self._cache_ttl,
                }
                self.logger.info(
                    "Metadata extraction completed",
                    duration=timer.duration,
                    tables_found=len(metadata.tables),
                    indexes_found=len(metadata.indexes),
                )
                return metadata
            except Exception as e:
                raise DatabaseConnectionError(
                    f"Failed to extract metadata: {e}",
                    code=ErrorCodes.METADATA_EXTRACT_FAILED,
                    context={"platform": self.platform, "database": self.config.database},
                    cause=e,
                )

    def _is_metadata_cache_valid(self) -> bool:
        """Return True if metadata cache is valid."""
        if "timestamp" not in self._metadata_cache:
            return False
        return (time.time() - self._metadata_cache["timestamp"]) < self._cache_ttl

    @abstractmethod
    async def _create_connection_pool(self) -> Any:
        """Create and return a platform-specific async connection pool."""
        pass

    @abstractmethod
    async def _execute_query_impl(self, query: str, params: Optional[Dict[str, Any]]) -> QueryResult:
        """Execute the query using a raw connection. Return standardized result."""
        pass

    @abstractmethod
    async def _extract_metadata_impl(self) -> DatabaseMetadata:
        """Extract and return platform-specific metadata."""
        pass
