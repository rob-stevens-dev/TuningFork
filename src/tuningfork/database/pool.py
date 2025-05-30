"""
Connection pool implementation for TuningFork database connectors.

Provides async connection pooling with resource management, health checking,
and comprehensive monitoring for all database platforms.
"""

import asyncio
import time
import weakref
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Dict, Optional, Set, Union
from datetime import datetime, timedelta

from tuningfork.logging import get_logger, get_performance_logger
from tuningfork.core.exceptions import (
    DatabaseConnectionError,
    ErrorCodes
)


class PooledConnection:
    """Wrapper for pooled database connections with metadata."""
    
    def __init__(self, connection: Any, pool: 'ConnectionPool'):
        self.connection = connection
        self.pool = weakref.ref(pool)
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.use_count = 0
        self.is_healthy = True
        self.is_in_use = False
        self.connection_id = id(connection)
    
    def mark_used(self) -> None:
        """Mark connection as used and update statistics."""
        self.last_used = datetime.now()
        self.use_count += 1
        self.is_in_use = True
    
    def mark_returned(self) -> None:
        """Mark connection as returned to pool."""
        self.is_in_use = False
    
    def get_age(self) -> timedelta:
        """Get connection age."""
        return datetime.now() - self.created_at
    
    def get_idle_time(self) -> timedelta:
        """Get time since last use."""
        return datetime.now() - self.last_used
    
    async def close(self) -> None:
        """Close the underlying connection."""
        try:
            if hasattr(self.connection, 'close'):
                if asyncio.iscoroutinefunction(self.connection.close):
                    await self.connection.close()
                else:
                    self.connection.close()
        except Exception:
            pass  # Ignore errors during close


class ConnectionPool:
    """Async connection pool with advanced features."""
    
    def __init__(
        self,
        connector: 'BaseDatabaseConnector',
        min_size: int = 1,
        max_size: int = 10,
        pool_timeout: int = 30,
        idle_timeout: int = 300,
        max_lifetime: int = 3600,
        health_check_interval: int = 60
    ):
        self.connector = connector
        self.min_size = min_size
        self.max_size = max_size
        self.pool_timeout = pool_timeout
        self.idle_timeout = idle_timeout
        self.max_lifetime = max_lifetime
        self.health_check_interval = health_check_interval
        
        # Pool state
        self._pool: asyncio.Queue[PooledConnection] = asyncio.Queue(maxsize=max_size)
        self._connections: Set[PooledConnection] = set()
        self._total_connections = 0
        self._lock = asyncio.Lock()
        self._closed = False
        
        # Health checking
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_health_check = datetime.now()
        
        # Statistics
        self._stats = {
            'total_created': 0,
            'total_closed': 0,
            'total_acquired': 0,
            'total_released': 0,
            'total_health_checks': 0,
            'unhealthy_connections_closed': 0,
            'pool_exhausted_count': 0,
            'average_wait_time': 0.0,
            'max_wait_time': 0.0
        }
        
        # Logging
        self.logger = get_logger(f"pool.{connector.platform}")
        self.perf_logger = get_performance_logger(f"pool.{connector.platform}")
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        self.logger.info("Initializing connection pool",
                        min_size=self.min_size,
                        max_size=self.max_size)
        
        try:
            # Create minimum connections
            for _ in range(self.min_size):
                pooled_conn = await self._create_connection()
                await self._pool.put(pooled_conn)
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.logger.info("Connection pool initialized",
                           initial_connections=self.min_size)
            
        except Exception as e:
            await self.close()
            raise DatabaseConnectionError(
                f"Failed to initialize connection pool: {e}",
                code=ErrorCodes.CONNECTION_REFUSED,  # Use Phase 1 ErrorCode
                context={
                    "min_size": self.min_size,
                    "max_size": self.max_size,
                    "platform": self.connector.platform
                }
            ) from e
    
    @asynccontextmanager
    async def acquire(self) -> AsyncContextManager[Any]:
        """Acquire a connection from the pool."""
        if self._closed:
            raise DatabaseConnectionError(
                "Connection pool is closed",
                code=ErrorCodes.CONNECTION_REFUSED  # Use Phase 1 ErrorCode
            )
        
        start_time = time.time()
        pooled_conn = None
        
        try:
            # Try to get connection from pool or create new one
            pooled_conn = await self._get_or_create_connection()
            
            wait_time = time.time() - start_time
            self._update_wait_stats(wait_time)
            
            # Mark as in use and update stats
            pooled_conn.mark_used()
            self._stats['total_acquired'] += 1
            
            self.logger.debug("Connection acquired",
                            connection_id=pooled_conn.connection_id,
                            wait_time_ms=wait_time * 1000,
                            use_count=pooled_conn.use_count)
            
            yield pooled_conn.connection
            
        except Exception as e:
            self.logger.error("Failed to acquire connection", error=str(e))
            if pooled_conn:
                await self._close_connection(pooled_conn)
            raise
        finally:
            if pooled_conn:
                await self._release_connection(pooled_conn)
    
    async def _get_or_create_connection(self) -> PooledConnection:
        """Get connection from pool or create new one if needed."""
        # Try to get from pool first
        try:
            pooled_conn = await asyncio.wait_for(
                self._pool.get(),
                timeout=min(self.pool_timeout, 1.0)
            )
            
            # Validate connection health
            if await self._is_connection_healthy(pooled_conn):
                return pooled_conn
            else:
                # Close unhealthy connection and try again
                await self._close_connection(pooled_conn)
                return await self._get_or_create_connection()
                
        except asyncio.TimeoutError:
            # Pool is empty, try to create new connection
            async with self._lock:
                if self._total_connections < self.max_size:
                    return await self._create_connection()
                else:
                    # Pool exhausted, wait longer
                    self._stats['pool_exhausted_count'] += 1
                    self.logger.warning("Connection pool exhausted, waiting",
                                      active_connections=self._total_connections,
                                      max_size=self.max_size)
                    
                    try:
                        pooled_conn = await asyncio.wait_for(
                            self._pool.get(),
                            timeout=self.pool_timeout
                        )
                        
                        if await self._is_connection_healthy(pooled_conn):
                            return pooled_conn
                        else:
                            await self._close_connection(pooled_conn)
                            return await self._get_or_create_connection()
                            
                    except asyncio.TimeoutError:
                        raise DatabaseConnectionError(
                            f"Connection pool exhausted after {self.pool_timeout}s timeout",
                            code=ErrorCodes.POOL_EXHAUSTED,  # Use Phase 1 ErrorCode
                            context={
                                "pool_size": self.max_size,
                                "active_connections": self._total_connections,
                                "timeout": self.pool_timeout
                            }
                        )
    
    async def _create_connection(self) -> PooledConnection:
        """Create a new pooled connection."""
        # Measure creation time manually since TimingContext properties aren't available inside context
        start_time = time.time()
        
        try:
            # Use connector's platform-specific connection creation
            raw_connection = await self.connector._create_raw_connection()
            
            pooled_conn = PooledConnection(raw_connection, self)
            
            async with self._lock:
                self._connections.add(pooled_conn)
                self._total_connections += 1
                self._stats['total_created'] += 1
            
            creation_time = time.time() - start_time
            
            self.logger.debug("New connection created",
                            connection_id=pooled_conn.connection_id,
                            creation_time_ms=creation_time * 1000,
                            total_connections=self._total_connections)
            
            return pooled_conn
            
        except Exception as e:
            self.logger.error("Failed to create connection", error=str(e))
            raise DatabaseConnectionError(
                f"Failed to create database connection: {e}",
                code=ErrorCodes.CONNECTION_REFUSED,  # Use Phase 1 ErrorCode
                context={"platform": self.connector.platform}
            ) from e
    
    async def _release_connection(self, pooled_conn: PooledConnection) -> None:
        """Release connection back to the pool."""
        try:
            pooled_conn.mark_returned()
            self._stats['total_released'] += 1
            
            # Check if connection should be closed
            if (self._should_close_connection(pooled_conn) or 
                not await self._is_connection_healthy(pooled_conn)):
                await self._close_connection(pooled_conn)
                return
            
            # Return to pool if not full
            try:
                self._pool.put_nowait(pooled_conn)
                self.logger.debug("Connection returned to pool",
                                connection_id=pooled_conn.connection_id)
            except asyncio.QueueFull:
                # Pool is full, close the connection
                await self._close_connection(pooled_conn)
                
        except Exception as e:
            self.logger.error("Error releasing connection", 
                            error=str(e),
                            connection_id=getattr(pooled_conn, 'connection_id', 'unknown'))
            await self._close_connection(pooled_conn)
    
    async def _close_connection(self, pooled_conn: PooledConnection) -> None:
        """Close a pooled connection and remove from tracking."""
        try:
            await pooled_conn.close()
            
            async with self._lock:
                self._connections.discard(pooled_conn)
                self._total_connections -= 1
                self._stats['total_closed'] += 1
            
            self.logger.debug("Connection closed",
                            connection_id=pooled_conn.connection_id,
                            age_seconds=pooled_conn.get_age().total_seconds(),
                            use_count=pooled_conn.use_count)
            
        except Exception as e:
            self.logger.error("Error closing connection", error=str(e))
    
    async def _is_connection_healthy(self, pooled_conn: PooledConnection) -> bool:
        """Check if a connection is healthy."""
        try:
            # Use connector's platform-specific health check
            if hasattr(self.connector, '_check_connection_health'):
                return await self.connector._check_connection_health(pooled_conn.connection)
            
            # Default health check - just verify connection exists
            return pooled_conn.connection is not None and pooled_conn.is_healthy
            
        except Exception:
            return False
    
    def _should_close_connection(self, pooled_conn: PooledConnection) -> bool:
        """Determine if a connection should be closed instead of returned to pool."""
        # Close if too old
        if pooled_conn.get_age().total_seconds() > self.max_lifetime:
            self.logger.debug("Connection too old, closing",
                            connection_id=pooled_conn.connection_id,
                            age_seconds=pooled_conn.get_age().total_seconds())
            return True
        
        # Close if pool has too many connections
        if self._total_connections > self.max_size:
            return True
        
        return False
    
    async def _health_check_loop(self) -> None:
        """Background task to check connection health."""
        while not self._closed:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in health check loop", error=str(e))
    
    async def _perform_health_check(self) -> None:
        """Perform health check on idle connections."""
        self.logger.debug("Performing connection health check")
        
        self._stats['total_health_checks'] += 1
        self._last_health_check = datetime.now()
        
        # Check for idle connections that should be closed
        connections_to_close = []
        
        async with self._lock:
            for conn in list(self._connections):
                if not conn.is_in_use:
                    # Close if idle too long
                    if conn.get_idle_time().total_seconds() > self.idle_timeout:
                        connections_to_close.append(conn)
                    # Close if too old
                    elif conn.get_age().total_seconds() > self.max_lifetime:
                        connections_to_close.append(conn)
        
        # Close unhealthy connections
        for conn in connections_to_close:
            self.logger.debug("Closing idle/old connection",
                            connection_id=conn.connection_id,
                            idle_time=conn.get_idle_time().total_seconds(),
                            age=conn.get_age().total_seconds())
            await self._close_connection(conn)
        
        if connections_to_close:
            self._stats['unhealthy_connections_closed'] += len(connections_to_close)
        
        # Ensure minimum connections
        current_size = self._total_connections
        if current_size < self.min_size:
            needed = self.min_size - current_size
            for _ in range(needed):
                try:
                    pooled_conn = await self._create_connection()
                    await self._pool.put(pooled_conn)
                except Exception as e:
                    self.logger.error("Failed to create connection during health check", error=str(e))
                    break
    
    def _update_wait_stats(self, wait_time: float) -> None:
        """Update connection wait time statistics."""
        self._stats['max_wait_time'] = max(self._stats['max_wait_time'], wait_time)
        
        # Update average wait time
        total_acquired = self._stats['total_acquired']
        if total_acquired > 0:
            current_avg = self._stats['average_wait_time']
            self._stats['average_wait_time'] = (
                (current_avg * (total_acquired - 1) + wait_time) / total_acquired
            )
    
    async def close(self) -> None:
        """Close the connection pool and all connections."""
        if self._closed:
            return
        
        self.logger.info("Closing connection pool")
        self._closed = True
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        connections_to_close = list(self._connections)
        for conn in connections_to_close:
            await self._close_connection(conn)
        
        # Clear the pool queue
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await self._close_connection(conn)
            except asyncio.QueueEmpty:
                break
        
        self.logger.info("Connection pool closed",
                        total_created=self._stats['total_created'],
                        total_closed=self._stats['total_closed'])
    
    @property
    def is_closed(self) -> bool:
        """Check if pool is closed."""
        return self._closed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            'total_connections': self._total_connections,
            'active_connections': sum(1 for conn in self._connections if conn.is_in_use),
            'idle_connections': sum(1 for conn in self._connections if not conn.is_in_use),
            'pool_size': self._pool.qsize(),
            'min_size': self.min_size,
            'max_size': self.max_size,
            'is_closed': self._closed,
            'last_health_check': self._last_health_check.isoformat(),
            **self._stats
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        stats = self.get_stats()
        
        return {
            'pool_utilization_percent': (stats['total_connections'] / self.max_size) * 100,
            'active_connection_ratio': (
                stats['active_connections'] / max(stats['total_connections'], 1)
            ),
            'average_wait_time_ms': stats['average_wait_time'] * 1000,
            'max_wait_time_ms': stats['max_wait_time'] * 1000,
            'pool_exhausted_rate': stats['pool_exhausted_count'] / max(stats['total_acquired'], 1),
            'connection_creation_rate': stats['total_created'] / max(stats['total_acquired'], 1),
            'unhealthy_connection_rate': (
                stats['unhealthy_connections_closed'] / max(stats['total_created'], 1)
            ),
            'health_check_frequency': 1 / self.health_check_interval if self.health_check_interval > 0 else 0
        }