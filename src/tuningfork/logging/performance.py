"""Performance logging for TuningFork operations.

This module provides performance monitoring and timing capabilities
for tracking operation performance, detecting bottlenecks, and
generating performance reports.

Classes:
    PerformanceLogger: Main performance logging interface
    TimingContext: Context manager for operation timing
    PerformanceMetrics: Performance metrics collection
    PerformanceReport: Performance analysis and reporting

Example:
    >>> perf_logger = PerformanceLogger("database.analysis")
    >>> with perf_logger.measure("query_execution") as timer:
    ...     # Execute database query
    ...     result = execute_query("SELECT * FROM large_table")
    >>> timer.duration  # Access timing information
    2.45
"""

import time
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union
from datetime import datetime, timedelta
from collections import defaultdict

from .structured import StructuredLogger
from ..core.utils import FormatUtils


@dataclass
class TimingMetrics:
    """Metrics for a single timing measurement.
    
    Attributes:
        operation: Operation name
        start_time: Operation start timestamp
        end_time: Operation end timestamp  
        duration: Duration in seconds
        metadata: Additional metadata
        success: Whether operation succeeded
        error: Error information if failed
    """
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    
    def complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark timing as complete.
        
        Args:
            success: Whether operation succeeded
            error: Error message if failed
        """
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds.
        
        Returns:
            Duration in milliseconds or None if not completed
        """
        return self.duration * 1000 if self.duration is not None else None
    
    @property
    def is_complete(self) -> bool:
        """Check if timing is complete.
        
        Returns:
            True if timing has been completed
        """
        return self.end_time is not None


@dataclass 
class PerformanceMetrics:
    """Aggregated performance metrics for an operation.
    
    Attributes:
        operation: Operation name
        total_calls: Total number of calls
        successful_calls: Number of successful calls
        failed_calls: Number of failed calls
        total_duration: Total duration in seconds
        min_duration: Minimum duration
        max_duration: Maximum duration
        avg_duration: Average duration
        median_duration: Median duration
        p95_duration: 95th percentile duration
        p99_duration: 99th percentile duration
        calls_per_second: Average calls per second
        errors: List of error messages
    """
    operation: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    avg_duration: Optional[float] = None
    median_duration: Optional[float] = None
    p95_duration: Optional[float] = None
    p99_duration: Optional[float] = None
    calls_per_second: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    _durations: List[float] = field(default_factory=list, repr=False)
    _start_time: Optional[float] = field(default=None, repr=False)
    
    def add_timing(self, timing: TimingMetrics) -> None:
        """Add timing measurement to metrics.
        
        Args:
            timing: Timing measurement to add
        """
        if not timing.is_complete or timing.duration is None:
            return
        
        # Initialize start time tracking
        if self._start_time is None:
            self._start_time = timing.start_time
        
        # Update counters
        self.total_calls += 1
        if timing.success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if timing.error:
                self.errors.append(timing.error)
        
        # Update duration stats
        duration = timing.duration
        self.total_duration += duration
        self._durations.append(duration)
        
        # Update min/max
        if self.min_duration is None or duration < self.min_duration:
            self.min_duration = duration
        if self.max_duration is None or duration > self.max_duration:
            self.max_duration = duration
        
        # Recalculate statistics
        self._calculate_statistics()
    
    def _calculate_statistics(self) -> None:
        """Calculate statistical metrics from duration data."""
        if not self._durations:
            return
        
        # Basic statistics
        self.avg_duration = statistics.mean(self._durations)
        self.median_duration = statistics.median(self._durations)
        
        # Percentiles
        if len(self._durations) >= 20:  # Only calculate percentiles with sufficient data
            sorted_durations = sorted(self._durations)
            self.p95_duration = sorted_durations[int(len(sorted_durations) * 0.95)]
            self.p99_duration = sorted_durations[int(len(sorted_durations) * 0.99)]
        
        # Throughput calculation
        if self._start_time is not None and self.total_calls > 0:
            elapsed_time = time.perf_counter() - self._start_time
            if elapsed_time > 0:
                self.calls_per_second = self.total_calls / elapsed_time
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage.
        
        Returns:
            Success rate (0-100)
        """
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage.
        
        Returns:
            Error rate (0-100)
        """
        if self.total_calls == 0:
            return 0.0  # Changed from: return 100.0 - self.success_rate
        return (self.failed_calls / self.total_calls) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary.
        
        Returns:
            Dictionary representation of metrics
        """
        return {
            'operation': self.operation,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': self.success_rate,
            'error_rate': self.error_rate,
            'total_duration': self.total_duration,
            'min_duration': self.min_duration,
            'max_duration': self.max_duration,
            'avg_duration': self.avg_duration,
            'median_duration': self.median_duration,
            'p95_duration': self.p95_duration,
            'p99_duration': self.p99_duration,
            'calls_per_second': self.calls_per_second,
            'error_count': len(self.errors),
        }


class TimingContext:
    """Context manager for measuring operation timing.
    
    This class provides a context manager interface for measuring
    operation performance with automatic logging and error handling.
    
    Example:
        >>> with TimingContext("database_query") as timer:
        ...     result = execute_query("SELECT * FROM users")
        >>> print(f"Query took {timer.duration_ms:.2f}ms")
    """
    
    def __init__(
        self,
        operation: str,
        logger: Optional[StructuredLogger] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_log: bool = True,
    ) -> None:
        """Initialize timing context.
        
        Args:
            operation: Operation name
            logger: Logger for automatic logging
            metadata: Additional metadata
            auto_log: Whether to automatically log timing results
        """
        self.operation = operation
        self.logger = logger
        self.metadata = metadata or {}
        self.auto_log = auto_log
        self._timing: Optional[TimingMetrics] = None
    
    @property
    def timing(self) -> Optional[TimingMetrics]:
        """Get timing metrics.
        
        Returns:
            TimingMetrics instance or None if not started
        """
        return self._timing
    
    @property
    def duration(self) -> Optional[float]:
        """Get operation duration in seconds.
        
        Returns:
            Duration in seconds or None if not completed
        """
        return self._timing.duration if self._timing else None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get operation duration in milliseconds.
        
        Returns:
            Duration in milliseconds or None if not completed
        """
        return self._timing.duration_ms if self._timing else None
    
    def __enter__(self) -> 'TimingContext':
        """Enter timing context."""
        self._timing = TimingMetrics(
            operation=self.operation,
            start_time=time.perf_counter(),
            metadata=self.metadata
        )
        
        if self.logger and self.auto_log:
            self.logger.debug(
                "Operation started",
                operation=self.operation,
                **self.metadata
            )
        
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit timing context."""
        if self._timing is None:
            return
        
        # Mark timing as complete
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self._timing.complete(success=success, error=error)
        
        # Log results if enabled
        if self.logger and self.auto_log:
            if success:
                self.logger.info(
                    "Operation completed",
                    operation=self.operation,
                    duration_ms=self._timing.duration_ms,
                    success=True,
                    **self.metadata
                )
            else:
                self.logger.error(
                    "Operation failed",
                    operation=self.operation,
                    duration_ms=self._timing.duration_ms,
                    success=False,
                    error=error,
                    **self.metadata
                )


class PerformanceLogger:
    """Performance logger for monitoring and analyzing operation performance.
    
    This class provides comprehensive performance monitoring including
    timing measurements, throughput analysis, and performance reporting.
    
    Attributes:
        name: Logger name
        logger: Underlying structured logger
        metrics: Performance metrics by operation
        
    Example:
        >>> perf_logger = PerformanceLogger("database.operations")
        >>> with perf_logger.measure("query_execution"):
        ...     result = database.execute("SELECT * FROM users")
        >>> report = perf_logger.get_performance_report()
        >>> print(report.summary)
    """
    
    def __init__(
        self,
        name: str,
        *,
        auto_log: bool = True,
        track_metrics: bool = True,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        """Initialize performance logger.
        
        Args:
            name: Logger name
            auto_log: Whether to automatically log timing results
            track_metrics: Whether to track aggregated metrics
            logger: Custom structured logger instance
        """
        self.name = name
        self.auto_log = auto_log
        self.track_metrics = track_metrics
        self.logger = logger or StructuredLogger(f"perf.{name}")
        
        # Performance metrics storage
        self._metrics: Dict[str, PerformanceMetrics] = defaultdict(
            lambda: PerformanceMetrics(operation="unknown")
        )
        
        # Active timings
        self._active_timings: Dict[str, TimingMetrics] = {}
    
    @contextmanager
    def measure(
        self,
        operation: str,
        **metadata: Any
    ) -> Generator[TimingContext, None, None]:
        """Context manager for measuring operation performance.
        
        Args:
            operation: Operation name
            **metadata: Additional metadata
            
        Yields:
            TimingContext for the operation
            
        Example:
            >>> with perf_logger.measure("database_query", table="users") as timer:
            ...     result = database.execute("SELECT * FROM users")
            >>> print(f"Query took {timer.duration_ms:.2f}ms")
        """
        timing_context = TimingContext(
            operation=operation,
            logger=self.logger if self.auto_log else None,
            metadata=metadata,
            auto_log=self.auto_log,
        )
        
        try:
            with timing_context as ctx:
                yield ctx
        finally:
            # Add timing to metrics if tracking is enabled
            if self.track_metrics and timing_context.timing:
                self._add_timing_to_metrics(timing_context.timing)
    
    def _add_timing_to_metrics(self, timing: TimingMetrics) -> None:
        """Add timing measurement to aggregated metrics.
        
        Args:
            timing: Timing measurement to add
        """
        if timing.operation not in self._metrics:
            self._metrics[timing.operation] = PerformanceMetrics(operation=timing.operation)
        
        self._metrics[timing.operation].add_timing(timing)
    
    def start_timing(self, operation: str, **metadata: Any) -> str:
        """Start timing an operation manually.
        
        Args:
            operation: Operation name
            **metadata: Additional metadata
            
        Returns:
            Timing ID for stopping the timing
            
        Example:
            >>> timing_id = perf_logger.start_timing("long_operation", user_id="123")
            >>> # ... perform operation ...
            >>> perf_logger.stop_timing(timing_id)
        """
        import uuid
        timing_id = str(uuid.uuid4())
        
        timing = TimingMetrics(
            operation=operation,
            start_time=time.perf_counter(),
            metadata=metadata
        )
        
        self._active_timings[timing_id] = timing
        
        if self.auto_log:
            self.logger.debug(
                "Manual timing started",
                timing_id=timing_id,
                operation=operation,
                **metadata
            )
        
        return timing_id
    
    def stop_timing(
        self,
        timing_id: str,
        success: bool = True,
        error: Optional[str] = None,
    ) -> Optional[TimingMetrics]:
        """Stop timing an operation manually.
        
        Args:
            timing_id: Timing ID from start_timing
            success: Whether operation succeeded
            error: Error message if failed
            
        Returns:
            Completed TimingMetrics or None if timing not found
        """
        timing = self._active_timings.pop(timing_id, None)
        if timing is None:
            self.logger.warning("Timing not found", timing_id=timing_id)
            return None
        
        timing.complete(success=success, error=error)
        
        if self.auto_log:
            if success:
                self.logger.info(
                    "Manual timing completed",
                    timing_id=timing_id,
                    operation=timing.operation,
                    duration_ms=timing.duration_ms,
                    success=True,
                    **timing.metadata
                )
            else:
                self.logger.error(
                    "Manual timing failed",
                    timing_id=timing_id,
                    operation=timing.operation,
                    duration_ms=timing.duration_ms,
                    success=False,
                    error=error,
                    **timing.metadata
                )
        
        # Add to metrics if tracking is enabled
        if self.track_metrics:
            self._add_timing_to_metrics(timing)
        
        return timing
    
    def record_timing(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        error: Optional[str] = None,
        **metadata: Any
    ) -> None:
        """Record a timing measurement directly.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation succeeded
            error: Error message if failed
            **metadata: Additional metadata
        """
        timing = TimingMetrics(
            operation=operation,
            start_time=time.perf_counter() - duration,
            metadata=metadata,
            success=success,
            error=error
        )
        timing.complete(success=success, error=error)
        
        if self.auto_log:
            level = "info" if success else "error"
            self.logger.log(
                level,
                f"Timing recorded: {operation}",
                operation=operation,
                duration_ms=timing.duration_ms,
                success=success,
                error=error,
                **metadata
            )
        
        # Add to metrics if tracking is enabled
        if self.track_metrics:
            self._add_timing_to_metrics(timing)
    
    def get_metrics(self, operation: Optional[str] = None) -> Union[PerformanceMetrics, Dict[str, PerformanceMetrics]]:
        """Get performance metrics.
        
        Args:
            operation: Specific operation name, or None for all operations
            
        Returns:
            PerformanceMetrics for specific operation or dict of all metrics
        """
        if operation:
            return self._metrics.get(operation, PerformanceMetrics(operation=operation))
        else:
            return dict(self._metrics)
    
    def reset_metrics(self, operation: Optional[str] = None) -> None:
        """Reset performance metrics.
        
        Args:
            operation: Specific operation to reset, or None for all operations
        """
        if operation:
            if operation in self._metrics:
                del self._metrics[operation]
        else:
            self._metrics.clear()
        
        self.logger.info("Performance metrics reset", operation=operation or "all")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary across all operations.
        
        Returns:
            Summary dictionary with aggregated statistics
        """
        if not self._metrics:
            return {
                "total_operations": 0,
                "total_calls": 0,
                "total_duration": 0.0,
                "overall_success_rate": 0.0,
                "operations": {}
            }
        
        total_calls = sum(m.total_calls for m in self._metrics.values())
        total_successful = sum(m.successful_calls for m in self._metrics.values())
        total_duration = sum(m.total_duration for m in self._metrics.values())
        
        return {
            "total_operations": len(self._metrics),
            "total_calls": total_calls,
            "total_duration": total_duration,
            "overall_success_rate": (total_successful / total_calls * 100) if total_calls > 0 else 0.0,
            "operations": {
                name: metrics.to_dict() 
                for name, metrics in self._metrics.items()
            }
        }
    
    def get_top_operations(
        self,
        metric: str = "total_duration",
        limit: int = 10,
        reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """Get top operations by specified metric.
        
        Args:
            metric: Metric to sort by (total_duration, total_calls, avg_duration, etc.)
            limit: Maximum number of operations to return
            reverse: Whether to sort in descending order
            
        Returns:
            List of operation metrics sorted by specified metric
        """
        operations = []
        
        for name, metrics in self._metrics.items():
            op_dict = metrics.to_dict()
            op_dict['name'] = name
            operations.append(op_dict)
        
        # Sort by specified metric
        try:
            operations.sort(
                key=lambda x: x.get(metric, 0) or 0,
                reverse=reverse
            )
        except (TypeError, KeyError):
            self.logger.warning(f"Invalid sort metric: {metric}")
            return operations
        
        return operations[:limit]
    
    def log_performance_summary(self) -> None:
        """Log current performance summary."""
        summary = self.get_summary()
        
        self.logger.info(
            "Performance summary",
            total_operations=summary["total_operations"],
            total_calls=summary["total_calls"],
            total_duration_formatted=FormatUtils.format_duration(summary["total_duration"]),
            overall_success_rate=f"{summary['overall_success_rate']:.1f}%"
        )
        
        # Log top operations by duration
        top_by_duration = self.get_top_operations("total_duration", limit=5)
        for i, op in enumerate(top_by_duration, 1):
            self.logger.info(
                f"Top operation #{i} by duration",
                operation=op["name"],
                total_duration=FormatUtils.format_duration(op["total_duration"]),
                total_calls=op["total_calls"],
                avg_duration=FormatUtils.format_duration(op["avg_duration"] or 0),
                success_rate=f"{op['success_rate']:.1f}%"
            )
    
    def export_metrics(self, format: str = "json") -> str:
        """Export performance metrics in specified format.
        
        Args:
            format: Export format ("json", "csv")
            
        Returns:
            Formatted metrics data
        """
        summary = self.get_summary()
        
        if format.lower() == "json":
            import json
            return json.dumps(summary, indent=2, default=str)
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "operation", "total_calls", "successful_calls", "failed_calls",
                "success_rate", "total_duration", "avg_duration", "min_duration",
                "max_duration", "median_duration", "p95_duration", "calls_per_second"
            ])
            
            # Write data
            for name, metrics in summary["operations"].items():
                writer.writerow([
                    name,
                    metrics["total_calls"],
                    metrics["successful_calls"], 
                    metrics["failed_calls"],
                    f"{metrics['success_rate']:.2f}%",
                    metrics["total_duration"],
                    metrics["avg_duration"],
                    metrics["min_duration"],
                    metrics["max_duration"],
                    metrics["median_duration"],
                    metrics["p95_duration"],
                    metrics["calls_per_second"]
                ])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def __repr__(self) -> str:
        """Return string representation of performance logger."""
        return (
            f"PerformanceLogger("
            f"name={self.name!r}, "
            f"operations={len(self._metrics)}, "
            f"auto_log={self.auto_log})"
        )