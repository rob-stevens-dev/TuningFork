"""Tests for performance logging module."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from tuningfork.logging.performance import (
    PerformanceLogger,
    TimingContext,
    TimingMetrics,
    PerformanceMetrics,
)
from tuningfork.logging.structured import StructuredLogger


class TestTimingMetrics:
    """Test cases for TimingMetrics class."""
    
    def test_timing_metrics_initialization(self):
        """Test TimingMetrics initializes correctly."""
        start_time = time.perf_counter()
        metrics = TimingMetrics(
            operation="test_operation",
            start_time=start_time,
            metadata={"key": "value"}
        )
        
        assert metrics.operation == "test_operation"
        assert metrics.start_time == start_time
        assert metrics.end_time is None
        assert metrics.duration is None
        assert metrics.metadata == {"key": "value"}
        assert metrics.success is True
        assert metrics.error is None
        assert not metrics.is_complete
    
    def test_timing_metrics_completion(self):
        """Test completing timing metrics."""
        start_time = time.perf_counter()
        metrics = TimingMetrics("test_op", start_time)
        
        # Complete successfully
        metrics.complete(success=True)
        
        assert metrics.is_complete
        assert metrics.success is True
        assert metrics.error is None
        assert metrics.end_time is not None
        assert metrics.duration is not None
        assert metrics.duration > 0
        assert metrics.duration_ms is not None
        assert metrics.duration_ms > 0
    
    def test_timing_metrics_failure(self):
        """Test completing timing metrics with failure."""
        start_time = time.perf_counter()
        metrics = TimingMetrics("test_op", start_time)
        
        # Complete with failure
        metrics.complete(success=False, error="Test error occurred")
        
        assert metrics.is_complete
        assert metrics.success is False
        assert metrics.error == "Test error occurred"
        assert metrics.duration is not None
    
    def test_duration_properties(self):
        """Test duration property calculations."""
        start_time = time.perf_counter()
        metrics = TimingMetrics("test_op", start_time)
        
        # Before completion
        assert metrics.duration_ms is None
        
        # After completion
        time.sleep(0.01)  # Small delay
        metrics.complete()
        
        assert metrics.duration > 0
        assert metrics.duration_ms > 0
        assert metrics.duration_ms == metrics.duration * 1000


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics class."""
    
    def test_performance_metrics_initialization(self):
        """Test PerformanceMetrics initializes correctly."""
        metrics = PerformanceMetrics("test_operation")
        
        assert metrics.operation == "test_operation"
        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.total_duration == 0.0
        assert metrics.min_duration is None
        assert metrics.max_duration is None
        assert metrics.avg_duration is None
        assert metrics.success_rate == 0.0
        assert metrics.error_rate == 0.0  # Fixed: was expecting 100.0
        assert len(metrics.errors) == 0
    
    def test_add_successful_timing(self):
        """Test adding successful timing measurements."""
        metrics = PerformanceMetrics("test_op")
        
        # Create and complete a timing
        timing = TimingMetrics("test_op", time.perf_counter())
        time.sleep(0.01)
        timing.complete(success=True)
        
        # Add to metrics
        metrics.add_timing(timing)
        
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 0
        assert metrics.success_rate == 100.0
        assert metrics.error_rate == 0.0
        assert metrics.total_duration > 0
        assert metrics.min_duration == timing.duration
        assert metrics.max_duration == timing.duration
        assert metrics.avg_duration == timing.duration
    
    def test_add_failed_timing(self):
        """Test adding failed timing measurements."""
        metrics = PerformanceMetrics("test_op")
        
        # Create and complete a failed timing
        timing = TimingMetrics("test_op", time.perf_counter())
        time.sleep(0.01)
        timing.complete(success=False, error="Test error")
        
        # Add to metrics
        metrics.add_timing(timing)
        
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 1
        assert metrics.success_rate == 0.0
        assert metrics.error_rate == 100.0
        assert len(metrics.errors) == 1
        assert metrics.errors[0] == "Test error"
    
    def test_multiple_timings_statistics(self):
            """Test statistics calculation with multiple timings."""
            metrics = PerformanceMetrics("test_op")
            
            # Add multiple timings with controlled durations
            durations = [0.05, 0.1, 0.08, 0.15, 0.02]
            
            for duration in durations:
                timing = TimingMetrics("test_op", time.perf_counter())
                # Set exact timing to avoid measurement overhead
                timing.end_time = timing.start_time + duration
                timing.duration = duration
                timing.success = True
                timing.error = None
                
                metrics.add_timing(timing)
            
            assert metrics.total_calls == 5
            assert metrics.successful_calls == 5
            # Use approximate comparisons to handle floating point precision
            assert abs(metrics.min_duration - 0.02) < 0.001
            assert abs(metrics.max_duration - 0.15) < 0.001
            assert abs(metrics.avg_duration - 0.08) < 0.01  # Average of durations
            
            # Sort durations for median calculation: [0.02, 0.05, 0.08, 0.1, 0.15]
            # Median should be 0.08 (middle value)
            assert abs(metrics.median_duration - 0.08) < 0.001
    
    def test_percentile_calculations(self):
        """Test percentile calculations with sufficient data."""
        metrics = PerformanceMetrics("test_op")
        
        # Add 50 timings to enable percentile calculations
        for i in range(50):
            timing = TimingMetrics("test_op", time.perf_counter())
            timing.end_time = timing.start_time + (i * 0.01)  # Increasing durations
            timing.duration = i * 0.01
            timing.complete(success=True)
            metrics.add_timing(timing)
        
        assert metrics.total_calls == 50
        assert metrics.p95_duration is not None
        assert metrics.p99_duration is not None
        assert metrics.p95_duration <= metrics.p99_duration
        assert metrics.p99_duration <= metrics.max_duration
    
    def test_calls_per_second_calculation(self):
        """Test calls per second calculation."""
        metrics = PerformanceMetrics("test_op")
        
        # Simulate timing data
        start_time = time.perf_counter()
        metrics._start_time = start_time
        
        # Add timings
        for i in range(10):
            timing = TimingMetrics("test_op", start_time + i * 0.1)
            timing.complete(success=True)
            metrics.add_timing(timing)
        
        # Should have positive calls per second
        assert metrics.calls_per_second is not None
        assert metrics.calls_per_second > 0
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics("test_op")
        
        # Add a timing
        timing = TimingMetrics("test_op", time.perf_counter())
        timing.complete(success=True)
        metrics.add_timing(timing)
        
        # Convert to dict
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert result["operation"] == "test_op"
        assert result["total_calls"] == 1
        assert result["successful_calls"] == 1
        assert result["failed_calls"] == 0
        assert "success_rate" in result
        assert "error_rate" in result
        assert "total_duration" in result


class TestTimingContext:
    """Test cases for TimingContext class."""
    
    def test_timing_context_initialization(self):
        """Test TimingContext initializes correctly."""
        context = TimingContext("test_operation", auto_log=False)
        
        assert context.operation == "test_operation"
        assert context.auto_log is False
        assert context.timing is None
        assert context.duration is None
        assert context.duration_ms is None
    
    def test_timing_context_as_context_manager(self):
        """Test TimingContext as context manager."""
        context = TimingContext("test_operation", auto_log=False)
        
        with context as ctx:
            assert ctx is context
            assert context.timing is not None
            assert context.timing.operation == "test_operation"
            assert not context.timing.is_complete
            
            # Small delay to measure
            time.sleep(0.01)
        
        # After context exit
        assert context.timing.is_complete
        assert context.timing.success is True
        assert context.duration is not None
        assert context.duration > 0
        assert context.duration_ms is not None
    
    def test_timing_context_with_exception(self):
        """Test TimingContext handling exceptions."""
        context = TimingContext("test_operation", auto_log=False)
        
        with pytest.raises(ValueError):
            with context:
                time.sleep(0.01)
                raise ValueError("Test exception")
        
        # Should still complete timing even with exception
        assert context.timing.is_complete
        assert context.timing.success is False
        assert context.timing.error == "Test exception"
        assert context.duration is not None
    
    def test_timing_context_with_logger(self):
        """Test TimingContext with automatic logging."""
        mock_logger = Mock(spec=StructuredLogger)
        context = TimingContext(
            "test_operation",
            logger=mock_logger,
            metadata={"key": "value"},
            auto_log=True
        )
        
        with context:
            time.sleep(0.01)
        
        # Should have called logger methods
        assert mock_logger.debug.called
        assert mock_logger.info.called


class TestPerformanceLogger:
    """Test cases for PerformanceLogger class."""
    
    def test_performance_logger_initialization(self):
        """Test PerformanceLogger initializes correctly."""
        logger = PerformanceLogger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.auto_log is True
        assert logger.track_metrics is True
        assert isinstance(logger.logger, StructuredLogger)
        assert len(logger._metrics) == 0
    
    def test_measure_context_manager(self):
        """Test measure context manager."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        with logger.measure("test_operation", param="value") as timer:
            assert isinstance(timer, TimingContext)
            assert timer.operation == "test_operation"
            time.sleep(0.01)
        
        # Should have added to metrics
        assert "test_operation" in logger._metrics
        metrics = logger._metrics["test_operation"]
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
    
    def test_measure_with_exception(self):
        """Test measure context manager with exception."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        with pytest.raises(RuntimeError):
            with logger.measure("failing_operation"):
                time.sleep(0.01)
                raise RuntimeError("Test failure")
        
        # Should still track the failed operation
        assert "failing_operation" in logger._metrics
        metrics = logger._metrics["failing_operation"]
        assert metrics.total_calls == 1
        assert metrics.failed_calls == 1
        assert len(metrics.errors) == 1
    
    def test_manual_timing_operations(self):
        """Test manual timing start/stop operations."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        # Start timing
        timing_id = logger.start_timing("manual_operation", user_id="123")
        assert isinstance(timing_id, str)
        
        time.sleep(0.01)
        
        # Stop timing
        result = logger.stop_timing(timing_id, success=True)
        
        assert result is not None
        assert isinstance(result, TimingMetrics)
        assert result.operation == "manual_operation"
        assert result.is_complete
        assert result.success is True
    
    def test_stop_nonexistent_timing(self):
        """Test stopping nonexistent timing."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        # Try to stop timing that doesn't exist
        result = logger.stop_timing("nonexistent_id")
        
        assert result is None
    
    def test_record_timing_directly(self):
        """Test recording timing measurements directly."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        # Record a timing directly
        logger.record_timing(
            "direct_operation",
            duration=0.5,  # 500ms
            success=True,
            param1="value1"
        )
        
        # Should be in metrics
        assert "direct_operation" in logger._metrics
        metrics = logger._metrics["direct_operation"]
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert abs(metrics.total_duration - 0.5) < 0.01
    
    def test_get_metrics_single_operation(self):
        """Test getting metrics for single operation."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        # Add some measurements
        with logger.measure("test_op"):
            time.sleep(0.01)
        
        # Get metrics for specific operation
        metrics = logger.get_metrics("test_op")
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation == "test_op"
        assert metrics.total_calls == 1
    
    def test_get_metrics_all_operations(self):
        """Test getting metrics for all operations."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        # Add measurements for different operations
        with logger.measure("op1"):
            time.sleep(0.01)
        
        with logger.measure("op2"):
            time.sleep(0.01)
        
        # Get all metrics
        all_metrics = logger.get_metrics()
        assert isinstance(all_metrics, dict)
        assert len(all_metrics) == 2
        assert "op1" in all_metrics
        assert "op2" in all_metrics
    
    def test_reset_metrics(self):
        """Test resetting performance metrics."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        # Add some measurements
        with logger.measure("test_op"):
            time.sleep(0.01)
        
        assert len(logger._metrics) == 1
        
        # Reset specific operation
        logger.reset_metrics("test_op")
        assert "test_op" not in logger._metrics
        
        # Add measurements again
        with logger.measure("op1"):
            pass
        with logger.measure("op2"):
            pass
        
        assert len(logger._metrics) == 2
        
        # Reset all metrics
        logger.reset_metrics()
        assert len(logger._metrics) == 0
    
    def test_get_summary(self):
        """Test getting performance summary."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        # Add measurements
        for i in range(5):
            with logger.measure("test_op"):
                time.sleep(0.01)
        
        # Get summary
        summary = logger.get_summary()
        
        assert isinstance(summary, dict)
        assert summary["total_operations"] == 1
        assert summary["total_calls"] == 5
        assert summary["total_duration"] > 0
        assert summary["overall_success_rate"] == 100.0
        assert "operations" in summary
        assert "test_op" in summary["operations"]
    
    def test_get_top_operations(self):
        """Test getting top operations by metrics."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        # Add different operations with different durations
        operations = [
            ("fast_op", 0.01),
            ("slow_op", 0.1), 
            ("medium_op", 0.05)
        ]
        
        for op_name, duration in operations:
            logger.record_timing(op_name, duration=duration, success=True)
        
        # Get top operations by duration
        top_ops = logger.get_top_operations("total_duration", limit=2)
        
        assert len(top_ops) == 2
        assert top_ops[0]["name"] == "slow_op"  # Longest duration first
        assert top_ops[1]["name"] == "medium_op"
    
    def test_export_metrics_json(self):
        """Test exporting metrics as JSON."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        # Add some measurements
        with logger.measure("test_op"):
            time.sleep(0.01)
        
        # Export as JSON
        json_data = logger.export_metrics("json")
        
        assert isinstance(json_data, str)
        # Should be valid JSON
        import json
        parsed = json.loads(json_data)
        assert "operations" in parsed
        assert "test_op" in parsed["operations"]
    
    def test_export_metrics_csv(self):
        """Test exporting metrics as CSV."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        # Add some measurements
        with logger.measure("test_op"):
            time.sleep(0.01)
        
        # Export as CSV
        csv_data = logger.export_metrics("csv")
        
        assert isinstance(csv_data, str)
        assert "operation,total_calls" in csv_data
        assert "test_op" in csv_data
    
    def test_export_unsupported_format(self):
        """Test exporting with unsupported format raises error."""
        logger = PerformanceLogger("test_logger", auto_log=False)
        
        with pytest.raises(ValueError) as exc_info:
            logger.export_metrics("unsupported_format")
        
        assert "Unsupported export format" in str(exc_info.value)
    
    def test_performance_logger_repr(self):
        """Test performance logger string representation."""
        logger = PerformanceLogger("test_logger", auto_log=True)
        
        repr_str = repr(logger)
        
        assert "PerformanceLogger" in repr_str
        assert "test_logger" in repr_str
        assert "auto_log=True" in repr_str


@pytest.mark.integration
class TestPerformanceLoggerIntegration:
    """Integration tests for PerformanceLogger."""
    
    def test_integration_with_structured_logger(self):
        """Test integration with StructuredLogger."""
        structured_logger = StructuredLogger("perf.integration")
        perf_logger = PerformanceLogger(
            "integration_test",
            logger=structured_logger,
            auto_log=True
        )
        
        with perf_logger.measure("integration_operation"):
            time.sleep(0.01)
        
        # Should complete without errors
        metrics = perf_logger.get_metrics("integration_operation")
        assert metrics.total_calls == 1
    
    def test_concurrent_measurements(self):
        """Test concurrent performance measurements."""
        import threading
        
        logger = PerformanceLogger("concurrent_test", auto_log=False)
        results = []
        
        def measure_operation(thread_id):
            with logger.measure(f"concurrent_op_{thread_id}"):
                time.sleep(0.01)
            results.append(thread_id)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=measure_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All operations should be recorded
        assert len(results) == 5
        assert len(logger._metrics) == 5
        
        # Each operation should have one call
        for i in range(5):
            metrics = logger.get_metrics(f"concurrent_op_{i}")
            assert metrics.total_calls == 1


@pytest.mark.performance
class TestPerformanceLoggerPerformance:
    """Performance tests for PerformanceLogger."""
    
    def test_measurement_overhead(self, benchmark):
        """Test the overhead of performance measurement."""
        logger = PerformanceLogger("benchmark_test", auto_log=False, track_metrics=False)
        
        def measure_simple_operation():
            with logger.measure("benchmark_op"):
                # Simple operation
                x = sum(range(100))
                return x
        
        # Benchmark should show minimal overhead
        result = benchmark(measure_simple_operation)
        assert result is not None
    
    def test_metrics_collection_performance(self, benchmark):
        """Test performance of metrics collection."""
        logger = PerformanceLogger("metrics_test", auto_log=False, track_metrics=True)
        
        def collect_metrics():
            with logger.measure("metrics_op"):
                time.sleep(0.001)  # 1ms operations
        
        benchmark(collect_metrics)
        
        # Should have collected metrics
        assert len(logger._metrics) == 1
        metrics = logger._metrics["metrics_op"]
        # Fixed: Don't check exact count as benchmark runs multiple times
        assert metrics.total_calls >= 1