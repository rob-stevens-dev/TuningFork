"""Custom log handlers for TuningFork logging system.

This module provides specialized log handlers for different output
destinations and use cases including file rotation, remote logging,
and performance optimization.

Classes:
    ConsoleHandler: Enhanced console output handler
    FileHandler: Basic file output handler
    RotatingFileHandler: File handler with rotation
    TimedRotatingFileHandler: Time-based file rotation
    RemoteHandler: Handler for remote log aggregation
    BufferedHandler: Buffered handler for performance

Example:
    >>> handler = RotatingFileHandler(
    ...     filename="app.log",
    ...     maxBytes=10485760,  # 10MB
    ...     backupCount=5
    ... )
    >>> logger.addHandler(handler)
"""

import logging
import logging.handlers
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union
from datetime import datetime, timedelta
from queue import Queue, Empty
import json
import gzip
import urllib.request
import urllib.parse


class ConsoleHandler(logging.StreamHandler):
    """Enhanced console handler with color support and stream selection.
    
    This handler extends the standard StreamHandler with automatic
    stream selection based on log level and optional color support.
    """
    
    def __init__(
        self,
        *,
        use_stderr_for_errors: bool = True,
        colors: bool = True,
        color_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize console handler.
        
        Args:
            use_stderr_for_errors: Send ERROR/CRITICAL to stderr
            colors: Enable colored output
            color_map: Custom color mapping for log levels
        """
        # Start with stdout as default
        super().__init__(sys.stdout)
        
        self.use_stderr_for_errors = use_stderr_for_errors
        self.colors = colors and self._supports_color()
        
        # Default color mapping
        self.color_map = color_map or {
            'DEBUG': '\033[36m',     # Cyan
            'INFO': '\033[32m',      # Green
            'WARNING': '\033[33m',   # Yellow
            'ERROR': '\033[31m',     # Red
            'CRITICAL': '\033[35m',  # Magenta
            'RESET': '\033[0m',      # Reset
        }
    
    def _supports_color(self) -> bool:
        """Check if terminal supports color output.
        
        Returns:
            True if colors are supported
        """
        # Check if we're in a terminal
        if not (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()):
            return False
        
        # Check environment variables
        if os.environ.get('NO_COLOR'):
            return False
        
        if os.environ.get('FORCE_COLOR'):
            return True
        
        # Check TERM environment variable
        term = os.environ.get('TERM', '')
        if 'color' in term or term in ['xterm', 'xterm-256color', 'screen']:
            return True
        
        return False
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record with appropriate stream selection.
        
        Args:
            record: Log record to emit
        """
        # Select appropriate stream
        if (self.use_stderr_for_errors and 
            record.levelno >= logging.ERROR):
            original_stream = self.stream
            self.stream = sys.stderr
            try:
                super().emit(record)
            finally:
                self.stream = original_stream
        else:
            super().emit(record)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format record with optional colors.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        # Get formatted message from formatter
        formatted = super().format(record)
        
        # Add colors if enabled
        if self.colors and record.levelname in self.color_map:
            color_code = self.color_map[record.levelname]
            reset_code = self.color_map['RESET']
            formatted = f"{color_code}{formatted}{reset_code}"
        
        return formatted


class FileHandler(logging.FileHandler):
    """Enhanced file handler with directory creation and permissions.
    
    This handler extends the standard FileHandler with automatic
    directory creation and configurable file permissions.
    """
    
    def __init__(
        self,
        filename: Union[str, Path],
        *,
        mode: str = 'a',
        encoding: str = 'utf-8',
        delay: bool = False,
        create_dirs: bool = True,
        file_mode: Optional[int] = None,
    ) -> None:
        """Initialize file handler.
        
        Args:
            filename: Log file path
            mode: File open mode
            encoding: File encoding
            delay: Delay file opening until first emit
            create_dirs: Create parent directories if they don't exist
            file_mode: File permissions (octal, e.g., 0o644)
        """
        self.create_dirs = create_dirs
        self.file_mode = file_mode
        
        # Convert to Path for easier handling
        filename_path = Path(filename)
        
        # Create parent directories if needed
        if create_dirs and not filename_path.parent.exists():
            filename_path.parent.mkdir(parents=True, exist_ok=True)
        
        super().__init__(str(filename_path), mode=mode, encoding=encoding, delay=delay)
    
    def _open(self) -> TextIO:
        """Open file with optional permission setting.
        
        Returns:
            Opened file object
        """
        stream = super()._open()
        
        # Set file permissions if specified
        if self.file_mode is not None:
            try:
                os.chmod(self.baseFilename, self.file_mode)
            except OSError:
                # Ignore permission errors (might not have rights)
                pass
        
        return stream


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Enhanced rotating file handler with compression and cleanup.
    
    This handler extends the standard RotatingFileHandler with
    optional compression of rotated files and configurable cleanup.
    """
    
    def __init__(
        self,
        filename: Union[str, Path],
        *,
        maxBytes: int = 10485760,  # 10MB
        backupCount: int = 5,
        encoding: str = 'utf-8',
        delay: bool = False,
        compress_rotated: bool = False,
        create_dirs: bool = True,
    ) -> None:
        """Initialize rotating file handler.
        
        Args:
            filename: Log file path
            maxBytes: Maximum file size before rotation
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening until first emit
            compress_rotated: Compress rotated files with gzip
            create_dirs: Create parent directories if needed
        """
        self.compress_rotated = compress_rotated
        
        # Convert to Path and create directories
        filename_path = Path(filename)
        if create_dirs and not filename_path.parent.exists():
            filename_path.parent.mkdir(parents=True, exist_ok=True)
        
        super().__init__(
            str(filename_path),
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay
        )
    
    def doRollover(self) -> None:
        """Perform file rollover with optional compression."""
        # Perform standard rollover
        super().doRollover()
        
        # Compress rotated files if enabled
        if self.compress_rotated and self.backupCount > 0:
            # Compress the most recent backup (filename.1)
            backup_file = f"{self.baseFilename}.1"
            if os.path.exists(backup_file):
                compressed_file = f"{backup_file}.gz"
                
                try:
                    with open(backup_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            f_out.writelines(f_in)
                    
                    # Remove uncompressed file
                    os.remove(backup_file)
                    
                    # Rename other backup files to account for compression
                    for i in range(2, self.backupCount + 1):
                        old_name = f"{self.baseFilename}.{i}"
                        new_name = f"{self.baseFilename}.{i-1}.gz"
                        
                        if os.path.exists(old_name):
                            if os.path.exists(new_name):
                                os.remove(new_name)
                            os.rename(old_name, new_name.replace('.gz', ''))
                            
                            # Compress the renamed file
                            with open(new_name.replace('.gz', ''), 'rb') as f_in:
                                with gzip.open(new_name, 'wb') as f_out:
                                    f_out.writelines(f_in)
                            os.remove(new_name.replace('.gz', ''))
                
                except (OSError, IOError) as e:
                    # Log compression failure but don't stop logging
                    self.handleError(None)


class TimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """Enhanced timed rotating file handler with cleanup and compression.
    
    This handler extends the standard TimedRotatingFileHandler with
    improved cleanup and optional compression of old log files.
    """
    
    def __init__(
        self,
        filename: Union[str, Path],
        *,
        when: str = 'midnight',
        interval: int = 1,
        backupCount: int = 30,
        encoding: str = 'utf-8',
        delay: bool = False,
        utc: bool = False,
        compress_rotated: bool = False,
        create_dirs: bool = True,
        cleanup_days: Optional[int] = None,
    ) -> None:
        """Initialize timed rotating file handler.
        
        Args:
            filename: Log file path
            when: When to rotate ('S', 'M', 'H', 'D', 'midnight', 'W0'-'W6')
            interval: Rotation interval
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening until first emit
            utc: Use UTC time for rotation
            compress_rotated: Compress rotated files
            create_dirs: Create parent directories if needed
            cleanup_days: Additional cleanup based on file age (days)
        """
        self.compress_rotated = compress_rotated
        self.cleanup_days = cleanup_days
        
        # Convert to Path and create directories
        filename_path = Path(filename)
        if create_dirs and not filename_path.parent.exists():
            filename_path.parent.mkdir(parents=True, exist_ok=True)
        
        super().__init__(
            str(filename_path),
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            utc=utc
        )
    
    def doRollover(self) -> None:
        """Perform timed rollover with compression and cleanup."""
        # Perform standard rollover
        super().doRollover()
        
        # Compress rotated file if enabled
        if self.compress_rotated:
            # Find the most recently rotated file
            log_dir = Path(self.baseFilename).parent
            log_name = Path(self.baseFilename).name
            
            # Look for rotated files
            for log_file in log_dir.glob(f"{log_name}.*"):
                if not log_file.name.endswith('.gz'):
                    try:
                        compressed_file = log_file.with_suffix(log_file.suffix + '.gz')
                        
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(compressed_file, 'wb') as f_out:
                                f_out.writelines(f_in)
                        
                        # Remove uncompressed file
                        log_file.unlink()
                        
                    except (OSError, IOError):
                        # Ignore compression errors
                        self.handleError(None)
        
        # Additional cleanup based on file age
        if self.cleanup_days:
            self._cleanup_old_files()
    
    def _cleanup_old_files(self) -> None:
        """Clean up files older than cleanup_days."""
        if not self.cleanup_days:
            return
        
        cutoff_time = time.time() - (self.cleanup_days * 24 * 60 * 60)
        log_dir = Path(self.baseFilename).parent
        log_name = Path(self.baseFilename).name
        
        # Find and remove old log files
        for log_file in log_dir.glob(f"{log_name}.*"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
            except (OSError, IOError):
                # Ignore cleanup errors
                pass


class BufferedHandler(logging.Handler):
    """Buffered handler for high-performance logging.
    
    This handler buffers log records in memory and flushes them
    periodically or when the buffer reaches a certain size.
    """
    
    def __init__(
        self,
        target_handler: logging.Handler,
        *,
        buffer_size: int = 1000,
        flush_interval: float = 5.0,
        flush_level: int = logging.ERROR,
    ) -> None:
        """Initialize buffered handler.
        
        Args:
            target_handler: Handler to receive flushed records
            buffer_size: Maximum records to buffer before auto-flush
            flush_interval: Time interval for periodic flush (seconds)
            flush_level: Log level that triggers immediate flush
        """
        super().__init__()
        self.target_handler = target_handler
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.flush_level = flush_level
        
        self._buffer: List[logging.LogRecord] = []
        self._last_flush = time.time()
        self._lock = threading.Lock()
        
        # Start flush timer thread
        self._timer_thread = threading.Thread(target=self._flush_timer, daemon=True)
        self._stop_timer = threading.Event()
        self._timer_thread.start()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Add record to buffer and check flush conditions.
        
        Args:
            record: Log record to buffer
        """
        with self._lock:
            self._buffer.append(record)
            
            # Check flush conditions
            should_flush = (
                len(self._buffer) >= self.buffer_size or
                record.levelno >= self.flush_level or
                (time.time() - self._last_flush) >= self.flush_interval
            )
            
            if should_flush:
                self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Flush buffered records to target handler."""
        if not self._buffer:
            return
        
        records_to_flush = self._buffer[:]
        self._buffer.clear()
        self._last_flush = time.time()
        
        # Emit records to target handler
        for record in records_to_flush:
            try:
                self.target_handler.emit(record)
            except Exception:
                self.handleError(record)
    
    def _flush_timer(self) -> None:
        """Timer thread for periodic buffer flushing."""
        while not self._stop_timer.wait(self.flush_interval):
            with self._lock:
                if self._buffer and (time.time() - self._last_flush) >= self.flush_interval:
                    self._flush_buffer()
    
    def flush(self) -> None:
        """Manually flush the buffer."""
        with self._lock:
            self._flush_buffer()
        self.target_handler.flush()
    
    def close(self) -> None:
        """Close handler and clean up resources."""
        # Stop timer thread
        self._stop_timer.set()
        if self._timer_thread.is_alive():
            self._timer_thread.join(timeout=1.0)
        
        # Flush any remaining records
        self.flush()
        
        # Close target handler
        self.target_handler.close()
        super().close()


class RemoteHandler(logging.Handler):
    """Handler for sending logs to remote aggregation systems.
    
    This handler sends log records to remote systems like ELK stack,
    Splunk, or other log aggregation platforms via HTTP.
    """
    
    def __init__(
        self,
        url: str,
        *,
        method: str = 'POST',
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        batch_size: int = 100,
        flush_interval: float = 30.0,
        max_retries: int = 3,
        auth_token: Optional[str] = None,
    ) -> None:
        """Initialize remote handler.
        
        Args:
            url: Remote endpoint URL
            method: HTTP method to use
            headers: Additional HTTP headers
            timeout: Request timeout in seconds
            batch_size: Number of records to batch together
            flush_interval: Time interval for batch flushing
            max_retries: Maximum retry attempts for failed requests
            auth_token: Authentication token for the remote service
        """
        super().__init__()
        self.url = url
        self.method = method.upper()
        self.timeout = timeout
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        
        # Setup headers
        self.headers = headers or {}
        self.headers.setdefault('Content-Type', 'application/json')
        
        if auth_token:
            self.headers['Authorization'] = f'Bearer {auth_token}'
        
        # Batching setup
        self._batch: List[Dict[str, Any]] = []
        self._last_flush = time.time()
        self._lock = threading.Lock()
        
        # Start batch flush timer
        self._timer_thread = threading.Thread(target=self._flush_timer, daemon=True)
        self._stop_timer = threading.Event()
        self._timer_thread.start()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Add record to batch and check flush conditions.
        
        Args:
            record: Log record to send
        """
        try:
            # Convert record to dictionary
            log_dict = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
            }
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'message', 'exc_info', 
                    'exc_text', 'stack_info'
                }:
                    try:
                        json.dumps(value)  # Test serializability
                        log_dict[key] = value
                    except (TypeError, ValueError):
                        log_dict[key] = str(value)
            
            # Add exception info if present
            if record.exc_info:
                log_dict['exception'] = self.formatException(record.exc_info)
            
            with self._lock:
                self._batch.append(log_dict)
                
                # Check if we should flush
                if (len(self._batch) >= self.batch_size or
                    record.levelno >= logging.ERROR):
                    self._flush_batch()
        
        except Exception:
            self.handleError(record)
    
    def _flush_batch(self) -> None:
        """Flush current batch to remote endpoint."""
        if not self._batch:
            return
        
        batch_to_send = self._batch[:]
        self._batch.clear()
        self._last_flush = time.time()
        
        # Send batch with retries
        for attempt in range(self.max_retries + 1):
            try:
                self._send_batch(batch_to_send)
                break
            except Exception as e:
                if attempt == self.max_retries:
                    # Final attempt failed, handle error
                    self.handleError(None)
                else:
                    # Wait before retry (exponential backoff)
                    time.sleep(2 ** attempt)
    
    def _send_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Send batch of log records to remote endpoint.
        
        Args:
            batch: List of log record dictionaries to send
        """
        # Prepare request data
        if len(batch) == 1:
            data = json.dumps(batch[0])
        else:
            data = json.dumps({'logs': batch})
        
        # Create request
        req = urllib.request.Request(
            self.url,
            data=data.encode('utf-8'),
            headers=self.headers,
            method=self.method
        )
        
        # Send request
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            if response.status >= 400:
                raise Exception(f"HTTP {response.status}: {response.reason}")
    
    def _flush_timer(self) -> None:
        """Timer thread for periodic batch flushing."""
        while not self._stop_timer.wait(self.flush_interval):
            with self._lock:
                if (self._batch and 
                    (time.time() - self._last_flush) >= self.flush_interval):
                    self._flush_batch()
    
    def flush(self) -> None:
        """Manually flush current batch."""
        with self._lock:
            self._flush_batch()
    
    def close(self) -> None:
        """Close handler and clean up resources."""
        # Stop timer thread
        self._stop_timer.set()
        if self._timer_thread.is_alive():
            self._timer_thread.join(timeout=1.0)
        
        # Flush any remaining records
        self.flush()
        super().close()


class AsyncHandler(logging.Handler):
    """Asynchronous handler for non-blocking log processing.
    
    This handler queues log records and processes them in a
    separate thread to avoid blocking the main application.
    """
    
    def __init__(
        self,
        target_handler: logging.Handler,
        *,
        queue_size: int = 10000,
        timeout: float = 1.0,
        stop_timeout: float = 5.0,
    ) -> None:
        """Initialize async handler.
        
        Args:
            target_handler: Handler to process records asynchronously
            queue_size: Maximum queue size
            timeout: Timeout for queue operations
            stop_timeout: Timeout when stopping the handler
        """
        super().__init__()
        self.target_handler = target_handler
        self.timeout = timeout
        self.stop_timeout = stop_timeout
        
        # Setup queue and worker thread
        self._queue: Queue = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Queue record for asynchronous processing.
        
        Args:
            record: Log record to queue
        """
        try:
            self._queue.put_nowait(record)
        except:
            # Queue is full, drop the record or handle error
            self.handleError(record)
    
    def _worker(self) -> None:
        """Worker thread that processes queued records."""
        while not self._stop_event.is_set():
            try:
                # Get record from queue with timeout
                record = self._queue.get(timeout=self.timeout)
                
                # Process record with target handler
                try:
                    self.target_handler.emit(record)
                except Exception:
                    self.handleError(record)
                finally:
                    self._queue.task_done()
            
            except Empty:
                # Timeout occurred, continue loop
                continue
            except Exception:
                # Unexpected error in worker thread
                break
    
    def flush(self) -> None:
        """Wait for queue to be processed and flush target handler."""
        # Wait for queue to be empty
        self._queue.join()
        
        # Flush target handler
        self.target_handler.flush()
    
    def close(self) -> None:
        """Close handler and stop worker thread."""
        # Signal worker to stop
        self._stop_event.set()
        
        # Wait for worker thread to finish
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=self.stop_timeout)
        
        # Process any remaining records
        while not self._queue.empty():
            try:
                record = self._queue.get_nowait()
                self.target_handler.emit(record)
                self._queue.task_done()
            except Empty:
                break
            except Exception:
                break
        
        # Close target handler
        self.target_handler.close()
        super().close()


class MultiHandler(logging.Handler):
    """Handler that distributes records to multiple target handlers.
    
    This handler allows sending the same log record to multiple
    destinations with different formatting or filtering.
    """
    
    def __init__(
        self,
        handlers: List[logging.Handler],
        *,
        stop_on_error: bool = False,
    ) -> None:
        """Initialize multi-handler.
        
        Args:
            handlers: List of target handlers
            stop_on_error: Whether to stop processing on first error
        """
        super().__init__()
        self.handlers = handlers[:]
        self.stop_on_error = stop_on_error
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit record to all target handlers.
        
        Args:
            record: Log record to emit
        """
        for handler in self.handlers:
            try:
                # Check if handler should process this record
                if record.levelno >= handler.level:
                    handler.emit(record)
            except Exception:
                self.handleError(record)
                if self.stop_on_error:
                    break
    
    def flush(self) -> None:
        """Flush all target handlers."""
        for handler in self.handlers:
            try:
                handler.flush()
            except Exception:
                pass
    
    def close(self) -> None:
        """Close all target handlers."""
        for handler in self.handlers:
            try:
                handler.close()
            except Exception:
                pass
        super().close()
    
    def addHandler(self, handler: logging.Handler) -> None:
        """Add a handler to the list.
        
        Args:
            handler: Handler to add
        """
        if handler not in self.handlers:
            self.handlers.append(handler)
    
    def removeHandler(self, handler: logging.Handler) -> None:
        """Remove a handler from the list.
        
        Args:
            handler: Handler to remove
        """
        if handler in self.handlers:
            self.handlers.remove(handler)