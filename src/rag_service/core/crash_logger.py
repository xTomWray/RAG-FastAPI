"""Crash-safe logging for diagnosing system crashes.

This module provides comprehensive logging that:
1. Flushes immediately to disk after every write (survives crashes)
2. Logs system state (CPU, RAM, GPU, disk) before operations
3. Traces the full pipeline: ingestion -> chunking -> embedding -> storage
4. Creates timestamped logs for correlation with Windows Event Viewer
5. Captures enough context to pinpoint what caused a crash

Log files are written to ./logs/crash_*.log with immediate disk sync.

Usage:
    from rag_service.core.crash_logger import crash_logger, trace_operation

    # Simple logging
    crash_logger.info("Starting document processing")

    # With system state
    crash_logger.info("Before embedding", capture_state=True)

    # Trace an operation
    with trace_operation("process_file", file_path="/path/to/file.pdf"):
        # ... operation code ...
        pass
"""

import atexit
import datetime
import gc
import os
import platform
import sys
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TextIO


class LogLevel(Enum):
    """Log levels."""
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class SystemState:
    """Snapshot of system state."""
    timestamp: str = ""

    # CPU
    cpu_percent: float | None = None

    # Memory
    ram_total_gb: float | None = None
    ram_used_gb: float | None = None
    ram_percent: float | None = None

    # GPU
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_memory_total_gb: float | None = None
    gpu_memory_used_gb: float | None = None
    gpu_memory_percent: float | None = None
    gpu_temperature_c: float | None = None
    gpu_utilization: int | None = None
    # Enhanced GPU metrics
    gpu_power_draw_w: float | None = None
    gpu_power_limit_w: float | None = None
    gpu_clock_mhz: int | None = None

    # Process
    process_memory_mb: float | None = None
    thread_count: int | None = None

    # Disk
    disk_free_gb: float | None = None

    def to_log_lines(self) -> list[str]:
        """Convert to log lines."""
        lines = []

        # RAM
        if self.ram_percent is not None:
            lines.append(
                f"    RAM: {self.ram_used_gb:.1f}/{self.ram_total_gb:.1f} GB "
                f"({self.ram_percent:.1f}%)"
            )

        # CPU
        if self.cpu_percent is not None:
            lines.append(f"    CPU: {self.cpu_percent:.1f}%")

        # GPU
        if self.gpu_available:
            gpu_line = f"    GPU: {self.gpu_name}"
            if self.gpu_memory_percent is not None:
                gpu_line += (
                    f" | MEM: {self.gpu_memory_used_gb:.2f}/"
                    f"{self.gpu_memory_total_gb:.1f} GB ({self.gpu_memory_percent:.1f}%)"
                )
            if self.gpu_temperature_c is not None:
                gpu_line += f" | TEMP: {self.gpu_temperature_c}°C"
            if self.gpu_utilization is not None:
                gpu_line += f" | UTIL: {self.gpu_utilization}%"
            lines.append(gpu_line)

            # Second line for power and clock info (critical for crash diagnosis)
            if self.gpu_power_draw_w is not None:
                power_line = f"    GPU_PWR: {self.gpu_power_draw_w:.1f}W"
                if self.gpu_power_limit_w:
                    pct = (self.gpu_power_draw_w / self.gpu_power_limit_w) * 100
                    power_line += f"/{self.gpu_power_limit_w:.0f}W ({pct:.0f}%)"
                if self.gpu_clock_mhz:
                    power_line += f" | CLK: {self.gpu_clock_mhz}MHz"
                lines.append(power_line)

        # Process
        if self.process_memory_mb is not None:
            lines.append(
                f"    PROCESS: {self.process_memory_mb:.1f} MB | "
                f"threads={self.thread_count}"
            )

        # Disk
        if self.disk_free_gb is not None:
            lines.append(f"    DISK: {self.disk_free_gb:.1f} GB free")

        return lines


def capture_system_state() -> SystemState:
    """Capture current system state.

    Returns:
        SystemState with current metrics.
    """
    state = SystemState(
        timestamp=datetime.datetime.now().isoformat(timespec="milliseconds")
    )

    # RAM and CPU via psutil (if available)
    try:
        import psutil

        # RAM
        mem = psutil.virtual_memory()
        state.ram_total_gb = mem.total / (1024**3)
        state.ram_used_gb = mem.used / (1024**3)
        state.ram_percent = mem.percent

        # CPU
        state.cpu_percent = psutil.cpu_percent(interval=None)

        # Process info
        process = psutil.Process()
        state.process_memory_mb = process.memory_info().rss / (1024**2)
        state.thread_count = threading.active_count()

        # Disk
        disk = psutil.disk_usage('.')
        state.disk_free_gb = disk.free / (1024**3)

    except ImportError:
        pass
    except Exception:
        pass

    # GPU via PyTorch
    try:
        import torch

        if torch.cuda.is_available():
            state.gpu_available = True
            state.gpu_name = torch.cuda.get_device_name(0)

            props = torch.cuda.get_device_properties(0)
            state.gpu_memory_total_gb = props.total_memory / (1024**3)

            mem_reserved = torch.cuda.memory_reserved(0)
            state.gpu_memory_used_gb = mem_reserved / (1024**3)
            state.gpu_memory_percent = (
                100 * mem_reserved / props.total_memory
                if props.total_memory > 0 else 0
            )
    except Exception:
        pass

    # GPU temperature, utilization, power, and clocks via pynvml
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        try:
            state.gpu_temperature_c = float(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            pass

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            state.gpu_utilization = util.gpu
        except Exception:
            pass

        # Power draw (critical for crash diagnosis)
        try:
            state.gpu_power_draw_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            pass

        try:
            state.gpu_power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except Exception:
            pass

        # Clock speed
        try:
            state.gpu_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        except Exception:
            pass

        pynvml.nvmlShutdown()
    except Exception:
        pass

    return state


class CrashSafeLogger:
    """Logger that flushes immediately to survive system crashes."""

    def __init__(
        self,
        log_dir: Path | str = "./logs",
        log_prefix: str = "crash",
        max_size_mb: float = 100.0,
        min_level: LogLevel = LogLevel.DEBUG,
        enabled: bool = True,
        console_echo: bool = True,
    ):
        """Initialize crash-safe logger.

        Args:
            log_dir: Directory for log files.
            log_prefix: Prefix for log filenames.
            max_size_mb: Maximum log file size before rotation.
            min_level: Minimum log level to record.
            enabled: Whether logging is enabled.
            console_echo: Whether to also print to console.
        """
        self._enabled = enabled
        self._min_level = min_level
        self._console_echo = console_echo
        self._lock = threading.Lock()
        self._file: TextIO | None = None
        self._log_path: Path | None = None
        self._max_size = int(max_size_mb * 1024 * 1024)
        self._operation_stack: list[str] = []

        if enabled:
            self._setup_log_file(Path(log_dir), log_prefix)
            atexit.register(self.close)

    def _setup_log_file(self, log_dir: Path, log_prefix: str) -> None:
        """Create log directory and file."""
        try:
            log_dir.mkdir(parents=True, exist_ok=True)

            # Use date in filename
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            self._log_path = log_dir / f"{log_prefix}_{date_str}.log"

            # Open in append mode with line buffering
            self._file = open(self._log_path, "a", encoding="utf-8", buffering=1)

            # Write session header
            self._write_header()
        except Exception as e:
            print(f"WARNING: Could not create crash log: {e}", file=sys.stderr)
            self._enabled = False

    def _write_header(self) -> None:
        """Write session header with system info."""
        if not self._file:
            return

        now = datetime.datetime.now()

        lines = [
            "",
            "=" * 90,
            f"SESSION START: {now.isoformat()}",
            f"Log File: {self._log_path}",
            "-" * 90,
            f"Python: {sys.version}",
            f"Platform: {platform.platform()}",
            f"Machine: {platform.machine()}",
            f"Processor: {platform.processor()}",
        ]

        # Initial system state
        state = capture_system_state()
        if state.ram_total_gb:
            lines.append(f"Total RAM: {state.ram_total_gb:.1f} GB")
        if state.gpu_available:
            lines.append(f"GPU: {state.gpu_name}")
            lines.append(f"GPU Memory: {state.gpu_memory_total_gb:.1f} GB")

        lines.extend([
            "-" * 90,
            "FORMAT: [timestamp] [level] [operation_context] message",
            "        (system state on separate indented lines when captured)",
            "=" * 90,
            "",
        ])

        for line in lines:
            self._file.write(line + "\n")
        self._flush()

    def _flush(self) -> None:
        """Flush and sync to disk."""
        if self._file:
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except Exception:
                pass

    def _format_message(
        self,
        message: str,
        level: LogLevel,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Format a log message."""
        timestamp = datetime.datetime.now().isoformat(timespec="milliseconds")
        level_str = level.name

        # Build operation context
        if self._operation_stack:
            op_context = " > ".join(self._operation_stack)
            prefix = f"[{timestamp}] [{level_str:8}] [{op_context}]"
        else:
            prefix = f"[{timestamp}] [{level_str:8}]"

        # Add context dict if provided
        if context:
            ctx_str = " | ".join(f"{k}={v}" for k, v in context.items())
            return f"{prefix} {message} | {ctx_str}"

        return f"{prefix} {message}"

    def _log(
        self,
        message: str,
        level: LogLevel,
        capture_state: bool = False,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Internal logging method."""
        if not self._enabled or not self._file:
            return

        if level.value < self._min_level.value:
            return

        with self._lock:
            try:
                # Format and write message
                formatted = self._format_message(message, level, context)
                self._file.write(formatted + "\n")

                # Console echo
                if self._console_echo and level.value >= LogLevel.INFO.value:
                    print(formatted)

                # Capture system state if requested
                if capture_state:
                    state = capture_system_state()
                    for line in state.to_log_lines():
                        self._file.write(line + "\n")

                # CRITICAL: Flush to disk immediately
                self._flush()

                # Check rotation
                self._check_rotation()

            except Exception as e:
                print(f"Crash log write error: {e}", file=sys.stderr)

    def debug(
        self,
        message: str,
        capture_state: bool = False,
        **context: Any,
    ) -> None:
        """Log debug message."""
        self._log(message, LogLevel.DEBUG, capture_state, context or None)

    def info(
        self,
        message: str,
        capture_state: bool = False,
        **context: Any,
    ) -> None:
        """Log info message."""
        self._log(message, LogLevel.INFO, capture_state, context or None)

    def warn(
        self,
        message: str,
        capture_state: bool = False,
        **context: Any,
    ) -> None:
        """Log warning message."""
        self._log(message, LogLevel.WARN, capture_state, context or None)

    def error(
        self,
        message: str,
        capture_state: bool = True,
        exc_info: bool = False,
        **context: Any,
    ) -> None:
        """Log error message."""
        if exc_info:
            tb = traceback.format_exc()
            message = f"{message}\n{tb}"
        self._log(message, LogLevel.ERROR, capture_state, context or None)

    def critical(
        self,
        message: str,
        capture_state: bool = True,
        exc_info: bool = False,
        **context: Any,
    ) -> None:
        """Log critical message."""
        if exc_info:
            tb = traceback.format_exc()
            message = f"{message}\n{tb}"
        self._log(message, LogLevel.CRITICAL, capture_state, context or None)

    def push_context(self, operation: str) -> None:
        """Push an operation onto the context stack."""
        self._operation_stack.append(operation)

    def pop_context(self) -> None:
        """Pop an operation from the context stack."""
        if self._operation_stack:
            self._operation_stack.pop()

    def log_state(self, label: str = "STATE_CHECK") -> None:
        """Log current system state with a label."""
        self._log(label, LogLevel.INFO, capture_state=True)

    def _check_rotation(self) -> None:
        """Rotate log file if too large."""
        if not self._file or not self._log_path:
            return

        try:
            if self._log_path.stat().st_size > self._max_size:
                self._file.close()

                # Rename old file
                backup = self._log_path.with_suffix(".log.old")
                if backup.exists():
                    backup.unlink()
                self._log_path.rename(backup)

                # Create new file
                self._file = open(self._log_path, "a", encoding="utf-8", buffering=1)
                self._write_header()
        except Exception:
            pass

    def close(self) -> None:
        """Close the log file."""
        if self._file:
            try:
                self._log("SESSION END", LogLevel.INFO, capture_state=True)
                self._file.close()
            except Exception:
                pass
            self._file = None


class OperationTracer:
    """Context manager for tracing operations."""

    def __init__(
        self,
        logger: CrashSafeLogger,
        operation: str,
        capture_state_before: bool = True,
        capture_state_after: bool = True,
        **context: Any,
    ):
        self.logger = logger
        self.operation = operation
        self.capture_before = capture_state_before
        self.capture_after = capture_state_after
        self.context = context
        self.start_time: datetime.datetime | None = None

    def __enter__(self):
        self.start_time = datetime.datetime.now()
        self.logger.push_context(self.operation)
        self.logger.info(
            f">>> START",
            capture_state=self.capture_before,
            **self.context,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.datetime.now() - self.start_time).total_seconds()

        if exc_type is not None:
            self.logger.error(
                f"<<< FAILED after {duration:.3f}s | {exc_type.__name__}: {exc_val}",
                capture_state=True,
            )
        else:
            self.logger.info(
                f"<<< DONE in {duration:.3f}s",
                capture_state=self.capture_after,
            )

        self.logger.pop_context()
        return False  # Don't suppress exceptions

    def checkpoint(self, label: str, **extra_context: Any) -> None:
        """Log a checkpoint within the operation."""
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"... {label} (elapsed: {elapsed:.3f}s)",
            capture_state=True,
            **extra_context,
        )


# Global logger instance
_crash_logger: CrashSafeLogger | None = None
_logger_lock = threading.Lock()


def get_crash_logger(
    log_dir: Path | str = "./logs",
    enabled: bool = True,
    console_echo: bool = False,
) -> CrashSafeLogger:
    """Get or create the global crash logger.

    Args:
        log_dir: Directory for log files.
        enabled: Whether logging is enabled.
        console_echo: Whether to echo to console.

    Returns:
        The crash-safe logger instance.
    """
    global _crash_logger
    with _logger_lock:
        if _crash_logger is None:
            _crash_logger = CrashSafeLogger(
                log_dir=log_dir,
                enabled=enabled,
                console_echo=console_echo,
            )
    return _crash_logger


def configure_crash_logger(
    log_dir: Path | str = "./logs",
    enabled: bool = True,
    console_echo: bool = False,
    min_level: LogLevel = LogLevel.DEBUG,
) -> CrashSafeLogger:
    """Configure and return the global crash logger.

    Call this at application startup to configure logging settings.

    Args:
        log_dir: Directory for log files.
        enabled: Whether logging is enabled.
        console_echo: Whether to echo to console.
        min_level: Minimum log level.

    Returns:
        The configured crash-safe logger instance.
    """
    global _crash_logger
    with _logger_lock:
        if _crash_logger is not None:
            _crash_logger.close()
        _crash_logger = CrashSafeLogger(
            log_dir=log_dir,
            enabled=enabled,
            console_echo=console_echo,
            min_level=min_level,
        )
    return _crash_logger


# Convenience functions
def trace_operation(
    operation: str,
    capture_state_before: bool = True,
    capture_state_after: bool = True,
    **context: Any,
) -> OperationTracer:
    """Create a context manager for tracing an operation.

    Usage:
        with trace_operation("process_pdf", file="doc.pdf", size_mb=5.2):
            # ... operation code ...
            pass

    Args:
        operation: Name of the operation.
        capture_state_before: Capture system state at start.
        capture_state_after: Capture system state at end.
        **context: Additional context to log.

    Returns:
        Context manager that traces the operation.
    """
    return OperationTracer(
        get_crash_logger(),
        operation,
        capture_state_before,
        capture_state_after,
        **context,
    )


def log_state(label: str = "STATE_CHECK") -> None:
    """Log current system state."""
    get_crash_logger().log_state(label)


# Alias for convenience
crash_logger = property(lambda self: get_crash_logger())


# Module-level functions that use the global logger
def debug(message: str, capture_state: bool = False, **context: Any) -> None:
    """Log debug message."""
    get_crash_logger().debug(message, capture_state, **context)


def info(message: str, capture_state: bool = False, **context: Any) -> None:
    """Log info message."""
    get_crash_logger().info(message, capture_state, **context)


def warn(message: str, capture_state: bool = False, **context: Any) -> None:
    """Log warning message."""
    get_crash_logger().warn(message, capture_state, **context)


def error(
    message: str,
    capture_state: bool = True,
    exc_info: bool = False,
    **context: Any,
) -> None:
    """Log error message."""
    get_crash_logger().error(message, capture_state, exc_info, **context)


def critical(
    message: str,
    capture_state: bool = True,
    exc_info: bool = False,
    **context: Any,
) -> None:
    """Log critical message."""
    get_crash_logger().critical(message, capture_state, exc_info, **context)
