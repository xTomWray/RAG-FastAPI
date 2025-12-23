"""Crash-safe GPU logging for diagnosing system crashes.

This module provides logging that:
1. Flushes immediately to disk after every write (survives crashes)
2. Logs GPU state before each operation (so we know what caused crash)
3. Creates a dedicated crash log file with timestamps
4. Logs in a format easy to correlate with Windows Event Viewer

Usage:
    from rag_service.core.gpu_crash_logger import gpu_logger, log_gpu_state

    gpu_logger.log("Starting batch processing")
    log_gpu_state("before_batch")
    # ... GPU operation ...
    log_gpu_state("after_batch")
"""

import atexit
import datetime
import os
import sys
import threading
from pathlib import Path
from typing import Any, TextIO


class CrashSafeLogger:
    """Logger that flushes immediately to survive system crashes."""

    def __init__(
        self,
        log_dir: Path | str = "./logs",
        log_name: str = "gpu_crash",
        max_size_mb: float = 50.0,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._lock = threading.Lock()
        self._file: TextIO | None = None
        self._log_path: Path | None = None
        self._max_size = int(max_size_mb * 1024 * 1024)

        if enabled:
            self._setup_log_file(Path(log_dir), log_name)
            atexit.register(self.close)

    def _setup_log_file(self, log_dir: Path, log_name: str) -> None:
        """Create log directory and file."""
        try:
            log_dir.mkdir(parents=True, exist_ok=True)

            # Use date in filename for easy identification
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            self._log_path = log_dir / f"{log_name}_{date_str}.log"

            # Open in append mode with line buffering (kept open for crash logging)
            self._file = open(self._log_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

            # Write session header
            self._write_header()
        except Exception as e:
            print(f"WARNING: Could not create crash log: {e}", file=sys.stderr)
            self._enabled = False

    def _write_header(self) -> None:
        """Write session header with system info."""
        if not self._file:
            return

        header = [
            "",
            "=" * 80,
            f"SESSION START: {datetime.datetime.now().isoformat()}",
            f"Python: {sys.version}",
            f"Platform: {sys.platform}",
        ]

        # Add GPU info
        try:
            import torch

            if torch.cuda.is_available():
                header.append("CUDA Available: True")
                header.append(f"CUDA Version: {torch.version.cuda}")
                header.append(f"GPU: {torch.cuda.get_device_name(0)}")
                props = torch.cuda.get_device_properties(0)
                header.append(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")
            else:
                header.append("CUDA Available: False")
        except Exception as e:
            header.append(f"CUDA Info Error: {e}")

        header.append("=" * 80)
        header.append("")

        for line in header:
            self._file.write(line + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def log(
        self,
        message: str,
        level: str = "INFO",
        gpu_state: bool = False,
    ) -> None:
        """Log a message immediately to disk.

        Args:
            message: The message to log.
            level: Log level (INFO, WARN, ERROR, DEBUG, CRITICAL).
            gpu_state: If True, also log current GPU state.
        """
        if not self._enabled or not self._file:
            return

        with self._lock:
            try:
                timestamp = datetime.datetime.now().isoformat(timespec="milliseconds")
                line = f"[{timestamp}] [{level:8}] {message}"
                self._file.write(line + "\n")

                if gpu_state:
                    self._write_gpu_state()

                # CRITICAL: Flush AND fsync to ensure data reaches disk
                self._file.flush()
                os.fsync(self._file.fileno())

                # Rotate if too large
                self._check_rotation()

            except Exception as e:
                print(f"Crash log write error: {e}", file=sys.stderr)

    def _write_gpu_state(self) -> None:
        """Write current GPU state to log."""
        if not self._file:
            return

        try:
            import torch

            if torch.cuda.is_available():
                # Memory info
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9

                state = (
                    f"           GPU_MEM: allocated={mem_allocated:.2f}GB, "
                    f"reserved={mem_reserved:.2f}GB, total={mem_total:.1f}GB, "
                    f"percent={100 * mem_reserved / mem_total:.1f}%"
                )
                self._file.write(state + "\n")

                # Temperature (if available)
                temp = self._get_gpu_temp()
                if temp is not None:
                    self._file.write(f"           GPU_TEMP: {temp}C\n")

                # GPU utilization (if available)
                util = self._get_gpu_utilization()
                if util is not None:
                    self._file.write(f"           GPU_UTIL: {util}%\n")

        except Exception as e:
            self._file.write(f"           GPU_STATE_ERROR: {e}\n")

    def _get_gpu_temp(self) -> float | None:
        """Get GPU temperature."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()
            return float(temp)
        except Exception:
            return None

    def _get_gpu_utilization(self) -> int | None:
        """Get GPU utilization percentage."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            return int(util.gpu)
        except Exception:
            return None

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

                # Create new file (kept open for crash logging)
                self._file = open(self._log_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
                self._write_header()
        except Exception:
            pass

    def close(self) -> None:
        """Close the log file."""
        if self._file:
            try:
                self.log("SESSION END", level="INFO")
                self._file.close()
            except Exception:
                pass
            self._file = None


class GPUOperationTracer:
    """Context manager for tracing GPU operations."""

    def __init__(
        self,
        logger: CrashSafeLogger,
        operation: str,
        details: str = "",
    ):
        self.logger = logger
        self.operation = operation
        self.details = details
        self.start_time: datetime.datetime | None = None

    def __enter__(self) -> "GPUOperationTracer":
        self.start_time = datetime.datetime.now()
        msg = f">>> START: {self.operation}"
        if self.details:
            msg += f" | {self.details}"
        self.logger.log(msg, gpu_state=True)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        assert self.start_time is not None  # Set in __enter__
        duration = (datetime.datetime.now() - self.start_time).total_seconds()

        if exc_type is not None:
            self.logger.log(
                f"<<< ERROR: {self.operation} | duration={duration:.3f}s | "
                f"error={exc_type.__name__}: {exc_val}",
                level="ERROR",
                gpu_state=True,
            )
        else:
            self.logger.log(
                f"<<< DONE: {self.operation} | duration={duration:.3f}s",
                gpu_state=True,
            )
        # Don't suppress exceptions (return None)


# Global logger instance
_gpu_logger: CrashSafeLogger | None = None


def get_gpu_logger(
    log_dir: Path | str = "./logs",
    enabled: bool = True,
) -> CrashSafeLogger:
    """Get or create the global GPU crash logger.

    Args:
        log_dir: Directory for log files.
        enabled: Whether logging is enabled.

    Returns:
        The crash-safe logger instance.
    """
    global _gpu_logger
    if _gpu_logger is None:
        _gpu_logger = CrashSafeLogger(log_dir=log_dir, enabled=enabled)
    return _gpu_logger


def log_gpu_state(label: str = "") -> None:
    """Log current GPU state with an optional label.

    Args:
        label: Optional label for the log entry.
    """
    logger = get_gpu_logger()
    msg = f"GPU_STATE_CHECK: {label}" if label else "GPU_STATE_CHECK"
    logger.log(msg, gpu_state=True)


def trace_gpu_operation(operation: str, details: str = "") -> GPUOperationTracer:
    """Create a context manager for tracing a GPU operation.

    Usage:
        with trace_gpu_operation("embed_batch", "batch_size=32"):
            # GPU operation here
            embeddings = model.encode(texts)

    Args:
        operation: Name of the operation.
        details: Additional details to log.

    Returns:
        Context manager that logs start/end with GPU state.
    """
    return GPUOperationTracer(get_gpu_logger(), operation, details)


def log_batch_progress(
    batch_num: int,
    total_batches: int,
    items_processed: int,
    total_items: int,
) -> None:
    """Log batch processing progress with GPU state.

    Args:
        batch_num: Current batch number (1-indexed).
        total_batches: Total number of batches.
        items_processed: Number of items processed so far.
        total_items: Total number of items.
    """
    logger = get_gpu_logger()
    progress_pct = 100 * items_processed / total_items if total_items > 0 else 0
    logger.log(
        f"BATCH_PROGRESS: batch={batch_num}/{total_batches}, "
        f"items={items_processed}/{total_items} ({progress_pct:.1f}%)",
        gpu_state=True,
    )
