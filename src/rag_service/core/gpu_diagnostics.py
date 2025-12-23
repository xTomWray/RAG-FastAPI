"""Enhanced GPU diagnostics for troubleshooting system crashes.

This module provides deep diagnostic information about the GPU state that
can help identify the root cause of system crashes during GPU operations.

Key diagnostics captured:
1. CUDA driver/runtime versions - Version mismatches cause crashes
2. Power draw (TDP) - PSU insufficiency causes crashes during power spikes
3. PCIe bandwidth/link state - Bus errors cause instant system hangs
4. GPU ECC/XID errors - Hardware-level error flags
5. Memory allocation details - Track fragmentation and allocation failures
6. CUDA context health - Detect context corruption early
7. Throttling reasons - Thermal, power, or other throttling indicators
"""

import datetime
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

# Constants for NVML throttle reasons (bitmask)
NVML_GPU_THROTTLE_REASON_NONE = 0x0
NVML_GPU_THROTTLE_REASON_GPU_IDLE = 0x1
NVML_GPU_THROTTLE_REASON_APP_CLOCK_SETTING = 0x2
NVML_GPU_THROTTLE_REASON_SW_POWER_CAP = 0x4
NVML_GPU_THROTTLE_REASON_HW_SLOWDOWN = 0x8
NVML_GPU_THROTTLE_REASON_SYNC_BOOST = 0x10
NVML_GPU_THROTTLE_REASON_SW_THERMAL_SLOWDOWN = 0x20
NVML_GPU_THROTTLE_REASON_HW_THERMAL_SLOWDOWN = 0x40
NVML_GPU_THROTTLE_REASON_HW_POWER_BRAKE = 0x80
NVML_GPU_THROTTLE_REASON_DISPLAY_CLOCK_SETTING = 0x100


@dataclass
class GPUDiagnostics:
    """Comprehensive GPU diagnostic information."""

    timestamp: str = ""

    # Basic info
    gpu_name: str = ""
    gpu_uuid: str = ""

    # Driver/Runtime versions
    driver_version: str = ""
    cuda_version: str = ""
    nvml_version: str = ""
    pytorch_cuda_version: str = ""
    cudnn_version: str = ""

    # Memory (in GB)
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_free_gb: float = 0.0
    memory_percent: float = 0.0

    # PyTorch memory (more detailed)
    torch_allocated_gb: float = 0.0
    torch_reserved_gb: float = 0.0
    torch_max_allocated_gb: float = 0.0

    # Temperature
    temperature_c: float | None = None
    temperature_slowdown_c: float | None = None
    temperature_shutdown_c: float | None = None

    # Power
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    power_max_limit_w: float | None = None
    power_percent: float | None = None

    # Clocks
    clock_graphics_mhz: int | None = None
    clock_sm_mhz: int | None = None
    clock_memory_mhz: int | None = None
    clock_max_graphics_mhz: int | None = None
    clock_max_memory_mhz: int | None = None

    # Utilization
    gpu_utilization: int | None = None
    memory_utilization: int | None = None

    # PCIe
    pcie_gen: int | None = None
    pcie_max_gen: int | None = None
    pcie_width: int | None = None
    pcie_max_width: int | None = None
    pcie_tx_throughput_kb: int | None = None
    pcie_rx_throughput_kb: int | None = None

    # Throttling
    throttle_reasons: list[str] = field(default_factory=list)
    is_throttled: bool = False

    # Errors
    ecc_errors_single: int | None = None
    ecc_errors_double: int | None = None
    has_ecc_errors: bool = False

    # Performance state
    performance_state: str = ""  # P0-P12

    # Fan
    fan_speed_percent: int | None = None

    # Context health
    cuda_context_valid: bool = True
    cuda_last_error: str = ""

    def to_log_lines(self, verbose: bool = False) -> list[str]:
        """Convert to log lines for crash log.

        Args:
            verbose: If True, include all details. If False, only critical info.
        """
        lines = []

        # Always log memory and power (most likely crash causes)
        mem_line = f"    GPU_MEM: {self.memory_used_gb:.2f}/{self.memory_total_gb:.1f}GB ({self.memory_percent:.1f}%)"
        if self.torch_allocated_gb > 0:
            mem_line += f" | torch_alloc={self.torch_allocated_gb:.2f}GB"
        lines.append(mem_line)

        # Power is critical for crash diagnosis
        if self.power_draw_w is not None:
            power_line = f"    GPU_PWR: {self.power_draw_w:.1f}W"
            if self.power_limit_w:
                power_line += f"/{self.power_limit_w:.0f}W"
            if self.power_percent:
                power_line += f" ({self.power_percent:.1f}%)"
            lines.append(power_line)

        # Temperature with thresholds
        if self.temperature_c is not None:
            temp_line = f"    GPU_TMP: {self.temperature_c:.0f}°C"
            if self.temperature_slowdown_c:
                temp_line += f" (slowdown@{self.temperature_slowdown_c:.0f}°C)"
            lines.append(temp_line)

        # Throttling is a major warning sign
        if self.is_throttled and self.throttle_reasons:
            lines.append(f"    GPU_THROTTLE: {', '.join(self.throttle_reasons)}")

        # Clock speeds can indicate throttling even before it's reported
        if self.clock_graphics_mhz is not None:
            clock_line = f"    GPU_CLK: core={self.clock_graphics_mhz}MHz"
            if self.clock_max_graphics_mhz:
                clock_line += f" (max={self.clock_max_graphics_mhz}MHz)"
            if self.clock_memory_mhz:
                clock_line += f", mem={self.clock_memory_mhz}MHz"
            lines.append(clock_line)

        # PCIe status
        if self.pcie_gen is not None:
            pcie_line = f"    GPU_PCIE: Gen{self.pcie_gen}"
            if self.pcie_max_gen:
                pcie_line += f"/{self.pcie_max_gen}"
            if self.pcie_width:
                pcie_line += f" x{self.pcie_width}"
            if self.pcie_tx_throughput_kb:
                pcie_line += f" TX:{self.pcie_tx_throughput_kb}KB/s"
            if self.pcie_rx_throughput_kb:
                pcie_line += f" RX:{self.pcie_rx_throughput_kb}KB/s"
            lines.append(pcie_line)

        # ECC errors
        if self.has_ecc_errors:
            lines.append(f"    GPU_ECC_ERR: single={self.ecc_errors_single}, double={self.ecc_errors_double}")

        # Performance state
        if self.performance_state:
            lines.append(f"    GPU_PSTATE: {self.performance_state} | UTIL: {self.gpu_utilization}%")

        # CUDA errors
        if self.cuda_last_error:
            lines.append(f"    CUDA_ERR: {self.cuda_last_error}")

        if verbose:
            # Add version info in verbose mode
            lines.append(f"    VERSIONS: driver={self.driver_version}, cuda={self.cuda_version}")
            if self.fan_speed_percent is not None:
                lines.append(f"    FAN: {self.fan_speed_percent}%")

        return lines


def _decode_throttle_reasons(reasons_bitmask: int) -> list[str]:
    """Decode NVML throttle reasons bitmask to human-readable strings."""
    reasons = []

    if reasons_bitmask == NVML_GPU_THROTTLE_REASON_NONE:
        return []

    if reasons_bitmask & NVML_GPU_THROTTLE_REASON_HW_SLOWDOWN:
        reasons.append("HW_SLOWDOWN")
    if reasons_bitmask & NVML_GPU_THROTTLE_REASON_HW_THERMAL_SLOWDOWN:
        reasons.append("THERMAL_HW")
    if reasons_bitmask & NVML_GPU_THROTTLE_REASON_SW_THERMAL_SLOWDOWN:
        reasons.append("THERMAL_SW")
    if reasons_bitmask & NVML_GPU_THROTTLE_REASON_HW_POWER_BRAKE:
        reasons.append("POWER_BRAKE")
    if reasons_bitmask & NVML_GPU_THROTTLE_REASON_SW_POWER_CAP:
        reasons.append("POWER_CAP")
    if reasons_bitmask & NVML_GPU_THROTTLE_REASON_APP_CLOCK_SETTING:
        reasons.append("APP_CLOCK")
    if reasons_bitmask & NVML_GPU_THROTTLE_REASON_SYNC_BOOST:
        reasons.append("SYNC_BOOST")
    if reasons_bitmask & NVML_GPU_THROTTLE_REASON_GPU_IDLE:
        reasons.append("IDLE")

    return reasons


def collect_gpu_diagnostics() -> GPUDiagnostics | None:
    """Collect comprehensive GPU diagnostics.

    Returns:
        GPUDiagnostics object with all available information, or None if no GPU.
    """
    diag = GPUDiagnostics(
        timestamp=datetime.datetime.now().isoformat(timespec="milliseconds")
    )

    # Get PyTorch CUDA info
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()
        diag.gpu_name = torch.cuda.get_device_name(device)
        diag.pytorch_cuda_version = torch.version.cuda or ""

        # Memory from PyTorch
        props = torch.cuda.get_device_properties(device)
        diag.memory_total_gb = props.total_memory / (1024**3)

        diag.torch_allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
        diag.torch_reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
        diag.torch_max_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

        # Check for CUDA errors
        try:
            # This will raise if there's a pending error
            torch.cuda.synchronize()
            diag.cuda_context_valid = True
            diag.cuda_last_error = ""
        except RuntimeError as e:
            diag.cuda_context_valid = False
            diag.cuda_last_error = str(e)

        # cuDNN version
        if torch.backends.cudnn.is_available():
            diag.cudnn_version = str(torch.backends.cudnn.version())

    except ImportError:
        return None
    except Exception as e:
        diag.cuda_last_error = f"PyTorch error: {e}"

    # Get detailed info from NVML
    try:
        import pynvml

        pynvml.nvmlInit()

        try:
            diag.driver_version = pynvml.nvmlSystemGetDriverVersion()
            diag.nvml_version = pynvml.nvmlSystemGetNVMLVersion()
            diag.cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            # Convert to readable format (e.g., 12010 -> 12.1)
            if isinstance(diag.cuda_version, int):
                major = diag.cuda_version // 1000
                minor = (diag.cuda_version % 1000) // 10
                diag.cuda_version = f"{major}.{minor}"
        except Exception:
            pass

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # UUID
        try:
            diag.gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
        except Exception:
            pass

        # Memory
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            diag.memory_used_gb = mem.used / (1024**3)
            diag.memory_free_gb = mem.free / (1024**3)
            diag.memory_total_gb = mem.total / (1024**3)
            diag.memory_percent = (mem.used / mem.total) * 100 if mem.total > 0 else 0
        except Exception:
            pass

        # Temperature
        try:
            diag.temperature_c = float(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            pass

        try:
            diag.temperature_slowdown_c = float(
                pynvml.nvmlDeviceGetTemperatureThreshold(
                    handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
                )
            )
        except Exception:
            pass

        try:
            diag.temperature_shutdown_c = float(
                pynvml.nvmlDeviceGetTemperatureThreshold(
                    handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
                )
            )
        except Exception:
            pass

        # Power
        try:
            diag.power_draw_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            pass

        try:
            diag.power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except Exception:
            pass

        try:
            diag.power_max_limit_w = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
        except Exception:
            pass

        if diag.power_draw_w and diag.power_limit_w:
            diag.power_percent = (diag.power_draw_w / diag.power_limit_w) * 100

        # Clocks
        try:
            diag.clock_graphics_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        except Exception:
            pass

        try:
            diag.clock_sm_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except Exception:
            pass

        try:
            diag.clock_memory_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except Exception:
            pass

        try:
            diag.clock_max_graphics_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        except Exception:
            pass

        try:
            diag.clock_max_memory_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except Exception:
            pass

        # Utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            diag.gpu_utilization = util.gpu
            diag.memory_utilization = util.memory
        except Exception:
            pass

        # PCIe
        try:
            diag.pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
        except Exception:
            pass

        try:
            diag.pcie_max_gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
        except Exception:
            pass

        try:
            diag.pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
        except Exception:
            pass

        try:
            diag.pcie_max_width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
        except Exception:
            pass

        try:
            diag.pcie_tx_throughput_kb = pynvml.nvmlDeviceGetPcieThroughput(
                handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
            )
        except Exception:
            pass

        try:
            diag.pcie_rx_throughput_kb = pynvml.nvmlDeviceGetPcieThroughput(
                handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
            )
        except Exception:
            pass

        # Throttling
        try:
            throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            diag.throttle_reasons = _decode_throttle_reasons(throttle_reasons)
            diag.is_throttled = len(diag.throttle_reasons) > 0 and "IDLE" not in diag.throttle_reasons
        except Exception:
            pass

        # ECC errors
        try:
            diag.ecc_errors_single = pynvml.nvmlDeviceGetTotalEccErrors(
                handle,
                pynvml.NVML_SINGLE_BIT_ECC,
                pynvml.NVML_VOLATILE_ECC
            )
            diag.ecc_errors_double = pynvml.nvmlDeviceGetTotalEccErrors(
                handle,
                pynvml.NVML_DOUBLE_BIT_ECC,
                pynvml.NVML_VOLATILE_ECC
            )
            diag.has_ecc_errors = (diag.ecc_errors_single or 0) > 0 or (diag.ecc_errors_double or 0) > 0
        except pynvml.NVMLError:
            # ECC not supported on all GPUs
            pass
        except Exception:
            pass

        # Performance state
        try:
            pstate = pynvml.nvmlDeviceGetPerformanceState(handle)
            diag.performance_state = f"P{pstate}"
        except Exception:
            pass

        # Fan
        try:
            diag.fan_speed_percent = pynvml.nvmlDeviceGetFanSpeed(handle)
        except Exception:
            pass

        pynvml.nvmlShutdown()

    except ImportError:
        pass
    except Exception as e:
        diag.cuda_last_error = f"NVML error: {e}"

    return diag


class DiagnosticCrashLogger:
    """Enhanced crash logger with deep GPU diagnostics."""

    def __init__(
        self,
        log_dir: Path | str = "./logs",
        log_name: str = "gpu_diag",
        max_size_mb: float = 100.0,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._lock = threading.Lock()
        self._file: TextIO | None = None
        self._log_path: Path | None = None
        self._max_size = int(max_size_mb * 1024 * 1024)
        self._operation_id = 0

        if enabled:
            self._setup_log_file(Path(log_dir), log_name)

    def _setup_log_file(self, log_dir: Path, log_name: str) -> None:
        """Create log directory and file."""
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_path = log_dir / f"{log_name}_{date_str}.log"
            self._file = open(self._log_path, "a", encoding="utf-8", buffering=1)
            self._write_header()
        except Exception as e:
            print(f"WARNING: Could not create diagnostic log: {e}", file=sys.stderr)
            self._enabled = False

    def _write_header(self) -> None:
        """Write comprehensive header with system and GPU info."""
        if not self._file:
            return

        lines = [
            "",
            "=" * 100,
            f"GPU DIAGNOSTIC LOG - {datetime.datetime.now().isoformat()}",
            "=" * 100,
            "",
            "SYSTEM INFO:",
            f"  Python: {sys.version}",
            f"  Platform: {sys.platform}",
            "",
        ]

        # Collect initial diagnostics
        diag = collect_gpu_diagnostics()
        if diag:
            lines.extend([
                "GPU INFO:",
                f"  Name: {diag.gpu_name}",
                f"  UUID: {diag.gpu_uuid}",
                f"  Driver: {diag.driver_version}",
                f"  CUDA: {diag.cuda_version}",
                f"  PyTorch CUDA: {diag.pytorch_cuda_version}",
                f"  cuDNN: {diag.cudnn_version}",
                f"  Memory: {diag.memory_total_gb:.1f} GB",
                f"  Power Limit: {diag.power_limit_w:.0f}W (max: {diag.power_max_limit_w:.0f}W)" if diag.power_limit_w else "",
                f"  Temp Thresholds: slowdown={diag.temperature_slowdown_c}°C, shutdown={diag.temperature_shutdown_c}°C" if diag.temperature_slowdown_c else "",
                f"  PCIe: Gen{diag.pcie_max_gen} x{diag.pcie_max_width}" if diag.pcie_max_gen else "",
                "",
            ])
        else:
            lines.append("GPU INFO: No CUDA GPU available\n")

        lines.extend([
            "=" * 100,
            "LOG FORMAT: [timestamp] [op_id] [event] message",
            "=" * 100,
            "",
        ])

        for line in lines:
            if line:  # Skip empty strings from failed conditionals
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

    def _write(self, event: str, message: str, include_diag: bool = False, verbose: bool = False) -> None:
        """Write a log entry."""
        if not self._enabled or not self._file:
            return

        with self._lock:
            try:
                timestamp = datetime.datetime.now().isoformat(timespec="milliseconds")
                line = f"[{timestamp}] [{self._operation_id:06d}] [{event:20}] {message}"
                self._file.write(line + "\n")

                if include_diag:
                    diag = collect_gpu_diagnostics()
                    if diag:
                        for diag_line in diag.to_log_lines(verbose=verbose):
                            self._file.write(diag_line + "\n")

                self._flush()
            except Exception as e:
                print(f"Diagnostic log write error: {e}", file=sys.stderr)

    def start_operation(self, name: str, **context: Any) -> int:
        """Log start of an operation and return operation ID."""
        self._operation_id += 1
        ctx_str = " | ".join(f"{k}={v}" for k, v in context.items()) if context else ""
        self._write("OP_START", f"{name} | {ctx_str}" if ctx_str else name, include_diag=True, verbose=True)
        return self._operation_id

    def checkpoint(self, label: str, **context: Any) -> None:
        """Log a checkpoint within an operation."""
        ctx_str = " | ".join(f"{k}={v}" for k, v in context.items()) if context else ""
        self._write("CHECKPOINT", f"{label} | {ctx_str}" if ctx_str else label, include_diag=True)

    def end_operation(self, name: str, success: bool = True, error: str = "") -> None:
        """Log end of an operation."""
        if success:
            self._write("OP_END", f"{name} | SUCCESS", include_diag=True)
        else:
            self._write("OP_FAILED", f"{name} | ERROR: {error}", include_diag=True, verbose=True)

    def warn(self, message: str) -> None:
        """Log a warning with diagnostics."""
        self._write("WARNING", message, include_diag=True)

    def critical(self, message: str) -> None:
        """Log a critical issue with full diagnostics."""
        self._write("CRITICAL", message, include_diag=True, verbose=True)

    def log_pre_cuda_op(self, op_name: str, **context: Any) -> None:
        """Log immediately before a CUDA operation - this is the last line before potential crash."""
        ctx_str = " | ".join(f"{k}={v}" for k, v in context.items()) if context else ""
        self._write("PRE_CUDA_OP", f">>> {op_name} | {ctx_str}" if ctx_str else f">>> {op_name}", include_diag=True)

        # Extra sync to ensure this line survives
        if self._file:
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except Exception:
                pass

    def log_post_cuda_op(self, op_name: str, duration_ms: float = 0) -> None:
        """Log immediately after a CUDA operation completes."""
        self._write("POST_CUDA_OP", f"<<< {op_name} | duration={duration_ms:.1f}ms", include_diag=True)

    def close(self) -> None:
        """Close the log file."""
        if self._file:
            try:
                self._write("SESSION_END", "Logger closing", include_diag=True, verbose=True)
                self._file.close()
            except Exception:
                pass
            self._file = None


# Global instance
_diag_logger: DiagnosticCrashLogger | None = None
_diag_lock = threading.Lock()


def get_diagnostic_logger(
    log_dir: Path | str = "./logs",
    enabled: bool = True,
) -> DiagnosticCrashLogger:
    """Get or create the global diagnostic logger."""
    global _diag_logger
    with _diag_lock:
        if _diag_logger is None:
            _diag_logger = DiagnosticCrashLogger(log_dir=log_dir, enabled=enabled)
    return _diag_logger


def log_gpu_diagnostics(label: str = "DIAGNOSTIC_CHECK") -> GPUDiagnostics | None:
    """Log current GPU diagnostics and return the data."""
    logger = get_diagnostic_logger()
    logger.checkpoint(label)
    return collect_gpu_diagnostics()


# Convenience function for quick diagnostic dump
def dump_diagnostics() -> str:
    """Get a string dump of current GPU diagnostics."""
    diag = collect_gpu_diagnostics()
    if not diag:
        return "No GPU available"

    lines = [
        f"GPU: {diag.gpu_name}",
        f"Memory: {diag.memory_used_gb:.2f}/{diag.memory_total_gb:.1f}GB ({diag.memory_percent:.1f}%)",
    ]

    if diag.power_draw_w:
        lines.append(f"Power: {diag.power_draw_w:.1f}W/{diag.power_limit_w:.0f}W ({diag.power_percent:.1f}%)")

    if diag.temperature_c:
        lines.append(f"Temp: {diag.temperature_c:.0f}°C")

    if diag.is_throttled:
        lines.append(f"THROTTLING: {', '.join(diag.throttle_reasons)}")

    if diag.cuda_last_error:
        lines.append(f"CUDA Error: {diag.cuda_last_error}")

    return " | ".join(lines)
