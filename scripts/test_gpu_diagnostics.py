#!/usr/bin/env python3
"""Test script for GPU and system diagnostics.

Run this script to verify that diagnostics are working correctly
and to get a baseline of your system state before running embeddings.

Usage:
    python scripts/test_gpu_diagnostics.py

This will:
1. Display comprehensive GPU diagnostics (memory, power, temp, clocks)
2. Display CPU diagnostics (temp, frequency, utilization)
3. Display memory and system info
4. Test CUDA context health
5. Perform a small test computation
6. Show recommendations based on findings
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> None:
    """Run GPU diagnostics test."""
    print_header("GPU DIAGNOSTIC TEST")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize diagnostic logger
    try:
        from rag_service.core.gpu_diagnostics import get_diagnostic_logger

        diag_log = get_diagnostic_logger()
        diag_log.start_operation("diagnostic_test")
        print("Diagnostic logger initialized - logs will be written to logs/gpu_diag_*.log")
    except Exception as e:
        print(f"Warning: Could not initialize diagnostic logger: {e}")
        diag_log = None

    # System diagnostics first (CPU, RAM, etc.)
    print_header("1. SYSTEM DIAGNOSTICS")
    sys_diag = None
    try:
        from rag_service.core.system_diagnostics import (
            collect_system_diagnostics,
        )

        sys_diag = collect_system_diagnostics()

        print(f"Hostname: {sys_diag.hostname}")
        print(f"OS: {sys_diag.os_info}")
        if sys_diag.uptime_hours:
            print(f"Uptime: {sys_diag.uptime_hours:.1f} hours")
        print()

        # CPU
        cpu = sys_diag.cpu
        print("CPU:")
        print(f"  Model: {cpu.cpu_name}")
        print(f"  Cores: {cpu.physical_cores} physical, {cpu.logical_cores} logical")
        print(f"  Utilization: {cpu.total_utilization:.1f}%")
        if cpu.current_freq_mhz > 0:
            print(f"  Frequency: {cpu.current_freq_mhz:.0f} MHz (max: {cpu.max_freq_mhz:.0f} MHz)")
        if cpu.temperature_c is not None:
            print(f"  Temperature: {cpu.temperature_c:.1f}°C")
        else:
            print("  Temperature: Not available (requires admin or OpenHardwareMonitor)")
        print()

        # Memory
        mem = sys_diag.memory
        print("MEMORY:")
        print(f"  Total RAM: {mem.total_gb:.1f} GB")
        print(f"  Used: {mem.used_gb:.1f} GB ({mem.percent_used:.1f}%)")
        print(f"  Available: {mem.available_gb:.1f} GB")
        if mem.swap_total_gb > 0:
            print(
                f"  Swap: {mem.swap_used_gb:.1f}/{mem.swap_total_gb:.1f} GB ({mem.swap_percent:.1f}%)"
            )
        print()

        # Hardware sensors
        hw = sys_diag.hardware
        if hw.vrm_temp_c or hw.voltage_12v or hw.system_power_w:
            print("HARDWARE SENSORS:")
            if hw.vrm_temp_c:
                print(f"  VRM Temperature: {hw.vrm_temp_c:.0f}°C")
            if hw.chipset_temp_c:
                print(f"  Chipset Temperature: {hw.chipset_temp_c:.0f}°C")
            if hw.voltage_12v:
                print(f"  12V Rail: {hw.voltage_12v:.2f}V (nominal: 12.00V)")
            if hw.voltage_5v:
                print(f"  5V Rail: {hw.voltage_5v:.2f}V (nominal: 5.00V)")
            if hw.voltage_vcore:
                print(f"  CPU Vcore: {hw.voltage_vcore:.3f}V")
            if hw.system_power_w:
                print(f"  System Power: {hw.system_power_w:.0f}W")
            if hw.cpu_fan_rpm:
                print(f"  CPU Fan: {hw.cpu_fan_rpm} RPM")
        else:
            print("HARDWARE SENSORS:")
            print("  Not available (install OpenHardwareMonitor or LibreHardwareMonitor)")
        print()

        # Windows Events
        events = sys_diag.windows_events
        if events.last_bsod_code:
            print("!!! RECENT BSOD DETECTED !!!")
            print(f"  Code: {events.last_bsod_code}")
            print(f"  Time: {events.last_bsod_time}")
        if events.critical_events:
            print(f"RECENT CRITICAL EVENTS: {len(events.critical_events)}")
            for evt in events.critical_events[:3]:
                print(f"  - {evt.get('source')}: {evt.get('message', '')[:60]}...")

    except Exception as e:
        print(f"Error collecting system diagnostics: {e}")
        import traceback

        traceback.print_exc()

    # Check for CUDA availability
    print_header("2. CUDA AVAILABILITY")
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")

            # cuDNN
            if torch.backends.cudnn.is_available():
                print(f"cuDNN version: {torch.backends.cudnn.version()}")
                print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        else:
            print("WARNING: CUDA is not available!")
            print("The embedding model will run on CPU (much slower)")
            return
    except ImportError:
        print("ERROR: PyTorch is not installed!")
        return

    # Collect GPU diagnostics
    print_header("3. GPU DIAGNOSTICS")
    try:
        from rag_service.core.gpu_diagnostics import collect_gpu_diagnostics

        diag = collect_gpu_diagnostics()
        if diag:
            print(f"GPU Name: {diag.gpu_name}")
            print(f"GPU UUID: {diag.gpu_uuid}")
            print()
            print("VERSIONS:")
            print(f"  Driver: {diag.driver_version}")
            print(f"  CUDA: {diag.cuda_version}")
            print(f"  PyTorch CUDA: {diag.pytorch_cuda_version}")
            print(f"  cuDNN: {diag.cudnn_version}")
            print()
            print("MEMORY:")
            print(f"  Total: {diag.memory_total_gb:.2f} GB")
            print(f"  Used: {diag.memory_used_gb:.2f} GB ({diag.memory_percent:.1f}%)")
            print(f"  Free: {diag.memory_free_gb:.2f} GB")
            print(f"  Torch Allocated: {diag.torch_allocated_gb:.2f} GB")
            print(f"  Torch Reserved: {diag.torch_reserved_gb:.2f} GB")
            print()
            print("TEMPERATURE:")
            print(f"  Current: {diag.temperature_c}°C")
            if diag.temperature_slowdown_c:
                print(f"  Slowdown threshold: {diag.temperature_slowdown_c}°C")
            if diag.temperature_shutdown_c:
                print(f"  Shutdown threshold: {diag.temperature_shutdown_c}°C")
            print()
            print("POWER:")
            if diag.power_draw_w is not None:
                print(f"  Current draw: {diag.power_draw_w:.1f} W")
            if diag.power_limit_w:
                print(f"  Power limit: {diag.power_limit_w:.1f} W")
            if diag.power_max_limit_w:
                print(f"  Max power limit: {diag.power_max_limit_w:.1f} W")
            if diag.power_percent:
                print(f"  Power usage: {diag.power_percent:.1f}%")
            print()
            print("CLOCKS:")
            if diag.clock_graphics_mhz:
                print(
                    f"  Graphics: {diag.clock_graphics_mhz} MHz (max: {diag.clock_max_graphics_mhz} MHz)"
                )
            if diag.clock_memory_mhz:
                print(
                    f"  Memory: {diag.clock_memory_mhz} MHz (max: {diag.clock_max_memory_mhz} MHz)"
                )
            print()
            print("PCIe:")
            if diag.pcie_gen:
                print(f"  Current: Gen{diag.pcie_gen} x{diag.pcie_width}")
            if diag.pcie_max_gen:
                print(f"  Maximum: Gen{diag.pcie_max_gen} x{diag.pcie_max_width}")
            print()
            print("THROTTLING:")
            if diag.is_throttled:
                print(f"  STATUS: THROTTLED! Reasons: {', '.join(diag.throttle_reasons)}")
            else:
                print("  STATUS: Not throttled")
            print()
            print("PERFORMANCE:")
            print(f"  P-State: {diag.performance_state}")
            print(f"  GPU Utilization: {diag.gpu_utilization}%")
            print(f"  Memory Utilization: {diag.memory_utilization}%")
            if diag.fan_speed_percent is not None:
                print(f"  Fan Speed: {diag.fan_speed_percent}%")
            print()
            print("CUDA CONTEXT:")
            print(f"  Valid: {diag.cuda_context_valid}")
            if diag.cuda_last_error:
                print(f"  Last Error: {diag.cuda_last_error}")

            # ECC errors
            if diag.has_ecc_errors:
                print()
                print("!!! ECC ERRORS DETECTED !!!")
                print(f"  Single-bit errors: {diag.ecc_errors_single}")
                print(f"  Double-bit errors: {diag.ecc_errors_double}")

        else:
            print("ERROR: Could not collect GPU diagnostics")

    except Exception as e:
        print(f"ERROR collecting diagnostics: {e}")
        import traceback

        traceback.print_exc()

    # Test CUDA operations
    print_header("4. CUDA OPERATION TEST")
    try:
        import torch

        print("Testing basic CUDA operations...")

        # Allocate small tensor
        print("  Allocating small tensor (100MB)...")
        x = torch.randn(25000000, device="cuda")  # ~100MB
        torch.cuda.synchronize()
        print(f"  Success! Tensor shape: {x.shape}")

        # Matrix multiplication
        print("  Testing matrix multiplication...")
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        start = time.perf_counter()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        duration = (time.perf_counter() - start) * 1000
        print(f"  Success! 1000x1000 matmul in {duration:.2f}ms")

        # Clear memory
        del x, a, b, c
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  Memory cleared successfully")

    except Exception as e:
        print(f"ERROR during CUDA test: {e}")
        import traceback

        traceback.print_exc()

    # Check for potential issues
    print_header("5. RECOMMENDATIONS")
    issues = []
    warnings = []

    # System-level checks
    if sys_diag:
        # CPU temperature
        if sys_diag.cpu.temperature_c and sys_diag.cpu.temperature_c > 80:
            issues.append(f"CPU temperature is high ({sys_diag.cpu.temperature_c:.0f}°C)")

        # Memory pressure
        if sys_diag.memory.percent_used > 85:
            warnings.append(f"RAM usage is high ({sys_diag.memory.percent_used:.0f}%)")

        # Swap usage
        if sys_diag.memory.swap_percent > 50:
            warnings.append(
                f"Swap usage is high ({sys_diag.memory.swap_percent:.0f}%) - may indicate memory pressure"
            )

        # VRM temperature (critical for system stability)
        if sys_diag.hardware.vrm_temp_c and sys_diag.hardware.vrm_temp_c > 90:
            issues.append(
                f"VRM temperature is dangerously high ({sys_diag.hardware.vrm_temp_c:.0f}°C)"
            )
        elif sys_diag.hardware.vrm_temp_c and sys_diag.hardware.vrm_temp_c > 80:
            warnings.append(f"VRM temperature is elevated ({sys_diag.hardware.vrm_temp_c:.0f}°C)")

        # Voltage rails (should be within 5% of nominal)
        if sys_diag.hardware.voltage_12v:
            if sys_diag.hardware.voltage_12v < 11.4 or sys_diag.hardware.voltage_12v > 12.6:
                issues.append(
                    f"12V rail out of spec ({sys_diag.hardware.voltage_12v:.2f}V) - PSU issue!"
                )

        # Recent BSOD
        if sys_diag.windows_events.last_bsod_code:
            issues.append(f"Recent BSOD detected: {sys_diag.windows_events.last_bsod_code}")

    if diag:
        # Check temperature
        if diag.temperature_c and diag.temperature_c > 75:
            issues.append(
                f"GPU temperature is high ({diag.temperature_c}°C). Consider improving cooling."
            )

        # Check power
        if diag.power_percent and diag.power_percent > 95:
            issues.append(
                f"GPU power draw is near limit ({diag.power_percent:.0f}%). May cause throttling."
            )

        # Check throttling
        if diag.is_throttled and "IDLE" not in diag.throttle_reasons:
            issues.append(f"GPU is currently throttled: {', '.join(diag.throttle_reasons)}")

        # Check PCIe
        if diag.pcie_gen and diag.pcie_max_gen and diag.pcie_gen < diag.pcie_max_gen:
            warnings.append(f"PCIe running at Gen{diag.pcie_gen} instead of Gen{diag.pcie_max_gen}")

        if diag.pcie_width and diag.pcie_max_width and diag.pcie_width < diag.pcie_max_width:
            warnings.append(f"PCIe running at x{diag.pcie_width} instead of x{diag.pcie_max_width}")

        # Check memory
        if diag.memory_percent > 80:
            warnings.append(f"GPU memory usage is high ({diag.memory_percent:.0f}%)")

        # Check ECC
        if diag.has_ecc_errors:
            issues.append("ECC errors detected - possible hardware issue!")

        # Check CUDA context
        if not diag.cuda_context_valid:
            issues.append(f"CUDA context is invalid: {diag.cuda_last_error}")

    if issues:
        print("ISSUES (should address):")
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("No critical issues found.")

    if warnings:
        print()
        print("WARNINGS (may affect performance):")
        for warning in warnings:
            print(f"  ⚠️  {warning}")

    # Crash prevention recommendations
    print()
    print("CRASH PREVENTION RECOMMENDATIONS:")
    print("  1. If crashes occur during embedding, try reducing batch_size in config.yaml")
    print("     Current default: 256, try: 64 or 32")
    print("  2. Increase gpu_inter_batch_delay to 0.5 or 1.0 seconds")
    print("  3. Lower gpu_max_memory_percent to 60 or 70")
    print("  4. Monitor logs/gpu_diag_*.log for detailed crash diagnostics")
    print("  5. Check Windows Event Viewer for driver crashes after system restart")

    print_header("TEST COMPLETE")

    # Close diagnostic logger
    if diag_log:
        diag_log.end_operation("diagnostic_test", success=True)
        diag_log.close()
        print("Diagnostic logs written to: logs/gpu_diag_*.log")

    print("Crash logs will be written to: logs/crash_*.log")


if __name__ == "__main__":
    main()
