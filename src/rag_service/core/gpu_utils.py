"""GPU monitoring and safeguard utilities.

Provides utilities for monitoring GPU memory, temperature, and implementing
throttling to prevent system crashes during intensive embedding operations.
"""

import gc
import logging
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from rag_service.core.crash_logger import get_crash_logger, trace_operation
from rag_service.core.gpu_diagnostics import (
    collect_gpu_diagnostics,
    get_diagnostic_logger,
    dump_diagnostics,
)
from rag_service.core.system_diagnostics import (
    collect_cpu_diagnostics,
    collect_memory_diagnostics,
    get_system_logger,
)

logger = logging.getLogger(__name__)
crash_log = get_crash_logger()
diag_log = get_diagnostic_logger()
sys_log = get_system_logger()

T = TypeVar("T")


@dataclass
class GPUStatus:
    """Current GPU status information."""

    available: bool
    device_name: str = ""
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_free_gb: float = 0.0
    memory_percent: float = 0.0
    temperature_c: float | None = None


def get_gpu_status() -> GPUStatus:
    """Get current GPU status including memory and temperature.

    Returns:
        GPUStatus with current GPU information.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return GPUStatus(available=False)

        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)

        # Memory info
        memory_total = torch.cuda.get_device_properties(device).total_memory
        memory_reserved = torch.cuda.memory_reserved(device)
        memory_allocated = torch.cuda.memory_allocated(device)

        # Use reserved memory as "used" since that's what's actually allocated on GPU
        memory_used = memory_reserved
        memory_free = memory_total - memory_used
        memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0

        # Try to get temperature (requires pynvml)
        temperature = _get_gpu_temperature()

        return GPUStatus(
            available=True,
            device_name=device_name,
            memory_total_gb=memory_total / (1024**3),
            memory_used_gb=memory_used / (1024**3),
            memory_free_gb=memory_free / (1024**3),
            memory_percent=memory_percent,
            temperature_c=temperature,
        )
    except Exception as e:
        logger.debug(f"Could not get GPU status: {e}")
        return GPUStatus(available=False)


def _get_gpu_temperature() -> float | None:
    """Get GPU temperature using pynvml if available.

    Returns:
        Temperature in Celsius, or None if unavailable.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return float(temp)
    except Exception:
        return None


def clear_gpu_memory() -> None:
    """Clear GPU memory cache and run garbage collection.

    This helps free up memory between batches and can prevent OOM errors.
    """
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared GPU memory cache")
    except ImportError:
        pass


def wait_for_gpu_cooldown(
    max_memory_percent: float = 85.0,
    max_temperature_c: float = 80.0,
    check_interval: float = 1.0,
    max_wait: float = 30.0,
) -> bool:
    """Wait for GPU to cool down if memory or temperature is too high.

    Args:
        max_memory_percent: Maximum memory usage before waiting.
        max_temperature_c: Maximum temperature before waiting.
        check_interval: Seconds between status checks.
        max_wait: Maximum seconds to wait before giving up.

    Returns:
        True if GPU is within limits, False if timeout.
    """
    start_time = time.time()

    while time.time() - start_time < max_wait:
        status = get_gpu_status()

        if not status.available:
            return True  # No GPU, nothing to wait for

        memory_ok = status.memory_percent <= max_memory_percent
        temp_ok = status.temperature_c is None or status.temperature_c <= max_temperature_c

        if memory_ok and temp_ok:
            return True

        if not memory_ok:
            logger.info(
                f"GPU memory at {status.memory_percent:.1f}%, waiting... "
                f"(target: <{max_memory_percent}%)"
            )
            clear_gpu_memory()

        if not temp_ok:
            logger.info(
                f"GPU temperature at {status.temperature_c}°C, waiting... "
                f"(target: <{max_temperature_c}°C)"
            )

        time.sleep(check_interval)

    logger.warning(f"GPU cooldown timeout after {max_wait}s")
    return False


def throttled_batch_processor(
    items: list[T],
    processor: Callable[[list[T]], list],
    batch_size: int = 32,
    max_memory_percent: float = 80.0,
    max_temperature_c: float = 75.0,
    inter_batch_delay: float = 0.1,
    adaptive_batch_size: bool = True,
    min_batch_size: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list:
    """Process items in batches with GPU throttling and memory management.

    This is the main safeguard function that:
    1. Monitors GPU memory and temperature
    2. Waits for cooldown if thresholds exceeded
    3. Clears memory between batches
    4. Adaptively reduces batch size under pressure
    5. Adds inter-batch delays to prevent thermal runaway

    Args:
        items: List of items to process.
        processor: Function that processes a batch and returns results.
        batch_size: Initial batch size.
        max_memory_percent: Maximum GPU memory before throttling.
        max_temperature_c: Maximum GPU temperature before throttling.
        inter_batch_delay: Seconds to wait between batches.
        adaptive_batch_size: Whether to reduce batch size under pressure.
        min_batch_size: Minimum batch size when adapting.
        progress_callback: Optional callback(processed, total) for progress.

    Returns:
        Combined results from all batches.
    """
    if not items:
        return []

    results = []
    current_batch_size = batch_size
    total_items = len(items)
    processed = 0
    batch_num = 0
    total_batches = (total_items + batch_size - 1) // batch_size

    # Start diagnostic logging session
    diag_log.start_operation(
        "batch_processor",
        total_items=total_items,
        batch_size=batch_size,
        max_memory_percent=max_memory_percent,
        max_temperature_c=max_temperature_c,
    )

    # Log initial system state (CPU, RAM - helps diagnose if CPU/RAM is also under stress)
    sys_log.log_state("BATCH_PROCESSOR_START", include_all=True)

    # Crash-safe logging for entire batch processing operation
    crash_log.info(
        "BATCH_PROCESSOR_START",
        capture_state=True,
        total_items=total_items,
        batch_size=batch_size,
        max_memory_percent=max_memory_percent,
        max_temperature_c=max_temperature_c,
    )

    logger.info(
        f"Processing {total_items} items with GPU safeguards "
        f"(batch={batch_size}, max_mem={max_memory_percent}%, max_temp={max_temperature_c}°C)"
    )

    while processed < total_items:
        batch_num += 1

        # Get current GPU status
        status = get_gpu_status()

        # Check for throttling using enhanced diagnostics
        diag = collect_gpu_diagnostics()
        if diag and diag.is_throttled:
            diag_log.warn(f"GPU THROTTLING DETECTED: {', '.join(diag.throttle_reasons)}")
            crash_log.warn(
                f"BATCH_{batch_num}_THROTTLE_WARNING",
                capture_state=True,
                throttle_reasons=", ".join(diag.throttle_reasons),
            )

        # Check power draw if approaching limit
        if diag and diag.power_percent and diag.power_percent > 90:
            diag_log.warn(f"HIGH POWER DRAW: {diag.power_draw_w:.1f}W ({diag.power_percent:.1f}% of limit)")

        # Log state before each batch (crash-safe)
        crash_log.info(
            f"BATCH_{batch_num}_PRE_CHECK",
            capture_state=True,
            batch_num=batch_num,
            total_batches=total_batches,
            processed=processed,
            current_batch_size=current_batch_size,
        )

        # Adaptive batch size reduction under memory pressure
        if adaptive_batch_size and status.available:
            if status.memory_percent > max_memory_percent - 10:
                # Reduce batch size when approaching limit
                new_size = max(min_batch_size, current_batch_size // 2)
                if new_size != current_batch_size:
                    crash_log.warn(
                        f"BATCH_SIZE_REDUCE: {current_batch_size} -> {new_size}",
                        capture_state=True,
                        reason="memory_pressure",
                        memory_percent=status.memory_percent,
                    )
                    logger.info(
                        f"Reducing batch size: {current_batch_size} -> {new_size} "
                        f"(memory: {status.memory_percent:.1f}%)"
                    )
                    current_batch_size = new_size
            elif status.memory_percent < max_memory_percent - 30 and current_batch_size < batch_size:
                # Gradually restore batch size when memory is comfortable
                new_size = min(batch_size, current_batch_size * 2)
                if new_size != current_batch_size:
                    crash_log.debug(f"BATCH_SIZE_RESTORE: {current_batch_size} -> {new_size}")
                    logger.debug(f"Restoring batch size: {current_batch_size} -> {new_size}")
                    current_batch_size = new_size

        # Wait for GPU to be within limits
        wait_for_gpu_cooldown(
            max_memory_percent=max_memory_percent,
            max_temperature_c=max_temperature_c,
        )

        # Process batch
        batch_end = min(processed + current_batch_size, total_items)
        batch = items[processed:batch_end]
        batch_items = len(batch)

        # Log immediately before GPU operation (crash-safe)
        crash_log.info(
            f"BATCH_{batch_num}_EXECUTE",
            capture_state=True,
            batch_items=batch_items,
            items_range=f"{processed}-{batch_end}",
        )

        # Enhanced diagnostic logging - this is the last line before potential crash
        diag_log.log_pre_cuda_op(
            f"batch_{batch_num}_encode",
            batch_items=batch_items,
            items_range=f"{processed}-{batch_end}",
            current_batch_size=current_batch_size,
        )

        try:
            batch_start_time = time.perf_counter()
            batch_results = processor(batch)
            batch_duration_ms = (time.perf_counter() - batch_start_time) * 1000

            # Log successful completion with timing
            diag_log.log_post_cuda_op(f"batch_{batch_num}_encode", duration_ms=batch_duration_ms)

            results.extend(batch_results)

            # Log successful batch completion
            crash_log.info(
                f"BATCH_{batch_num}_COMPLETE",
                capture_state=True,
                results_count=len(batch_results),
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # GPU OOM - log and handle
                diag_log.critical(f"OOM during batch {batch_num}: {e}")
                crash_log.error(
                    f"BATCH_{batch_num}_OOM",
                    capture_state=True,
                    error=str(e),
                    batch_size=current_batch_size,
                )
                logger.warning(f"GPU OOM error, reducing batch size and retrying")
                clear_gpu_memory()
                current_batch_size = max(min_batch_size, current_batch_size // 2)

                if current_batch_size < min_batch_size:
                    diag_log.critical(f"Fatal OOM - cannot process with min batch size {min_batch_size}")
                    crash_log.critical(
                        "BATCH_PROCESSOR_FATAL_OOM",
                        capture_state=True,
                        min_batch_size=min_batch_size,
                    )
                    raise RuntimeError(
                        f"Cannot process even with minimum batch size ({min_batch_size}). "
                        "Consider using CPU or reducing document size."
                    ) from e
                continue  # Retry with smaller batch
            # Log unexpected errors with full diagnostics
            diag_log.critical(f"Unexpected error in batch {batch_num}: {e}")
            raise

        processed = batch_end

        # Progress callback
        if progress_callback:
            progress_callback(processed, total_items)

        # Log progress periodically
        if processed % (batch_size * 10) == 0 or processed == total_items:
            logger.info(f"Progress: {processed}/{total_items} items ({100*processed/total_items:.1f}%)")

        # Log full system state every 5 batches (helps see trends)
        if batch_num % 5 == 0:
            sys_log.log_state(f"BATCH_{batch_num}_SYSTEM_CHECK")

        # Clear memory and add delay between batches
        if processed < total_items:
            clear_gpu_memory()
            if inter_batch_delay > 0:
                time.sleep(inter_batch_delay)

    crash_log.info(
        "BATCH_PROCESSOR_COMPLETE",
        capture_state=True,
        total_processed=processed,
        total_batches=batch_num,
    )
    diag_log.end_operation("batch_processor", success=True)
    sys_log.log_state("BATCH_PROCESSOR_COMPLETE", include_all=True)
    logger.info(f"Completed processing {total_items} items")
    return results


def estimate_memory_for_batch(
    num_items: int,
    avg_text_length: int = 500,
    embedding_dim: int = 384,
    bytes_per_float: int = 4,
) -> float:
    """Estimate GPU memory needed for a batch in GB.

    This is a rough estimate based on typical transformer memory patterns.

    Args:
        num_items: Number of items in batch.
        avg_text_length: Average text length in characters.
        embedding_dim: Embedding dimension.
        bytes_per_float: Bytes per float (4 for fp32, 2 for fp16).

    Returns:
        Estimated memory in GB.
    """
    # Rough estimation factors (empirically determined)
    # - Tokenization typically produces ~0.3 tokens per character
    # - Transformer memory scales with sequence_length^2 for attention
    # - Plus embeddings and intermediate activations

    avg_tokens = avg_text_length * 0.3
    sequence_memory = num_items * avg_tokens * embedding_dim * bytes_per_float
    attention_memory = num_items * (avg_tokens ** 2) * bytes_per_float

    # Total with overhead
    total = (sequence_memory + attention_memory) * 2  # 2x for gradients/activations
    return total / (1024**3)


def get_safe_batch_size(
    total_items: int,
    target_memory_gb: float = 2.0,
    max_batch_size: int = 64,
    min_batch_size: int = 4,
) -> int:
    """Calculate a safe batch size based on available GPU memory.

    Args:
        total_items: Total number of items to process.
        target_memory_gb: Target memory usage in GB.
        max_batch_size: Maximum batch size.
        min_batch_size: Minimum batch size.

    Returns:
        Recommended batch size.
    """
    status = get_gpu_status()

    if not status.available:
        return max_batch_size  # CPU, no memory concerns

    # Use at most 50% of available memory for safety
    available_memory = status.memory_free_gb * 0.5
    target = min(target_memory_gb, available_memory)

    # Estimate batch size based on memory
    # Rough heuristic: ~50MB per item in batch for small transformer models
    items_per_gb = 20
    safe_size = int(target * items_per_gb)

    # Clamp to bounds
    safe_size = max(min_batch_size, min(max_batch_size, safe_size, total_items))

    logger.debug(
        f"Calculated safe batch size: {safe_size} "
        f"(free: {status.memory_free_gb:.1f}GB, target: {target:.1f}GB)"
    )

    return safe_size
