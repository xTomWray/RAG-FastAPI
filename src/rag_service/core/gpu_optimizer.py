"""GPU optimization and auto-configuration for any hardware.

This module implements production-style hardware profiling and dynamic
configuration, similar to approaches used by:
- HuggingFace Accelerate (device_map="auto")
- vLLM (automatic batch sizing based on VRAM)
- TensorRT (optimization profiles)

Key features:
1. Hardware detection - GPU capabilities, VRAM, power limits
2. Model profiling - Memory footprint at different precisions
3. Dynamic calculation - Optimal settings based on resource budgets
4. Calibration - Optional benchmark to fine-tune settings
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Type aliases
PrecisionType = Literal["fp32", "fp16", "auto"]


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""

    # GPU info
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_compute_capability: tuple[int, int] = (0, 0)

    # Memory (in GB)
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0

    # Power (in watts)
    power_limit_watts: float = 0.0
    power_default_watts: float = 0.0
    power_max_watts: float = 0.0

    # Capabilities
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_tf32: bool = False

    # CPU info
    cpu_cores: int = 1
    cpu_name: str = ""


@dataclass
class ModelProfile:
    """Memory and compute profile for an embedding model."""

    name: str
    embedding_dim: int

    # Memory footprint in GB at different precisions
    model_size_fp32_gb: float
    model_size_fp16_gb: float

    # Memory per item in batch (approximate, in MB)
    memory_per_item_fp32_mb: float
    memory_per_item_fp16_mb: float

    # Compute intensity (relative scale, higher = more compute per token)
    compute_intensity: float = 1.0

    # Power draw characteristics (relative scale)
    power_intensity: float = 1.0


# Pre-defined profiles for common embedding models
MODEL_PROFILES: dict[str, ModelProfile] = {
    # Small models
    "sentence-transformers/all-MiniLM-L6-v2": ModelProfile(
        name="all-MiniLM-L6-v2",
        embedding_dim=384,
        model_size_fp32_gb=0.09,
        model_size_fp16_gb=0.045,
        memory_per_item_fp32_mb=2.0,
        memory_per_item_fp16_mb=1.0,
        compute_intensity=0.3,
        power_intensity=0.3,
    ),
    "sentence-transformers/all-mpnet-base-v2": ModelProfile(
        name="all-mpnet-base-v2",
        embedding_dim=768,
        model_size_fp32_gb=0.42,
        model_size_fp16_gb=0.21,
        memory_per_item_fp32_mb=4.0,
        memory_per_item_fp16_mb=2.0,
        compute_intensity=0.6,
        power_intensity=0.5,
    ),
    # Large models
    "BAAI/bge-large-en-v1.5": ModelProfile(
        name="bge-large-en-v1.5",
        embedding_dim=1024,
        model_size_fp32_gb=1.34,
        model_size_fp16_gb=0.67,
        memory_per_item_fp32_mb=8.0,
        memory_per_item_fp16_mb=4.0,
        compute_intensity=1.0,
        power_intensity=1.0,
    ),
    "BAAI/bge-base-en-v1.5": ModelProfile(
        name="bge-base-en-v1.5",
        embedding_dim=768,
        model_size_fp32_gb=0.44,
        model_size_fp16_gb=0.22,
        memory_per_item_fp32_mb=4.0,
        memory_per_item_fp16_mb=2.0,
        compute_intensity=0.6,
        power_intensity=0.5,
    ),
    "BAAI/bge-small-en-v1.5": ModelProfile(
        name="bge-small-en-v1.5",
        embedding_dim=384,
        model_size_fp32_gb=0.13,
        model_size_fp16_gb=0.065,
        memory_per_item_fp32_mb=2.0,
        memory_per_item_fp16_mb=1.0,
        compute_intensity=0.3,
        power_intensity=0.3,
    ),
    # Instructor models
    "hkunlp/instructor-large": ModelProfile(
        name="instructor-large",
        embedding_dim=768,
        model_size_fp32_gb=1.34,
        model_size_fp16_gb=0.67,
        memory_per_item_fp32_mb=8.0,
        memory_per_item_fp16_mb=4.0,
        compute_intensity=1.0,
        power_intensity=1.0,
    ),
    # E5 models
    "intfloat/e5-large-v2": ModelProfile(
        name="e5-large-v2",
        embedding_dim=1024,
        model_size_fp32_gb=1.34,
        model_size_fp16_gb=0.67,
        memory_per_item_fp32_mb=8.0,
        memory_per_item_fp16_mb=4.0,
        compute_intensity=1.0,
        power_intensity=1.0,
    ),
}

# Default profile for unknown models (conservative estimates)
DEFAULT_MODEL_PROFILE = ModelProfile(
    name="unknown",
    embedding_dim=768,
    model_size_fp32_gb=0.5,
    model_size_fp16_gb=0.25,
    memory_per_item_fp32_mb=5.0,
    memory_per_item_fp16_mb=2.5,
    compute_intensity=0.7,
    power_intensity=0.7,
)


@dataclass
class OptimizedSettings:
    """Calculated optimal settings for the hardware/model combination."""

    # Device and precision
    device: str = "cpu"
    precision: PrecisionType = "fp32"

    # Batch size
    batch_size: int = 32
    min_batch_size: int = 4

    # Power management
    power_limit_watts: int | None = None
    enable_gpu_warmup: bool = True

    # Memory management
    max_memory_percent: float = 80.0
    max_temperature_c: float = 80.0
    inter_batch_delay: float = 0.1
    adaptive_batch_size: bool = True

    # Metadata
    hardware_profile: HardwareProfile = field(default_factory=HardwareProfile)
    model_profile: ModelProfile = field(default_factory=lambda: DEFAULT_MODEL_PROFILE)
    optimization_notes: list[str] = field(default_factory=list)


def detect_hardware() -> HardwareProfile:
    """Detect hardware capabilities.

    Returns:
        HardwareProfile with detected capabilities.
    """
    profile = HardwareProfile()

    # Detect CPU
    try:
        import psutil

        profile.cpu_cores = psutil.cpu_count(logical=False) or 1
        profile.ram_total_gb = psutil.virtual_memory().total / (1024**3)
        profile.ram_free_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        import os

        profile.cpu_cores = os.cpu_count() or 1

    # Detect GPU via PyTorch
    try:
        import torch

        if torch.cuda.is_available():
            profile.gpu_available = True
            profile.gpu_name = torch.cuda.get_device_name(0)

            # Compute capability
            cc = torch.cuda.get_device_capability(0)
            profile.gpu_compute_capability = cc

            # FP16 support (compute capability >= 5.3 for good FP16)
            profile.supports_fp16 = cc[0] > 5 or (cc[0] == 5 and cc[1] >= 3)

            # BF16 support (Ampere and newer, compute capability >= 8.0)
            profile.supports_bf16 = cc[0] >= 8

            # TF32 support (Ampere and newer)
            profile.supports_tf32 = cc[0] >= 8

            # VRAM
            props = torch.cuda.get_device_properties(0)
            profile.vram_total_gb = props.total_memory / (1024**3)

            # Free VRAM (approximate - PyTorch reserved vs total)
            profile.vram_free_gb = (props.total_memory - torch.cuda.memory_reserved(0)) / (1024**3)

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            profile.gpu_available = True
            profile.gpu_name = "Apple Silicon (MPS)"
            profile.supports_fp16 = True
            # MPS doesn't expose memory info easily
            profile.vram_total_gb = profile.ram_total_gb * 0.75  # Unified memory estimate
            profile.vram_free_gb = profile.ram_free_gb * 0.5

    except ImportError:
        pass

    # Detect power limits via nvidia-smi
    if profile.gpu_available and "NVIDIA" in profile.gpu_name:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.limit,power.default_limit,power.max_limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = [float(p.strip()) for p in result.stdout.strip().split(",")]
                if len(parts) >= 3:
                    profile.power_limit_watts = parts[0]
                    profile.power_default_watts = parts[1]
                    profile.power_max_watts = parts[2]
        except Exception:
            pass

    return profile


def get_model_profile(model_name: str) -> ModelProfile:
    """Get profile for a model, with fallback to defaults.

    Args:
        model_name: HuggingFace model name.

    Returns:
        ModelProfile for the model.
    """
    # Direct match
    if model_name in MODEL_PROFILES:
        return MODEL_PROFILES[model_name]

    # Partial match (e.g., "BAAI/bge-large-en-v1.5" matches "bge-large")
    model_lower = model_name.lower()
    for key, profile in MODEL_PROFILES.items():
        if key.lower() in model_lower or profile.name.lower() in model_lower:
            return profile

    # Estimate based on model name patterns
    if "large" in model_lower:
        estimated = ModelProfile(
            name=model_name,
            embedding_dim=1024,
            model_size_fp32_gb=1.3,
            model_size_fp16_gb=0.65,
            memory_per_item_fp32_mb=8.0,
            memory_per_item_fp16_mb=4.0,
            compute_intensity=1.0,
            power_intensity=1.0,
        )
    elif "base" in model_lower:
        estimated = ModelProfile(
            name=model_name,
            embedding_dim=768,
            model_size_fp32_gb=0.44,
            model_size_fp16_gb=0.22,
            memory_per_item_fp32_mb=4.0,
            memory_per_item_fp16_mb=2.0,
            compute_intensity=0.6,
            power_intensity=0.6,
        )
    elif "small" in model_lower or "mini" in model_lower:
        estimated = ModelProfile(
            name=model_name,
            embedding_dim=384,
            model_size_fp32_gb=0.1,
            model_size_fp16_gb=0.05,
            memory_per_item_fp32_mb=2.0,
            memory_per_item_fp16_mb=1.0,
            compute_intensity=0.3,
            power_intensity=0.3,
        )
    else:
        estimated = DEFAULT_MODEL_PROFILE

    logger.info(f"Using estimated profile for unknown model: {model_name}")
    return estimated


def calculate_optimal_settings(
    model_name: str,
    # Resource budget (percentages)
    vram_budget_percent: float = 70.0,
    power_budget_percent: float = 75.0,
    # Preferences
    precision: PrecisionType = "auto",
    stability_priority: Literal["performance", "balanced", "stable"] = "balanced",
    # Overrides
    hardware_profile: HardwareProfile | None = None,
) -> OptimizedSettings:
    """Calculate optimal settings for hardware/model combination.

    This is the main entry point for auto-configuration. It:
    1. Detects hardware if not provided
    2. Looks up model profile
    3. Calculates optimal batch size based on VRAM budget
    4. Calculates power limit based on power budget
    5. Selects precision based on hardware support

    Args:
        model_name: HuggingFace model name.
        vram_budget_percent: Target VRAM usage (0-100).
        power_budget_percent: Target power usage (0-100).
        precision: "fp32", "fp16", or "auto" to detect.
        stability_priority: Trade-off preference.
        hardware_profile: Pre-detected hardware (optional).

    Returns:
        OptimizedSettings with calculated values.
    """
    # Detect hardware
    hw = hardware_profile or detect_hardware()
    model = get_model_profile(model_name)

    settings = OptimizedSettings(
        hardware_profile=hw,
        model_profile=model,
    )

    # Stability multipliers
    stability_factors = {
        "performance": {"vram": 1.0, "power": 1.0, "batch": 1.0},
        "balanced": {"vram": 0.85, "power": 0.85, "batch": 0.8},
        "stable": {"vram": 0.7, "power": 0.7, "batch": 0.6},
    }
    factors = stability_factors[stability_priority]

    # Determine device
    if hw.gpu_available:
        if "NVIDIA" in hw.gpu_name:
            settings.device = "cuda"
        elif "Apple" in hw.gpu_name or "MPS" in hw.gpu_name:
            settings.device = "mps"
        else:
            settings.device = "cuda"  # Assume CUDA-compatible
    else:
        settings.device = "cpu"
        settings.optimization_notes.append("No GPU detected, using CPU")

    # Determine precision
    if precision == "auto":
        if hw.supports_fp16 and settings.device != "cpu":
            settings.precision = "fp16"
            settings.optimization_notes.append(
                f"Using FP16 (compute capability {hw.gpu_compute_capability})"
            )
        else:
            settings.precision = "fp32"
    else:
        settings.precision = precision
        if precision == "fp16" and not hw.supports_fp16:
            settings.optimization_notes.append(
                "Warning: FP16 requested but GPU may not support it efficiently"
            )

    # Calculate batch size based on VRAM budget
    if settings.device != "cpu" and hw.vram_total_gb > 0:
        # Available VRAM for batching (after model load)
        model_size = (
            model.model_size_fp16_gb if settings.precision == "fp16" else model.model_size_fp32_gb
        )
        memory_per_item = (
            model.memory_per_item_fp16_mb
            if settings.precision == "fp16"
            else model.memory_per_item_fp32_mb
        )

        # VRAM budget
        vram_budget_gb = hw.vram_total_gb * (vram_budget_percent / 100.0) * factors["vram"]
        available_for_batch_gb = vram_budget_gb - model_size

        if available_for_batch_gb > 0:
            # Calculate max batch size
            max_batch = int(available_for_batch_gb * 1024 / memory_per_item)

            # Apply stability factor
            target_batch = int(max_batch * factors["batch"])

            # Clamp to reasonable bounds
            settings.batch_size = max(4, min(256, target_batch))
            settings.min_batch_size = max(2, settings.batch_size // 8)
        else:
            settings.optimization_notes.append(
                f"Warning: Model may not fit in VRAM budget. Model: {model_size:.2f}GB, "
                f"Budget: {vram_budget_gb:.2f}GB"
            )
            settings.batch_size = 4
    else:
        # CPU - use conservative batch size based on RAM
        settings.batch_size = 32

    # Calculate power limit
    if hw.power_default_watts > 0:
        target_power = int(
            hw.power_default_watts
            * (power_budget_percent / 100.0)
            * factors["power"]
            * model.power_intensity
        )
        # Ensure within valid range
        min_power = 100  # Most GPUs won't go below this
        settings.power_limit_watts = max(min_power, target_power)

        settings.optimization_notes.append(
            f"Power limit: {settings.power_limit_watts}W "
            f"({power_budget_percent:.0f}% of {hw.power_default_watts:.0f}W default)"
        )

    # Memory and thermal settings based on stability
    if stability_priority == "stable":
        settings.max_memory_percent = 70.0
        settings.max_temperature_c = 75.0
        settings.inter_batch_delay = 0.2
    elif stability_priority == "balanced":
        settings.max_memory_percent = 80.0
        settings.max_temperature_c = 80.0
        settings.inter_batch_delay = 0.1
    else:  # performance
        settings.max_memory_percent = 90.0
        settings.max_temperature_c = 85.0
        settings.inter_batch_delay = 0.05

    settings.adaptive_batch_size = True
    settings.enable_gpu_warmup = settings.device != "cpu"

    return settings


@dataclass
class CalibrationResult:
    """Results from calibration benchmark."""

    optimal_batch_size: int
    optimal_power_limit: int | None
    throughput_items_per_sec: float
    peak_memory_gb: float
    peak_power_watts: float
    peak_temperature_c: float
    recommended_settings: OptimizedSettings


def run_calibration(
    model_name: str,
    test_batch_sizes: list[int] | None = None,
    warmup_iterations: int = 2,
    test_iterations: int = 3,
    target_memory_percent: float = 80.0,
    target_power_percent: float = 80.0,
) -> CalibrationResult:
    """Run calibration benchmark to find optimal settings.

    This performs actual inference to measure:
    - Memory usage at different batch sizes
    - Power draw at different batch sizes
    - Throughput at different batch sizes

    Args:
        model_name: Model to calibrate.
        test_batch_sizes: Batch sizes to test (auto if None).
        warmup_iterations: Warmup runs per batch size.
        test_iterations: Test runs per batch size.
        target_memory_percent: Target max memory usage.
        target_power_percent: Target max power usage.

    Returns:
        CalibrationResult with optimal settings.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    logger.info(f"Starting calibration for {model_name}")

    # Detect hardware
    hw = detect_hardware()
    get_model_profile(model_name)

    # Determine test batch sizes
    if test_batch_sizes is None:
        if hw.gpu_available:
            test_batch_sizes = [8, 16, 32, 64, 96, 128, 192, 256]
        else:
            test_batch_sizes = [8, 16, 32, 64]

    # Load model
    device = "cuda" if hw.gpu_available and "NVIDIA" in hw.gpu_name else "cpu"
    logger.info(f"Loading model on {device}...")
    model = SentenceTransformer(model_name, device=device)

    # Test text
    test_text = "This is a calibration test sentence for measuring embedding performance."

    # Results storage
    results: list[dict[str, Any]] = []

    for batch_size in test_batch_sizes:
        logger.info(f"Testing batch size {batch_size}...")

        # Prepare test batch
        test_batch = [test_text] * batch_size

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        for _ in range(warmup_iterations):
            _ = model.encode(test_batch, show_progress_bar=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Measure
        peak_memory = 0.0
        peak_power = 0.0
        peak_temp = 0.0
        total_time = 0.0

        for _ in range(test_iterations):
            start_time = time.perf_counter()
            _ = model.encode(test_batch, show_progress_bar=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            total_time += elapsed

            # Measure GPU stats
            if torch.cuda.is_available():
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / (1024**3))

            # Power and temp via nvidia-smi
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    peak_power = max(peak_power, float(parts[0].strip()))
                    peak_temp = max(peak_temp, float(parts[1].strip()))
            except Exception:
                pass

        avg_time = total_time / test_iterations
        throughput = batch_size / avg_time
        memory_percent = (peak_memory / hw.vram_total_gb * 100) if hw.vram_total_gb > 0 else 0
        power_percent = (
            (peak_power / hw.power_default_watts * 100) if hw.power_default_watts > 0 else 0
        )

        results.append(
            {
                "batch_size": batch_size,
                "throughput": throughput,
                "peak_memory_gb": peak_memory,
                "memory_percent": memory_percent,
                "peak_power_watts": peak_power,
                "power_percent": power_percent,
                "peak_temp_c": peak_temp,
                "avg_time_ms": avg_time * 1000,
            }
        )

        logger.info(
            f"  Batch {batch_size}: {throughput:.1f} items/s, "
            f"mem={peak_memory:.2f}GB ({memory_percent:.1f}%), "
            f"power={peak_power:.0f}W ({power_percent:.1f}%), "
            f"temp={peak_temp:.0f}C"
        )

        # Stop if we exceed targets significantly
        if memory_percent > target_memory_percent * 1.2:
            logger.info("  Stopping: exceeded memory target")
            break
        if power_percent > target_power_percent * 1.2:
            logger.info("  Stopping: exceeded power target")
            break

    # Find optimal batch size (highest throughput within targets)
    valid_results = [
        r
        for r in results
        if r["memory_percent"] <= target_memory_percent
        and r["power_percent"] <= target_power_percent
    ]

    if valid_results:
        optimal = max(valid_results, key=lambda r: r["throughput"])
    else:
        # Fall back to smallest batch size
        optimal = results[0]
        logger.warning("No batch size met targets, using minimum")

    # Calculate recommended power limit
    recommended_power = None
    if hw.power_default_watts > 0:
        # Set power limit to observed peak + 10% headroom, capped at target
        recommended_power = int(
            min(
                optimal["peak_power_watts"] * 1.1,
                hw.power_default_watts * (target_power_percent / 100.0),
            )
        )

    # Build recommended settings
    settings = calculate_optimal_settings(
        model_name=model_name,
        vram_budget_percent=target_memory_percent,
        power_budget_percent=target_power_percent,
        hardware_profile=hw,
    )
    settings.batch_size = optimal["batch_size"]
    settings.power_limit_watts = recommended_power

    return CalibrationResult(
        optimal_batch_size=optimal["batch_size"],
        optimal_power_limit=recommended_power,
        throughput_items_per_sec=optimal["throughput"],
        peak_memory_gb=optimal["peak_memory_gb"],
        peak_power_watts=optimal["peak_power_watts"],
        peak_temperature_c=optimal["peak_temp_c"],
        recommended_settings=settings,
    )


def print_hardware_report() -> None:
    """Print a detailed hardware report."""
    hw = detect_hardware()

    print("\n" + "=" * 70)
    print("HARDWARE PROFILE")
    print("=" * 70)

    print(f"\nCPU: {hw.cpu_cores} cores")
    print(f"RAM: {hw.ram_total_gb:.1f} GB total, {hw.ram_free_gb:.1f} GB free")

    if hw.gpu_available:
        print(f"\nGPU: {hw.gpu_name}")
        print(
            f"  Compute Capability: {hw.gpu_compute_capability[0]}.{hw.gpu_compute_capability[1]}"
        )
        print(f"  VRAM: {hw.vram_total_gb:.1f} GB total, {hw.vram_free_gb:.1f} GB free")
        print(f"  FP16 Support: {'Yes' if hw.supports_fp16 else 'No'}")
        print(f"  BF16 Support: {'Yes' if hw.supports_bf16 else 'No'}")
        print(f"  TF32 Support: {'Yes' if hw.supports_tf32 else 'No'}")

        if hw.power_default_watts > 0:
            print("\nPower:")
            print(f"  Current Limit: {hw.power_limit_watts:.0f}W")
            print(f"  Default Limit: {hw.power_default_watts:.0f}W")
            print(f"  Maximum Limit: {hw.power_max_watts:.0f}W")
    else:
        print("\nNo GPU detected")

    print("=" * 70 + "\n")


def print_optimization_report(
    model_name: str,
    vram_budget_percent: float = 70.0,
    power_budget_percent: float = 75.0,
    stability_priority: Literal["performance", "balanced", "stable"] = "balanced",
) -> None:
    """Print optimization recommendations for a model."""
    settings = calculate_optimal_settings(
        model_name=model_name,
        vram_budget_percent=vram_budget_percent,
        power_budget_percent=power_budget_percent,
        stability_priority=stability_priority,
    )

    print("\n" + "=" * 70)
    print(f"OPTIMIZATION REPORT: {model_name}")
    print("=" * 70)

    print("\nTarget Budgets:")
    print(f"  VRAM: {vram_budget_percent:.0f}%")
    print(f"  Power: {power_budget_percent:.0f}%")
    print(f"  Priority: {stability_priority}")

    print("\nRecommended Settings:")
    print(f"  Device: {settings.device}")
    print(f"  Precision: {settings.precision}")
    print(f"  Batch Size: {settings.batch_size}")
    print(f"  Min Batch Size: {settings.min_batch_size}")
    print(
        f"  Power Limit: {settings.power_limit_watts}W"
        if settings.power_limit_watts
        else "  Power Limit: Not set"
    )
    print(f"  GPU Warmup: {settings.enable_gpu_warmup}")
    print(f"  Max Memory: {settings.max_memory_percent:.0f}%")
    print(f"  Max Temperature: {settings.max_temperature_c:.0f}C")
    print(f"  Inter-batch Delay: {settings.inter_batch_delay}s")

    if settings.optimization_notes:
        print("\nNotes:")
        for note in settings.optimization_notes:
            print(f"  * {note}")

    print("\n" + "-" * 70)
    print("Config YAML settings:")
    print("-" * 70)
    print(
        f"""
embedding_model: {model_name}
device: {settings.device}
embedding_batch_size: {settings.batch_size}
precision: {settings.precision}

# GPU safeguard settings
enable_gpu_safeguards: true
gpu_max_memory_percent: {settings.max_memory_percent:.0f}
gpu_max_temperature_c: {settings.max_temperature_c:.0f}
gpu_inter_batch_delay: {settings.inter_batch_delay}
gpu_adaptive_batch_size: {settings.adaptive_batch_size}
gpu_min_batch_size: {settings.min_batch_size}

# GPU Power Management
gpu_power_limit_watts: {settings.power_limit_watts if settings.power_limit_watts else "null"}
enable_gpu_warmup: {str(settings.enable_gpu_warmup).lower()}
"""
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if not args or args[0] == "hardware":
        print_hardware_report()

    elif args[0] == "optimize":
        model = args[1] if len(args) > 1 else "BAAI/bge-large-en-v1.5"
        vram = float(args[2]) if len(args) > 2 else 70.0
        power = float(args[3]) if len(args) > 3 else 75.0
        priority = args[4] if len(args) > 4 else "balanced"
        # Cast to Literal type for mypy
        stability = priority if priority in ("performance", "balanced", "stable") else "balanced"
        print_optimization_report(model, vram, power, stability)  # type: ignore[arg-type]

    elif args[0] == "calibrate":
        model = args[1] if len(args) > 1 else "BAAI/bge-large-en-v1.5"
        result = run_calibration(model)
        print("\nCalibration complete!")
        print(f"  Optimal batch size: {result.optimal_batch_size}")
        print(f"  Optimal power limit: {result.optimal_power_limit}W")
        print(f"  Throughput: {result.throughput_items_per_sec:.1f} items/s")
        print(f"  Peak memory: {result.peak_memory_gb:.2f} GB")
        print(f"  Peak power: {result.peak_power_watts:.0f}W")

    else:
        print("Usage:")
        print("  python -m rag_service.core.gpu_optimizer hardware")
        print(
            "  python -m rag_service.core.gpu_optimizer optimize [model] [vram%] [power%] [priority]"
        )
        print("  python -m rag_service.core.gpu_optimizer calibrate [model]")
