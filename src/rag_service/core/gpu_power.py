"""GPU power management to prevent system crashes from power spikes.

The RTX 30-series GPUs can spike from ~35W idle to 400W+ in milliseconds,
which can trip PSU overcurrent protection or cause voltage sag crashes.
Power limiting caps the maximum draw, reducing transient severity.
"""

import logging
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


def get_power_info() -> dict[str, Any] | None:
    """Get current GPU power information using nvidia-smi.

    Returns:
        Dict with power info, or None if unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,power.draw,power.limit,power.default_limit,power.max_limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return None

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 5:
            return {
                "name": parts[0],
                "power_draw": float(parts[1]),
                "power_limit": float(parts[2]),
                "default_limit": float(parts[3]),
                "max_limit": float(parts[4]),
            }
    except Exception as e:
        logger.debug(f"Could not get power info: {e}")

    return None


def set_power_limit(watts: int) -> tuple[bool, str]:
    """Set GPU power limit using nvidia-smi.

    Note: Requires admin/root privileges.

    Args:
        watts: Power limit in watts.

    Returns:
        Tuple of (success, message).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-pl", str(watts)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            return True, f"Power limit set to {watts}W"
        else:
            error = result.stderr.strip() or result.stdout.strip()
            if "Insufficient Permissions" in error or "permission" in error.lower():
                return (
                    False,
                    f"Requires admin privileges. Run as Administrator or use: nvidia-smi -pl {watts}",
                )
            return False, f"Failed: {error}"

    except FileNotFoundError:
        return False, "nvidia-smi not found - ensure NVIDIA drivers are installed"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timed out"
    except Exception as e:
        return False, f"Error: {e}"


def get_recommended_settings(gpu_name: str | None = None) -> dict[str, Any]:
    """Get recommended power and batch settings for GPU/model combination.

    Args:
        gpu_name: GPU name (auto-detected if None).

    Returns:
        Dict with recommended settings.
    """
    if gpu_name is None:
        info = get_power_info()
        gpu_name = info["name"] if info else "Unknown"

    # RTX 3090 specific recommendations (your GPU)
    if "3090" in gpu_name:
        return {
            "gpu_name": gpu_name,
            # Power limiting - reduces spike from 420W to ~300W
            "power_limit_stable": 280,  # Very stable, ~70% default
            "power_limit_balanced": 320,  # Balanced stability/performance
            "power_limit_performance": 370,  # Near stock, some risk
            # Batch sizes for bge-large-en-v1.5 (335M params)
            "batch_size_stable": 32,
            "batch_size_balanced": 64,
            "batch_size_performance": 128,
            # Other settings
            "inter_batch_delay_ms": 100,  # Let power settle between batches
            "notes": [
                "RTX 3090 is notorious for transient power spikes",
                "Power limiting to 280-320W significantly reduces crash risk",
                "Batch size 64 is good balance for bge-large model",
            ],
        }

    # RTX 4090
    elif "4090" in gpu_name:
        return {
            "gpu_name": gpu_name,
            "power_limit_stable": 300,
            "power_limit_balanced": 370,
            "power_limit_performance": 430,
            "batch_size_stable": 48,
            "batch_size_balanced": 96,
            "batch_size_performance": 192,
            "inter_batch_delay_ms": 50,
            "notes": ["RTX 4090 has better power delivery than 3090"],
        }

    # RTX 3080
    elif "3080" in gpu_name:
        return {
            "gpu_name": gpu_name,
            "power_limit_stable": 240,
            "power_limit_balanced": 280,
            "power_limit_performance": 320,
            "batch_size_stable": 24,
            "batch_size_balanced": 48,
            "batch_size_performance": 96,
            "inter_batch_delay_ms": 100,
            "notes": ["Similar to 3090 but lower power headroom"],
        }

    # Generic/Unknown
    else:
        info = get_power_info()
        default_limit = info["default_limit"] if info else 300
        return {
            "gpu_name": gpu_name,
            "power_limit_stable": int(default_limit * 0.70),
            "power_limit_balanced": int(default_limit * 0.80),
            "power_limit_performance": int(default_limit * 0.90),
            "batch_size_stable": 16,
            "batch_size_balanced": 32,
            "batch_size_performance": 64,
            "inter_batch_delay_ms": 150,
            "notes": ["Using conservative defaults for unknown GPU"],
        }


def apply_power_limit_for_stability(mode: str = "balanced") -> tuple[bool, str, int | None]:
    """Apply recommended power limit based on stability mode.

    Args:
        mode: "stable", "balanced", or "performance"

    Returns:
        Tuple of (success, message, applied_watts).
    """
    settings = get_recommended_settings()

    limit_key = f"power_limit_{mode}"
    if limit_key not in settings:
        return False, f"Unknown mode: {mode}", None

    target_watts = settings[limit_key]
    success, message = set_power_limit(target_watts)

    return success, message, target_watts if success else None


def print_power_status() -> None:
    """Print current power status and recommendations."""
    info = get_power_info()

    if not info:
        print("Could not get GPU power information")
        return

    print(f"\n{'=' * 60}")
    print(f"GPU: {info['name']}")
    print(f"{'=' * 60}")
    print(f"Current power draw:   {info['power_draw']:.1f}W")
    print(f"Current power limit:  {info['power_limit']:.1f}W")
    print(f"Default power limit:  {info['default_limit']:.1f}W")
    print(f"Maximum power limit:  {info['max_limit']:.1f}W")
    print(f"Power usage:          {info['power_draw'] / info['power_limit'] * 100:.1f}%")

    settings = get_recommended_settings(info["name"])
    print(f"\n{'-' * 60}")
    print("Recommended Power Limits for Stability:")
    print(f"  Stable mode:      {settings['power_limit_stable']}W (safest, ~15-20% slower)")
    print(f"  Balanced mode:    {settings['power_limit_balanced']}W (recommended)")
    print(f"  Performance mode: {settings['power_limit_performance']}W (near stock)")
    print("\nRecommended Batch Sizes:")
    print(f"  Stable:      {settings['batch_size_stable']}")
    print(f"  Balanced:    {settings['batch_size_balanced']}")
    print(f"  Performance: {settings['batch_size_performance']}")

    if settings.get("notes"):
        print(f"\n{'-' * 60}")
        print("Notes:")
        for note in settings["notes"]:
            print(f"  * {note}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # CLI usage: python -m rag_service.core.gpu_power [status|set WATTS|stable|balanced|performance]
    import sys

    if len(sys.argv) < 2 or sys.argv[1] == "status":
        print_power_status()
    elif sys.argv[1] == "set" and len(sys.argv) > 2:
        watts = int(sys.argv[2])
        success, msg = set_power_limit(watts)
        print(msg)
    elif sys.argv[1] in ("stable", "balanced", "performance"):
        success, msg, applied_watts = apply_power_limit_for_stability(sys.argv[1])
        print(msg)
        if success and applied_watts is not None:
            print(f"Applied {sys.argv[1]} mode ({applied_watts}W)")
    else:
        print(
            "Usage: python -m rag_service.core.gpu_power [status|set WATTS|stable|balanced|performance]"
        )
