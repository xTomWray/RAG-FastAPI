"""Installation commands for managing dependencies.

Provides commands for installing production, development, and GPU
dependencies using pip internally.
"""

import subprocess
import sys

import typer

from rag_service.cli.utils import (
    console,
    exit_with_error,
    print_error,
    print_info,
    print_success,
    print_warning,
)

app = typer.Typer(help="Installation commands")

# Check if running on Windows
IS_WINDOWS = sys.platform == "win32"


def _run_pip_install(extras: list[str] | None = None, editable: bool = True) -> bool:
    """Run pip install with optional extras.

    Args:
        extras: List of optional extras to install (e.g., ["dev", "gpu"]).
        editable: Whether to install in editable mode.

    Returns:
        True if installation succeeded.
    """
    package_spec = "."
    if extras:
        package_spec = f".[{','.join(extras)}]"

    args = [sys.executable, "-m", "pip", "install"]
    if editable:
        args.append("-e")
    args.append(package_spec)

    console.print(f"[dim]Running: {' '.join(args)}[/dim]")
    console.print()

    result = subprocess.run(args)

    # If it failed on Windows due to file locking, provide helpful message
    if result.returncode != 0 and IS_WINDOWS:
        console.print()
        print_warning(
            "Installation failed - likely due to Windows file locking.\n"
            "\n"
            "This happens because rag-service CLI is part of the package\n"
            "being reinstalled, and Windows locks files in use.\n"
            "\n"
            "Workarounds:\n"
            f"  1. Run pip directly:  pip install -e {package_spec}\n"
            f"  2. Use Makefile:      make {'dev' if 'dev' in (extras or []) else 'gpu' if 'gpu' in (extras or []) else 'install'}"
        )

    return result.returncode == 0


def _setup_pre_commit() -> bool:
    """Set up pre-commit hooks.

    Returns:
        True if setup succeeded.
    """
    print_info("Setting up pre-commit hooks...")
    result = subprocess.run(
        [sys.executable, "-m", "pre_commit", "install"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Try with pre-commit directly
        result = subprocess.run(
            ["pre-commit", "install"],
            capture_output=True,
            text=True,
        )
    return result.returncode == 0


def _install_cuda_pytorch() -> bool:
    """Install PyTorch with CUDA support from the PyTorch index.

    This installs PyTorch with CUDA 12.4 from PyTorch's wheel index,
    which is required for GPU acceleration.

    Returns:
        True if installation succeeded.
    """
    print_info("Installing PyTorch with CUDA 12.4 support...")
    console.print("[dim]This may take a few minutes (PyTorch is ~2.5GB)[/dim]")
    console.print()

    # First uninstall any existing CPU-only PyTorch
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"],
        capture_output=True,
    )

    # Install CUDA-enabled PyTorch from PyTorch's index
    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
    ]

    console.print("[dim]Running: pip install torch torchvision --index-url .../cu124[/dim]")
    console.print()

    result = subprocess.run(args)
    return result.returncode == 0


def _verify_cuda() -> bool:
    """Verify CUDA is working with PyTorch.

    Returns:
        True if CUDA is available and working.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; exit(0 if torch.cuda.is_available() else 1)"],
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


@app.command("install")  # type: ignore[untyped-decorator]
def install(
    dev: bool = typer.Option(
        False, "--dev", "-d", help="Install development dependencies and pre-commit hooks."
    ),
    gpu: bool = typer.Option(False, "--gpu", "-g", help="Install GPU support (CUDA 12.4)."),
    all_extras: bool = typer.Option(
        False, "--all", "-a", help="Install all optional dependencies."
    ),
) -> None:
    """Install project dependencies.

    By default, installs production dependencies only. Use flags to
    include additional extras.

    Examples:
        rag-service install           # Production dependencies
        rag-service install --dev     # Development environment
        rag-service install --gpu     # GPU support (CUDA 12.4)
        rag-service install --all     # Everything
    """
    extras: list[str] = []

    # Handle GPU installation specially - need to install from PyTorch index
    if gpu or all_extras:
        if not _install_cuda_pytorch():
            print_error("Failed to install CUDA PyTorch.")
            print_info(
                "Try manually: pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
            exit_with_error("GPU installation failed.")

        console.print()
        if _verify_cuda():
            print_success("CUDA PyTorch installed and verified!")
        else:
            print_warning("PyTorch installed but CUDA not detected. Check your NVIDIA drivers.")

        console.print()

    if all_extras:
        extras = ["all"]
        print_info("Installing all dependencies (dev, gpu, ocr, graphrag)...")
    else:
        if dev:
            extras.append("dev")
        if gpu:
            extras.append("gpu")

        if extras:
            print_info(f"Installing dependencies with extras: {', '.join(extras)}...")
        else:
            print_info("Installing production dependencies...")

    console.print()

    if not _run_pip_install(extras if extras else None):
        exit_with_error("Installation failed.")

    # Set up pre-commit hooks for dev installs
    if dev or all_extras:
        console.print()
        if not _setup_pre_commit():
            print_error("Failed to set up pre-commit hooks.")
            print_info("You can set them up manually with: pre-commit install")
        else:
            print_success("Pre-commit hooks installed.")

    console.print()
    print_success("Installation complete!")
