"""Shared utilities for CLI commands.

Provides common functionality like console output, process management,
and cross-platform helpers used across CLI commands.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Shared console instance for consistent output
console = Console()

# PID file location
PID_FILE_NAME = ".rag_service.pid"


def get_pid_file_path() -> Path:
    """Get the PID file path, checking multiple locations.

    Returns:
        Path to the PID file, preferring existing files over default location.
    """
    locations = [
        Path.cwd() / PID_FILE_NAME,
        Path(__file__).parent.parent.parent.parent / PID_FILE_NAME,
    ]
    for path in locations:
        if path.exists():
            return path
    return Path.cwd() / PID_FILE_NAME


def write_pid_file() -> None:
    """Write the current process ID to the PID file."""
    pid_path = Path.cwd() / PID_FILE_NAME
    pid_path.write_text(str(os.getpid()))


def read_pid_file() -> int | None:
    """Read the process ID from the PID file.

    Returns:
        The PID if found and valid, None otherwise.
    """
    pid_path = get_pid_file_path()
    if pid_path.exists():
        try:
            return int(pid_path.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


def remove_pid_file() -> None:
    """Remove the PID file if it exists."""
    pid_path = get_pid_file_path()
    if pid_path.exists():
        pid_path.unlink()


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running.

    Args:
        pid: Process ID to check.

    Returns:
        True if the process is running, False otherwise.
    """
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def check_service_health(port: int, timeout: float = 2.0) -> bool:
    """Check if the service is responding to health checks.

    Args:
        port: Port number to check.
        timeout: Request timeout in seconds.

    Returns:
        True if the service responds with status 200.
    """
    try:
        response = httpx.get(f"http://localhost:{port}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def terminate_process(pid: int, force: bool = False) -> bool:
    """Terminate a process by PID. Cross-platform compatible.

    Args:
        pid: Process ID to terminate.
        force: If True, force kill the process.

    Returns:
        True if termination was successful.
    """
    if sys.platform == "win32":
        try:
            args = ["taskkill", "/PID", str(pid)]
            if force:
                args.append("/F")
            result = subprocess.run(args, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    else:
        try:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)
            return True
        except (OSError, ProcessLookupError):
            return False


def wait_for_process_stop(pid: int, timeout: float = 3.0) -> bool:
    """Wait for a process to stop.

    Args:
        pid: Process ID to wait for.
        timeout: Maximum time to wait in seconds.

    Returns:
        True if process stopped within timeout.
    """
    iterations = int(timeout / 0.1)
    for _ in range(iterations):
        time.sleep(0.1)
        if not is_process_running(pid):
            return True
    return False


def run_subprocess(
    args: list[str],
    capture_output: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with consistent settings.

    Args:
        args: Command and arguments to run.
        capture_output: Whether to capture stdout/stderr.
        check: Whether to raise on non-zero exit code.

    Returns:
        CompletedProcess instance with results.
    """
    return subprocess.run(
        args,
        capture_output=capture_output,
        text=True,
        check=check,
    )


def print_success(message: str) -> None:
    """Print a success message in green."""
    console.print(f"[green]{message}[/green]")


def print_error(message: str) -> None:
    """Print an error message in red."""
    console.print(f"[red]{message}[/red]")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    console.print(f"[yellow]{message}[/yellow]")


def print_info(message: str) -> None:
    """Print an info message in blue."""
    console.print(f"[blue]{message}[/blue]")


def exit_with_error(message: str, code: int = 1) -> NoReturn:
    """Print an error message and exit with the given code.

    Args:
        message: Error message to display.
        code: Exit code (default 1).
    """
    print_error(message)
    raise SystemExit(code)


def create_spinner(description: str) -> Progress:
    """Create a spinner progress indicator.

    Args:
        description: Text to display next to the spinner.

    Returns:
        Progress instance configured as a spinner.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )
