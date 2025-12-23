"""Entry point for running the RAG service as a module.

This module serves as a thin wrapper around the Typer CLI application.
All functionality is delegated to the cli package.

Supports in-place restart via exit code 42 (RESTART_EXIT_CODE).
When the service exits with code 42, it will automatically restart
in the same console window.

Usage:
    python -m rag_service start           # Start the service
    python -m rag_service stop            # Stop the service
    python -m rag_service restart         # Restart the service
    python -m rag_service status          # Check service status
    python -m rag_service test            # Run tests
    python -m rag_service --help          # Show all commands
"""

import subprocess
import sys

# Exit code that signals "please restart me"
# This is used by the UI's "Apply & Restart" button
RESTART_EXIT_CODE = 42


def main() -> None:
    """Main entry point - delegates to Typer CLI app."""
    from rag_service.cli.main import app

    app()


def main_with_restart_loop() -> None:
    """Entry point with restart loop support.

    Runs the service as a subprocess. If it exits with RESTART_EXIT_CODE (42),
    automatically restart it in the same console window.

    This is used when you want automatic restart support (e.g., from GUI config changes).
    """
    python_exe = sys.executable
    # Pass through all arguments except the script name
    base_cmd = [python_exe, "-m", "rag_service.cli.main"] + sys.argv[1:]

    while True:
        print(f"[Launcher] Starting service: {' '.join(base_cmd)}")
        result = subprocess.run(base_cmd)

        if result.returncode == RESTART_EXIT_CODE:
            print("\n🔄 Restarting service...\n")
            continue  # Loop back to restart
        else:
            # Normal exit or error - propagate the exit code
            sys.exit(result.returncode)


if __name__ == "__main__":
    # Check if we should use the restart loop
    # Use loop if "start" command is used (the primary use case for restart)
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        main_with_restart_loop()
    else:
        main()
