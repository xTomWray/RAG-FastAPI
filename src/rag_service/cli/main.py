"""Main CLI application for the RAG Documentation Service.

Provides a unified command-line interface using Typer with Rich
integration for beautiful terminal output.

Usage:
    rag-service start [OPTIONS]       Start the service
    rag-service stop                  Stop the service
    rag-service test [OPTIONS]        Run tests
    rag-service install [OPTIONS]     Install dependencies
    rag-service --help                Show help
"""

import typer
from rich.console import Console

from rag_service.cli import (
    clean_commands,
    commands,
    dev_commands,
    docker_commands,
    install_commands,
)

# Create the main Typer app
app = typer.Typer(
    name="rag-service",
    help="RAG Documentation Service CLI - A cross-platform tool for managing the RAG API service.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
)

console = Console()

# Register service lifecycle commands (start, stop, restart, status, stats)
app.command(name="start")(commands.start)
app.command(name="stop")(commands.stop)
app.command(name="restart")(commands.restart)
app.command(name="status")(commands.status)
app.command(name="stats")(commands.stats)

# Register install command
app.command(name="install")(install_commands.install)

# Register development commands (test, lint, format, typecheck, check)
app.command(name="test")(dev_commands.test)
app.command(name="lint")(dev_commands.lint)
app.command(name="format")(dev_commands.format_code)
app.command(name="typecheck")(dev_commands.typecheck)
app.command(name="check")(dev_commands.check)

# Register docker commands
app.command(name="docker-build")(docker_commands.docker_build)
app.command(name="docker-run")(docker_commands.docker_run)

# Register cleanup command
app.command(name="clean")(clean_commands.clean)


@app.callback(invoke_without_command=True)  # type: ignore[untyped-decorator]
def main(ctx: typer.Context) -> None:
    """RAG Documentation Service CLI.

    A cross-platform command-line tool for managing the RAG API service,
    running development tools, and deploying with Docker.

    Use 'rag-service COMMAND --help' for more information on a command.
    """
    # If no command is provided, show help
    if ctx.invoked_subcommand is None:
        console.print()
        console.print("[bold blue]RAG Documentation Service[/bold blue]")
        console.print()
        console.print("Use [green]rag-service --help[/green] to see available commands.")
        console.print()
        raise typer.Exit(0)


def run_with_restart_loop() -> None:
    """Run the CLI with restart loop support for the 'start' command.

    When the service exits with code 42 (RESTART_EXIT_CODE), it will
    automatically restart in the same console window. This enables
    seamless config reloads from the GUI.

    For other commands, runs normally without the loop.
    """
    import subprocess
    import sys

    from rag_service.__main__ import RESTART_EXIT_CODE

    # Only use restart loop for 'start' command
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        python_exe = sys.executable
        # Run the actual CLI as a subprocess
        base_cmd = [python_exe, "-c", "from rag_service.cli.main import app; app()", *sys.argv[1:]]

        while True:
            console.print("[dim][Launcher] Starting service...[/dim]")
            result = subprocess.run(base_cmd)

            if result.returncode == RESTART_EXIT_CODE:
                console.print()
                console.print("[bold green]🔄 Restarting service...[/bold green]")
                console.print()
                continue  # Loop back to restart
            else:
                # Normal exit or error
                sys.exit(result.returncode)
    else:
        # For non-start commands, just run directly
        app()


if __name__ == "__main__":
    app()
