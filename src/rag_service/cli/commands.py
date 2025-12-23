"""Service lifecycle commands: start, stop, restart, status.

Provides commands for managing the RAG service lifecycle,
including starting the server, stopping running instances,
and checking service status.
"""

import os
import time
from typing import Any

import typer
import uvicorn

from rag_service.cli.utils import (
    check_service_health,
    console,
    exit_with_error,
    is_process_running,
    print_info,
    print_success,
    print_warning,
    read_pid_file,
    remove_pid_file,
    terminate_process,
    wait_for_process_stop,
    write_pid_file,
)
from rag_service.config import get_settings

app = typer.Typer(help="Service lifecycle commands")


@app.command()
def start(
    host: str | None = typer.Option(
        None, "--host", "-h", help="Host to bind to (overrides config)."
    ),
    port: int | None = typer.Option(
        None, "--port", "-p", help="Port to bind to (overrides config)."
    ),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable hot reload for development."),
    no_ui: bool = typer.Option(
        False, "--no-ui", help="Disable embedded Gradio UI (API-only mode)."
    ),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of uvicorn workers."),
) -> None:
    """Start the RAG service.

    Launches the FastAPI server with optional Gradio UI. By default,
    runs with the UI enabled on the configured host/port.

    Examples:
        rag-service start                    # Start with defaults
        rag-service start --reload           # Development mode
        rag-service start --no-ui -w 4       # Production API-only
    """
    settings = get_settings()
    actual_host = host or settings.host
    actual_port = port or settings.port

    # Set environment variable for UI control
    if no_ui:
        os.environ["ENABLE_GUI"] = "false"

    # Check if already running
    if check_service_health(actual_port):
        exit_with_error(
            f"Service is already running on port {actual_port}. "
            "Use 'rag-service restart' to restart or 'rag-service stop' first."
        )

    # Write PID file
    write_pid_file()

    console.print()
    console.print("[bold green]Starting RAG Documentation Service[/bold green]")
    console.print(f"  [dim]Host:[/dim] {actual_host}")
    console.print(f"  [dim]Port:[/dim] {actual_port}")
    console.print(f"  [dim]Workers:[/dim] {workers}")
    console.print(f"  [dim]UI:[/dim] {'disabled' if no_ui else 'enabled'}")
    console.print(f"  [dim]Reload:[/dim] {'enabled' if reload else 'disabled'}")
    console.print()
    print_info(f"API docs: http://{actual_host}:{actual_port}/docs")
    if not no_ui:
        print_info(f"UI: http://{actual_host}:{actual_port}/ui")
    console.print()

    try:
        uvicorn.run(
            "rag_service.main:app",
            host=actual_host,
            port=actual_port,
            reload=reload,
            workers=workers if not reload else 1,  # Reload only works with 1 worker
        )
    finally:
        remove_pid_file()


@app.command()
def stop() -> None:
    """Stop the running RAG service.

    Sends a termination signal to the running service process.
    Falls back to force kill if graceful shutdown fails.
    """
    settings = get_settings()
    port = settings.port
    pid = read_pid_file()

    if pid and is_process_running(pid):
        print_info(f"Stopping service (PID: {pid})...")

        terminate_process(pid, force=False)

        if wait_for_process_stop(pid, timeout=3.0):
            remove_pid_file()
            print_success("Service stopped.")
            return

        # Force kill if still running
        print_warning("Graceful shutdown timed out, force killing...")
        terminate_process(pid, force=True)
        remove_pid_file()
        print_success("Service force-stopped.")
        return

    # Check if service is responding but PID file is missing
    if check_service_health(port):
        print_warning(f"Service is running on port {port} but PID file not found.")
        print_info("Please stop the service manually or kill the process.")
        raise typer.Exit(1)

    print_info("No running service found.")
    remove_pid_file()


@app.command()
def restart(
    host: str | None = typer.Option(
        None, "--host", "-h", help="Host to bind to (overrides config)."
    ),
    port: int | None = typer.Option(
        None, "--port", "-p", help="Port to bind to (overrides config)."
    ),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable hot reload for development."),
    no_ui: bool = typer.Option(
        False, "--no-ui", help="Disable embedded Gradio UI (API-only mode)."
    ),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of uvicorn workers."),
) -> None:
    """Restart the RAG service.

    Stops any running instance and starts a new one with the
    specified configuration.
    """
    print_info("Restarting RAG Documentation Service...")

    # Stop existing service
    settings = get_settings()
    actual_port = port or settings.port
    pid = read_pid_file()

    if pid and is_process_running(pid):
        terminate_process(pid, force=False)
        wait_for_process_stop(pid, timeout=3.0)
        remove_pid_file()
    elif check_service_health(actual_port):
        print_warning("Cannot stop existing service (no PID file). Proceeding anyway...")

    # Wait for port to be released
    time.sleep(0.5)

    # Start new service
    start(host=host, port=port, reload=reload, no_ui=no_ui, workers=workers)


@app.command()
def status() -> None:
    """Show the current service status.

    Displays information about the running service including
    PID, health status, and access URLs.
    """
    settings = get_settings()
    port = settings.port
    pid = read_pid_file()
    health_ok = check_service_health(port)

    console.print()
    console.print("[bold]RAG Documentation Service Status[/bold]")
    console.print("=" * 40)
    console.print(f"  [dim]Port:[/dim] {port}")

    if pid:
        running = is_process_running(pid)
        status_text = "[green]running[/green]" if running else "[red]not running[/red]"
        console.print(f"  [dim]PID:[/dim] {pid} ({status_text})")
    else:
        console.print("  [dim]PID:[/dim] not found")

    if health_ok:
        console.print("  [dim]Health:[/dim] [green]OK (responding)[/green]")
        console.print()
        print_info(f"API: http://localhost:{port}/docs")
        print_info(f"UI: http://localhost:{port}/ui")
    else:
        console.print("  [dim]Health:[/dim] [red]not responding[/red]")

    console.print()


@app.command()
def stats(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON instead of formatted text."
    ),
    reset: bool = typer.Option(False, "--reset", "-r", help="Reset statistics after displaying."),
) -> None:
    """Show service runtime statistics.

    Displays performance metrics including:
    - Service uptime and health status
    - GPU utilization and memory
    - Embedding throughput and latency
    - Search query statistics
    - Document ingestion metrics

    Examples:
        rag-service stats              # Show formatted stats
        rag-service stats --json       # Output as JSON
        rag-service stats --reset      # Show then reset
    """
    import json as json_module

    import httpx

    settings = get_settings()
    port = settings.port
    api_url = f"http://localhost:{port}"

    # Check if service is running
    if not check_service_health(port):
        exit_with_error(
            f"Service not responding on port {port}. Use 'rag-service start' to start the service."
        )

    try:
        # Fetch stats from API
        response = httpx.get(f"{api_url}/api/v1/stats", timeout=10.0)

        if response.status_code != 200:
            exit_with_error(f"Failed to fetch stats: HTTP {response.status_code}")

        data = response.json()

        if json_output:
            # Output raw JSON
            console.print(json_module.dumps(data, indent=2))
        else:
            # Use the formatted output from stats module

            # Create a temporary collector just for formatting
            # (we display data from API, not local stats)
            _format_stats_output(data)

        # Reset if requested
        if reset:
            reset_response = httpx.post(f"{api_url}/api/v1/stats/reset", timeout=5.0)
            if reset_response.status_code == 200:
                print_success("Statistics reset.")
            else:
                print_warning("Failed to reset statistics.")

    except httpx.ConnectError:
        exit_with_error(f"Cannot connect to service on port {port}.")
    except Exception as e:
        exit_with_error(f"Error fetching stats: {e}")


def _format_stats_output(data: dict[str, Any]) -> None:
    """Format and print statistics to console.

    Args:
        data: Statistics dictionary from API.
    """
    from rich.panel import Panel
    from rich.table import Table

    # Service status
    service = data.get("service", {})
    health = data.get("health", {})

    console.print()
    console.print(
        Panel.fit(
            f"[bold]Uptime:[/bold] {service.get('uptime', 'N/A')}  |  "
            f"[bold]Health:[/bold] {health.get('status', 'unknown').upper()} ({health.get('score', 0)}%)  |  "
            f"[bold]Operations:[/bold] {health.get('total_operations', 0):,}",
            title="📊 RAG Service Statistics",
            border_style="blue",
        )
    )

    # GPU info
    gpu = data.get("gpu", {})
    if gpu.get("available"):
        gpu_table = Table(show_header=False, box=None, padding=(0, 2))
        gpu_table.add_column("Metric", style="dim")
        gpu_table.add_column("Value", style="green")

        gpu_table.add_row("Device", gpu.get("device_name", "Unknown"))
        mem_pct = gpu.get("memory_percent", 0)
        gpu_table.add_row(
            "Memory",
            f"{gpu.get('memory_allocated_gb', 0):.1f} / {gpu.get('memory_total_gb', 0):.1f} GB ({mem_pct:.0f}%)",
        )
        if "temperature_c" in gpu:
            temp = gpu["temperature_c"]
            temp_style = "red" if temp > 80 else "yellow" if temp > 70 else "green"
            gpu_table.add_row("Temperature", f"[{temp_style}]{temp:.0f}°C[/{temp_style}]")
        if "power_draw_watts" in gpu:
            gpu_table.add_row(
                "Power", f"{gpu['power_draw_watts']:.0f}W / {gpu.get('power_limit_watts', 'N/A')}W"
            )
        if "utilization_percent" in gpu:
            gpu_table.add_row("Utilization", f"{gpu['utilization_percent']:.0f}%")

        console.print()
        console.print(Panel(gpu_table, title="🎮 GPU", border_style="cyan"))

    # Operations table
    ops = data.get("operations", {})

    # Embeddings
    emb = ops.get("embeddings", {})
    if emb.get("count", 0) > 0:
        emb_table = Table(show_header=False, box=None, padding=(0, 2))
        emb_table.add_column("Metric", style="dim")
        emb_table.add_column("Value", style="green")

        emb_table.add_row("Operations", f"{emb.get('count', 0):,}")
        emb_table.add_row("Texts Embedded", f"{emb.get('total_texts', 0):,}")
        emb_table.add_row("Throughput", f"{emb.get('texts_per_second', 0):.1f} texts/sec")
        emb_table.add_row("Avg Latency", f"{emb.get('avg_duration_ms', 0):.1f}ms")
        emb_table.add_row("P95 Latency", f"{emb.get('p95_duration_ms', 0):.1f}ms")
        emb_table.add_row("Success Rate", f"{emb.get('success_rate', 100):.1f}%")

        console.print()
        console.print(Panel(emb_table, title="🧠 Embeddings", border_style="green"))

    # Searches
    search = ops.get("searches", {})
    if search.get("count", 0) > 0:
        search_table = Table(show_header=False, box=None, padding=(0, 2))
        search_table.add_column("Metric", style="dim")
        search_table.add_column("Value", style="green")

        search_table.add_row("Total Queries", f"{search.get('total_queries', 0):,}")
        search_table.add_row("Avg Results", f"{search.get('avg_results_per_query', 0):.1f}")
        search_table.add_row("Avg Latency", f"{search.get('avg_duration_ms', 0):.1f}ms")
        search_table.add_row("P95 Latency", f"{search.get('p95_duration_ms', 0):.1f}ms")

        strategy_counts = search.get("strategy_counts", {})
        if strategy_counts:
            strategies = ", ".join(f"{k}={v}" for k, v in strategy_counts.items())
            search_table.add_row("Strategies", strategies)

        console.print()
        console.print(Panel(search_table, title="🔍 Searches", border_style="yellow"))

    # Ingestions
    ingest = ops.get("ingestions", {})
    if ingest.get("count", 0) > 0:
        ingest_table = Table(show_header=False, box=None, padding=(0, 2))
        ingest_table.add_column("Metric", style="dim")
        ingest_table.add_column("Value", style="green")

        ingest_table.add_row("Documents", f"{ingest.get('total_documents', 0):,}")
        ingest_table.add_row("Chunks Created", f"{ingest.get('total_chunks', 0):,}")
        ingest_table.add_row("Data Processed", f"{ingest.get('total_mb', 0):.1f} MB")
        ingest_table.add_row("Avg Chunks/Doc", f"{ingest.get('avg_chunks_per_doc', 0):.1f}")

        console.print()
        console.print(Panel(ingest_table, title="📥 Ingestions", border_style="magenta"))

    # Errors
    errors = data.get("recent_errors", [])
    if errors:
        console.print()
        console.print(f"[yellow]⚠️ Recent Errors ({len(errors)}):[/yellow]")
        for err in errors[-5:]:
            console.print(
                f"  [dim]{err.get('timestamp', '')}[/dim] [red]{err.get('type', 'Error')}[/red]: {err.get('message', '')[:80]}"
            )

    console.print()
