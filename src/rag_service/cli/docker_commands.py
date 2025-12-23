"""Docker-related commands: docker-build, docker-run.

Provides commands for building and running Docker containers
for the RAG service.
"""

import subprocess

import typer

from rag_service.cli.utils import (
    console,
    exit_with_error,
    print_info,
    print_success,
)

app = typer.Typer(help="Docker commands")

# Default Docker image name
IMAGE_NAME = "rag-documentation-service"


def _run_docker_command(args: list[str]) -> int:
    """Run a docker command and return the exit code.

    Args:
        args: Docker command arguments.

    Returns:
        Exit code of the docker command.
    """
    console.print(f"[dim]Running: docker {' '.join(args)}[/dim]")
    console.print()
    result = subprocess.run(["docker"] + args)
    return result.returncode


@app.command("docker-build")
def docker_build(
    gpu: bool = typer.Option(
        False, "--gpu", "-g", help="Build GPU-enabled image (CUDA)."
    ),
    tag: str = typer.Option(
        "latest", "--tag", "-t", help="Image tag."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Build without using cache."
    ),
) -> None:
    """Build Docker image for the RAG service.

    Builds a Docker image using the multi-stage Dockerfile.
    Use --gpu for CUDA-enabled builds.

    Examples:
        rag-service docker-build              # CPU image
        rag-service docker-build --gpu        # GPU image
        rag-service docker-build -t v1.0.0    # Custom tag
    """
    target = "gpu" if gpu else "production"
    full_tag = f"{IMAGE_NAME}:{tag}"
    if gpu:
        full_tag = f"{IMAGE_NAME}:gpu-{tag}"

    print_info(f"Building Docker image: {full_tag}")
    print_info(f"Target: {target}")
    console.print()

    args = [
        "build",
        "--target", target,
        "-t", full_tag,
        ".",
    ]

    if no_cache:
        args.insert(1, "--no-cache")

    exit_code = _run_docker_command(args)

    if exit_code == 0:
        print_success(f"Successfully built: {full_tag}")
    else:
        exit_with_error(f"Docker build failed with exit code {exit_code}")


@app.command("docker-run")
def docker_run(
    gpu: bool = typer.Option(
        False, "--gpu", "-g", help="Run GPU-enabled image."
    ),
    tag: str = typer.Option(
        "latest", "--tag", "-t", help="Image tag to run."
    ),
    port: int = typer.Option(
        8080, "--port", "-p", help="Host port to bind."
    ),
    detach: bool = typer.Option(
        False, "--detach", "-d", help="Run in background."
    ),
    name: str = typer.Option(
        "rag-service", "--name", "-n", help="Container name."
    ),
) -> None:
    """Run the RAG service in a Docker container.

    Starts a container from the built image with appropriate
    port mappings and volume mounts.

    Examples:
        rag-service docker-run              # Interactive
        rag-service docker-run -d           # Background
        rag-service docker-run --gpu        # GPU-enabled
    """
    full_tag = f"{IMAGE_NAME}:{tag}"
    if gpu:
        full_tag = f"{IMAGE_NAME}:gpu-{tag}"

    print_info(f"Running Docker image: {full_tag}")
    console.print()

    args = [
        "run",
        "--rm",
        "-p", f"{port}:8080",
        "-v", "rag-data:/app/data",
        "--name", name,
    ]

    if detach:
        args.append("-d")

    if gpu:
        args.extend(["--gpus", "all"])

    args.append(full_tag)

    exit_code = _run_docker_command(args)

    if exit_code == 0 and detach:
        print_success(f"Container '{name}' started in background.")
        print_info(f"Access at: http://localhost:{port}")
    elif exit_code != 0:
        exit_with_error(f"Docker run failed with exit code {exit_code}")
