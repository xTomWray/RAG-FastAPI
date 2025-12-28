"""Cleanup commands for removing build artifacts and data.

Provides commands for cleaning up generated files, caches,
and optionally vector store data.
"""

import shutil
from pathlib import Path

import typer

from rag_service.cli.utils import (
    console,
    print_info,
    print_success,
    print_warning,
)

app = typer.Typer(help="Cleanup commands")


def _remove_directory(path: Path, description: str) -> bool:
    """Remove a directory if it exists.

    Args:
        path: Directory path to remove.
        description: Description for logging.

    Returns:
        True if directory was removed.
    """
    if path.exists() and path.is_dir():
        try:
            shutil.rmtree(path, ignore_errors=False)
            console.print(f"  [dim]Removed:[/dim] {description} ({path})")
            return True
        except OSError:
            # On Windows, files may be locked by other processes
            # Try with ignore_errors as fallback
            shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                console.print(f"  [dim]Removed:[/dim] {description} ({path})")
                return True
            console.print(f"  [dim]Skipped:[/dim] {description} (files in use)")
            return False
    return False


def _remove_pattern(base_path: Path, pattern: str, description: str) -> int:
    """Remove files/directories matching a pattern.

    Args:
        base_path: Base directory to search.
        pattern: Glob pattern to match.
        description: Description for logging.

    Returns:
        Number of items removed.
    """
    count = 0
    skipped = 0

    # Collect paths first, handling inaccessible directories (e.g., WSL symlinks on Windows)
    paths_to_remove = []
    try:
        for path in base_path.rglob(pattern):
            paths_to_remove.append(path)
    except OSError:
        # rglob may fail on inaccessible paths (e.g., WSL symlinks on Windows)
        pass

    for path in paths_to_remove:
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=False)
            else:
                path.unlink()
            count += 1
        except OSError:
            # On Windows, files may be locked by other processes
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                if not path.exists():
                    count += 1
                else:
                    skipped += 1
            else:
                skipped += 1

    if count > 0:
        console.print(f"  [dim]Removed:[/dim] {count} {description}")
    if skipped > 0:
        console.print(f"  [dim]Skipped:[/dim] {skipped} {description} (files in use)")
    return count


@app.command()  # type: ignore[untyped-decorator]
def clean(
    data: bool = typer.Option(False, "--data", help="Also remove vector store data and indices."),
    all_clean: bool = typer.Option(False, "--all", "-a", help="Remove everything including data."),
) -> None:
    """Clean build artifacts and caches.

    Removes Python bytecode, pytest cache, build directories,
    and other generated files.

    Examples:
        rag-service clean           # Clean artifacts only
        rag-service clean --data    # Also clean vector data
        rag-service clean --all     # Clean everything
    """
    project_root = Path.cwd()
    removed_count = 0

    print_info("Cleaning build artifacts...")
    console.print()

    # Python bytecode
    removed_count += _remove_pattern(project_root, "__pycache__", "__pycache__ directories")
    removed_count += _remove_pattern(project_root, "*.pyc", ".pyc files")
    removed_count += _remove_pattern(project_root, "*.pyo", ".pyo files")

    # Build directories
    for dir_name in ["build", "dist", "*.egg-info", ".eggs"]:
        for path in project_root.glob(dir_name):
            if _remove_directory(path, dir_name):
                removed_count += 1

    # Cache directories
    for cache_dir in [".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", ".coverage"]:
        cache_path = project_root / cache_dir
        if cache_path.exists():
            if cache_path.is_dir():
                if _remove_directory(cache_path, cache_dir):
                    removed_count += 1
            else:
                try:
                    cache_path.unlink()
                    console.print(f"  [dim]Removed:[/dim] {cache_dir}")
                    removed_count += 1
                except OSError:
                    # On Windows, file may be locked by another process
                    console.print(f"  [dim]Skipped:[/dim] {cache_dir} (file in use)")

    # Src egg-info
    for path in (project_root / "src").glob("*.egg-info"):
        if _remove_directory(path, path.name):
            removed_count += 1

    # Data directories (optional)
    if data or all_clean:
        console.print()
        print_warning("Cleaning data directories...")

        data_dirs = [
            project_root / "data" / "vector_store",
            project_root / "data" / "faiss_index",
            project_root / "data" / "chroma_db",
            project_root / "data" / "graph_store",
        ]

        for data_dir in data_dirs:
            if _remove_directory(data_dir, data_dir.name):
                removed_count += 1

    console.print()
    if removed_count > 0:
        print_success(f"Cleaned {removed_count} items.")
    else:
        print_info("Nothing to clean.")
