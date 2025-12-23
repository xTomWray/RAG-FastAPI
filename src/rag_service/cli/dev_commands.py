"""Development tool commands: test, lint, format, typecheck, check.

Provides commands for running tests, linting, formatting, and
type checking during development.
"""

import subprocess
import sys

import typer

from rag_service.cli.utils import (
    console,
    print_error,
    print_info,
    print_success,
)

app = typer.Typer(help="Development tool commands")


def _run_command(args: list[str], check: bool = True) -> int:
    """Run a command and return the exit code.

    Args:
        args: Command and arguments to run.
        check: Whether to print error on failure.

    Returns:
        Exit code of the command.
    """
    console.print(f"[dim]Running: {' '.join(args)}[/dim]")
    console.print()
    result = subprocess.run(args)
    if check and result.returncode != 0:
        print_error(f"Command failed with exit code {result.returncode}")
    return result.returncode


@app.command()
def test(
    unit: bool = typer.Option(False, "--unit", "-u", help="Run unit tests only."),
    integration: bool = typer.Option(
        False, "--integration", "-i", help="Run integration tests only."
    ),
    coverage: bool = typer.Option(False, "--coverage", "-c", help="Generate coverage report."),
) -> None:
    """Run tests with pytest.

    By default runs all tests. Use flags to run specific test types
    or generate coverage reports.

    Examples:
        rag-service test              # All tests
        rag-service test --unit       # Unit tests only
        rag-service test --coverage   # With coverage report
    """
    args = [sys.executable, "-m", "pytest"]

    if unit:
        args.append("tests/unit/")
        print_info("Running unit tests...")
    elif integration:
        args.append("tests/integration/")
        print_info("Running integration tests...")
    else:
        args.append("tests/")
        print_info("Running all tests...")

    args.extend(["-v", "--tb=short"])

    if coverage:
        args.extend(
            [
                "--cov=src/rag_service",
                "--cov-report=term-missing",
                "--cov-report=html",
            ]
        )
        print_info("Coverage report will be generated.")

    console.print()
    exit_code = _run_command(args)

    if exit_code == 0:
        print_success("All tests passed!")
        if coverage:
            print_info("Coverage report: htmlcov/index.html")

    raise typer.Exit(exit_code)


@app.command()
def lint(
    fix: bool = typer.Option(False, "--fix", "-f", help="Automatically fix issues."),
) -> None:
    """Run linter (ruff) on the codebase.

    Checks for code style issues and potential bugs.

    Examples:
        rag-service lint         # Check for issues
        rag-service lint --fix   # Auto-fix issues
    """
    args = ["ruff", "check", "src/", "tests/"]

    if fix:
        args.append("--fix")
        print_info("Running linter with auto-fix...")
    else:
        print_info("Running linter...")

    console.print()
    exit_code = _run_command(args)

    if exit_code == 0:
        print_success("No linting issues found!")

    raise typer.Exit(exit_code)


@app.command("format")
def format_code(
    check: bool = typer.Option(False, "--check", help="Check formatting without modifying files."),
) -> None:
    """Format code with ruff.

    Applies consistent formatting to all Python files.

    Examples:
        rag-service format         # Format files
        rag-service format --check # Check only
    """
    args = ["ruff", "format", "src/", "tests/"]

    if check:
        args.append("--check")
        print_info("Checking code formatting...")
    else:
        print_info("Formatting code...")

    console.print()
    exit_code = _run_command(args)

    if exit_code == 0:
        if check:
            print_success("Code is properly formatted!")
        else:
            print_success("Code formatted successfully!")

    raise typer.Exit(exit_code)


@app.command()
def typecheck() -> None:
    """Run type checker (mypy) on the codebase.

    Performs static type analysis to catch type-related bugs.
    Matches CI configuration exactly.
    """
    print_info("Running type checker...")
    console.print()

    # Match CI exactly: mypy src/ --ignore-missing-imports
    args = [sys.executable, "-m", "mypy", "src/", "--ignore-missing-imports"]
    exit_code = _run_command(args)

    if exit_code == 0:
        print_success("No type errors found!")

    raise typer.Exit(exit_code)


@app.command()
def check() -> None:
    """Run all quality checks (lint, format, typecheck).

    Runs the complete suite of code quality checks. This is
    equivalent to running lint, format --check, and typecheck.
    """
    print_info("Running all quality checks...")
    console.print()

    all_passed = True

    # Lint
    console.print("[bold]1. Linting...[/bold]")
    if _run_command(["ruff", "check", "src/", "tests/"], check=False) != 0:
        all_passed = False
    console.print()

    # Format check
    console.print("[bold]2. Format check...[/bold]")
    if _run_command(["ruff", "format", "--check", "src/", "tests/"], check=False) != 0:
        all_passed = False
    console.print()

    # Type check (match CI: mypy src/ --ignore-missing-imports)
    console.print("[bold]3. Type check...[/bold]")
    if (
        _run_command(
            [sys.executable, "-m", "mypy", "src/", "--ignore-missing-imports"], check=False
        )
        != 0
    ):
        all_passed = False
    console.print()

    if all_passed:
        print_success("All checks passed!")
        raise typer.Exit(0)
    else:
        print_error("Some checks failed.")
        raise typer.Exit(1)
