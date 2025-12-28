"""Tests for the CLI commands.

Uses Typer's CliRunner for testing CLI commands without
actually starting servers or running subprocesses.
"""

import re

from typer.testing import CliRunner

from rag_service.cli.main import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text.

    This is needed because Rich/Typer output may contain color codes
    that interfere with string matching in tests.
    """
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_pattern.sub("", text)


class TestCLIHelp:
    """Test CLI help output."""

    def test_main_help(self) -> None:
        """Test that --help shows all commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "RAG Documentation Service CLI" in result.stdout
        assert "start" in result.stdout
        assert "stop" in result.stdout
        assert "test" in result.stdout
        assert "install" in result.stdout

    def test_start_help(self) -> None:
        """Test start command help."""
        result = runner.invoke(app, ["start", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "Start the RAG service" in output
        assert "--reload" in output
        assert "--no-ui" in output
        assert "--workers" in output

    def test_stop_help(self) -> None:
        """Test stop command help."""
        result = runner.invoke(app, ["stop", "--help"])
        assert result.exit_code == 0
        assert "Stop the running RAG service" in result.stdout

    def test_status_help(self) -> None:
        """Test status command help."""
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "Show the current service status" in result.stdout

    def test_install_help(self) -> None:
        """Test install command help."""
        result = runner.invoke(app, ["install", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "Install project dependencies" in output
        assert "--dev" in output
        assert "--gpu" in output
        assert "--all" in output

    def test_test_help(self) -> None:
        """Test test command help."""
        result = runner.invoke(app, ["test", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "Run tests with pytest" in output
        assert "--unit" in output
        assert "--coverage" in output

    def test_lint_help(self) -> None:
        """Test lint command help."""
        result = runner.invoke(app, ["lint", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "Run linter" in output
        assert "--fix" in output

    def test_format_help(self) -> None:
        """Test format command help."""
        result = runner.invoke(app, ["format", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "Format code" in output
        assert "--check" in output

    def test_check_help(self) -> None:
        """Test check command help."""
        result = runner.invoke(app, ["check", "--help"])
        assert result.exit_code == 0
        assert "Run all quality checks" in result.stdout

    def test_clean_help(self) -> None:
        """Test clean command help."""
        result = runner.invoke(app, ["clean", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "Clean build artifacts" in output
        assert "--data" in output

    def test_docker_build_help(self) -> None:
        """Test docker-build command help."""
        result = runner.invoke(app, ["docker-build", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "Build Docker image" in output
        assert "--gpu" in output

    def test_docker_run_help(self) -> None:
        """Test docker-run command help."""
        result = runner.invoke(app, ["docker-run", "--help"])
        assert result.exit_code == 0
        assert "Run the RAG service in a Docker container" in result.stdout


class TestStatusCommand:
    """Test status command behavior."""

    def test_status_no_service_running(self) -> None:
        """Test status when no service is running."""
        result = runner.invoke(app, ["status"])
        # Should complete without error
        assert result.exit_code == 0
        assert "RAG Documentation Service Status" in result.stdout


class TestCleanCommand:
    """Test clean command behavior."""

    def test_clean_basic(self) -> None:
        """Test basic clean command."""
        result = runner.invoke(app, ["clean"])
        # Should complete without error
        assert result.exit_code == 0
        # Should report cleaning activity
        assert "Cleaning" in result.stdout or "Nothing to clean" in result.stdout
