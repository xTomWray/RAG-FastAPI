"""Tests for the CLI commands.

Uses Typer's CliRunner for testing CLI commands without
actually starting servers or running subprocesses.
"""

from typer.testing import CliRunner

from rag_service.cli.main import app

runner = CliRunner()


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
        assert "Start the RAG service" in result.stdout
        assert "--reload" in result.stdout
        assert "--no-ui" in result.stdout
        assert "--workers" in result.stdout

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
        assert "Install project dependencies" in result.stdout
        assert "--dev" in result.stdout
        assert "--gpu" in result.stdout
        assert "--all" in result.stdout

    def test_test_help(self) -> None:
        """Test test command help."""
        result = runner.invoke(app, ["test", "--help"])
        assert result.exit_code == 0
        assert "Run tests with pytest" in result.stdout
        assert "--unit" in result.stdout
        assert "--coverage" in result.stdout

    def test_lint_help(self) -> None:
        """Test lint command help."""
        result = runner.invoke(app, ["lint", "--help"])
        assert result.exit_code == 0
        assert "Run linter" in result.stdout
        assert "--fix" in result.stdout

    def test_format_help(self) -> None:
        """Test format command help."""
        result = runner.invoke(app, ["format", "--help"])
        assert result.exit_code == 0
        assert "Format code" in result.stdout
        assert "--check" in result.stdout

    def test_check_help(self) -> None:
        """Test check command help."""
        result = runner.invoke(app, ["check", "--help"])
        assert result.exit_code == 0
        assert "Run all quality checks" in result.stdout

    def test_clean_help(self) -> None:
        """Test clean command help."""
        result = runner.invoke(app, ["clean", "--help"])
        assert result.exit_code == 0
        assert "Clean build artifacts" in result.stdout
        assert "--data" in result.stdout

    def test_docker_build_help(self) -> None:
        """Test docker-build command help."""
        result = runner.invoke(app, ["docker-build", "--help"])
        assert result.exit_code == 0
        assert "Build Docker image" in result.stdout
        assert "--gpu" in result.stdout

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
