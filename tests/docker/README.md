# Local CI Testing

This directory contains Docker configurations for running CI tests locally before pushing to GitHub. This helps catch Linux-specific issues when developing on Windows or macOS.

## Quick Start

```bash
# Run pre-commit checks (Windows native + Linux Docker + Production Docker build)
make pre-commit

# Full CI matrix (8/11 jobs - matches GitHub CI, skips macOS)
make pre-commit-full

# Quick pre-commit (Windows only, faster)
make pre-commit-quick
```

## Available Make Targets

| Target | Description |
|--------|-------------|
| `make pre-commit` | **Standard**: Windows + Linux (single Python) + Docker build |
| `make pre-commit-full` | **Full CI matrix**: 8/11 GitHub jobs (Python 3.11, 3.12, coverage) |
| `make pre-commit-quick` | **Quick**: Windows only (no Docker) |
| `make pre-commit-hooks` | Run pre-commit hooks only |
| `make test-linux` | Linux tests in Docker (Python 3.11) |
| `make test-linux-all` | Linux tests for Python 3.11, 3.12 |
| `make test-windows` | Native Windows tests |
| `make ci-local` | Full Linux CI pipeline in Docker |
| `make ci-local-full` | Full CI on both Windows and Linux + Production build |
| `make docker-build-test` | Test production Docker image builds |

## GitHub CI Job Coverage

| GitHub CI Job | `pre-commit` | `pre-commit-full` |
|---------------|:------------:|:-----------------:|
| lint | ✅ | ✅ |
| type-check | ✅ | ✅ |
| test (ubuntu, py3.11) | ✅ | ✅ |
| test (ubuntu, py3.12) | ❌ | ✅ |
| test (windows, py3.11-3.12) | ✅ (local Python) | ✅ (local Python) |
| test (macos, py3.11-3.12) | ❌ | ❌ (skipped) |
| coverage | ❌ | ✅ |
| docker build | ✅ | ✅ |
| **Total** | **~5/11** | **8/11** |

## Files

```
tests/docker/
├── Dockerfile.linux-test      # Linux test environment
├── docker-compose.test.yml    # Multi-service orchestration
├── run-local-ci.ps1           # PowerShell script (Windows)
├── run-local-ci.sh            # Bash script (Linux/macOS)
└── README.md                  # This file
```

## Docker Compose Services

The `docker-compose.test.yml` provides these services:

| Service | Description |
|---------|-------------|
| `test-py311` | Python 3.11 unit tests (default) |
| `test-py312` | Python 3.12 unit tests |
| `lint` | Ruff linter + formatter check |
| `typecheck` | mypy type checking |
| `coverage` | Run tests with coverage report |
| `integration` | Integration tests only |
| `ci-full` | Full CI pipeline (lint → typecheck → tests) |

## Usage Examples

### Basic Local CI Check

```bash
# Before committing, run:
make pre-commit
```

### Test Specific Python Version

```bash
# Test with Python 3.12 only
docker-compose -f tests/docker/docker-compose.test.yml up --build test-py312
```

### Run Full CI Pipeline

```bash
# Mimics GitHub Actions exactly
make ci-local
```

### PowerShell Script (Windows)

```powershell
# Windows tests only
.\tests\docker\run-local-ci.ps1

# Linux Docker tests only
.\tests\docker\run-local-ci.ps1 -Linux

# Both Windows and Linux
.\tests\docker\run-local-ci.ps1 -All

# Full CI pipeline
.\tests\docker\run-local-ci.ps1 -Full
```

### Bash Script (Linux/macOS)

```bash
# Default Linux tests
./tests/docker/run-local-ci.sh

# Full CI pipeline
./tests/docker/run-local-ci.sh --full

# All Python versions
./tests/docker/run-local-ci.sh --all-py
```

## Test Results

Test results are saved to `test-results/` directory (git-ignored):

- `py311-results.xml` - Python 3.11 JUnit XML
- `py312-results.xml` - Python 3.12 JUnit XML
- `linux-results.xml` - Full CI run results
- `coverage.xml` - Coverage report

## Requirements

- **Docker Desktop** (Windows/macOS) or Docker Engine (Linux)
- **docker-compose** (usually included with Docker Desktop)

## Comparison with GitHub Actions

| GitHub CI Job | Local Equivalent | Included in |
|---------------|------------------|-------------|
| `lint` | `make lint` or Docker `lint` service | `pre-commit`, `pre-commit-full` |
| `type-check` | `make typecheck` or Docker `typecheck` service | `pre-commit`, `pre-commit-full` |
| `test` (ubuntu matrix) | `make test-linux-all` | `pre-commit-full` |
| `test` (windows matrix) | `make test-windows` | `pre-commit`, `pre-commit-full` |
| `test` (macos matrix) | N/A (requires macOS hardware) | ❌ |
| `coverage` | Docker `coverage` service | `pre-commit-full` |
| `docker` (build) | `make docker-build-test` | `pre-commit`, `pre-commit-full` |

**Recommendations:**
- Use `make pre-commit` for quick validation before commits (~3-5 min)
- Use `make pre-commit-full` before pushing to catch multi-Python issues (~10-15 min)
- macOS-specific issues are rare if Windows + Linux pass

## Troubleshooting

### Docker not found
Ensure Docker Desktop is installed and running.

### Permission denied (Linux/macOS)
```bash
chmod +x tests/docker/run-local-ci.sh
```

### Build cache issues
```bash
docker-compose -f tests/docker/docker-compose.test.yml build --no-cache
```

### View container logs
```bash
docker-compose -f tests/docker/docker-compose.test.yml logs
```
