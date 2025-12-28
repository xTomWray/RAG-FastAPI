# RAG Documentation Service - Local CI Test Runner (PowerShell)
# Runs tests locally on Windows and optionally in Linux Docker containers
#
# Usage:
#   .\tests\docker\run-local-ci.ps1              # Windows tests only
#   .\tests\docker\run-local-ci.ps1 -Linux       # Linux Docker tests only
#   .\tests\docker\run-local-ci.ps1 -All         # Both Windows and Linux
#   .\tests\docker\run-local-ci.ps1 -Full        # Full CI pipeline (lint, typecheck, tests)

param(
    [switch]$Linux,
    [switch]$All,
    [switch]$Full,
    [switch]$SkipBuild,
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  RAG Documentation Service - Local CI" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Track results
$Results = @{
    WindowsLint = $null
    WindowsTypeCheck = $null
    WindowsTests = $null
    LinuxTests = $null
}

function Invoke-WindowsCI {
    param([bool]$FullPipeline)

    Write-Host "`n[Windows] Running CI checks..." -ForegroundColor Yellow
    Push-Location $ProjectRoot

    try {
        if ($FullPipeline) {
            # Lint
            Write-Host "`n--- Lint Check ---" -ForegroundColor Magenta
            & ruff check src/ tests/
            if ($LASTEXITCODE -ne 0) { throw "Lint failed" }
            & ruff format --check src/ tests/
            if ($LASTEXITCODE -ne 0) { throw "Format check failed" }
            $Results.WindowsLint = "PASSED"
            Write-Host "Lint: PASSED" -ForegroundColor Green

            # Type Check
            Write-Host "`n--- Type Check ---" -ForegroundColor Magenta
            & mypy src/ --ignore-missing-imports
            if ($LASTEXITCODE -ne 0) { throw "Type check failed" }
            $Results.WindowsTypeCheck = "PASSED"
            Write-Host "Type Check: PASSED" -ForegroundColor Green
        }

        # Tests
        Write-Host "`n--- Unit Tests ---" -ForegroundColor Magenta
        $env:CLICK_DISABLE_COLORS = "1"
        & pytest tests/unit/ -v --tb=short
        if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
        $Results.WindowsTests = "PASSED"
        Write-Host "Tests: PASSED" -ForegroundColor Green

    } catch {
        Write-Host "Windows CI FAILED: $_" -ForegroundColor Red
        if ($FullPipeline -and -not $Results.WindowsLint) { $Results.WindowsLint = "FAILED" }
        if ($FullPipeline -and -not $Results.WindowsTypeCheck) { $Results.WindowsTypeCheck = "FAILED" }
        if (-not $Results.WindowsTests) { $Results.WindowsTests = "FAILED" }
        return $false
    } finally {
        Pop-Location
    }
    return $true
}

function Invoke-LinuxCI {
    param([bool]$FullPipeline, [bool]$SkipBuild)

    Write-Host "`n[Linux/Docker] Running CI checks..." -ForegroundColor Yellow
    Push-Location $ProjectRoot

    try {
        # Check if Docker is available
        $dockerVersion = & docker --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Docker is not available. Please install Docker Desktop." -ForegroundColor Red
            return $false
        }
        Write-Host "Using: $dockerVersion" -ForegroundColor Gray

        # Create test-results directory
        New-Item -ItemType Directory -Force -Path "$ProjectRoot\test-results" | Out-Null

        $composeFile = "tests/docker/docker-compose.test.yml"
        $buildArg = if ($SkipBuild) { "" } else { "--build" }

        if ($FullPipeline) {
            Write-Host "`n--- Full Linux CI Pipeline ---" -ForegroundColor Magenta
            & docker-compose -f $composeFile up $buildArg --abort-on-container-exit ci-full
        } else {
            Write-Host "`n--- Linux Unit Tests (Python $PythonVersion) ---" -ForegroundColor Magenta
            $service = "test-py$($PythonVersion -replace '\.','')"
            & docker-compose -f $composeFile up $buildArg --abort-on-container-exit $service
        }

        if ($LASTEXITCODE -ne 0) { throw "Linux tests failed" }
        $Results.LinuxTests = "PASSED"
        Write-Host "Linux Tests: PASSED" -ForegroundColor Green

    } catch {
        Write-Host "Linux CI FAILED: $_" -ForegroundColor Red
        $Results.LinuxTests = "FAILED"
        return $false
    } finally {
        Pop-Location
    }
    return $true
}

# Main execution
$Success = $true
$RunWindows = -not $Linux
$RunLinux = $Linux -or $All

if ($RunWindows) {
    if (-not (Invoke-WindowsCI -FullPipeline $Full)) {
        $Success = $false
    }
}

if ($RunLinux) {
    if (-not (Invoke-LinuxCI -FullPipeline $Full -SkipBuild $SkipBuild)) {
        $Success = $false
    }
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Results Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

foreach ($key in $Results.Keys) {
    if ($Results[$key]) {
        $color = if ($Results[$key] -eq "PASSED") { "Green" } else { "Red" }
        Write-Host "  $key : $($Results[$key])" -ForegroundColor $color
    }
}

Write-Host ""
if ($Success) {
    Write-Host "All checks PASSED!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Some checks FAILED!" -ForegroundColor Red
    exit 1
}
