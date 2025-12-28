#!/bin/bash
# RAG Documentation Service - Local CI Test Runner (Bash)
# Runs tests in Linux Docker containers
#
# Usage:
#   ./tests/docker/run-local-ci.sh              # Default: Linux tests
#   ./tests/docker/run-local-ci.sh --full       # Full CI pipeline
#   ./tests/docker/run-local-ci.sh --all-py     # All Python versions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/tests/docker/docker-compose.test.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
FULL=false
ALL_PY=false
SKIP_BUILD=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --full) FULL=true ;;
        --all-py) ALL_PY=true ;;
        --skip-build) SKIP_BUILD=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo -e "\n${CYAN}========================================"
echo "  RAG Documentation Service - Local CI"
echo -e "========================================${NC}\n"

cd "$PROJECT_ROOT"

# Create test-results directory
mkdir -p test-results

BUILD_ARG=""
if [ "$SKIP_BUILD" = false ]; then
    BUILD_ARG="--build"
fi

if [ "$FULL" = true ]; then
    echo -e "${YELLOW}[Linux/Docker] Running Full CI Pipeline...${NC}\n"
    docker-compose -f "$COMPOSE_FILE" up $BUILD_ARG --abort-on-container-exit ci-full
elif [ "$ALL_PY" = true ]; then
    echo -e "${YELLOW}[Linux/Docker] Running tests for all Python versions...${NC}\n"
    docker-compose -f "$COMPOSE_FILE" up $BUILD_ARG --abort-on-container-exit test-py310 test-py311 test-py312
else
    echo -e "${YELLOW}[Linux/Docker] Running tests (Python 3.11)...${NC}\n"
    docker-compose -f "$COMPOSE_FILE" up $BUILD_ARG --abort-on-container-exit test-py311
fi

echo -e "\n${GREEN}All Linux CI checks PASSED!${NC}\n"
