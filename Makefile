# RAG Documentation Service - Makefile
# All targets are thin wrappers around: python -m rag_service
#
# This Makefile provides convenience shortcuts for common commands.
# All logic is implemented in the Python CLI for cross-platform support.

.PHONY: install dev gpu all test test-unit test-integration test-cov \
        lint lint-fix format format-check typecheck check pre-commit \
        run run-prod docker-build docker-run clean clean-data help

# Default target
.DEFAULT_GOAL := help

# Auto-detect Python in virtual environment
PYTHON := $(shell \
	if [ -f ./rag_venv/bin/python ]; then echo ./rag_venv/bin/python; \
	elif [ -f ./rag_venv/Scripts/python.exe ]; then echo ./rag_venv/Scripts/python.exe; \
	elif [ -f ./.venv/bin/python ]; then echo ./.venv/bin/python; \
	elif [ -f ./.venv/Scripts/python.exe ]; then echo ./.venv/Scripts/python.exe; \
	else echo python; fi)

# Set PYTHONPATH to include src directory so rag_service module can be found
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

#---------------------------------------------------------------------------
# Installation
#---------------------------------------------------------------------------

install: ## Install production dependencies
	$(PYTHON) -m rag_service install

dev: ## Install development dependencies
	$(PYTHON) -m rag_service install --dev

gpu: ## Install with GPU support
	$(PYTHON) -m rag_service install --gpu

all: ## Install all optional dependencies
	$(PYTHON) -m rag_service install --all

#---------------------------------------------------------------------------
# Testing
#---------------------------------------------------------------------------

test: ## Run all tests
	$(PYTHON) -m rag_service test

test-unit: ## Run unit tests only
	$(PYTHON) -m rag_service test --unit

test-integration: ## Run integration tests only
	$(PYTHON) -m rag_service test --integration

test-cov: ## Run tests with coverage report
	$(PYTHON) -m rag_service test --coverage

#---------------------------------------------------------------------------
# Code Quality
#---------------------------------------------------------------------------

lint: ## Run linter
	$(PYTHON) -m rag_service lint

lint-fix: ## Run linter and fix issues
	$(PYTHON) -m rag_service lint --fix

format: ## Format code
	$(PYTHON) -m rag_service format

format-check: ## Check code formatting
	$(PYTHON) -m rag_service format --check

typecheck: ## Run type checker
	$(PYTHON) -m rag_service typecheck

check: ## Run all checks (lint, format, typecheck)
	$(PYTHON) -m rag_service check

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

#---------------------------------------------------------------------------
# Running
#---------------------------------------------------------------------------

run: ## Run the development server
	$(PYTHON) -m rag_service start --reload

run-prod: ## Run production server (API only)
	$(PYTHON) -m rag_service start --workers 4 --no-ui

#---------------------------------------------------------------------------
# Docker
#---------------------------------------------------------------------------

docker-build: ## Build Docker image
	$(PYTHON) -m rag_service docker-build

docker-build-gpu: ## Build GPU Docker image
	$(PYTHON) -m rag_service docker-build --gpu

docker-run: ## Run Docker container
	$(PYTHON) -m rag_service docker-run

#---------------------------------------------------------------------------
# Cleanup
#---------------------------------------------------------------------------

clean: ## Clean build artifacts
	$(PYTHON) -m rag_service clean

clean-data: ## Clean data directories (careful!)
	$(PYTHON) -m rag_service clean --data

#---------------------------------------------------------------------------
# Help
#---------------------------------------------------------------------------

help: ## Show this help message
	@echo "RAG Documentation Service"
	@echo ""
	@echo "All targets delegate to: $(PYTHON) -m rag_service"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "For more options, run: $(PYTHON) -m rag_service --help"
