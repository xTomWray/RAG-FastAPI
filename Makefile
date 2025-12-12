.PHONY: install dev test test-unit test-integration lint format type-check pre-commit run clean help

# Default target
.DEFAULT_GOAL := help

# Python executable
PYTHON := python
PIP := pip

#---------------------------------------------------------------------------
# Installation
#---------------------------------------------------------------------------

install: ## Install production dependencies
	$(PIP) install -e .

dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

gpu: ## Install with GPU support
	$(PIP) install -e ".[gpu]"

all: ## Install all optional dependencies
	$(PIP) install -e ".[all]"

#---------------------------------------------------------------------------
# Testing
#---------------------------------------------------------------------------

test: ## Run all tests
	pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests only
	pytest tests/integration/ -v --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src/rag_service --cov-report=term-missing --cov-report=html

#---------------------------------------------------------------------------
# Code Quality
#---------------------------------------------------------------------------

lint: ## Run linter
	ruff check src/ tests/

lint-fix: ## Run linter and fix issues
	ruff check src/ tests/ --fix

format: ## Format code
	ruff format src/ tests/

format-check: ## Check code formatting
	ruff format src/ tests/ --check

type-check: ## Run type checker
	mypy src/

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

check: lint format-check type-check ## Run all checks

#---------------------------------------------------------------------------
# Running
#---------------------------------------------------------------------------

run: ## Run the development server
	uvicorn rag_service.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run production server
	uvicorn rag_service.main:app --host 0.0.0.0 --port 8000 --workers 4

#---------------------------------------------------------------------------
# Docker
#---------------------------------------------------------------------------

docker-build: ## Build Docker image
	docker build -t rag-documentation-service .

docker-run: ## Run Docker container
	docker run -p 8000:8000 -v $(PWD)/data:/app/data rag-documentation-service

docker-compose-up: ## Start with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop docker-compose
	docker-compose down

#---------------------------------------------------------------------------
# Cleanup
#---------------------------------------------------------------------------

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-data: ## Clean data directories (careful!)
	rm -rf data/index/
	rm -rf data/chroma/

#---------------------------------------------------------------------------
# Help
#---------------------------------------------------------------------------

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

