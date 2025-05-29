# TuningFork Development Makefile
# 
# This Makefile provides convenient commands for development tasks
# including testing, code quality, and environment setup.

.PHONY: help install install-dev test test-unit test-integration test-performance test-coverage
.PHONY: quality format lint typecheck clean build docs
.PHONY: full-test pre-commit setup-dev

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED    := \033[31m
GREEN  := \033[32m
YELLOW := \033[33m
BLUE   := \033[34m
RESET  := \033[0m

help: ## Show this help message
	@echo "$(BLUE)TuningFork Development Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Examples:$(RESET)"
	@echo "  make setup-dev    # Set up development environment"
	@echo "  make test         # Run all tests"
	@echo "  make quality      # Run all quality checks"
	@echo "  make full-test    # Run complete test suite"

# Environment Setup
setup-dev: ## Set up development environment with all dependencies
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(RESET)"

install: ## Install package in production mode
	@echo "$(BLUE)Installing TuningFork...$(RESET)"
	pip install -e .
	@echo "$(GREEN)Installation complete!$(RESET)"

install-dev: ## Install package in development mode
	@echo "$(BLUE)Installing TuningFork in development mode...$(RESET)"
	pip install -e ".[dev]"
	@echo "$(GREEN)Development installation complete!$(RESET)"

# Testing
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(RESET)"
	python scripts/test.py

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	python scripts/test.py --type unit -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	python scripts/test.py --type integration -v

test-performance: ## Run performance tests only
	@echo "$(BLUE)Running performance tests...$(RESET)"
	python scripts/test.py --type performance -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	python scripts/test.py --coverage --html
	@echo "$(GREEN)Coverage report generated in htmlcov/$(RESET)"

test-benchmark: ## Run benchmark tests only
	@echo "$(BLUE)Running benchmark tests...$(RESET)"
	python scripts/test.py --benchmark

# Code Quality
quality: ## Run all quality checks (format, lint, typecheck)
	@echo "$(BLUE)Running quality checks...$(RESET)"
	python scripts/test.py --quality

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	python scripts/test.py --format

lint: ## Run linting with flake8
	@echo "$(BLUE)Running linter...$(RESET)"
	flake8 src tests

typecheck: ## Run type checking with mypy
	@echo "$(BLUE)Running type checker...$(RESET)"
	mypy src

# Full Test Suite
full-test: ## Run complete test suite (format, quality, tests with coverage)
	@echo "$(BLUE)Running full test suite...$(RESET)"
	python scripts/test.py --full

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

# Development Utilities
clean: ## Clean up generated files and caches
	@echo "$(BLUE)Cleaning up...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/ coverage.xml
	rm -rf build/ dist/
	@echo "$(GREEN)Cleanup complete!$(RESET)"

build: ## Build package distributions
	@echo "$(BLUE)Building package...$(RESET)"
	python -m build
	@echo "$(GREEN)Package built in dist/$(RESET)"

docs: ## Generate documentation (placeholder)
	@echo "$(BLUE)Generating documentation...$(RESET)"
	@echo "$(YELLOW)Documentation generation not yet implemented$(RESET)"

# Quick Development Workflow
dev-check: ## Quick development check (format + unit tests)
	@echo "$(BLUE)Running quick development check...$(RESET)"
	$(MAKE) format
	$(MAKE) test-unit

dev-ready: ## Check if code is ready for commit (quality + unit tests)
	@echo "$(BLUE)Checking if code is ready for commit...$(RESET)"
	$(MAKE) quality
	$(MAKE) test-unit
	@echo "$(GREEN)Code is ready for commit!$(RESET)"

# CI/CD simulation
ci: ## Simulate CI pipeline (full quality and test suite)
	@echo "$(BLUE)Simulating CI pipeline...$(RESET)"
	$(MAKE) clean
	$(MAKE) quality
	$(MAKE) test-coverage
	@echo "$(GREEN)CI simulation complete!$(RESET)"

# Information targets
info: ## Show development environment information
	@echo "$(BLUE)Development Environment Information$(RESET)"
	@echo ""
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Project root: $$(pwd)"
	@echo ""
	@echo "Installed packages:"
	@pip list | grep -E "(tuningfork|pytest|black|flake8|mypy)" || echo "No development packages found"

check-deps: ## Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(RESET)"
	pip list --outdated

# Database-specific targets (for future phases)
db-setup: ## Set up test databases (placeholder for future phases)
	@echo "$(YELLOW)Database setup not yet implemented (Phase 2)$(RESET)"

db-teardown: ## Tear down test databases (placeholder for future phases)
	@echo "$(YELLOW)Database teardown not yet implemented (Phase 2)$(RESET)"

# Performance monitoring
perf-baseline: ## Establish performance baseline (placeholder)
	@echo "$(BLUE)Establishing performance baseline...$(RESET)"
	python scripts/test.py --type performance --benchmark
	@echo "$(GREEN)Performance baseline established$(RESET)"

# Security checks (future enhancement)
security: ## Run security checks (placeholder)
	@echo "$(YELLOW)Security checks not yet implemented$(RESET)"
	@echo "Consider adding: bandit, safety, etc."

# Release preparation (future enhancement)
release-check: ## Check if ready for release
	@echo "$(BLUE)Checking release readiness...$(RESET)"
	$(MAKE) clean
	$(MAKE) full-test
	$(MAKE) build
	@echo "$(GREEN)Release check complete!$(RESET)"

# Troubleshooting
debug-env: ## Debug development environment issues
	@echo "$(BLUE)Development Environment Debug Information$(RESET)"
	@echo ""
	@echo "Python executable: $$(which python)"
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Virtual environment: $$VIRTUAL_ENV"
	@echo ""
	@echo "Package installation status:"
	@pip show tuningfork 2>/dev/null || echo "TuningFork not installed"
	@echo ""
	@echo "Test dependencies:"
	@pip show pytest pytest-cov pytest-asyncio 2>/dev/null || echo "Test dependencies missing"
	@echo ""
	@echo "Quality dependencies:"
	@pip show black flake8 mypy isort 2>/dev/null || echo "Quality dependencies missing"