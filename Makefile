# ΨQRH Quaternionic Transformer Framework - Unified Makefile
# Author: Klenio Araujo Padilha
# Project: Reformulating Transformers for LLMs

# Configuration
DOCKER_IMAGE := psiqrh-transformer
DOCKER_TAG := latest
COMPOSE_FILE := docker-compose.yml
PROJECT_NAME := psiqrh

# Colors for output
RESET := \033[0m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m

# Helper function for colored output
define print_info
	@echo "$(BLUE)[INFO]$(RESET) $(1)"
endef

define print_success
	@echo "$(GREEN)[SUCCESS]$(RESET) $(1)"
endef

define print_warning
	@echo "$(YELLOW)[WARNING]$(RESET) $(1)"
endef

define print_error
	@echo "$(RED)[ERROR]$(RESET) $(1)"
endef

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "$(MAGENTA)ΨQRH Quaternionic Transformer Framework$(RESET)"
	@echo "$(CYAN)Unified Makefile for Docker and Project Management$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(CYAN)Quick Start:$(RESET)"
	@echo "  make setup     # Build and start everything"
	@echo "  make test      # Run all tests"
	@echo "  make shell     # Interactive development"
	@echo "  make clean     # Clean everything"

# Docker Management
.PHONY: build
build: ## Build Docker image
	$(call print_info,"Building ΨQRH Docker image...")
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	$(call print_success,"Docker image built successfully!")

.PHONY: up
up: ## Start all services with docker-compose
	$(call print_info,"Starting ΨQRH services...")
	@docker-compose -f $(COMPOSE_FILE) up -d
	$(call print_success,"Services started! Use 'make shell' to access container")

.PHONY: down
down: ## Stop all services
	$(call print_info,"Stopping ΨQRH services...")
	@docker-compose -f $(COMPOSE_FILE) down
	$(call print_success,"Services stopped!")

.PHONY: restart
restart: down up ## Restart all services

.PHONY: setup
setup: build up ## Complete setup: build and start services
	$(call print_success,"ΨQRH Framework ready! Use 'make test' to validate")

# Development Environment
.PHONY: shell
shell: ## Open interactive shell in container
	$(call print_info,"Opening interactive shell...")
	@docker-compose -f $(COMPOSE_FILE) exec psiqrh bash

.PHONY: shell-root
shell-root: ## Open interactive shell as root
	$(call print_info,"Opening root shell...")
	@docker-compose -f $(COMPOSE_FILE) exec --user root psiqrh bash

.PHONY: logs
logs: ## View container logs
	@docker-compose -f $(COMPOSE_FILE) logs -f psiqrh

.PHONY: status
status: ## Show container and volume status
	$(call print_info,"Container status:")
	@docker-compose -f $(COMPOSE_FILE) ps
	@echo ""
	$(call print_info,"Volume usage:")
	@docker volume ls | grep $(PROJECT_NAME) || echo "No ΨQRH volumes found"

# Testing and Validation
.PHONY: test
test: ## Run complete test suite
	$(call print_info,"Running ΨQRH comprehensive test suite...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh bash -c " \
		echo '=== ΨQRH Framework Test Suite ===' && \
		python simple_validation_test.py && \
		echo '--- Comprehensive Integration Test ---' && \
		python comprehensive_integration_test.py && \
		echo '--- Robust Validation Test ---' && \
		python robust_validation_test.py && \
		echo '=== All tests completed successfully ==='"
	$(call print_success,"All tests passed!")

.PHONY: test-simple
test-simple: ## Run simple validation test
	$(call print_info,"Running simple validation test...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python simple_validation_test.py

.PHONY: test-comprehensive
test-comprehensive: ## Run comprehensive integration test
	$(call print_info,"Running comprehensive integration test...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python comprehensive_integration_test.py

.PHONY: test-robust
test-robust: ## Run robust validation test
	$(call print_info,"Running robust validation test...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python robust_validation_test.py

.PHONY: test-4d
test-4d: ## Run 4D Unitary Layer tests
	$(call print_info,"Running 4D Unitary Layer tests...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python test_4d_unitary_layer.py

# Demonstrations and Analysis
.PHONY: demo
demo: ## Run all demonstrations
	$(call print_info,"Running ΨQRH demonstrations...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh bash -c " \
		echo '=== ΨQRH Framework Demonstrations ===' && \
		python fractal_pytorch_integration.py && \
		echo '--- Spider Evolution Simulation ---' && \
		python emergence_simulation.py && \
		echo '--- Habitat Demo ---' && \
		python show_habitat_demo.py && \
		echo '=== Demonstrations completed ==='"
	$(call print_success,"Demonstrations completed!")

.PHONY: fractal
fractal: ## Run fractal dimension analysis
	$(call print_info,"Running fractal dimension analysis...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python needle_fractal_dimension.py
	$(call print_success,"Fractal analysis completed! Check images/ directory")

.PHONY: spider
spider: ## Run spider evolution simulation
	$(call print_info,"Running spider evolution simulation...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python emergence_simulation.py
	$(call print_success,"Spider simulation completed!")

.PHONY: integration
integration: ## Run fractal-PyTorch integration
	$(call print_info,"Running fractal-PyTorch integration...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python fractal_pytorch_integration.py

.PHONY: habitat
habitat: ## Run habitat demonstration
	$(call print_info,"Running habitat demonstration...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python show_habitat_demo.py

.PHONY: ecosystem
ecosystem: ## Start live ecosystem server
	$(call print_info,"Starting live ecosystem server...")
	@docker-compose -f $(COMPOSE_FILE) run --rm -p 8080:8080 psiqrh python live_ecosystem_server.py

# Development and Utilities
.PHONY: install
install: ## Install additional Python packages
	$(call print_info,"Installing packages in container...")
	@docker-compose -f $(COMPOSE_FILE) exec psiqrh pip install $(PACKAGES)

.PHONY: jupyter
jupyter: ## Start Jupyter notebook server
	$(call print_info,"Starting Jupyter notebook server...")
	@docker-compose -f $(COMPOSE_FILE) run --rm -p 8888:8888 psiqrh jupyter notebook --ip=0.0.0.0 --allow-root --no-browser

.PHONY: format
format: ## Format Python code
	$(call print_info,"Formatting Python code...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python -m black . || echo "Black not installed"

.PHONY: lint
lint: ## Run linting
	$(call print_info,"Running code linting...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python -m flake8 . || echo "Flake8 not installed"

# Data and Results Management
.PHONY: results
results: ## Show generated results
	$(call print_info,"Generated images:")
	@docker-compose -f $(COMPOSE_FILE) exec psiqrh ls -la images/ || echo "No images directory"
	@echo ""
	$(call print_info,"Generated logs:")
	@docker-compose -f $(COMPOSE_FILE) exec psiqrh ls -la logs/ || echo "No logs directory"

.PHONY: backup
backup: ## Backup volumes and results
	$(call print_info,"Creating backup of volumes...")
	@mkdir -p backup
	@docker run --rm -v $(PROJECT_NAME)_psiqrh-models:/source -v $(PWD)/backup:/backup alpine tar czf /backup/models-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /source .
	@docker run --rm -v $(PROJECT_NAME)_psiqrh-images:/source -v $(PWD)/backup:/backup alpine tar czf /backup/images-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /source .
	$(call print_success,"Backup completed in backup/ directory")

# Cleanup
.PHONY: clean
clean: ## Clean containers and volumes
	$(call print_warning,"This will remove all ΨQRH containers and volumes")
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	$(call print_info,"Cleaning containers and volumes...")
	@docker-compose -f $(COMPOSE_FILE) down -v
	@docker system prune -f
	$(call print_success,"Cleanup completed!")

.PHONY: clean-images
clean-images: ## Clean generated images
	$(call print_info,"Cleaning generated images...")
	@docker-compose -f $(COMPOSE_FILE) exec psiqrh rm -rf images/*.png || true
	$(call print_success,"Images cleaned!")

.PHONY: clean-logs
clean-logs: ## Clean log files
	$(call print_info,"Cleaning log files...")
	@docker-compose -f $(COMPOSE_FILE) exec psiqrh rm -rf logs/*.log || true
	$(call print_success,"Logs cleaned!")

.PHONY: reset
reset: clean setup ## Complete reset: clean and rebuild everything

# GPU Support
.PHONY: gpu-setup
gpu-setup: ## Setup with GPU support
	$(call print_info,"Setting up with GPU support...")
	@docker-compose -f $(COMPOSE_FILE) -f docker-compose.gpu.yml up --build -d
	$(call print_success,"GPU-enabled services started!")

.PHONY: gpu-test
gpu-test: ## Test GPU functionality
	$(call print_info,"Testing GPU functionality...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# CI/CD Support
.PHONY: ci-test
ci-test: build ## CI/CD test pipeline
	$(call print_info,"Running CI/CD test pipeline...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh bash -c " \
		python simple_validation_test.py && \
		python comprehensive_integration_test.py"
	$(call print_success,"CI/CD tests passed!")

.PHONY: docker-push
docker-push: build ## Push Docker image to registry
	$(call print_info,"Pushing Docker image...")
	@docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@docker push $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	$(call print_success,"Image pushed to registry!")

# Documentation
.PHONY: docs
docs: ## Generate documentation
	$(call print_info,"Generating documentation...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python -c "print('ΨQRH Framework Documentation'); print('See README.md and README.Docker.md')"

# Advanced Operations
.PHONY: benchmark
benchmark: ## Run performance benchmarks
	$(call print_info,"Running performance benchmarks...")
	@docker-compose -f $(COMPOSE_FILE) run --rm psiqrh python debug_performance.py

.PHONY: validate-all
validate-all: test demo benchmark ## Run complete validation suite
	$(call print_success,"Complete validation finished!")

.PHONY: production
production: ## Build production-ready image
	$(call print_info,"Building production image...")
	@docker build --target production -t $(DOCKER_IMAGE):prod .
	$(call print_success,"Production image ready!")

# Information
.PHONY: info
info: ## Show system information
	@echo "$(MAGENTA)ΨQRH Framework Information$(RESET)"
	@echo "Docker Image: $(DOCKER_IMAGE):$(DOCKER_TAG)"
	@echo "Compose File: $(COMPOSE_FILE)"
	@echo "Project Name: $(PROJECT_NAME)"
	@echo ""
	@echo "$(CYAN)Docker Version:$(RESET)"
	@docker --version
	@echo ""
	@echo "$(CYAN)Docker Compose Version:$(RESET)"
	@docker-compose --version
	@echo ""
	@echo "$(CYAN)Available Services:$(RESET)"
	@docker-compose -f $(COMPOSE_FILE) config --services

# Quick aliases for common operations
.PHONY: run
run: up ## Alias for 'up'

.PHONY: stop
stop: down ## Alias for 'down'

.PHONY: exec
exec: shell ## Alias for 'shell'

.PHONY: bash
bash: shell ## Alias for 'shell'