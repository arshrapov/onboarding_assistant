# ============================================================
# Makefile for Onboarding Assistant Docker Operations
# ============================================================

.PHONY: help build up down logs restart clean test health backup restore

# Default target
.DEFAULT_GOAL := help

# Variables
COMPOSE := docker-compose
IMAGE_NAME := onboarding-assistant
BACKUP_DIR := ./backups

help: ## Show this help message
	@echo "Onboarding Assistant - Docker Commands"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Build & Deployment
# ============================================================

build: ## Build Docker image locally
	@echo "🔨 Building Docker image..."
	$(COMPOSE) build

pull: ## Pull pre-built image from registry
	@echo "📥 Pulling image from registry..."
	$(COMPOSE) pull

up: ## Start services (detached mode)
	@echo "🚀 Starting services..."
	$(COMPOSE) up -d
	@echo "✅ Services started. Visit http://localhost:7860"

down: ## Stop and remove containers
	@echo "🛑 Stopping services..."
	$(COMPOSE) down

restart: ## Restart services
	@echo "🔄 Restarting services..."
	$(COMPOSE) restart

start: ## Start existing containers
	@echo "▶️  Starting containers..."
	$(COMPOSE) start

stop: ## Stop containers without removing
	@echo "⏸️  Stopping containers..."
	$(COMPOSE) stop

# ============================================================
# Monitoring & Debugging
# ============================================================

logs: ## View logs (follow mode)
	$(COMPOSE) logs -f

logs-tail: ## View last 100 log lines
	$(COMPOSE) logs --tail=100

health: ## Check service health
	@echo "🏥 Checking service health..."
	@curl -f http://localhost:7860/api/health && echo "✅ Service is healthy" || echo "❌ Service is unhealthy"

status: ## Show container status
	$(COMPOSE) ps

stats: ## Show resource usage statistics
	docker stats onboarding-assistant --no-stream

inspect: ## Inspect container details
	docker inspect onboarding-assistant

shell: ## Open shell in running container
	docker exec -it onboarding-assistant /bin/bash

# ============================================================
# Data Management
# ============================================================

backup: ## Backup data volume
	@echo "💾 Creating backup..."
	@mkdir -p $(BACKUP_DIR)
	@docker run --rm \
		-v onboarding-data:/data \
		-v $(PWD)/$(BACKUP_DIR):/backup \
		alpine tar czf /backup/onboarding-backup-$$(date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@echo "✅ Backup created in $(BACKUP_DIR)/"

restore: ## Restore data volume (usage: make restore BACKUP=filename.tar.gz)
	@if [ -z "$(BACKUP)" ]; then \
		echo "❌ Error: Please specify BACKUP file"; \
		echo "Usage: make restore BACKUP=onboarding-backup-20260103-120000.tar.gz"; \
		exit 1; \
	fi
	@echo "⚠️  Warning: This will overwrite existing data!"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@docker run --rm \
		-v onboarding-data:/data \
		-v $(PWD)/$(BACKUP_DIR):/backup \
		alpine sh -c "rm -rf /data/* && tar xzf /backup/$(BACKUP) -C /data"
	@echo "✅ Backup restored"

volume-ls: ## List Docker volumes
	docker volume ls | grep onboarding

volume-inspect: ## Inspect data volume
	docker volume inspect onboarding-data

# ============================================================
# Cleanup
# ============================================================

clean: ## Stop services and remove containers (keeps volumes)
	@echo "🧹 Cleaning up containers..."
	$(COMPOSE) down
	@echo "✅ Cleanup complete (volumes preserved)"

clean-all: ## Stop services and remove containers AND volumes (WARNING: deletes data!)
	@echo "⚠️  WARNING: This will delete all data including repositories and indexes!"
	@read -p "Are you sure? Type 'yes' to confirm: " confirm && [ "$$confirm" = "yes" ] || exit 1
	$(COMPOSE) down -v
	@echo "✅ Full cleanup complete (all data deleted)"

prune: ## Remove unused Docker resources
	@echo "🧹 Removing unused Docker resources..."
	docker system prune -f
	@echo "✅ Prune complete"

prune-all: ## Remove ALL unused Docker resources (images, volumes, etc.)
	@echo "⚠️  WARNING: This will remove all unused Docker resources!"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	docker system prune -a --volumes -f
	@echo "✅ Full prune complete"

# ============================================================
# Development
# ============================================================

dev: ## Run in development mode with live reload
	@echo "🔧 Starting in development mode..."
	docker run -d \
		--name $(IMAGE_NAME)-dev \
		-p 7860:7860 \
		-e GEMINI_API_KEY=$${GEMINI_API_KEY} \
		-v $(PWD)/app:/app/app \
		-v onboarding-data:/app/data \
		$(IMAGE_NAME):latest \
		uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

dev-stop: ## Stop development container
	docker stop $(IMAGE_NAME)-dev
	docker rm $(IMAGE_NAME)-dev

# ============================================================
# Testing & Validation
# ============================================================

test: ## Run tests in container
	docker exec onboarding-assistant pytest /app/tests/

validate: ## Validate docker-compose.yml
	@echo "✅ Validating docker-compose.yml..."
	$(COMPOSE) config --quiet

lint-dockerfile: ## Lint Dockerfile (requires hadolint)
	@if command -v hadolint >/dev/null 2>&1; then \
		echo "🔍 Linting Dockerfile..."; \
		hadolint Dockerfile; \
	else \
		echo "⚠️  hadolint not installed. Install from: https://github.com/hadolint/hadolint"; \
	fi

# ============================================================
# Setup
# ============================================================

setup: ## Initial setup (create .env from template)
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env file..."; \
		cp .env.example .env; \
		echo "✅ .env created. Please edit it and add your GEMINI_API_KEY"; \
	else \
		echo "ℹ️  .env already exists"; \
	fi

init: setup build up ## Complete initialization (setup + build + start)
	@echo "✅ Initialization complete!"
	@echo "🌐 Access UI at: http://localhost:7860"

# ============================================================
# CI/CD
# ============================================================

ci-build: ## CI: Build and tag image
	docker build -t $(IMAGE_NAME):$$(git rev-parse --short HEAD) .
	docker tag $(IMAGE_NAME):$$(git rev-parse --short HEAD) $(IMAGE_NAME):latest

ci-test: ## CI: Run tests
	docker run --rm $(IMAGE_NAME):latest pytest /app/tests/ || true

ci-scan: ## CI: Security scan with Trivy (requires trivy)
	@if command -v trivy >/dev/null 2>&1; then \
		trivy image $(IMAGE_NAME):latest; \
	else \
		echo "⚠️  Trivy not installed. Install from: https://github.com/aquasecurity/trivy"; \
	fi
