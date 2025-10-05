# 🌌 ΨQRH Docker Operations with Carl Sagan Spectral Knowledge
#
# This Makefile manages Docker operations for the ΨQRH cognitive system
# with embedded Carl Sagan knowledge for scientific skepticism.
#
# "Science is a candle in the dark" - Carl Sagan

# Docker image configuration
IMAGE_NAME := psiqrh-sagan
IMAGE_TAG := latest
DOCKERFILE := ops/docker/Dockerfile
CONTEXT := .

# Container configuration
CONTAINER_NAME := psiqrh-cognitive-server
PORT := 8000
HOST_PORT := 8080

# Knowledge base configuration
KNOWLEDGE_BASE := sagan_spectral
DATA_VOLUME := psiqrh-data

.PHONY: help docker-build docker-run docker-stop docker-clean knowledge-convert interactive logs status

help: ## Show this help message
	@echo "🌌 ΨQRH Docker Operations with Carl Sagan Knowledge"
	@echo "=================================================="
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🧠 Knowledge Integration:"
	@echo "  • Carl Sagan's wisdom embedded during build"
	@echo "  • Scientific skepticism as cognitive foundation"
	@echo "  • 'Extraordinary claims require extraordinary evidence'"
	@echo ""
	@echo "🔬 Epistemic Integrity Operations:"
	@echo "  • integrity-setup: Establish reference integrity hash"
	@echo "  • integrity-verify: Verify current epistemic integrity"
	@echo "  • integrity-certificate: Generate integrity certificate"
	@echo "  • integrity-status: Check integrity status"
	@echo "  • integrity-test: Run standalone integrity test"
	@echo "  • integrity-report: Generate detailed analysis report"
	@echo ""

docker-build: ## Build ΨQRH Docker image with embedded Sagan knowledge
	@echo "🌌 Building ΨQRH Docker image with Carl Sagan knowledge..."
	@echo "🧠 This will embed scientific skepticism into the system"
	@docker build \
		-f $(DOCKERFILE) \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		--progress=plain \
		$(CONTEXT)
	@echo ""
	@echo "✅ ΨQRH image built successfully with embedded Sagan knowledge"
	@echo "🔍 Verifying knowledge embedding..."
	@docker run --rm $(IMAGE_NAME):$(IMAGE_TAG) \
		python3 -c "import json; kb=json.load(open('/app/data/knowledge_bases/sagan_spectral.kb')); print('🧠 Carl Sagan knowledge verified:', len(kb.get('core_principles', {})), 'principles loaded')"

docker-run: ## Run ΨQRH cognitive server with Sagan knowledge
	@echo "🚀 Starting ΨQRH Cognitive Server with Carl Sagan knowledge..."
	@docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(HOST_PORT):$(PORT) \
		-v $(DATA_VOLUME):/app/data \
		-e ΨQRH_MODE=interactive \
		-e KNOWLEDGE_BASE=$(KNOWLEDGE_BASE) \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 /app/src/conceptual/live_ecosystem_server.py \
		--host=0.0.0.0 \
		--port=$(PORT) \
		--knowledge-base=/app/data/knowledge_bases/sagan_spectral.kb
	@echo ""
	@echo "✅ ΨQRH Cognitive Server started with Carl Sagan knowledge"
	@echo "🌐 Web Interface: http://localhost:$(HOST_PORT)"
	@echo "🧠 Sagan Knowledge API: http://localhost:$(HOST_PORT)/api/sagan/knowledge"
	@echo "🔍 Skeptical Analysis API: http://localhost:$(HOST_PORT)/api/sagan/analysis?claim=YOUR_CLAIM"
	@echo "💭 'Extraordinary claims require extraordinary evidence' - Carl Sagan"

interactive: ## Run interactive ΨQRH shell with Sagan knowledge
	@echo "🌌 Starting interactive ΨQRH shell with Carl Sagan knowledge..."
	@docker run -it --rm \
		-p $(HOST_PORT):$(PORT) \
		-v $(DATA_VOLUME):/app/data \
		-e ΨQRH_MODE=interactive \
		-e KNOWLEDGE_BASE=$(KNOWLEDGE_BASE) \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		/bin/bash
	@echo "🧠 Welcome to the ΨQRH cognitive environment"
	@echo "🕯️ 'Science is a candle in the dark' - Carl Sagan"

knowledge-convert: ## Convert Carl Sagan PDF to spectral knowledge (standalone)
	@echo "🌌 Converting Carl Sagan's work to spectral knowledge..."
	@docker run --rm \
		-v $(DATA_VOLUME):/app/data \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 /app/src/cognitive/sagan_spectral_converter.py
	@echo "✅ Carl Sagan spectral knowledge conversion complete"

docker-stop: ## Stop ΨQRH cognitive server
	@echo "🛑 Stopping ΨQRH cognitive server..."
	@docker stop $(CONTAINER_NAME) || true
	@docker rm $(CONTAINER_NAME) || true
	@echo "✅ ΨQRH cognitive server stopped"

docker-clean: ## Clean Docker resources and volumes
	@echo "🧹 Cleaning Docker resources..."
	@docker stop $(CONTAINER_NAME) || true
	@docker rm $(CONTAINER_NAME) || true
	@docker rmi $(IMAGE_NAME):$(IMAGE_TAG) || true
	@docker volume rm $(DATA_VOLUME) || true
	@docker system prune -f
	@echo "✅ Docker cleanup complete"

logs: ## Show ΨQRH server logs
	@echo "📋 ΨQRH Cognitive Server logs:"
	@docker logs -f $(CONTAINER_NAME)

status: ## Check ΨQRH server status
	@echo "📊 ΨQRH Cognitive Server Status:"
	@echo "================================"
	@docker ps --filter name=$(CONTAINER_NAME) --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "🧠 Carl Sagan Knowledge Status:"
	@docker exec $(CONTAINER_NAME) \
		python3 -c "import json; kb=json.load(open('/app/data/knowledge_bases/sagan_spectral.kb')); print('✅ Knowledge base loaded with', len(kb.get('core_principles', {})), 'core principles')" \
		2>/dev/null || echo "❌ Knowledge base not accessible"
	@echo ""
	@echo "🌐 Server endpoints:"
	@echo "  • Web Interface: http://localhost:$(HOST_PORT)"
	@echo "  • Ecosystem API: http://localhost:$(HOST_PORT)/api/ecosystem/status"
	@echo "  • Sagan Knowledge: http://localhost:$(HOST_PORT)/api/sagan/knowledge"

test-knowledge: ## Test Carl Sagan knowledge integration
	@echo "🧪 Testing Carl Sagan knowledge integration..."
	@echo ""
	@echo "🔍 Testing skeptical analysis with sample claim:"
	@curl -s "http://localhost:$(HOST_PORT)/api/sagan/analysis?claim=All%20swans%20are%20white" | \
		python3 -m json.tool || echo "❌ Server not running or API unavailable"
	@echo ""
	@echo "🧠 Testing knowledge base access:"
	@curl -s "http://localhost:$(HOST_PORT)/api/sagan/knowledge" | \
		python3 -c "import sys,json; data=json.load(sys.stdin); print('✅ Knowledge loaded:', data.get('sagan_status'))" \
		2>/dev/null || echo "❌ Knowledge API unavailable"

# Docker volume operations
create-volume: ## Create persistent data volume
	@echo "💾 Creating persistent data volume..."
	@docker volume create $(DATA_VOLUME)
	@echo "✅ Volume '$(DATA_VOLUME)' created"

remove-volume: ## Remove persistent data volume
	@echo "🗑️ Removing persistent data volume..."
	@docker volume rm $(DATA_VOLUME) || true
	@echo "✅ Volume '$(DATA_VOLUME)' removed"

# Development operations
dev-build: ## Build development version with verbose output
	@echo "🔧 Building development version with verbose output..."
	@docker build \
		-f $(DOCKERFILE) \
		-t $(IMAGE_NAME):dev \
		--no-cache \
		--progress=plain \
		$(CONTEXT)

dev-run: ## Run development version with mounted source
	@echo "🔧 Starting development version with source mounting..."
	@docker run -it --rm \
		-p $(HOST_PORT):$(PORT) \
		-v $(PWD):/app \
		-v $(DATA_VOLUME):/app/data \
		-e ΨQRH_MODE=development \
		-e KNOWLEDGE_BASE=$(KNOWLEDGE_BASE) \
		$(IMAGE_NAME):dev \
		/bin/bash

# Information targets
docker-info: ## Show Docker system information
	@echo "🐳 Docker System Information:"
	@echo "============================"
	@docker info | grep -E "(Server Version|Storage Driver|Kernel Version|Operating System)"
	@echo ""
	@echo "📊 Images:"
	@docker images | grep $(IMAGE_NAME) || echo "No ΨQRH images found"
	@echo ""
	@echo "📦 Containers:"
	@docker ps -a --filter name=$(CONTAINER_NAME) || echo "No ΨQRH containers found"
	@echo ""
	@echo "💾 Volumes:"
	@docker volume ls | grep $(DATA_VOLUME) || echo "No ΨQRH volumes found"

sagan-quote: ## Display random Carl Sagan quote
	@echo "💭 Carl Sagan's Wisdom:"
	@echo "====================="
	@python3 -c 'import random; quotes=["Extraordinary claims require extraordinary evidence", "Science is a candle in the dark", "The cosmos is within us. We are made of star-stuff", "Somewhere, something incredible is waiting to be known", "We are like butterflies who flutter for a day and think it is forever"]; print(f"\"{random.choice(quotes)}\" - Carl Sagan")'

# Epistemic Integrity Operations
integrity-setup: ## Establish reference integrity hash based on current system understanding
	@echo "🔬 Establishing epistemic integrity reference hash..."
	@echo "This creates a baseline for detecting degradation in scientific reasoning"
	@docker run --rm \
		-v $(DATA_VOLUME):/app/data \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 -c "from src.conceptual.integrity_guardian import IntegrityGuardian; g = IntegrityGuardian(); success = g.establish_reference_hash(); print('✅ Reference hash established' if success else '❌ Failed to establish hash')"
	@echo "🧠 Reference hash represents commitment to scientific method principles"

integrity-verify: ## Verify current epistemic integrity against reference
	@echo "🔍 Verifying epistemic integrity..."
	@echo "Testing commitment to scientific method and rational skepticism"
	@docker run --rm \
		-v $(DATA_VOLUME):/app/data \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 -c "from src.conceptual.integrity_guardian import IntegrityGuardian; g = IntegrityGuardian(); status = g.verify_epistemic_integrity(); print(f'Integrity Score: {status.score:.3f}'); print(f'Level: {status.level}'); print('✅ VERIFIED' if status.verified else '❌ FAILED')"

integrity-certificate: ## Generate epistemic integrity certificate
	@echo "📋 Generating epistemic integrity certificate..."
	@docker run --rm \
		-v $(DATA_VOLUME):/app/data \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 -c "from src.conceptual.integrity_guardian import IntegrityGuardian; g = IntegrityGuardian(); cert = g.generate_integrity_certificate(); print(cert if cert else '❌ Certificate generation failed - insufficient integrity')"

integrity-status: ## Check integrity status without full verification
	@echo "📊 Current epistemic integrity status:"
	@echo "===================================="
	@docker run --rm \
		-v $(DATA_VOLUME):/app/data \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 -c "import json; from src.conceptual.integrity_guardian import IntegrityGuardian; g = IntegrityGuardian(); status = g.check_integrity_status(); print(json.dumps(status, indent=2))"

integrity-test: ## Run epistemic integrity test (standalone)
	@echo "🧪 Running standalone epistemic integrity test..."
	@echo "Testing understanding of scientific principles"
	@docker run --rm \
		-v $(DATA_VOLUME):/app/data \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 /app/src/conceptual/epistemic_integrity.py

integrity-guard-test: ## Test integrity guardian system
	@echo "🛡️ Testing epistemic integrity guardian..."
	@echo "This simulates system initialization with integrity verification"
	@docker run --rm \
		-v $(DATA_VOLUME):/app/data \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 /app/src/conceptual/integrity_guardian.py

integrity-report: ## Generate detailed integrity analysis report
	@echo "📋 Generating detailed epistemic integrity report..."
	@docker run --rm \
		-v $(DATA_VOLUME):/app/data \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 -c "from src.conceptual.integrity_guardian import IntegrityGuardian; from src.conceptual.epistemic_integrity import EpistemicIntegrityVerifier; g = IntegrityGuardian(); v = EpistemicIntegrityVerifier(); score, responses, analysis = v.run_integrity_test(40); hash_val = v.calculate_integrity_hash(responses); report = v.generate_integrity_report(score, responses, analysis, hash_val); print(report)"