# ΨQRH — Makefile
# Run from project root: make build, make up, etc.

.PHONY: build up down test clean integrity convert-pdf Ψcws-stats demo-pdf-Ψcws list-Ψcws test-native-reader convert-wiki-topic list-wiki-topics convert-all-wiki-topics test-ΨQRH test-math docker-build docker-up docker-down docker-logs docker-shell docker-api docker-frontend dev-build dev-up dev-down dev-shell dev-jupyter dev-api dev-test dev-clean

# Docker Commands
docker-build:
	docker-compose -f ops/docker/docker-compose.yml build

docker-up:
	docker-compose -f ops/docker/docker-compose.yml up -d

docker-down:
	docker-compose -f ops/docker/docker-compose.yml down

docker-logs:
	docker-compose -f ops/docker/docker-compose.yml logs -f

docker-shell:
	docker-compose -f ops/docker/docker-compose.yml exec psiqrh-api /bin/bash

docker-api:
	docker-compose -f ops/docker/docker-compose.yml up -d psiqrh-api

docker-frontend:
	docker-compose -f ops/docker/docker-compose.yml up -d psiqrh-frontend

# Legacy Docker Commands (for backward compatibility)
build:
	docker-compose -f ops/docker/docker-compose.yml build

up:
	docker-compose -f ops/docker/docker-compose.yml run --rm psiqrh

down:
	docker-compose -f ops/docker/docker-compose.yml down

test:
	docker-compose -f ops/docker/docker-compose.yml run --rm psiqrh-test

integrity:
	docker-compose -f ops/docker/docker-compose.yml run --rm psiqrh make -f ops/Makefile integrity-verify

clean:
	docker-compose -f ops/docker/docker-compose.yml down -v --rmi all
	docker builder prune -f

# Development Commands
dev-build:
	docker-compose -f ops/docker/docker-compose.dev.yml build

dev-up:
	docker-compose -f ops/docker/docker-compose.dev.yml up -d

dev-down:
	docker-compose -f ops/docker/docker-compose.dev.yml down

dev-shell:
	docker-compose -f ops/docker/docker-compose.dev.yml exec psiqrh-dev /bin/bash

dev-jupyter:
	docker-compose -f ops/docker/docker-compose.dev.yml exec psiqrh-dev jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=dev123

dev-api:
	docker-compose -f ops/docker/docker-compose.dev.yml exec psiqrh-dev python app.py

dev-test:
	docker-compose -f ops/docker/docker-compose.dev.yml exec psiqrh-dev python -m pytest tests/ -v

dev-clean:
	docker-compose -f ops/docker/docker-compose.dev.yml down -v --rmi all

# Quick Start Commands
start: docker-build docker-up
	@echo "🚀 ΨQRH System Started!"
	@echo "🌐 Frontend: http://localhost:8080"
	@echo "🔧 API: http://localhost:5000"
	@echo "📊 API via proxy: http://localhost:8080/api/"

stop: docker-down
	@echo "🛑 ΨQRH System Stopped!"

restart: stop start
	@echo "🔄 ΨQRH System Restarted!"

status:
	@docker-compose -f ops/docker/docker-compose.yml ps
	@echo ""
	@echo "🌐 Frontend: http://localhost:8080"
	@echo "🔧 API: http://localhost:5000"
	@echo "📊 API via proxy: http://localhost:8080/api/"

help:
	@echo "ΨQRH Makefile Commands:"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  make docker-build     - Build all Docker images"
	@echo "  make docker-up        - Start all services in background"
	@echo "  make docker-down      - Stop all services"
	@echo "  make docker-logs      - Follow logs from all services"
	@echo "  make docker-shell     - Open shell in API container"
	@echo "  make docker-api       - Start only API service"
	@echo "  make docker-frontend  - Start only frontend service"
	@echo ""
	@echo "🔬 Development Commands:"
	@echo "  make dev-build        - Build development environment"
	@echo "  make dev-up           - Start development environment"
	@echo "  make dev-down         - Stop development environment"
	@echo "  make dev-shell        - Open shell in development container"
	@echo "  make dev-jupyter      - Start Jupyter notebook in container"
	@echo "  make dev-api          - Run API in development container"
	@echo "  make dev-test         - Run tests in development container"
	@echo "  make dev-clean        - Clean development environment"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make start           - Build and start all services"
	@echo "  make stop            - Stop all services"
	@echo "  make restart         - Restart all services"
	@echo "  make status          - Show service status and URLs"
	@echo ""
	@echo "🧪 Testing & Development:"
	@echo "  make test            - Run tests"
	@echo "  make integrity       - Run integrity verification"
	@echo "  make clean           - Clean all Docker resources"
	@echo ""
	@echo "📄 ΨCWS Operations:"
	@echo "  make convert-pdf PDF=path/to/file.pdf"
	@echo "  make Ψcws-stats"
	@echo "  make list-Ψcws"
	@echo ""
	@echo "🌐 URLs:"
	@echo "  Frontend: http://localhost:8080"
	@echo "  API: http://localhost:5000"
	@echo "  API via proxy: http://localhost:8080/api/"
	@echo "  Jupyter: http://localhost:8888 (dev)"

# ΨQRH PDF to ΨCWS Conversion Commands
# Variables
ΨCWS_OUTPUT_DIR = data/Ψcws_cache

# Convert PDF to ΨCWS format: make convert-pdf PDF=path/to/file.pdf
convert-pdf:
	@if [ -z "$(PDF)" ]; then \
		echo "❌ Usage: make convert-pdf PDF=path/to/file.pdf"; \
		echo "📖 Example: make convert-pdf PDF=src/conceptual/models/d41d8cd98f00b204e9800998ecf8427e.pdf"; \
		exit 1; \
	fi
	@mkdir -p $(ΨCWS_OUTPUT_DIR)
	@echo "🔄 Converting $(PDF) to .Ψcws format using ΨQRH consciousness pipeline..."
	@python3 -c "\
import sys; \
sys.path.append('src'); \
from pathlib import Path; \
from conscience.conscious_wave_modulator import ConsciousWaveModulator; \
import hashlib; \
pdf_path = Path('$(PDF)'); \
config = {'cache_dir': '$(ΨCWS_OUTPUT_DIR)', 'embedding_dim': 256, 'sequence_length': 64, 'device': 'cpu'}; \
modulator = ConsciousWaveModulator(config); \
cwm_file = modulator.process_file(pdf_path); \
file_stat = pdf_path.stat(); \
hash_input = f'{pdf_path.absolute()}_{file_stat.st_mtime}'; \
file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]; \
output_path = Path('$(ΨCWS_OUTPUT_DIR)') / f'{file_hash}_{pdf_path.stem}.Ψcws'; \
cwm_file.save(output_path); \
print(f'✅ Generated: {output_path}'); \
print(f'🧠 Consciousness metrics: {cwm_file.spectral_data.consciousness_metrics}'); \
"

# Show CWM cache statistics and consciousness metrics
Ψcws-stats:
	@echo "📊 ΨQRH ΨCWS Cache Statistics:"
	@echo "Cache directory: $(ΨCWS_OUTPUT_DIR)"
	@echo "Number of .Ψcws files: $$(find $(ΨCWS_OUTPUT_DIR) -name '*.Ψcws' 2>/dev/null | wc -l)"
	@echo "Total cache size: $$(du -h $(ΨCWS_OUTPUT_DIR) 2>/dev/null | cut -f1 || echo '0B')"
	@find $(ΨCWS_OUTPUT_DIR) -name '*.Ψcws' -type f -exec ls -lht {} + 2>/dev/null | head -3 || echo "No .Ψcws files found"

# Demo: Convert test PDF and show stats
demo-pdf-Ψcws:
	@echo "🎬 ΨQRH PDF→ΨCWS Demo with d41d8cd98f00b204e9800998ecf8427e.pdf"
	@make convert-pdf PDF=src/conceptual/models/d41d8cd98f00b204e9800998ecf8427e.pdf
	@make Ψcws-stats

# List available .Ψcws files using native reader
list-Ψcws:
	@echo "📋 Listando arquivos .Ψcws disponíveis via leitura nativa:"
	@python3 -c "\
import sys; sys.path.append('src'); \
from conscience.psicws_native_reader import list_Ψcws_files; \
files = list_Ψcws_files(); \
print(f'Total: {len(files)} arquivos .Ψcws'); \
print('\\n📄 Arquivos encontrados:'); \
[print(f'  {i+1}. {f[\"original_name\"]} ({f[\"size_kb\"]} KB)\\n     Hash: {f[\"hash\"]}\\n     Modificado: {f[\"modified_time\"]}') for i, f in enumerate(files)]"

# Test native reader functionality
test-native-reader:
	@echo "🧪 Testando funcionalidade de leitura nativa .Ψcws:"
	@python3 test_native_reader.py

# Show consciousness analysis of .Ψcws cache
analyze-Ψcws-consciousness:
	@python3 analyze_consciousness.py

# List available Wikipedia topics for conversion
list-wiki-topics:
	@echo "📋 Available Wikipedia topics for conversion:"
	@python3 wiki_to_psicws_converter.py list

# Convert single Wikipedia topic to .Ψcws format
convert-wiki-topic:
	@if [ -z "$(TOPIC)" ]; then \
		echo "❌ Usage: make convert-wiki-topic TOPIC=consciousness"; \
		echo "📖 Available topics: run 'make list-wiki-topics'"; \
		exit 1; \
	fi
	@echo "🔄 Converting Wikipedia topic $(TOPIC) to .Ψcws format..."
	@python3 wiki_to_psicws_converter.py "$(TOPIC)"

# Convert all supported Wikipedia topics
convert-all-wiki-topics:
	@echo "🔄 Converting ALL Wikipedia topics to .Ψcws format..."
	@echo "⚠️  This will make multiple API requests to Wikipedia (may take 10+ minutes)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	@python3 wiki_to_psicws_converter.py all

# Demo: Convert a consciousness topic
demo-wiki-conversion:
	@echo "🎬 Demo: Converting Consciousness (Wikipedia topic)"
	@make convert-wiki-topic TOPIC=consciousness
	@make analyze-Ψcws-consciousness

# Test ΨQRH architecture with comprehensive test suite
test-ΨQRH:
	@echo "🧪 Running ΨQRH Comprehensive Test Suite..."
	@python3 Enhanced_Transparency_Framework.py

# Run advanced mathematical tests
test-math:
	@echo "🧮 Running Advanced Mathematical Tests..."
	@python3 -c "\
import sys; \
sys.path.insert(0, '.'); \
sys.path.insert(0, 'src'); \
from src.testing.advanced_mathematical_tests import AdvancedMathematicalTests; \
from src.core.qrh_layer import QRHConfig; \
config = QRHConfig(embed_dim=64); \
math_tests = AdvancedMathematicalTests(config); \
results = math_tests.run_dynamic_comprehensive_validation(); \
print(f'✅ Mathematical tests completed: {len(results)} tests run'); \
"