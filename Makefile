# ΨQRH — Makefile
# Run from project root: make build, make up, etc.

.PHONY: build up down test clean integrity convert-pdf Ψcws-stats demo-pdf-Ψcws list-Ψcws test-native-reader convert-wiki-topic list-wiki-topics convert-all-wiki-topics test-ΨQRH test-math docker-build docker-up docker-down docker-logs docker-shell docker-api docker-frontend dev-build dev-up dev-down dev-shell dev-jupyter dev-api dev-test dev-clean restart restart-full restart-prod restart-dev dev-restart dev-restart-fast dev-reload dev-logs dev-logs-app test-integration test-chat stop-all start stop status help train-model validate-model chat-model test-deep-dive train-full validate-core train-complete test-physics validate-complete train-full-complete

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

restart: docker-down docker-up
	@echo "🔄 ΨQRH Production System Restarted!"
	@echo "🌐 Frontend: http://localhost:8080"
	@echo "🔧 API: http://localhost:5000"

restart-full: docker-down docker-build docker-up
	@echo "🔄 ΨQRH Production System Fully Rebuilt and Restarted!"
	@echo "🌐 Frontend: http://localhost:8080"
	@echo "🔧 API: http://localhost:5000"

dev-restart: dev-down dev-up
	@echo "🔄 Development Environment Restarted!"
	@echo "🌐 Frontend: http://localhost:3000 and http://localhost:8081"
	@echo "🔧 API: http://localhost:5000 and http://localhost:8080"
	@echo "📊 Jupyter: http://localhost:8888"
	@echo "💾 PostgreSQL: localhost:5432"
	@echo "🔴 Redis: localhost:6379"

# Fix frontend nginx proxy to connect with dev API
fix-frontend:
	@echo "🔧 Fixing frontend nginx configuration..."
	@docker-compose -f ops/docker/docker-compose.yml stop psiqrh-frontend 2>/dev/null || true
	@docker-compose -f ops/docker/docker-compose.yml rm -f psiqrh-frontend 2>/dev/null || true
	@echo "🔄 Starting frontend with updated config..."
	@docker-compose -f ops/docker/docker-compose.yml up -d --no-deps psiqrh-frontend
	@echo "🔗 Connecting frontend to dev network..."
	@docker network connect psiqrh-dev-network psiqrh-frontend 2>/dev/null || echo "⚠️  Already connected or network not found"
	@docker-compose -f ops/docker/docker-compose.yml restart psiqrh-frontend
	@sleep 3
	@echo "✅ Frontend fixed!"
	@echo "🧪 Testing connection..."
	@curl -s http://localhost:3000/api/health > /dev/null && echo "✅ API proxy working!" || echo "❌ API proxy still not working"
	@echo ""
	@echo "🌐 Frontend: http://localhost:3000"
	@echo "🔧 API Health: http://localhost:3000/api/health"

stop-all:
	@echo "🛑 Stopping ALL services (production + development)..."
	-docker-compose -f ops/docker/docker-compose.yml down 2>/dev/null || true
	-docker-compose -f ops/docker/docker-compose.dev.yml down 2>/dev/null || true
	@echo "✅ All services stopped!"

restart-prod: stop-all
	@echo "🚀 Starting PRODUCTION environment only..."
	docker-compose -f ops/docker/docker-compose.yml up -d
	@echo "✅ Production started!"
	@echo "🌐 Frontend: http://localhost:8080"
	@echo "🔧 API: http://localhost:5000"

restart-dev: stop-all
	@echo "🚀 Starting DEVELOPMENT environment only..."
	docker-compose -f ops/docker/docker-compose.dev.yml up -d
	@echo "✅ Development started!"
	@echo "🌐 Frontend: http://localhost:3000 and http://localhost:8081"
	@echo "🔧 API: http://localhost:5000 and http://localhost:8080"
	@echo "📊 Jupyter: http://localhost:8888"

# Fast restart - only restart app container (keeps DB/Redis running)
dev-restart-fast:
	@echo "⚡ Fast restarting app container only..."
	docker-compose -f ops/docker/docker-compose.dev.yml restart psiqrh-dev
	@sleep 2
	@echo "✅ App restarted!"
	@echo "🔧 API: http://localhost:5000"

# Reload Flask app (even faster - no container restart)
dev-reload:
	@echo "⚡ Reloading Flask app..."
	@docker-compose -f ops/docker/docker-compose.dev.yml exec psiqrh-dev pkill -HUP -f "python.*app.py" || true
	@echo "✅ Flask reloaded!"
	@echo "🔧 API: http://localhost:5000"

# Test integrated system
test-integration:
	@echo "🧪 Testing ΨQRH Integrated System"
	@echo "=================================="
	@echo ""
	@echo "1. Testing Index Page (Split View)..."
	@curl -s http://localhost:5000/ | grep -q "Sistema Integrado" && echo "   ✅ Index: OK" || echo "   ❌ Index: Failed"
	@echo ""
	@echo "2. Testing Chat Interface..."
	@curl -s http://localhost:5000/chat.html | grep -q "ΨQRH Chat" && echo "   ✅ Chat: OK" || echo "   ❌ Chat: Failed"
	@echo ""
	@echo "3. Testing GLS Visualization..."
	@curl -s http://localhost:5000/harmonic_gls_demo.html | grep -q "Harmonic GLS" && echo "   ✅ GLS: OK" || echo "   ❌ GLS: Failed"
	@echo ""
	@echo "4. Testing API Health..."
	@curl -s http://localhost:5000/api/health | jq -r '.status' | grep -q "healthy" && echo "   ✅ API: healthy" || echo "   ❌ API: unhealthy"
	@echo ""
	@echo "=================================="
	@echo "🌐 URLs Available:"
	@echo "   • Main (Split View):  http://localhost:5000/"
	@echo "   • Chat Only:          http://localhost:5000/chat.html"
	@echo "   • GLS Only:           http://localhost:5000/harmonic_gls_demo.html"
	@echo ""
	@echo "📊 How to use:"
	@echo "   1. Open http://localhost:5000/ in browser"
	@echo "   2. Type message in chat (left panel)"
	@echo "   3. Watch GLS visualization update (right panel)"
	@echo "   4. Use tabs at bottom to switch views"
	@echo ""

# Test chat API with sample message
test-chat:
	@echo "🧪 Testing Chat API..."
	@echo "Sending message: 'ola mundo'"
	@curl -s -X POST http://localhost:5000/api/chat \
		-H "Content-Type: application/json" \
		-d '{"message": "ola mundo"}' | jq '{status: .status, fci: .consciousness_metrics.fci, state: .consciousness_metrics.state}'
	@echo ""
	@echo "✅ Chat API test complete"

# Show logs for debugging
dev-logs:
	@docker-compose -f ops/docker/docker-compose.dev.yml logs -f psiqrh-dev

# Show only app container logs (filtered)
dev-logs-app:
	@docker-compose -f ops/docker/docker-compose.dev.yml logs -f psiqrh-dev 2>&1 | grep -v "GET /static"

status:
	@docker-compose -f ops/docker/docker-compose.yml ps
	@echo ""
	@echo "🌐 Frontend: http://localhost:8080"
	@echo "🔧 API: http://localhost:5000"
	@echo "📊 API via proxy: http://localhost:8080/api/"

help:
	@echo "ΨQRH Makefile Commands:"
	@echo ""
	@echo "🛡️ Operações de Dados Seguros:"
	@echo "  make new-secure-asset SOURCE=... NAME=... LEVEL=<personal|enterprise|government> [KEY=...]"
	@echo "    - Cria ativo .Ψcws criptografado com manifesto e certificação"
	@echo "  make list-secure-assets"
	@echo "    - Lista todos os ativos seguros disponíveis"
	@echo "  make audit-asset NAME=..."
	@echo "    - Exibe manifesto de auditoria de um ativo"
	@echo "  make validate-secure-asset NAME=... [KEY=...]"
	@echo "    - Valida certificação e integridade de um ativo"
	@echo "  make train-with-secure-asset NAME=... KEY=..."
	@echo "    - Treina modelo usando ativo seguro (requer chave)"
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
	@echo "  make dev-restart-fast - ⚡ Fast restart (app only, keeps DB/Redis)"
	@echo "  make dev-reload       - ⚡ Ultra-fast reload (Flask only, no restart)"
	@echo "  make dev-shell        - Open shell in development container"
	@echo "  make dev-jupyter      - Start Jupyter notebook in container"
	@echo "  make dev-api          - Run API in development container"
	@echo "  make dev-test         - Run tests in development container"
	@echo "  make dev-logs         - Show development logs"
	@echo "  make dev-logs-app     - Show app logs (filtered)"
	@echo "  make dev-clean        - Clean development environment"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make start           - Build and start production services"
	@echo "  make stop            - Stop production services"
	@echo "  make stop-all        - Stop ALL services (prod + dev)"
	@echo "  make restart         - Restart production (fast)"
	@echo "  make restart-full    - Rebuild and restart production"
	@echo "  make restart-prod    - Stop all and start production only"
	@echo "  make restart-dev     - Stop all and start development only"
	@echo "  make dev-restart     - Restart development (without stopping prod)"
	@echo "  make status          - Show service status and URLs"
	@echo ""
	@echo "🧪 Testing & Development:"
	@echo "  make test              - Run tests"
	@echo "  make test-integration  - Test integrated system (chat + GLS)"
	@echo "  make test-chat         - Test chat API with sample message"
	@echo "  make test-physics      - Run physics validation tests (fractal, spectral, SO(4))"
	@echo "  make validate-core     - Validate core mathematical properties"
	@echo "  make integrity         - Run integrity verification"
	@echo "  make clean             - Clean all Docker resources"
	@echo ""
	@echo "🚀 Pipeline de Ponta a Ponta:"
	@echo "  make new-model SOURCE=<source> [NAME=nome] - Pipeline completo: adquire, converte, treina, certifica, ativa e inicia chat"
	@echo "    Por padrão usa PsiQRHTransformerComplete (física rigorosa) e conversão espectral"
	@echo "    Exemplos:"
	@echo "      make new-model SOURCE=gpt2-medium NAME=gpt2_qa           # Nome descritivo: psiqrh_gpt2_qa"
	@echo "      make new-model SOURCE=bert-base NAME=bert_sentiment      # Nome descritivo: psiqrh_bert_sentiment"
	@echo "      make new-model SOURCE=gpt2-medium                        # Usa timestamp automático"
	@echo "      make new-model SOURCE=gpt2-medium USE_COMPLETE=false     # Implementação padrão"
	@echo "      make new-model SOURCE=gpt2-medium USE_SPECTRAL=false     # Conversão genérica"
	@echo "      make new-model SOURCE=https://github.com/user/model-repo"
	@echo "      make new-model SOURCE=./local/model"
	@echo ""
	@echo "🔬 Conversão Espectral:"
	@echo "  make convert-model SOURCE=<source> OUTPUT=<dir> - Converte modelo usando análise espectral física"
	@echo "    Método: FFT → Espectro de Potência → Lei de Potência → Dimensão Fractal → α"
	@echo "    Inclui: Correção Leech (Λ₂₄) + Validação Energética + Quaternions"
	@echo "    Exemplos:"
	@echo "      make convert-model SOURCE=gpt2 OUTPUT=./models/gpt2_psiqrh"
	@echo "      make convert-model SOURCE=bert-base-uncased OUTPUT=./models/bert_psiqrh"
	@echo "      make convert-model SOURCE=./local/model OUTPUT=./models/converted"
	@echo ""
	@echo "🤖 Model Management & Quality Assurance:"
	@echo "  make model-discover    - Discover new models in models directory"
	@echo "  make model-list        - List all models with certification status"
	@echo "  make model-set-active MODEL=... - Set a model as active"
	@echo "  make model-certify MODEL=...   - (IMPORTANTE) Roda testes para certificar um modelo como 'apto'"
	@echo ""
	@echo "🤖 Model Training & Validation:"
	@echo "  make train-model       - Train ΨQRH model on WikiText-103"
	@echo "  make validate-model    - Validate trained model (all phases)"
	@echo "  make chat-model        - Chat interativo (usa modelo ativo automaticamente)"
	@echo "  make chat-model MODEL=nome - Chat com modelo específico"
	@echo "  make chat-model MODEL_DIR=caminho - Chat com diretório específico"
	@echo "  make chat-model-verbose - Chat com todos os detalhes de processamento"
	@echo "  make train-full        - Full pipeline (train + validate + test)"
	@echo "  make train-examples    - Show training command examples"
	@echo "  make test-model-quality - Run qualitative tests on model (requires certification)"
	@echo "  make validate-model-quick - Quick validation (skip perplexity)"
	@echo "  make test-deep-dive    - Test deep dive metrics with model (requires certification)"
	@echo "  make test-model-echo   - Teste de eco rápido no modelo ativo"
	@echo ""
	@echo "🌟 Complete ΨQRH Implementation (Física Rigorosa):"
	@echo "  make train-complete    - Train with PsiQRHTransformerComplete"
	@echo "  make validate-complete - Validate complete implementation model"
	@echo "  make train-full-complete - Full pipeline (train + validate + physics)"
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

# ========================================================================
# 🛡️ Operações de Dados Seguros (Confidencialidade e Integridade)
# ========================================================================
.PHONY: new-secure-asset list-secure-assets audit-asset train-with-secure-asset validate-secure-asset

# Cria um novo ativo .Ψcws criptografado e seu manifesto de auditoria.
# Uso: make new-secure-asset SOURCE=... NAME=... LEVEL=<personal|enterprise|government> [KEY=...]
new-secure-asset:
	@if [ -z "$(SOURCE)" ]; then \
		echo "❌ ERRO: O argumento SOURCE é obrigatório."; \
		echo "   Uso: make new-secure-asset SOURCE=path/to/file.txt NAME=nome_do_ativo LEVEL=<personal|enterprise|government>"; \
		echo "   Exemplo: make new-secure-asset SOURCE=relatorio_interno.txt NAME=relatorio-q3 LEVEL=enterprise KEY=CHAVE_SECRETA"; \
		exit 1; \
	fi
	@if [ -z "$(NAME)" ]; then \
		echo "❌ ERRO: O argumento NAME é obrigatório."; \
		echo "   Uso: make new-secure-asset SOURCE=path/to/file.txt NAME=nome_do_ativo LEVEL=<personal|enterprise|government>"; \
		exit 1; \
	fi
	@if [ -z "$(LEVEL)" ]; then \
		echo "❌ ERRO: O argumento LEVEL é obrigatório."; \
		echo "   Uso: make new-secure-asset SOURCE=path/to/file.txt NAME=nome_do_ativo LEVEL=<personal|enterprise|government>"; \
		exit 1; \
	fi
	@echo "🔒 Criando ativo seguro: $(NAME) (Nível: $(LEVEL))..."
	@python3 scripts/create_secure_asset.py \
		--source "$(SOURCE)" \
		--name "$(NAME)" \
		--level "$(LEVEL)" \
		$(if $(KEY),--key "$(KEY)",) \
		$(if $(AUTHOR),--author "$(AUTHOR)",) \
		$(if $(DESCRIPTION),--description "$(DESCRIPTION)",) \
		$(if $(CLASSIFICATION),--classification "$(CLASSIFICATION)",)

# Lista todos os ativos seguros disponíveis, lendo seus manifestos.
list-secure-assets:
	@echo "📦 Listando Ativos de Dados Seguros..."
	@python3 scripts/secure_asset_validator.py --list

# Exibe o manifesto de um ativo específico para auditoria.
# Uso: make audit-asset NAME=<nome_do_ativo>
audit-asset:
	@if [ -z "$(NAME)" ]; then \
		echo "❌ ERRO: O argumento NAME é obrigatório."; \
		echo "   Uso: make audit-asset NAME=nome_do_ativo"; \
		exit 1; \
	fi
	@echo "🔍 Auditando o ativo: $(NAME)"
	@python3 scripts/secure_asset_validator.py --name "$(NAME)"

# Valida um ativo seguro e sua certificação
# Uso: make validate-secure-asset NAME=<nome_do_ativo> [KEY=<chave>]
validate-secure-asset:
	@if [ -z "$(NAME)" ]; then \
		echo "❌ ERRO: O argumento NAME é obrigatório."; \
		echo "   Uso: make validate-secure-asset NAME=nome_do_ativo [KEY=chave]"; \
		exit 1; \
	fi
	@echo "🔐 Validando ativo seguro: $(NAME)"
	@python3 scripts/secure_asset_validator.py --name "$(NAME)" $(if $(KEY),--key "$(KEY)",)

# Treina um modelo usando um ativo seguro (requer a chave para "desbloquear").
# Uso: make train-with-secure-asset NAME=<nome_do_ativo> KEY=<chave>
train-with-secure-asset:
	@if [ -z "$(NAME)" ]; then \
		echo "❌ ERRO: O argumento NAME é obrigatório."; \
		echo "   Uso: make train-with-secure-asset NAME=nome_do_ativo KEY=chave"; \
		exit 1; \
	fi
	@if [ -z "$(KEY)" ]; then \
		echo "❌ ERRO: O argumento KEY é obrigatório."; \
		echo "   Uso: make train-with-secure-asset NAME=nome_do_ativo KEY=chave"; \
		exit 1; \
	fi
	@echo "🎓 Treinando com o ativo seguro: $(NAME)..."
	@echo "⚠️  Esta funcionalidade será implementada no sistema de treinamento"
	@echo "   Validando primeiro o ativo..."
	@python3 scripts/secure_asset_validator.py --name "$(NAME)" --key "$(KEY)"
	@echo "✅ Ativo validado. Pronto para treinamento seguro."

# ========================================================================
# 🤖 MODEL MANAGEMENT & CERTIFICATION COMMANDS
# ========================================================================

# Discover models in the models directory
model-discover:
	@echo "🔍 Discovering models..."
	@python3 tools/model_manager.py discover
	@echo "✅ Model discovery completed"

# List all models with certification status
model-list:
	@python3 tools/model_manager.py list

# Set a model as active
model-set-active:
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ Usage: make model-set-active MODEL=<model_name>"; \
		echo "💡 Example: make model-set-active MODEL=psiqrh_native_v1"; \
		exit 1; \
	fi
	@python3 tools/model_manager.py set-active "$(MODEL)"

# Certify a model as "apt" by running quality tests
model-certify:
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ Usage: make model-certify MODEL=<model_name>"; \
		echo "💡 Example: make model-certify MODEL=psiqrh_native_v1"; \
		exit 1; \
	fi
	@echo "🔬 Certificando modelo '$(MODEL)' como apto..."
	@python3 tools/certify_model.py "$(MODEL)"
	@echo "✅ Processo de certificação concluído."
	@make model-list

# Prune models from registry based on criteria
model-prune:
	@if [ -z "$(ARGS)" ]; then \
		echo "❌ Usage: make model-prune ARGS=<prune_criteria>"; \
		echo "💡 Example: make model-prune ARGS=\"--failed --uncertified\""; \
		echo "💡 Example: make model-prune ARGS=\"--empty-dirs\""; \
		exit 1; \
	fi
	@echo "🧹 Pruning models from registry with criteria: $(ARGS)"
	@python3 tools/model_manager.py prune $(ARGS)
	@echo "✅ Registry pruning completed"
	@make model-list

# ========================================================================
# 🚀 MODEL TRAINING & VALIDATION COMMANDS
# ========================================================================

# Variables for model training
# MODEL_DIR não tem padrão - usa modelo ativo automaticamente
MODEL_DIR ?=
TEXT_FILE ?= data/train.txt
EPOCHS ?= 3
BATCH_SIZE ?= 8
LEARNING_RATE ?= 1e-4
DEVICE ?= auto
SEQ_LENGTH ?= 256
D_MODEL ?= 256
N_LAYERS ?= 4
N_HEADS ?= 8

# Train ΨQRH model natively (character-level, no HuggingFace)
train-model:
	@echo "🚀 Training Native ΨQRH Transformer"
	@echo "===================================="
	@echo "Text file: $(TEXT_FILE)"
	@echo "Output directory: $(MODEL_DIR)"
	@echo "Epochs: $(EPOCHS)"
	@echo "Batch size: $(BATCH_SIZE)"
	@echo "Learning rate: $(LEARNING_RATE)"
	@echo "Device: $(DEVICE)"
	@echo "Model config: d_model=$(D_MODEL), n_layers=$(N_LAYERS), n_heads=$(N_HEADS)"
	@echo ""
	@python3 train_psiqrh_native.py \
		--text_file $(TEXT_FILE) \
		--output_dir $(MODEL_DIR) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--learning_rate $(LEARNING_RATE) \
		--device $(DEVICE) \
		--seq_length $(SEQ_LENGTH) \
		--d_model $(D_MODEL) \
		--n_layers $(N_LAYERS) \
		--n_heads $(N_HEADS)
	@echo ""
	@echo "✅ Training completed!"
	@echo "📁 Model saved to: $(MODEL_DIR)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. make validate-model MODEL_DIR=$(MODEL_DIR)"
	@echo "  2. make test-model-quality MODEL_DIR=$(MODEL_DIR)"

# Validate trained model (all phases)
validate-model:
	@echo "🔍 Validating ΨQRH Model"
	@echo "========================="
	@echo "Model directory: $(MODEL_DIR)"
	@echo ""
	@if [ ! -d "$(MODEL_DIR)" ]; then \
		echo "❌ Error: Model directory not found: $(MODEL_DIR)"; \
		echo "💡 Train a model first: make train-model"; \
		exit 1; \
	fi
	@python3 validate_training_output.py --model_dir $(MODEL_DIR) --device cpu --text_file $(TEXT_FILE)
	@echo ""
	@echo "Next step: make chat-model MODEL_DIR=$(MODEL_DIR)"

# Quick validation (skip perplexity benchmark)
validate-model-quick:
	@echo "⚡ Quick Validation (skipping perplexity benchmark)"
	@python3 validate_training_output.py \
		--model_dir $(MODEL_DIR) \
		--skip_benchmark
	@echo ""
	@echo "✅ Quick validation completed!"

# ========================================================================
# 🔬 CALIBRATION & PHYSICAL OPTIMIZATION COMMANDS
# ========================================================================
.PHONY: calibrate-model test-calibrated-echo

# Calibrate model using physics-informed gradient descent
calibrate-model:
	@echo "🔬 Iniciando calibração física por gradiente..."
	@python3 scripts/physics_gradient_calibrator.py
	@echo "✅ Calibração física concluída!"
	@echo "📁 Configurações calibradas salvas em: configs/gradient_calibrated/"

# Test echo with calibrated model configuration
test-calibrated-echo:
	@echo "🧪 Testando eco com modelo calibrado..."
	@python3 scripts/test_calibrated_echo.py
	@echo "✅ Teste de eco pós-calibração concluído!"

# Roda um teste de eco rápido no modelo ativo e certificado
test-model-echo:
	@echo "🎤 Executando teste de eco no modelo ativo..."
	@python3 psiqrh.py --test-echo

# Debug mode for model certification with verbose output
debug-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ Usage: make debug-model MODEL=<model_name>"; \
		echo "💡 Example: make debug-model MODEL=psiqrh_native_v1"; \
		exit 1; \
	fi
	@echo "🔧 Debug Mode: Running certification with verbose output for '$(MODEL)'..."
	@python3 tools/certify_model.py "$(MODEL)" --debug

# ========================================================================
# 🚀 Pipeline de Ponta a Ponta
# ========================================================================
.PHONY: new-model

# Pipeline completo: Adquire, converte, treina, certifica, ativa e inicia o chat com um novo modelo.
# Uso: make new-model SOURCE=<caminho_local | id_huggingface | url_git>
new-model:
	@if [ -z "$(SOURCE)" ]; then \
		echo "❌ ERRO: O argumento SOURCE é obrigatório."; \
		echo "   Uso: make new-model SOURCE=<path/to/model | gpt2-medium | https://github.com/...> [NAME=nome_descritivo]"; \
		echo "   Opcional: NAME=nome_descritivo (ex: gpt2_qa, bert_sentiment)"; \
		echo "   Opcional: USE_COMPLETE=false (padrão: true)"; \
		echo "   Opcional: USE_SPECTRAL=false (padrão: true)"; \
		exit 1; \
	fi
	@echo "🚀 Iniciando pipeline completo de ponta a ponta para a fonte: $(SOURCE)..."
	@echo "Este processo é totalmente automatizado e pode levar um tempo considerável."
	@PIPELINE_ARGS="--source \"$(SOURCE)\""; \
	if [ ! -z "$(NAME)" ]; then \
		echo "🏷️  Nome do modelo: $(NAME)"; \
		PIPELINE_ARGS="$$PIPELINE_ARGS --name \"$(NAME)\""; \
	fi; \
	if [ "$(USE_COMPLETE)" = "false" ]; then \
		echo "📋 Usando PsiQRHTransformer (implementação padrão)"; \
		PIPELINE_ARGS="$$PIPELINE_ARGS --no-complete"; \
	else \
		echo "🌟 Usando PsiQRHTransformerComplete (física rigorosa) - PADRÃO"; \
		PIPELINE_ARGS="$$PIPELINE_ARGS --use-complete"; \
	fi; \
	if [ "$(USE_SPECTRAL)" = "false" ]; then \
		echo "🔄 Usando conversão genérica"; \
		PIPELINE_ARGS="$$PIPELINE_ARGS --no-spectral"; \
	else \
		echo "🔬 Usando conversão espectral (FFT + Lei de Potência + Leech) - PADRÃO"; \
		PIPELINE_ARGS="$$PIPELINE_ARGS --use-spectral"; \
	fi; \
	python3 scripts/pipeline_from_source.py $$PIPELINE_ARGS
	@echo ""
	@echo "🔬 Iniciando calibração física por gradiente do modelo..."
	@make calibrate-model
	@echo ""
	@echo "🧪 Executando teste de eco pós-calibração..."
	@make test-calibrated-echo

# Convert a model to ΨQRH format using spectral analysis
# Uso: make convert-model SOURCE=gpt2 OUTPUT=./models/gpt2_psiqrh
convert-model:
	@if [ -z "$(SOURCE)" ]; then \
		echo "❌ ERRO: O argumento SOURCE é obrigatório."; \
		echo "   Uso: make convert-model SOURCE=<model_name | path> OUTPUT=<output_dir>"; \
		echo "   Exemplos:"; \
		echo "     make convert-model SOURCE=gpt2 OUTPUT=./models/gpt2_psiqrh"; \
		echo "     make convert-model SOURCE=bert-base-uncased OUTPUT=./models/bert_psiqrh"; \
		echo "     make convert-model SOURCE=./local/model OUTPUT=./models/converted"; \
		exit 1; \
	fi
	@if [ -z "$(OUTPUT)" ]; then \
		echo "❌ ERRO: O argumento OUTPUT é obrigatório."; \
		echo "   Uso: make convert-model SOURCE=<model_name | path> OUTPUT=<output_dir>"; \
		exit 1; \
	fi
	@echo "🔬 Convertendo modelo usando análise espectral física..."
	@echo "   Fonte: $(SOURCE)"
	@echo "   Destino: $(OUTPUT)"
	@python3 scripts/convert_model_spectral.py \
		--source "$(SOURCE)" \
		--output "$(OUTPUT)" \
		--use-leech \
		--validate-energy
	@echo ""
	@echo "✅ Conversão concluída! Próximos passos:"
	@echo "   1. Treinar: make train-complete MODEL_DIR=$(OUTPUT)"
	@echo "   2. Certificar: make model-certify MODEL=$$(basename $(OUTPUT))"
	@echo "   3. Ativar: make model-set-active MODEL=$$(basename $(OUTPUT))"

# Interactive chat with trained model
chat-model:
	@if [ ! -z "$(MODEL)" ]; then \
		echo "💬 ΨQRH Chat Interativo"; \
		echo "📁 Modelo: models/$(MODEL)"; \
		echo ""; \
		python3 psiqrh.py --interactive --model-dir "models/$(MODEL)" --quiet; \
	elif [ ! -z "$(MODEL_DIR)" ]; then \
		echo "💬 ΨQRH Chat Interativo"; \
		echo "📁 Modelo: $(MODEL_DIR)"; \
		echo ""; \
		python3 psiqrh.py --interactive --model-dir "$(MODEL_DIR)" --quiet; \
	else \
		echo "💬 ΨQRH Chat Interativo"; \
		echo "📁 Modelo: psiqrh_gpt2_MEDIO (ativo)"; \
		echo ""; \
		python3 psiqrh.py --interactive --model-dir "models/psiqrh_gpt2_MEDIO" --quiet; \
	fi

# Chat com modo verbose (mostra todos os detalhes)
chat-model-verbose:
	@if [ ! -z "$(MODEL)" ]; then \
		python3 psiqrh.py --interactive --model-dir "models/$(MODEL)" --verbose; \
	else \
		MODEL_SELECTED=$$(python3 tools/select_model.py); \
		if [ $$? -eq 0 ] && [ ! -z "$$MODEL_SELECTED" ]; then \
			python3 psiqrh.py --interactive --model-dir "models/$$MODEL_SELECTED" --verbose; \
		fi \
	fi

# Run qualitative tests on model
test-model-quality:
	@echo "🧪 Running Qualitative Tests"
	@echo "============================="
	@if [ ! -d "$(MODEL_DIR)" ]; then \
		echo "❌ Error: Model directory not found: $(MODEL_DIR)"; \
		echo "💡 Train a model first: make train-model"; \
		exit 1; \
	fi
	@IS_CERTIFIED=$$(python3 tools/model_manager.py is-certified "$(MODEL_DIR)" 2>/dev/null || echo "false"); \
	if [ "$$IS_CERTIFIED" != "true" ]; then \
		echo "❌ ERRO: O modelo '$(MODEL_DIR)' não é certificado como 'apto'."; \
		echo "💡 Execute 'make model-certify MODEL=$(MODEL_DIR)' para tentar certificá-lo."; \
		echo "⚠️  Use por sua conta e risco ou certifique o modelo primeiro."; \
		read -p "Continuar mesmo assim? [y/N]: " confirm; \
		if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
			echo "❌ Execução cancelada pelo usuário."; \
			exit 1; \
		fi; \
		echo "⚠️  Executando modelo NÃO CERTIFICADO - use por sua conta e risco!"; \
	else \
		echo "✅ Modelo CERTIFICADO: $(MODEL_DIR)"; \
	fi
	@python3 chat_with_model.py \
		--model_dir $(MODEL_DIR) \
		--device $(DEVICE) \
		--test_mode \
		--save_results validation_results.json
	@echo ""
	@echo "📊 Results saved to: validation_results.json"

# Test deep dive metrics with trained model
test-deep-dive:
	@echo "🔬 Testing Deep Dive Metrics"
	@echo "============================"
	@if [ ! -d "$(MODEL_DIR)" ]; then \
		echo "❌ Error: Model directory not found: $(MODEL_DIR)"; \
		echo "💡 Train a model first: make train-model"; \
		exit 1; \
	fi
	@IS_CERTIFIED=$$(python3 tools/model_manager.py is-certified "$(MODEL_DIR)" 2>/dev/null || echo "false"); \
	if [ "$$IS_CERTIFIED" != "true" ]; then \
		echo "❌ ERRO: O modelo '$(MODEL_DIR)' não é certificado como 'apto'."; \
		echo "💡 Execute 'make model-certify MODEL=$(MODEL_DIR)' para tentar certificá-lo."; \
		echo "⚠️  Use por sua conta e risco ou certifique o modelo primeiro."; \
		read -p "Continuar mesmo assim? [y/N]: " confirm; \
		if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
			echo "❌ Execução cancelada pelo usuário."; \
			exit 1; \
		fi; \
		echo "⚠️  Executando modelo NÃO CERTIFICADO - use por sua conta e risco!"; \
	else \
		echo "✅ Modelo CERTIFICADO: $(MODEL_DIR)"; \
	fi
	@echo ""
	@echo "⚠️  Make sure app.py is running with the trained model loaded!"
	@echo ""
	@echo "To update app.py, add this line around line 33:"
	@echo "  qrh_factory = QRHFactory(model_path='$(MODEL_DIR)')"
	@echo ""
	@read -p "Press Enter when app.py is ready, or Ctrl+C to cancel..."
	@echo ""
	@python3 test_deep_dive_metrics.py
	@echo ""
	@echo "✅ Deep dive metrics test completed!"

# Validate core mathematical properties of ΨQRH
validate-core:
	@echo "🔬 Validando as propriedades matemáticas e de eficiência do núcleo ΨQRH..."
	@python3 VALIDACAO/validate_core_properties.py
	@echo "✅ Validação do núcleo concluída."

# Full training pipeline (train + validate + test)
train-full:
	@echo "🎯 FULL TRAINING PIPELINE"
	@echo "========================="
	@echo ""
	@echo "This will:"
	@echo "  1. Train model on WikiText-103 ($(EPOCHS) epochs)"
	@echo "  2. Validate model artifacts and perplexity"
	@echo "  3. Run qualitative tests"
	@echo ""
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo ""
	@make train-model MODEL_DIR=$(MODEL_DIR) EPOCHS=$(EPOCHS) BATCH_SIZE=$(BATCH_SIZE)
	@echo ""
	@echo "⏸️  Training complete. Starting validation..."
	@sleep 2
	@make validate-model MODEL_DIR=$(MODEL_DIR)
	@echo ""
	@echo "⏸️  Validation complete. Running qualitative tests..."
	@sleep 2
	@make test-model-quality MODEL_DIR=$(MODEL_DIR)
	@echo ""
	@echo "======================================"
	@echo "🎉 FULL PIPELINE COMPLETED!"
	@echo "======================================"
	@echo ""
	@echo "Model ready at: $(MODEL_DIR)"
	@echo ""
	@echo "Try it out: make chat-model MODEL_DIR=$(MODEL_DIR)"

# Show available training examples
train-examples:
	@echo "📚 ΨQRH NATIVE TRAINING EXAMPLES"
	@echo "================================="
	@echo ""
	@echo "1. Quick test (default settings):"
	@echo "   make train-model"
	@echo ""
	@echo "2. Custom text file:"
	@echo "   make train-model TEXT_FILE=my_corpus.txt"
	@echo ""
	@echo "3. Larger model:"
	@echo "   make train-model D_MODEL=512 N_LAYERS=6 N_HEADS=8"
	@echo ""
	@echo "4. Longer sequences:"
	@echo "   make train-model SEQ_LENGTH=512 BATCH_SIZE=4"
	@echo ""
	@echo "5. Extended training:"
	@echo "   make train-model EPOCHS=10 LEARNING_RATE=5e-5"
	@echo ""
	@echo "6. GPU training:"
	@echo "   make train-model DEVICE=cuda BATCH_SIZE=16"
	@echo ""
	@echo "7. Full pipeline:"
	@echo "   make train-full EPOCHS=5 D_MODEL=384 N_LAYERS=6"
	@echo ""
	@echo "8. Validate existing model:"
	@echo "   make validate-model MODEL_DIR=./models/my_model"
	@echo ""
	@echo "🔬 Calibration & Optimization:"
	@echo "   make calibrate-model          - Calibrate model using physics-informed gradient descent"
	@echo "   make test-calibrated-echo     - Test echo with calibrated model configuration"
	@echo "   make test-model-echo          - Test echo with active certified model"
	@echo ""
	@echo "🌟 New Complete ΨQRH Implementation:"
	@echo "   make train-complete           - Train with PsiQRHTransformerComplete (física rigorosa)"
	@echo "   make test-physics             - Run physics validation tests"
	@echo "   make validate-complete        - Validate PsiQRHTransformerComplete model"
	@echo ""
	@echo "📝 Variables disponíveis:"
	@echo "   TEXT_FILE    - Arquivo de texto (padrão: data/train.txt)"
	@echo "   MODEL_DIR    - Diretório de saída (padrão: ./models/psiqrh_native_v1)"
	@echo "   EPOCHS       - Número de épocas (padrão: 3)"
	@echo "   BATCH_SIZE   - Tamanho do batch (padrão: 8)"
	@echo "   SEQ_LENGTH   - Comprimento de sequência (padrão: 256)"
	@echo "   D_MODEL      - Dimensão do modelo (padrão: 256)"
	@echo "   N_LAYERS     - Número de camadas (padrão: 4)"
	@echo "   N_HEADS      - Número de heads (padrão: 8)"
	@echo "   DEVICE       - Dispositivo (auto/cpu/cuda, padrão: auto)"

# ============================================================================
# NEW: Complete ΨQRH Implementation (Física Rigorosa)
# ============================================================================

# Train with PsiQRHTransformerComplete
train-complete:
	@echo "🌟 Training with PsiQRHTransformerComplete (Física Rigorosa)"
	@echo "============================================================="
	@echo "Text file: $(TEXT_FILE)"
	@echo "Output directory: $(MODEL_DIR)"
	@echo "Epochs: $(EPOCHS)"
	@echo "Batch size: $(BATCH_SIZE)"
	@echo ""
	@echo "🔬 Features:"
	@echo "   ✅ Fractal Quantum Embedding"
	@echo "   ✅ Spectral Attention with α(D) adaptation"
	@echo "   ✅ SO(4) Harmonic Evolution"
	@echo "   ✅ Optical Probe Generation"
	@echo ""
	@python3 train_psiqrh_native.py \
		--text_file $(TEXT_FILE) \
		--output_dir $(MODEL_DIR) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--learning_rate $(LEARNING_RATE) \
		--device $(DEVICE) \
		--seq_length $(SEQ_LENGTH) \
		--d_model $(D_MODEL) \
		--n_layers $(N_LAYERS) \
		--n_heads $(N_HEADS) \
		--use_complete
	@echo ""
	@echo "✅ Training with Complete Implementation completed!"
	@echo "📁 Model saved to: $(MODEL_DIR)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. make test-physics"
	@echo "  2. make validate-complete MODEL_DIR=$(MODEL_DIR)"

# Physics validation tests
test-physics:
	@echo "🔬 Running Physics Validation Tests"
	@echo "===================================="
	@python3 examples/test_complete_psiqrh.py
	@echo ""
	@echo "✅ Physics tests completed!"

# Validate Complete model
validate-complete:
	@echo "🔍 Validating PsiQRHTransformerComplete Model"
	@echo "=============================================="
	@echo "Model directory: $(MODEL_DIR)"
	@echo ""
	@if [ ! -d "$(MODEL_DIR)" ]; then \
		echo "❌ Error: Model directory not found: $(MODEL_DIR)"; \
		echo "💡 Train a model first: make train-complete"; \
		exit 1; \
	fi
	@python3 validate_training_output.py --model_dir $(MODEL_DIR) --device cpu --text_file $(TEXT_FILE)
	@echo ""
	@echo "📊 Next step: make test-physics to run physics validation"

# Full pipeline with Complete implementation
train-full-complete:
	@echo "🚀 FULL PIPELINE: Train + Validate + Physics Tests"
	@echo "===================================================="
	@make train-complete MODEL_DIR=$(MODEL_DIR) EPOCHS=$(EPOCHS) BATCH_SIZE=$(BATCH_SIZE)
	@echo ""
	@echo "⏳ Waiting 2 seconds before validation..."
	@sleep 2
	@make validate-complete MODEL_DIR=$(MODEL_DIR)
	@echo ""
	@echo "⏳ Waiting 2 seconds before physics tests..."
	@sleep 2
	@make test-physics
	@echo ""
	@echo "🎉 COMPLETE PIPELINE FINISHED!"
	@echo "✅ Training: Done"
	@echo "✅ Validation: Done"
	@echo "✅ Physics Tests: Done"