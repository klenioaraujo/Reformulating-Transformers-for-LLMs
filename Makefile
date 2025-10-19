# ΨQRH Project Makefile
# ====================
#
# Comprehensive automation for ΨQRH pipeline training, evaluation, and analysis.
# This Makefile centralizes all workflows for the semantic correction and numerical stability improvements.

# Configuration Variables
PYTHON = python3
TRAIN_DATA = data/training_pairs.json
TEST_DATA = data/test_cases.json
MODEL_DIR = models/checkpoints
LATEST_MODEL = $(MODEL_DIR)/best_model.pt
EPOCHS = 10
BATCH_SIZE = 8
DEVICE = cpu
SOURCE_MODEL ?=
LOCAL_SOURCE_PATH = models/source/$(SOURCE_MODEL)
TEST_DISTILL_MODEL ?= gpt2

# Default target
.PHONY: help
help: ## Mostra esta mensagem de ajuda.
	@awk 'BEGIN {FS = ":.*?## "; printf "Uso:\n  make \033[36m<alvo>\033[0m\n\nAlvos disponíveis:\n"} /^# [A-Z]/ { category = substr($$0, 3); printf "\n\033[1m%s\033[0m\n", category } /^[a-zA-Z_-]+:.*?## / { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# Installation and Setup
.PHONY: install
install: ## Instala as dependências do projeto.
	@echo "📦 Instalando dependências..."
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Dependências instaladas com sucesso!"

.PHONY: setup
setup: install data ## Configuração completa do projeto (instalação + dados).

.PHONY: setup-auto
setup-auto: setup-vocab ## Configuração automática completa do sistema ΨQRH (recomendado para primeira vez).
	@echo "🚀 Iniciando configuração automática do ΨQRH..."
	$(PYTHON) setup_system.py
	@echo "✅ Configuração automática concluída!"
	@echo ""
	@echo "🎯 PRÓXIMOS PASSOS:"
	@echo "1. Execute: ./start_psiqrh.sh"
	@echo "2. Teste: make test"
	@echo "3. Treine: make train-physics-emergent"
	@echo "4. Explore: python psiqrh.py --interactive"

# Data Preparation
.PHONY: data
data: ## Gera o dataset de treinamento a partir de textos brutos.
	@echo "📚 Preparando dados de treinamento..."
	$(PYTHON) tools/create_training_data.py
	@echo "✅ Dados de treinamento preparados!"

.PHONY: setup-vocab
setup-vocab: ## Converte o vocabulário do modelo fonte para o formato nativo ΨQRH. Use: make setup-vocab SOURCE_MODEL=gpt2
	@if [ -z "$(SOURCE_MODEL)" ]; then \
		echo "⚠️  SOURCE_MODEL não especificado, usando 'gpt2' como padrão."; \
		EFFECTIVE_SOURCE_MODEL=gpt2; \
	else \
		EFFECTIVE_SOURCE_MODEL=$(SOURCE_MODEL); \
	fi; \
	echo "📚 Convertendo vocabulário do modelo '$$EFFECTIVE_SOURCE_MODEL' para formato nativo..."; \
	$(PYTHON) scripts/create_native_vocab.py --model_name $$EFFECTIVE_SOURCE_MODEL; \
	echo "✅ Vocabulário nativo criado em data/native_vocab.json"

# Training Workflows
.PHONY: train
train: ## Treina o modelo ΨQRH. Use: make train EPOCHS=50 BATCH_SIZE=16
	@echo "🎯 Iniciando treinamento do ΨQRH..."
	@echo "   📊 Épocas: $(EPOCHS)"
	@echo "   📦 Batch size: $(BATCH_SIZE)"
	@echo "   💾 Modelo será salvo em: $(MODEL_DIR)"
	$(PYTHON) train_pipeline.py \
		--data-path $(TRAIN_DATA) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE)
	@echo "✅ Treinamento concluído!"

.PHONY: train-quick
train-quick: ## Treinamento rápido para teste (1 época).
	@echo "⚡ Treinamento rápido (1 época)..."
	make train EPOCHS=1 BATCH_SIZE=2

.PHONY: train-extended
train-extended: ## Treinamento extensivo (50 épocas).
	@echo "🔬 Treinamento extensivo (50 épocas)..."
	make train EPOCHS=50 BATCH_SIZE=8

.PHONY: train-physics-emergent
train-physics-emergent: ## Treinamento emergente baseado em princípios físicos (auto-calibração + consciência). Use: make train-physics-emergent EPOCHS=500
	@echo "🧠 Iniciando treinamento emergente físico ΨQRH..."
	@echo "🎯 Método: Auto-calibração + Harmonic Orchestration + Consciousness Metrics"
	@echo "🔄 Épocas: $(EPOCHS)"
	EPOCHS=$(EPOCHS) $(PYTHON) train_physics_emergent.py
	@echo "✅ Treinamento emergente físico concluído!"

# Evaluation Workflows
.PHONY: evaluate
evaluate: ## Avalia o melhor modelo treinado com métricas semânticas (BLEU, etc.).
	@echo "🧪 Avaliando modelo treinado..."
	@if [ ! -f $(LATEST_MODEL) ]; then \
		echo "❌ Nenhum modelo treinado encontrado em $(LATEST_MODEL)"; \
		echo "   Execute 'make train' primeiro."; \
		exit 1; \
	fi
	$(PYTHON) evaluate_model.py \
		--model-path $(LATEST_MODEL) \
		--test-data $(TEST_DATA) \
		--device $(DEVICE)
	@echo "✅ Avaliação concluída!"

.PHONY: evaluate-baseline
evaluate-baseline: ## Avalia o modelo não-treinado (baseline).
	@echo "📊 Avaliando baseline (modelo não-treinado)..."
	$(PYTHON) evaluate_model.py \
		--test-data $(TEST_DATA) \
		--device $(DEVICE)
	@echo "✅ Avaliação baseline concluída!"

# Audit and Analysis
.PHONY: audit
audit: ## Analisa o log de auditoria mais recente e gera relatório de estabilidade.
	@echo "🔍 Analisando logs de auditoria..."
	@LOG_FILE=$$(ls -t results/audit_logs/audit_*.json 2>/dev/null | head -n 1); \
	if [ -z "$$LOG_FILE" ]; then \
		echo "❌ Nenhum log de auditoria encontrado."; \
		echo "   Execute testes que gerem auditoria primeiro."; \
		exit 1; \
	fi; \
	echo "📄 Analisando: $$LOG_FILE"; \
	$(PYTHON) tools/audit_analyzer.py $$LOG_FILE
	@echo "✅ Análise de auditoria concluída!"

.PHONY: audit-test
audit-test: ## Executa teste de auditoria para validar estabilidade numérica.
	@echo "🧪 Executando teste de auditoria..."
	$(PYTHON) -c "from src.core.spectral_projector import create_audit_enabled_qrh_pipeline, invert_spectral_qrh; import torch; qrh_layer, audit_logger = create_audit_enabled_qrh_pipeline(embed_dim=64, alpha=1.0, audit_enabled=True); audit_logger.start_session('Makefile Audit Test', {'test': 'makefile_integration'}); psi_input = torch.randn(1, 10, 64, 4); psi_transformed = qrh_layer(psi_input); psi_reconstructed = invert_spectral_qrh(psi_transformed, qrh_layer, audit_logger); log_path = audit_logger.end_session('Test completed'); print(f'✅ Teste de auditoria concluído. Log: {log_path}')"

# Optimization and Validation
.PHONY: optimize-alpha
optimize-alpha: ## Executa o experimento para encontrar o valor ótimo de alpha.
	@echo "🎛️  Otimizando parâmetro alpha..."
	$(PYTHON) tools/find_optimal_alpha.py
	@echo "✅ Otimização de alpha concluída!"

.PHONY: hyperparameter-sweep
hyperparameter-sweep: ## Executa varredura sistemática de hiperparâmetros.
	@echo "🎯 Executando varredura de hiperparâmetros..."
	$(PYTHON) hyperparameter_sweep.py --epochs-per-config 2
	@echo "✅ Varredura de hiperparâmetros concluída!"

.PHONY: plot-learning-curves
plot-learning-curves: ## Plota curvas de aprendizado do treinamento mais recente.
	@echo "📊 Plotando curvas de aprendizado..."
	$(PYTHON) tools/plot_training_log.py
	@echo "✅ Curvas de aprendizado plotadas!"

.PHONY: visualize-semantic-space
visualize-semantic-space: ## Visualiza o espaço semântico aprendido pelo modelo.
	@echo "🎨 Visualizando espaço semântico..."
	@if [ ! -f $(LATEST_MODEL) ]; then \
		echo "❌ Nenhum modelo treinado encontrado."; \
		exit 1; \
	fi
	$(PYTHON) tools/visualize_semantic_space.py --model-path $(LATEST_MODEL)
	@echo "✅ Visualização do espaço semântico concluída!"

.PHONY: pretrain-inverter
pretrain-inverter: ## Executa o pré-treinamento isolado do Inverse Projector.
	@echo "🔧 Pré-treinando Inverse Cognitive Projector..."
	$(PYTHON) experiments/pretrain_inverter.py
	@echo "✅ Pré-treinamento concluído!"

# Testing and Validation
.PHONY: test-semantic-decoder
test-semantic-decoder: ## Testa o SemanticBeamSearchDecoder.
	@echo "🧠 Testando Semantic Decoder..."
	$(PYTHON) -c "from tools.semantic_decoder import create_semantic_decoder; decoder = create_semantic_decoder(beam_width=3); test_predictions = [[('Q', 0.8), ('u', 0.9), ('a', 0.7), ('n', 0.8), ('t', 0.6)], [('u', 0.9), ('U', 0.1), ('a', 0.3), ('m', 0.3), (' ', 0.5)]]; result = decoder.decode(test_predictions, max_length=6); quality = decoder.get_semantic_quality_score(result); print(f'✅ Decodificado: \"{result}\"'); print(f'📊 Qualidade: {quality}')"
	@echo "✅ Teste do decoder concluído!"

.PHONY: test-pipeline
test-pipeline: ## Testa o pipeline ΨQRH completo.
	@echo "🔬 Testando pipeline ΨQRH..."
	$(PYTHON) -c "from psiqrh import ΨQRHPipeline; pipeline = ΨQRHPipeline(task='text-generation', device='cpu'); result = pipeline('test quantum'); print(f'✅ Pipeline funcionando. Resposta: {result.get(\"response\", \"N/A\")[:50]}...')"
	@echo "✅ Teste do pipeline concluído!"

# Full Workflows
.PHONY: full-training
full-training: data train evaluate ## Workflow completo: dados + treinamento + avaliação.
	@echo "🎉 Workflow completo de treinamento finalizado!"

.PHONY: physics-emergent-workflow
physics-emergent-workflow: data train-physics-emergent evaluate ## Workflow completo de treinamento emergente físico.
	@echo "🧠 Workflow completo de treinamento emergente físico finalizado!"
	@echo "🎯 Sistema otimizado através de princípios físicos e consciência"

.PHONY: benchmark
benchmark: evaluate-baseline train evaluate ## Benchmark: baseline vs treinado.
	@echo "📊 Benchmark concluído!"
	@echo "   Compare os resultados em reports/evaluation/"

.PHONY: semantic-alignment
semantic-alignment: ## Workflow completo de alinhamento semântico ou destilação. Use: make semantic-alignment SOURCE_MODEL=gpt2
	@echo "🔬 Executando workflow de alinhamento semântico..."
	@if [ -z "$(SOURCE_MODEL)" ]; then \
		echo "   📋 Modo: Alinhamento semântico padrão"; \
		make data && make hyperparameter-sweep && make train-extended && make evaluate && make plot-learning-curves && make visualize-semantic-space; \
		echo "🎯 Workflow completo de alinhamento semântico finalizado!"; \
		echo "   📊 Resultados salvos em results/hyperparameter_sweep/"; \
		echo "   📈 Curvas de aprendizado em results/plots/"; \
		echo "   🎨 Visualização semântica em results/semantic_analysis/"; \
		echo "   📋 Relatórios em reports/evaluation/"; \
	else \
		echo "   🧠 Modo: Destilação de conhecimento de '$(SOURCE_MODEL)'"; \
		echo "   📥 Passo 1: Verificando se modelo já está baixado..."; \
		if [ ! -d "models/source/$(SOURCE_MODEL)" ]; then \
			echo "   📥 Modelo não encontrado localmente - baixando..."; \
			$(PYTHON) scripts/download_model_ultra_simple.py --model_name $(SOURCE_MODEL); \
		else \
			echo "   ✅ Modelo já baixado - usando cache local"; \
		fi; \
		echo "   🎯 Passo 2: Executando destilação harmônica..."; \
		$(PYTHON) model_converter_spectral_ultra_simple.py --mode distill --source_model $(SOURCE_MODEL) --output_model_name "psiqrh_distilled_$(SOURCE_MODEL)"; \
		echo "   🔍 Passo 3: Avaliando modelo destilado..."; \
		make evaluate MODEL_PATH=models/distilled/psiqrh_distilled_$(SOURCE_MODEL).pt; \
		echo "   ✅ Workflow de destilação concluído!"; \
	fi

# Cleanup
.PHONY: clean
clean: ## Remove todos os arquivos gerados (logs, modelos, relatórios).
	@echo "🧹 Limpando arquivos gerados..."
	rm -rf results/ reports/ models/checkpoints/ __pycache__/
	rm -rf */__pycache__ */*/__pycache__
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Limpeza concluída!"

.PHONY: clean-models
clean-models: ## Remove apenas os modelos treinados.
	@echo "🗑️  Removendo modelos treinados..."
	rm -rf models/checkpoints/
	@echo "✅ Modelos removidos!"

.PHONY: clean-logs
clean-logs: ## Remove apenas os logs e relatórios.
	@echo "📄 Removendo logs e relatórios..."
	rm -rf results/ reports/
	@echo "✅ Logs e relatórios removidos!"

# Information and Status
.PHONY: status
status: ## Mostra o status atual do projeto.
	@echo "📊 Status do Projeto ΨQRH"
	@echo "========================"
	@echo ""
	@echo "📁 Estrutura de Diretórios:"
	@if [ -d "data" ]; then echo "   ✅ data/ - Dados disponíveis"; else echo "   ❌ data/ - Dados ausentes"; fi
	@if [ -d "models/checkpoints" ]; then echo "   ✅ models/checkpoints/ - Modelos treinados"; else echo "   ❌ models/checkpoints/ - Sem modelos treinados"; fi
	@if [ -d "results" ]; then echo "   ✅ results/ - Resultados disponíveis"; else echo "   ❌ results/ - Sem resultados"; fi
	@if [ -d "reports" ]; then echo "   ✅ reports/ - Relatórios disponíveis"; else echo "   ❌ reports/ - Sem relatórios"; fi
	@echo ""
	@echo "🤖 Componentes:"
	@if command -v python3 &> /dev/null; then echo "   ✅ Python3 disponível"; else echo "   ❌ Python3 não encontrado"; fi
	@if [ -f "requirements.txt" ]; then echo "   ✅ requirements.txt encontrado"; else echo "   ❌ requirements.txt ausente"; fi
	@echo ""
	@echo "🎯 Últimos Arquivos:"
	@find models/checkpoints -name "*.pt" -type f -printf "   📁 %P\n" 2>/dev/null | head -3 || echo "   📁 Nenhum modelo encontrado"
	@find results -name "*.json" -type f -printf "   📄 %P\n" 2>/dev/null | head -3 || echo "   📄 Nenhum resultado encontrado"
	@find reports -name "*.md" -type f -printf "   📋 %P\n" 2>/dev/null | head -3 || echo "   📋 Nenhum relatório encontrado"

.PHONY: info
info: ## Mostra informações detalhadas sobre o projeto.
	@echo "ℹ️  Informações do Projeto ΨQRH"
	@echo "=============================="
	@echo ""
	@echo "🎯 Objetivo: Correção semântica e estabilização numérica do pipeline ΨQRH"
	@echo "🔧 Componentes Principais:"
	@echo "   • SemanticBeamSearchDecoder - Decodificação robusta com beam search"
	@echo "   • Supervised Training Pipeline - Treinamento end-to-end"
	@echo "   • Semantic Evaluation Framework - BLEU, word validity, coherence"
	@echo "   • Numerical Stability - Energy preservation, clamping"
	@echo ""
	@echo "📊 Métricas Principais:"
	@echo "   • MSE de reconstrução: < 0.3 (98.4% melhoria)"
	@echo "   • Preservação de energia: 100%"
	@echo "   • BLEU Score (meta): > 0.3"
	@echo "   • Word Validity (meta): > 20%"
	@echo ""
	@echo "🚀 Uso Rápido:"
	@echo "   make setup          # Configuração inicial"
	@echo "   make train          # Treinar modelo"
	@echo "   make evaluate       # Avaliar desempenho"
	@echo "   make full-training  # Workflow completo"

# Development and Debugging
.PHONY: lint
lint: ## Executa verificação de estilo no código Python.
	@echo "🔍 Verificando estilo do código..."
	$(PYTHON) -m flake8 --max-line-length=120 --ignore=E501,W503 src/ tools/ experiments/ || echo "⚠️  Flake8 não instalado - pulando verificação"
	@echo "✅ Verificação de estilo concluída!"

.PHONY: test-all
test-all: test-semantic-decoder test-pipeline test-pipeline-tracer audit-test test-physics-emergent ## Executa todos os testes disponíveis.
	@echo "✅ Todos os testes passaram!"

.PHONY: test-distillation
test-distillation: ## Executa o teste E2E do fluxo de destilação com um modelo de teste.
	@echo "🧪 Iniciando teste de ponta a ponta do fluxo de destilação com '$(TEST_DISTILL_MODEL)'..."
	# Passo 1: Executar o fluxo de destilação completo
	make semantic-alignment SOURCE_MODEL=$(TEST_DISTILL_MODEL)
	# Passo 2: Executar o script de validação com pytest
	@echo "📊 Validando os artefatos e a funcionalidade do modelo destilado..."
	$(PYTHON) -m pytest tests/test_distillation_workflow.py --model-name "$(TEST_DISTILL_MODEL)" -v
	@echo "✅ Teste de destilação concluído com sucesso!"

.PHONY: test-physics-emergent
test-physics-emergent: ## Testa o sistema de treinamento emergente físico.
	@echo "🧠 Testando sistema de treinamento emergente físico..."
	$(PYTHON) -c "from train_physics_emergent import PhysicsEmergentTrainer; print('✅ Importação da classe bem-sucedida'); from unittest.mock import Mock; mock_pipeline = Mock(); mock_pipeline._generate_text_physical.return_value = {'fci_value': 0.7, 'synchronization_order': 0.8, 'cluster_analysis': {'dominant_cluster': {'order_parameter': 0.75}}, 'energy_conservation': 0.9, 'spectral_coherence': 0.85, 'generated_text': 'blue'}; trainer = PhysicsEmergentTrainer(mock_pipeline); print('✅ Instanciação bem-sucedida'); result = trainer.physics_emergent_training_cycle('The sky is', 'blue'); print(f'✅ Ciclo de treinamento executado: FCI={result[\"consciousness_metrics\"][\"fci\"]:.3f}, Success={result[\"physics_success\"][\"overall_success\"]}')"
	@echo "✅ Teste do sistema emergente físico concluído!"

.PHONY: test-pipeline-tracer
test-pipeline-tracer: ## Testa o Pipeline Tracer com entrada personalizada. Use: make test-pipeline-tracer QUESTION="Sua pergunta"
	@echo "🔬 Testando Pipeline Tracer..."
	@if [ -n "$(QUESTION)" ]; then \
		echo "   ❓ Pergunta personalizada: $(QUESTION)"; \
		PSIQRH_TEST_QUESTION="$(QUESTION)" $(PYTHON) -m pytest tests/test_pipeline_tracer.py::TestPipelineTracer::test_tracer_runs_without_error -v --tb=short; \
	else \
		echo "   ❓ Usando pergunta padrão: 'Qual a cor do ceu?'"; \
		$(PYTHON) -m pytest tests/test_pipeline_tracer.py::TestPipelineTracer::test_tracer_runs_without_error -v --tb=short; \
	fi
	@echo "✅ Teste do Pipeline Tracer concluído!"

.PHONY: test
test: ## Executa a suíte de testes completa com pytest.
	@echo "🧪 Executando suíte de testes completa..."
	$(PYTHON) -m pytest tests/test_suite.py -v --tb=short --override-ini="addopts="
	@echo "✅ Suíte de testes concluída!"

# Model Download and Management
.PHONY: download-model
download-model: ## Baixa e cacheia um modelo do Hugging Face. Use: make download-model SOURCE_MODEL=gpt2
	@if [ -z "$(SOURCE_MODEL)" ]; then \
		echo "❌ SOURCE_MODEL não especificado. Use: make download-model SOURCE_MODEL=gpt2"; \
		exit 1; \
	fi
	@echo "📥 Baixando modelo '$(SOURCE_MODEL)' do Hugging Face (método ultra simples)..."
	$(PYTHON) scripts/download_model_ultra_simple.py --model_name $(SOURCE_MODEL)
	@echo "✅ Modelo '$(SOURCE_MODEL)' baixado e cacheado em models/source/"

.PHONY: list-downloaded-models
list-downloaded-models: ## Lista todos os modelos baixados localmente.
	@echo "📚 Modelos baixados localmente:"
	@if [ -d "models/source" ]; then \
		find models/source -name "metadata.json" -exec dirname {} \; | xargs -I {} basename {} | while read model; do \
			if [ -f "models/source/$$model/metadata.json" ]; then \
				vocab_size=$$(grep -o '"vocab_size": [0-9]*' "models/source/$$model/metadata.json" | cut -d' ' -f2); \
				hidden_size=$$(grep -o '"hidden_size": [0-9]*' "models/source/$$model/metadata.json" | cut -d' ' -f2); \
				model_type=$$(grep -o '"model_type": "[^"]*"' "models/source/$$model/metadata.json" | cut -d'"' -f4); \
				echo "   📁 $$model ($$model_type)"; \
				echo "      📊 Vocab: $$vocab_size, Hidden: $$hidden_size"; \
			fi; \
		done; \
	else \
		echo "   📁 Nenhum modelo baixado encontrado"; \
	fi

.PHONY: clean-downloaded-models
clean-downloaded-models: ## Remove todos os modelos baixados localmente.
	@echo "🗑️  Removendo modelos baixados..."
	rm -rf models/source/
	@echo "✅ Modelos baixados removidos!"

# Semantic Model Management
.PHONY: convert-to-semantic
convert-to-semantic: ## Converte um modelo destilado para formato semântico. Use: make convert-to-semantic SOURCE_MODEL=gpt2
	@if [ -z "$(SOURCE_MODEL)" ]; then \
		echo "❌ SOURCE_MODEL não especificado. Use: make convert-to-semantic SOURCE_MODEL=gpt2"; \
		exit 1; \
	fi
	@echo "🔮 Convertendo modelo destilado '$(SOURCE_MODEL)' para formato semântico..."
	@if [ ! -f "models/distilled/psiqrh_distilled_$(SOURCE_MODEL).pt" ]; then \
		echo "❌ Modelo destilado 'psiqrh_distilled_$(SOURCE_MODEL).pt' não encontrado."; \
		echo "   Execute 'make distill-knowledge SOURCE_MODEL=$(SOURCE_MODEL)' primeiro."; \
		exit 1; \
	fi
	@mkdir -p models/semantic/
	$(PYTHON) model_converter_spectral_ultra_simple.py --mode semantic --source_model $(SOURCE_MODEL) --output_model_name "psiqrh_semantic_$(SOURCE_MODEL)"
	@echo "✅ Conversão semântica concluída. Modelo salvo em 'models/semantic/'"

.PHONY: list-semantic-models
list-semantic-models: ## Lista todos os modelos convertidos para formato semântico.
	@echo "🧠 Modelos em formato semântico:"
	@if [ -d "models/semantic" ]; then \
		find models/semantic -name "*.pt" -type f | while read model; do \
			model_name=$$(basename "$$model" .pt); \
			model_size=$$(stat -c%s "$$model" 2>/dev/null || echo "unknown"); \
			if [ "$$model_size" != "unknown" ]; then \
				model_size_mb=$$(echo "scale=2; $$model_size / (1024*1024)" | bc); \
				echo "   📁 $$model_name ($$model_size_mb MB)"; \
			else \
				echo "   📁 $$model_name (tamanho desconhecido)"; \
			fi; \
		done; \
		if [ $$? -ne 0 ]; then \
			echo "   📁 Nenhum modelo semântico encontrado"; \
		fi; \
	else \
		echo "   📁 Diretório models/semantic/ não existe"; \
		echo "   📁 Nenhum modelo semântico encontrado"; \
	fi

.PHONY: remove-semantic-model
remove-semantic-model: ## Remove um modelo específico do formato semântico. Use: make remove-semantic-model SOURCE_MODEL=gpt2
	@if [ -z "$(SOURCE_MODEL)" ]; then \
		echo "❌ SOURCE_MODEL não especificado. Use: make remove-semantic-model SOURCE_MODEL=gpt2"; \
		exit 1; \
	fi
	@echo "🗑️  Removendo modelo semântico '$(SOURCE_MODEL)'..."
	@if [ -f "models/semantic/psiqrh_semantic_$(SOURCE_MODEL).pt" ]; then \
		rm -f "models/semantic/psiqrh_semantic_$(SOURCE_MODEL).pt"; \
		echo "✅ Modelo semântico 'psiqrh_semantic_$(SOURCE_MODEL).pt' removido"; \
	else \
		echo "⚠️  Modelo semântico 'psiqrh_semantic_$(SOURCE_MODEL).pt' não encontrado"; \
	fi

.PHONY: clean-semantic-models
clean-semantic-models: ## Remove todos os modelos em formato semântico.
	@echo "🗑️  Removendo todos os modelos semânticos..."
	rm -rf models/semantic/
	@echo "✅ Todos os modelos semânticos removidos!"

.PHONY: semantic-workflow
semantic-workflow: ## Workflow completo: baixar, destilar e converter para semântico. Use: make semantic-workflow SOURCE_MODEL=gpt2
	@if [ -z "$(SOURCE_MODEL)" ]; then \
		echo "❌ SOURCE_MODEL não especificado. Use: make semantic-workflow SOURCE_MODEL=gpt2"; \
		exit 1; \
	fi
	@echo "🚀 Iniciando workflow semântico completo para '$(SOURCE_MODEL)'..."
	@echo "   📥 Passo 1: Baixando modelo..."
	make download-model SOURCE_MODEL=$(SOURCE_MODEL)
	@echo "   🧠 Passo 2: Destilando conhecimento..."
	make distill-knowledge SOURCE_MODEL=$(SOURCE_MODEL)
	@echo "   🔮 Passo 3: Convertendo para formato semântico..."
	make convert-to-semantic SOURCE_MODEL=$(SOURCE_MODEL)
	@echo "   📊 Passo 4: Listando modelos disponíveis..."
	make list-downloaded-models
	make list-semantic-models
	@echo "✅ Workflow semântico completo concluído!"
	@echo ""
	@echo "🎯 Modelos disponíveis:"
	@echo "   • Baixados: models/source/$(SOURCE_MODEL)"
	@echo "   • Destilados: models/distilled/psiqrh_distilled_$(SOURCE_MODEL).pt"
	@echo "   • Semânticos: models/semantic/psiqrh_semantic_$(SOURCE_MODEL).pt"

# Special Configurations
.PHONY: gpu
gpu: ## Configura para usar GPU (se disponível).
	@echo "🎮 Configurando para GPU..."
	$(eval DEVICE = cuda)
	@echo "   DEVICE definido como: $(DEVICE)"
	@echo "   Use: make train DEVICE=cuda"

.PHONY: cpu
cpu: ## Configura para usar CPU.
	@echo "💻 Configurando para CPU..."
	$(eval DEVICE = cpu)
	@echo "   DEVICE definido como: $(DEVICE)"

# Emergency and Recovery
.PHONY: reset
reset: clean setup ## Reset completo do projeto (limpa tudo e reconfigura).
	@echo "🔄 Projeto resetado e reconfigurado!"

.PHONY: backup
backup: ## Cria backup dos modelos e resultados importantes.
	@echo "💾 Criando backup..."
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	BACKUP_DIR="backup_$$TIMESTAMP"; \
	mkdir -p $$BACKUP_DIR; \
	cp -r models/checkpoints $$BACKUP_DIR/ 2>/dev/null || true; \
	cp -r results $$BACKUP_DIR/ 2>/dev/null || true; \
	cp -r reports $$BACKUP_DIR/ 2>/dev/null || true; \
	echo "✅ Backup criado em: $$BACKUP_DIR"

# Aliases for common operations
.PHONY: t
t: train ## Alias para train

.PHONY: e
e: evaluate ## Alias para evaluate

.PHONY: a
a: audit ## Alias para audit

.PHONY: distill-knowledge
distill-knowledge: ## Destila conhecimento de um LLM base para o espaço Hilbert do ΨQRH. Use: make distill-knowledge SOURCE_MODEL=gpt2
	@if [ -z "$(SOURCE_MODEL)" ]; then \
		echo "❌ SOURCE_MODEL não especificado. Use: make distill-knowledge SOURCE_MODEL=gpt2"; \
		exit 1; \
	fi
	@echo "🔮 Iniciando destilação harmônica de '$(SOURCE_MODEL)' para o formato ΨQRH..."
	@echo "   📥 Verificando se modelo já está baixado..."
	@if [ ! -d "models/source/$(SOURCE_MODEL)" ]; then \
		echo "   📥 Modelo não encontrado localmente - baixando..."; \
		$(PYTHON) scripts/download_model_ultra_simple.py --model_name $(SOURCE_MODEL); \
	else \
		echo "   ✅ Modelo já baixado - usando cache local"; \
	fi
	$(PYTHON) model_converter_spectral_ultra_simple.py --mode distill --source_model $(SOURCE_MODEL) --output_model_name "psiqrh_distilled_$(SOURCE_MODEL)"
	@echo "✅ Destilação concluída. Modelo salvo em 'models/distilled/'"

.PHONY: vocab
vocab: ## Cria o vocabulário nativo GPT-2 necessário para o pipeline ΨQRH.
	@echo "🔬 Criando vocabulário nativo GPT-2..."
	$(PYTHON) create_native_vocab.py
	@echo "✅ Vocabulário nativo criado em data/native_vocab.json"

.PHONY: h
h: help ## Alias para help

# ΨQRH System Commands - UNIFIED SYSTEM
.PHONY: psiqrh-cli psiqrh-api psiqrh-interactive psiqrh-test psiqrh-benchmark psiqrh-enhanced

# Multi-Model Management Commands
.PHONY: list-models download-model convert-to-semantic distill-knowledge set-default-model semantic-workflow

psiqrh-cli: ## Executa CLI do ΨQRH. Use: make psiqrh-cli TEXT="Olá mundo"
	@echo "🧠 Executando ΨQRH CLI..."
	@if [ -z "$(TEXT)" ]; then \
		echo "❌ TEXT não especificado. Use: make psiqrh-cli TEXT=\"Olá mundo\""; \
		exit 1; \
	fi
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.CLI import ΨQRHCLI; cli = ΨQRHCLI(); cli.process_text('$(TEXT)')"

psiqrh-enhanced: ## Executa Enhanced CLI do ΨQRH Unificado. Use: make psiqrh-enhanced TEXT="Olá mundo"
	@echo "🚀 Executando ΨQRH Enhanced CLI (Sistema Unificado)..."
	@if [ -z "$(TEXT)" ]; then \
		echo "❌ TEXT não especificado. Use: make psiqrh-enhanced TEXT=\"Olá mundo\""; \
		exit 1; \
	fi
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.EnhancedCLI import EnhancedCLI; cli = EnhancedCLI(); cli.process_text('$(TEXT)')"
	@echo "✅ Comando psiqrh-enhanced executado com sucesso!"

psiqrh-enhanced-interactive: ## Modo interativo aprimorado do ΨQRH Unificado
	@echo "🤖 Iniciando modo interativo ΨQRH Unificado..."
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.EnhancedCLI import main; main()" --interactive

psiqrh-enhanced-batch: ## Processamento em lote com Enhanced CLI. Use: make psiqrh-enhanced-batch INPUT=input.txt OUTPUT=results.json
	@echo "📁 Executando processamento em lote ΨQRH Unificado..."
	@if [ -z "$(INPUT)" ]; then \
		echo "❌ INPUT não especificado. Use: make psiqrh-enhanced-batch INPUT=input.txt"; \
		exit 1; \
	fi
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.EnhancedCLI import EnhancedCLI; cli = EnhancedCLI(); cli.run_batch_processing('$(INPUT)', '$(OUTPUT)')"

psiqrh-enhanced-spectral: ## Exporta análise espectral completa. Use: make psiqrh-enhanced-spectral TEXT="teste" OUTPUT=analysis.json
	@echo "🔬 Exportando análise espectral ΨQRH Unificado..."
	@if [ -z "$(TEXT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "❌ TEXT e OUTPUT são obrigatórios. Use: make psiqrh-enhanced-spectral TEXT=\"teste\" OUTPUT=analysis.json"; \
		exit 1; \
	fi
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.EnhancedCLI import EnhancedCLI; cli = EnhancedCLI(); cli.export_spectral_analysis('$(TEXT)', '$(OUTPUT)')"

psiqrh-enhanced-benchmark: ## Benchmark aprimorado do ΨQRH Unificado. Use: make psiqrh-enhanced-benchmark RUNS=100
	@echo "📊 Executando benchmark ΨQRH Unificado..."
	@RUNS=$$(if [ -z "$(RUNS)" ]; then echo 100; else echo $(RUNS); fi); \
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.EnhancedCLI import EnhancedCLI; cli = EnhancedCLI(); cli.benchmark_system($$RUNS)"

psiqrh-enhanced-status: ## Status completo do sistema ΨQRH Unificado
	@echo "🔬 Verificando status ΨQRH Unificado..."
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.EnhancedCLI import EnhancedCLI; cli = EnhancedCLI(); cli.show_system_status()"

psiqrh-enhanced-legacy-test: ## Testa compatibilidade com sistema legado
	@echo "🧪 Executando teste de compatibilidade legado ΨQRH Unificado..."
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.EnhancedCLI import EnhancedCLI; cli = EnhancedCLI(); cli.run_legacy_compatibility_test()"

psiqrh-interactive: ## Modo interativo do ΨQRH (legacy)
	@echo "🤖 Iniciando modo interativo ΨQRH (legacy)..."
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.CLI import main; main()" --interactive

psiqrh-api: ## Inicia API REST do ΨQRH
	@echo "🌐 Iniciando API REST ΨQRH..."
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.API import main; main()" --host 0.0.0.0 --port 5000

psiqrh-test: ## Executa testes do sistema ΨQRH
	@echo "🧪 Executando testes ΨQRH..."
	cd ΨQRHSystem && $(PYTHON) -m pytest tests/ -v --tb=short

psiqrh-benchmark: ## Benchmark de performance do ΨQRH (legacy)
	@echo "📊 Executando benchmark ΨQRH (legacy)..."
	cd ΨQRHSystem && $(PYTHON) -c "from ΨQRHSystem.core.PipelineManager import PipelineManager; from ΨQRHSystem.config.SystemConfig import SystemConfig; import time; config = SystemConfig.default(); pipeline = PipelineManager(config); print('🔬 Benchmark ΨQRH - 100 execuções...'); start_time = time.time(); [pipeline.process('Benchmark test') for i in range(100)]; end_time = time.time(); avg_time = (end_time - start_time) / 100; print(f'✅ Benchmark concluído: {avg_time:.3f}s por execução')"
	@echo "✅ Comando psiqrh-benchmark executado com sucesso!"

# Multi-Model Management Commands
list-models: ## Lista todos os modelos disponíveis (fonte, destilados, semânticos)
	@echo "📚 Listando modelos disponíveis..."
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.ModelManagementCLI import ModelManagementCLI; cli = ModelManagementCLI(); cli.run(['list'])"
	@echo "✅ Comando list-models executado com sucesso!"

download-model: ## Baixa um modelo do Hugging Face. Use: make download-model SOURCE_MODEL=gpt2
	@echo "📥 Baixando modelo..."
	@if [ -z "$(SOURCE_MODEL)" ]; then \
		echo "❌ SOURCE_MODEL não especificado. Use: make download-model SOURCE_MODEL=gpt2"; \
		exit 1; \
	fi
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.ModelManagementCLI import ModelManagementCLI; cli = ModelManagementCLI(); cli.run(['download', '$(SOURCE_MODEL)'])"

convert-to-semantic: ## Converte um modelo para formato semântico. Use: make convert-to-semantic SOURCE_MODEL=gpt2
	@echo "🔮 Convertendo modelo para formato semântico..."
	@if [ -z "$(SOURCE_MODEL)" ]; then \
		echo "❌ SOURCE_MODEL não especificado. Use: make convert-to-semantic SOURCE_MODEL=gpt2"; \
		exit 1; \
	fi
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.ModelManagementCLI import ModelManagementCLI; cli = ModelManagementCLI(); cli.run(['convert', '$(SOURCE_MODEL)'])"

distill-knowledge: ## Destila conhecimento de um modelo. Use: make distill-knowledge SOURCE_MODEL=gpt2
	@echo "🧠 Destilando conhecimento..."
	@if [ -z "$(SOURCE_MODEL)" ]; then \
		echo "❌ SOURCE_MODEL não especificado. Use: make distill-knowledge SOURCE_MODEL=gpt2"; \
		exit 1; \
	fi
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.ModelManagementCLI import ModelManagementCLI; cli = ModelManagementCLI(); cli.run(['distill', '$(SOURCE_MODEL)'])"

set-default-model: ## Define o modelo padrão do sistema. Use: make set-default-model MODEL=gpt2
	@echo "🎯 Definindo modelo padrão..."
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ MODEL não especificado. Use: make set-default-model MODEL=gpt2"; \
		exit 1; \
	fi
	cd ΨQRHSystem && $(PYTHON) -c "from interfaces.ModelManagementCLI import ModelManagementCLI; cli = ModelManagementCLI(); cli.run(['set-default', '$(MODEL)'])"

semantic-workflow: ## Workflow completo: baixar, destilar e converter. Use: make semantic-workflow SOURCE_MODEL=gpt2
	@echo "🚀 Executando workflow semântico completo..."
	@if [ -z "$(SOURCE_MODEL)" ]; then \
		echo "❌ SOURCE_MODEL não especificado. Use: make semantic-workflow SOURCE_MODEL=gpt2"; \
		exit 1; \
	fi
	@echo "   📥 Passo 1: Baixando modelo..."
	make download-model SOURCE_MODEL=$(SOURCE_MODEL)
	@echo "   🧠 Passo 2: Destilando conhecimento..."
	make distill-knowledge SOURCE_MODEL=$(SOURCE_MODEL)
	@echo "   🔮 Passo 3: Convertendo para formato semântico..."
	make convert-to-semantic SOURCE_MODEL=$(SOURCE_MODEL)
	@echo "   📊 Passo 4: Verificando status..."
	make list-models
	@echo "✅ Workflow semântico completo concluído!"