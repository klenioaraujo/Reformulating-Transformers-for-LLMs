#!/usr/bin/env python3
"""
Teste de Inicialização com Sistema de Log de Dependências
========================================================

Este arquivo demonstra o uso do sistema de logging de dependências
com prompt engine para detectar e resolver conflitos de versões.

ΨQRH-PROMPT-ENGINE Usage Example
"""

import sys
import os
import time
from pathlib import Path

# Adicionar caminho do projeto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Inicializar o logger ANTES de qualquer import
from src.core.dependency_logger import DependencyLogger, function_context, log_function_dependencies
import tempfile

# Usar diretório temporário para logs
temp_dir = tempfile.mkdtemp()
log_dir = os.path.join(temp_dir, "dependency_logs")

print("🚀 Iniciando teste do sistema de log de dependências ΨQRH")
print("=" * 60)

# Inicializar logger
logger = DependencyLogger(log_dir=log_dir)
print(f"📝 Logger inicializado - Session ID: {logger.session_id}")
print(f"📁 Logs salvos em: {log_dir}")

# Simular função de processamento de dados
print("\n📊 Simulando função de processamento de dados...")
logger.set_function_context("data_processing")
    log_function_dependencies("data_processing_main", {
        "numpy": "1.26.0",
        "pandas": "2.0.3",
        "matplotlib": "3.7.0"
    })

    try:
        import numpy as np
        import pandas as pd
        print(f"✅ NumPy {np.__version__} importado com sucesso")
        print(f"✅ Pandas {pd.__version__} importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar: {e}")

# Simular função de machine learning
print("\n🤖 Simulando função de machine learning...")
with function_context("ml_training"):
    log_function_dependencies("neural_network_training", {
        "torch": "2.1.2",
        "numpy": "1.24.0",  # Versão DIFERENTE - vai gerar conflito!
        "matplotlib": "3.7.5"  # Versão ligeiramente diferente
    })

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} importado com sucesso")

        # Usar numpy novamente (mas logger já detectou o conflito)
        import numpy as np
        print(f"⚠️ NumPy {np.__version__} re-importado (possível conflito)")
    except ImportError as e:
        print(f"❌ Erro ao importar: {e}")

# Simular função de visualização
print("\n📈 Simulando função de visualização...")
with function_context("visualization"):
    log_function_dependencies("plot_generation", {
        "matplotlib": "3.7.5",
        "seaborn": "0.13.0",
        "plotly": "5.17.0"
    })

    try:
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib {plt.matplotlib.__version__} importado")
    except ImportError as e:
        print(f"❌ Erro ao importar matplotlib: {e}")

# Simular imports problemáticos
print("\n⚠️ Simulando imports com problemas conhecidos...")
with function_context("problematic_imports"):
    # Tentar importar algo que sabemos que vai dar erro
    try:
        import nonexistent_library
    except ImportError:
        print("❌ Import esperado de biblioteca inexistente (normal)")

    # Import lento simulado
    print("⏱️ Simulando import lento...")
    time.sleep(0.1)  # Simular demora

print("\n" + "=" * 60)
print("📊 GERANDO RELATÓRIO DE COMPATIBILIDADE")
print("=" * 60)

# Gerar relatório completo
compatibility_report = logger.generate_compatibility_report()
print(compatibility_report)

print("\n" + "=" * 60)
print("💾 SALVANDO LOGS E ANÁLISES")
print("=" * 60)

# Salvar todos os dados
logger.save_log()

# Análise de dados históricos (se existirem)
try:
    historical_analysis = logger.analyze_historical_conflicts()
    print("\n📈 ANÁLISE HISTÓRICA:")
    print(f"Total de sessões anteriores: {historical_analysis['total_sessions']}")
    print(f"Total de conflitos históricos: {historical_analysis['total_conflicts']}")

    if historical_analysis['recurring_conflicts']:
        print("\n🔄 Conflitos recorrentes:")
        for lib, count in historical_analysis['recurring_conflicts'].items():
            print(f"  - {lib}: {count} ocorrências")

    if historical_analysis['most_problematic']:
        lib, count = historical_analysis['most_problematic']
        print(f"\n🚨 Biblioteca mais problemática: {lib} ({count} conflitos)")

except Exception as e:
    print(f"ℹ️ Análise histórica não disponível: {e}")

# Dados para cruzamento
cross_ref_data = logger.get_cross_reference_data()
print(f"\n🔗 Dados de referência cruzada gerados para sessão {cross_ref_data['session_id']}")
print(f"Dependencies tracked: {len(cross_ref_data['dependencies'])}")
print(f"Conflicts logged: {len(cross_ref_data['conflicts'])}")

print("\n" + "=" * 60)
print("✅ TESTE DE INICIALIZAÇÃO CONCLUÍDO")
print("=" * 60)

# PROMPT ESPECÍFICO PARA ANÁLISE
ANALYSIS_PROMPT = """
ΨQRH-PROMPT-ENGINE: {
  "context": "Análise completa do sistema de log de dependências",
  "analysis": "Sistema detectou conflitos entre versões de numpy em funções diferentes",
  "solution": "Implementar estratégia de unificação de versões",
  "implementation": [
    "Detectar automaticamente conflitos de versão",
    "Sugerir versões compatíveis via IA",
    "Manter histórico de conflitos",
    "Gerar relatórios de compatibilidade"
  ],
  "validation": "Sistema funcionando corretamente com detecção de conflitos"
}

Este sistema permite:

1. 🔍 **Detecção Automática**: Monitora todas as importações em tempo real
2. ⚠️ **Conflitos de Versão**: Identifica quando diferentes funções precisam de versões diferentes
3. 🤖 **Análise Inteligente**: Usa prompt engine para sugerir resoluções
4. 📊 **Relatórios Detalhados**: Gera logs e relatórios de compatibilidade
5. 🔄 **Histórico**: Mantém registro para análise de padrões
6. 🔗 **Cruzamento de Dados**: Permite comparar entre sessões diferentes

Casos de Uso:
- Função A precisa numpy 1.26.0 para novos recursos
- Função B precisa numpy 1.24.0 por compatibilidade legada
- Sistema detecta conflito e sugere versão compatível
- Mantém log para futuras referências

O prompt engine analisa os conflitos e sugere resoluções inteligentes
baseadas no contexto específico de cada biblioteca e função.
"""

print("\n📋 PROMPT DE ANÁLISE GERADO:")
print(ANALYSIS_PROMPT)

print(f"\n🎯 Session ID para referência: {logger.session_id}")
print("Logs salvos em: logs/dependencies/")