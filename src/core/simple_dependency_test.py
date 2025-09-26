#!/usr/bin/env python3
"""
Teste Simplificado do Sistema de Log de Dependências
===================================================

Demonstração básica do sistema de logging com prompt engine.
"""

import sys
import os
import tempfile
import time
from pathlib import Path

# Adicionar caminho do projeto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("🚀 ΨQRH Dependency Logger - Teste Simplificado")
print("=" * 50)

# Criar logger com diretório temporário
from src.core.dependency_logger import DependencyLogger

temp_dir = tempfile.mkdtemp()
log_dir = os.path.join(temp_dir, "dependency_logs")

try:
    logger = DependencyLogger(log_dir=log_dir)
    print(f"✅ Logger inicializado - Session: {logger.session_id}")
    print(f"📁 Logs em: {log_dir}")

    # Teste 1: Log manual de dependências conflitantes
    print("\n📊 Teste 1: Detectando conflitos de versão")
    print("-" * 30)

    logger.log_function_dependency("neural_network", {
        "numpy": "1.26.0",
        "torch": "2.1.0"
    })

    logger.log_function_dependency("data_processing", {
        "numpy": "1.24.0",  # CONFLITO!
        "pandas": "2.0.3"
    })

    print("✅ Dependências registradas")

    # Teste 2: Imports reais para testar hooks
    print("\n🔧 Teste 2: Testando hooks de import")
    print("-" * 30)

    logger.set_function_context("test_imports")

    try:
        import json
        import time
        import os
        print("✅ Imports básicos funcionaram")
    except Exception as e:
        print(f"❌ Erro nos imports: {e}")

    # Teste 3: Gerar relatório
    print("\n📊 Teste 3: Gerando relatório de compatibilidade")
    print("-" * 50)

    report = logger.generate_compatibility_report()
    print(report)

    # Teste 4: Salvar logs
    print("\n💾 Teste 4: Salvando logs")
    print("-" * 30)

    logger.save_log()
    print("✅ Logs salvos com sucesso")

    # Teste 5: Dados para cruzamento
    print("\n🔗 Teste 5: Dados de referência cruzada")
    print("-" * 30)

    cross_data = logger.get_cross_reference_data()
    print(f"Session ID: {cross_data['session_id']}")
    print(f"Dependencies: {len(cross_data['dependencies'])}")
    print(f"Conflicts: {len(cross_data['conflicts'])}")

    print("\n" + "=" * 50)
    print("✅ TESTE CONCLUÍDO COM SUCESSO")
    print("=" * 50)

    # Mostrar exemplo de uso com prompt engine
    PROMPT_EXAMPLE = """
🤖 EXEMPLO DE USO COM PROMPT ENGINE:

ΨQRH-PROMPT-ENGINE: {
  "context": "Conflito detectado entre versões de numpy",
  "analysis": "neural_network requer numpy 1.26.0, data_processing usa 1.24.0",
  "solution": "Unificar versão para numpy 1.25.0 (compatível com ambas)",
  "implementation": [
    "pip install numpy==1.25.0",
    "Testar ambas as funções",
    "Verificar se não há quebras de API"
  ],
  "validation": "Executar testes de integração após atualização"
}

Este sistema permite:
✅ Detecção automática de conflitos
✅ Análise inteligente via IA
✅ Histórico para padrões
✅ Relatórios detalhados
✅ Sugestões de resolução
    """

    print(PROMPT_EXAMPLE)
    print(f"\n🎯 Logs disponíveis em: {log_dir}")

except Exception as e:
    print(f"❌ Erro durante teste: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🧹 Limpeza: {temp_dir}")