#!/usr/bin/env python3
"""
Teste de integração da API com o novo ΨQRHPipeline
"""

import sys
import os

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def test_pipeline_initialization():
    """Testa se o pipeline inicializa corretamente"""
    print("🧪 Testando inicialização do ΨQRHPipeline...")

    try:
        from psiqrh import ΨQRHPipeline
        pipeline = ΨQRHPipeline(task="text-generation", device="cpu")
        print("✅ Pipeline inicializado com sucesso")
        return pipeline
    except Exception as e:
        print(f"❌ Erro na inicialização do pipeline: {e}")
        return None

def test_pipeline_processing(pipeline):
    """Testa se o pipeline processa texto corretamente"""
    print("\n🧪 Testando processamento de texto...")

    if pipeline is None:
        print("❌ Pipeline não disponível para teste")
        return False

    try:
        test_text = "Olá, este é um teste de integração."
        result = pipeline(test_text)

        if result.get('status') == 'success':
            print("✅ Processamento bem-sucedido")
            print(f"   📝 Resposta: {result.get('response', '')[:100]}...")
            print(f"   📊 Métricas físicas: {result.get('physical_metrics', {})}")
            return True
        else:
            print(f"❌ Processamento falhou: {result.get('error', 'Erro desconhecido')}")
            return False

    except Exception as e:
        print(f"❌ Erro no processamento: {e}")
        return False

def test_api_structure():
    """Testa se a estrutura da API está correta"""
    print("\n🧪 Testando estrutura da API...")

    try:
        # Simular a estrutura da API sem Flask
        from psiqrh import ΨQRHPipeline

        # Simular inicialização da API
        qrh_pipeline = ΨQRHPipeline(task="text-generation", device="cpu")

        # Simular processamento de chat
        test_message = "Teste de mensagem"

        # Simular o processamento que aconteceria na API
        result = qrh_pipeline(test_message)

        # Verificar se a resposta tem a estrutura esperada
        expected_keys = ['status', 'response', 'physical_metrics', 'mathematical_validation']
        actual_keys = list(result.keys())

        # Verificar se pelo menos as chaves essenciais estão presentes
        essential_keys = ['status', 'response']
        missing_essential = [key for key in essential_keys if key not in actual_keys]

        if missing_essential:
            print(f"❌ Chaves essenciais faltando na resposta: {missing_essential}")
            print(f"   Chaves disponíveis: {actual_keys}")
            return False

        # Verificar se response é uma string não vazia
        if not isinstance(result.get('response'), str) or not result.get('response').strip():
            print(f"❌ Resposta inválida: {result.get('response')}")
            return False

        print("✅ Estrutura da API compatível")
        return True

    except Exception as e:
        print(f"❌ Erro na estrutura da API: {e}")
        return False

def main():
    """Função principal de teste"""
    print("🚀 Iniciando testes de integração ΨQRH API ↔ Pipeline")
    print("=" * 60)

    # Teste 1: Inicialização do pipeline
    pipeline = test_pipeline_initialization()

    # Teste 2: Processamento de texto
    processing_ok = test_pipeline_processing(pipeline)

    # Teste 3: Estrutura da API
    api_ok = test_api_structure()

    print("\n" + "=" * 60)
    print("📊 RESULTADO DOS TESTES:")

    if pipeline is not None and processing_ok and api_ok:
        print("✅ Todos os testes passaram! Integração bem-sucedida.")
        return 0
    else:
        print("❌ Alguns testes falharam. Verificar implementação.")
        return 1

if __name__ == "__main__":
    sys.exit(main())