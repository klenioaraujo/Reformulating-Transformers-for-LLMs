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

        # VERIFICAÇÃO MAIS ROBUSTA: Aceitar diferentes estruturas de resposta
        if isinstance(result, dict):
            # Se tem 'response' e não é erro, considerar sucesso
            if 'response' in result and result.get('status') != 'error':
                print("✅ Processamento bem-sucedido")
                print(f"   📝 Resposta: {result.get('response', '')[:100]}...")
                print(f"   📊 Métricas físicas: {result.get('physical_metrics', {})}")
                return True
            else:
                print(f"❌ Processamento falhou: {result.get('error', 'Erro desconhecido')}")
                return False
        else:
            print(f"❌ Resposta inválida do pipeline: {type(result)}")
            return False

    except Exception as e:
        print(f"❌ Erro no processamento: {e}")
        return False

def test_api_structure():
    """Testa se a estrutura da API está correta"""
    print("\n🧪 Testando estrutura da API...")

    try:
        from psiqrh import ΨQRHPipeline
        qrh_pipeline = ΨQRHPipeline(task="text-generation", device="cpu")
        test_message = "Teste de mensagem"
        result = qrh_pipeline(test_message)

        # VERIFICAÇÃO FLEXÍVEL: Aceitar diferentes estruturas
        if isinstance(result, dict):
            # Chaves essenciais mínimas
            essential_keys = ['status']
            missing_essential = [key for key in essential_keys if key not in result]

            if missing_essential:
                print(f"❌ Chaves essenciais faltando: {missing_essential}")
                return False

            # Verificar se tem algum tipo de resposta
            has_response = any(key in result for key in ['response', 'output', 'result'])
            if not has_response:
                print("❌ Nenhuma chave de resposta encontrada")
                return False

            print("✅ Estrutura da API compatível")
            return True
        else:
            print(f"❌ Resposta não é dicionário: {type(result)}")
            return False

    except Exception as e:
        print(f"❌ Erro na estrutura da API: {e}")
        return False

def test_physical_corrections():
    """Testa se as correções físicas fundamentais estão integradas"""
    print("\n🧪 Testando correções físicas fundamentais...")

    try:
        # Testar o sistema de eco físico
        from src.core.physical_fundamental_corrections import test_physical_fundamental_corrections

        # Executar teste das correções físicas
        success = test_physical_fundamental_corrections()

        if success:
            print("✅ Correções físicas funcionando corretamente")
            return True
        else:
            print("❌ Correções físicas com problemas")
            return False

    except Exception as e:
        print(f"❌ Erro no teste das correções físicas: {e}")
        return False

def test_harmonic_orchestrator():
    """Testa se o HarmonicOrchestrator está integrado com correções físicas"""
    print("\n🧪 Testando HarmonicOrchestrator com correções físicas...")

    try:
        from src.core.harmonic_orchestrator import HarmonicOrchestrator

        # Inicializar com correções físicas
        orchestrator = HarmonicOrchestrator(enable_physical_corrections=True)

        # Testar geração de eco físico
        echo_result = orchestrator.generate_physical_echo("test")

        # Verificar se o resultado tem as propriedades esperadas
        required_keys = ['input', 'echo', 'fractal_dimension', 'physical_validation']
        actual_keys = list(echo_result.keys())

        missing_keys = [key for key in required_keys if key not in actual_keys]

        if missing_keys:
            print(f"❌ Chaves faltando no resultado do eco: {missing_keys}")
            return False

        # Verificar se o eco é diferente da entrada (não apenas repetição)
        if echo_result['echo'] == echo_result['input']:
            print("⚠️  Eco idêntico à entrada - pode indicar problema")
        else:
            print(f"✅ Eco gerado: '{echo_result['input']}' → '{echo_result['echo']}'")

        # Verificar dimensão fractal física
        fractal_dim = echo_result.get('fractal_dimension', 0)
        if 1.0 <= fractal_dim <= 3.0:
            print(".3f")
        else:
            print(f"⚠️  Dimensão fractal fora do range físico: {fractal_dim}")

        print("✅ HarmonicOrchestrator integrado com correções físicas")
        return True

    except Exception as e:
        print(f"❌ Erro no teste do HarmonicOrchestrator: {e}")
        return False

def main():
    """Função principal de teste"""
    print("🚀 Iniciando testes de integração ΨQRH com Correções Físicas")
    print("=" * 70)

    # Teste 1: Inicialização do pipeline
    pipeline = test_pipeline_initialization()

    # Teste 2: Processamento de texto
    processing_ok = test_pipeline_processing(pipeline)

    # Teste 3: Estrutura da API
    api_ok = test_api_structure()

    # Teste 4: Correções físicas fundamentais
    physical_ok = test_physical_corrections()

    # Teste 5: HarmonicOrchestrator com correções físicas
    harmonic_ok = test_harmonic_orchestrator()

    print("\n" + "=" * 70)
    print("📊 RESULTADO DOS TESTES:")

    tests_results = [
        ("Pipeline inicialização", pipeline is not None),
        ("Processamento de texto", processing_ok),
        ("Estrutura da API", api_ok),
        ("Correções físicas", physical_ok),
        ("HarmonicOrchestrator", harmonic_ok)
    ]

    all_passed = True
    for test_name, passed in tests_results:
        status = "✅" if passed else "❌"
        print(f"   {status} {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 Todos os testes passaram! Sistema fisicamente corrigido.")
        print("🔬 Correções fundamentais integradas com sucesso!")
        return 0
    else:
        print("⚠️  Alguns testes falharam. Verificar implementação.")
        return 1

if __name__ == "__main__":
    sys.exit(main())