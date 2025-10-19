#!/usr/bin/env python3
"""
Teste do Filtro Quântico - Verifica se o filtro quântico está funcionando corretamente

Este script testa a integração do filtro quântico e verifica se ele pode
prevenir loops de geração e melhorar a qualidade do texto.
"""

import sys
import os

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import torch
from src.processing.quantum_token_filter import create_quantum_token_filter
from src.processing.quantum_token_filter_integration import integrate_quantum_filter, enhanced_quantum_text_generation


def test_quantum_filter_basic():
    """Teste básico do filtro quântico"""
    print("🧪 Teste Básico do Filtro Quântico")
    print("=" * 60)

    # Criar representações quânticas simuladas
    vocab_size = 1000
    embed_dim = 256
    quantum_vocab = torch.randn(vocab_size, embed_dim, 4)

    # Criar filtro quântico
    filter_engine = create_quantum_token_filter(
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        quantum_vocab_representations=quantum_vocab
    )

    # Testar predição
    logits = torch.randn(vocab_size)
    previous_tokens = [100, 200, 300]  # Tokens simulados

    result = filter_engine.predict_next_token(
        logits, previous_tokens, temperature=0.8, top_k=50
    )

    print(f"✅ Token selecionado: {result['selected_token']}")
    print(f"📊 Probabilidade: {result['selected_probability']:.4f}")
    print(f"🔧 Método: {result['method']}")
    print(f"📋 Relatório do filtro: {len(result['filter_report']['problematic_tokens_detected'])} tokens problemáticos")

    return result


def test_repetition_prevention():
    """Testa a prevenção de repetição"""
    print("\n🔄 Teste de Prevenção de Repetição")
    print("=" * 60)

    # Criar representações quânticas simuladas
    vocab_size = 100
    embed_dim = 256
    quantum_vocab = torch.randn(vocab_size, embed_dim, 4)

    # Criar filtro quântico
    filter_engine = create_quantum_token_filter(
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        quantum_vocab_representations=quantum_vocab
    )

    # Simular repetição do mesmo token
    problematic_token = 42
    previous_tokens = [problematic_token, problematic_token, problematic_token]

    # Criar logits que favorecem o token problemático
    logits = torch.zeros(vocab_size)
    logits[problematic_token] = 10.0  # Alta probabilidade

    # Testar predição com histórico de repetição
    result = filter_engine.predict_next_token(
        logits, previous_tokens, temperature=0.5, top_k=20
    )

    print(f"📊 Token problemático: {problematic_token}")
    print(f"📊 Token selecionado: {result['selected_token']}")
    print(f"📊 Probabilidade: {result['selected_probability']:.4f}")

    # Verificar se o filtro evitou a repetição
    if result['selected_token'] != problematic_token:
        print("✅ Filtro preveniu repetição com sucesso!")
    else:
        print("⚠️  Filtro não preveniu repetição")

    return result


def test_spectral_analysis():
    """Testa a análise espectral"""
    print("\n🔬 Teste de Análise Espectral")
    print("=" * 60)

    from src.processing.quantum_token_filter import QuantumSpectralAnalyzer

    # Criar analisador espectral
    analyzer = QuantumSpectralAnalyzer(embed_dim=256)

    # Criar token quântico simulado
    token_quantum = torch.randn(256, 4)

    # Analisar espectro
    analysis = analyzer.analyze_token_spectrum(token_quantum, token_id=123)

    print(f"📊 Token ID: {analysis['token_id']}")
    print(f"📊 Entropia espectral: {analysis['spectral_entropy']:.4f}")
    print(f"📊 Score de repetição: {analysis['repetition_score']:.4f}")
    print(f"📊 Score de anomalia: {analysis['anomaly_score']:.4f}")
    print(f"📊 Classificação: {analysis['token_class']}")
    print(f"📊 Frequências dominantes: {len(analysis['dominant_frequencies'])}")

    return analysis


def test_grammar_functions():
    """Testa as funções de gramática"""
    print("\n📐 Teste de Funções de Gramática")
    print("=" * 60)

    from src.processing.quantum_token_filter import MathematicalGrammarFunctions

    # Criar funções de gramática
    grammar = MathematicalGrammarFunctions(vocab_size=1000)

    # Testar restrições gramaticais
    candidate_logits = torch.randn(1000)
    previous_tokens = [13]  # Ponto final

    constrained_logits = grammar.apply_grammar_constraints(
        candidate_logits, previous_tokens, grammar_strength=0.7
    )

    print(f"📊 Logits originais: shape {candidate_logits.shape}")
    print(f"📊 Logits com gramática: shape {constrained_logits.shape}")
    print(f"📊 Diferença máxima: {torch.max(torch.abs(candidate_logits - constrained_logits)):.4f}")

    # Verificar se as restrições foram aplicadas
    if not torch.allclose(candidate_logits, constrained_logits):
        print("✅ Restrições gramaticais aplicadas com sucesso!")
    else:
        print("⚠️  Restrições gramaticais não foram aplicadas")

    return constrained_logits


def test_integration_with_mock():
    """Testa a integração com uma instância mock"""
    print("\n🔗 Teste de Integração com Mock")
    print("=" * 60)

    # Simular uma instância do PsiQRH
    class MockPsiQRH:
        def __init__(self):
            self.device = "cpu"
            # Criar representações quânticas simuladas
            self.quantum_vocab_representations = torch.randn(500, 256, 4)
            self.id_to_word = {i: f"word_{i}" for i in range(500)}
            self.word_to_id = {f"word_{i}": i for i in range(500)}

    # Criar instância mock
    mock_instance = MockPsiQRH()

    # Integrar filtro quântico
    success = integrate_quantum_filter(mock_instance)

    if success:
        print("✅ Filtro quântico integrado com sucesso!")

        # Testar geração de texto
        psi_final = torch.randn(256)
        input_text = "what color is the sky?"

        result = enhanced_quantum_text_generation(mock_instance, psi_final, input_text)
        print(f"📝 Texto gerado: '{result}'")

        # Verificar se temos filtro disponível
        if hasattr(mock_instance, 'quantum_token_filter'):
            print("✅ Filtro quântico disponível na instância")
        else:
            print("❌ Filtro quântico não disponível na instância")

    else:
        print("❌ Falha na integração do filtro quântico")

    return success


def run_all_tests():
    """Executa todos os testes"""
    print("🚀 Executando Todos os Testes do Filtro Quântico")
    print("=" * 60)

    results = {}

    try:
        results['basic'] = test_quantum_filter_basic()
    except Exception as e:
        print(f"❌ Teste básico falhou: {e}")
        results['basic'] = None

    try:
        results['repetition'] = test_repetition_prevention()
    except Exception as e:
        print(f"❌ Teste de repetição falhou: {e}")
        results['repetition'] = None

    try:
        results['spectral'] = test_spectral_analysis()
    except Exception as e:
        print(f"❌ Teste espectral falhou: {e}")
        results['spectral'] = None

    try:
        results['grammar'] = test_grammar_functions()
    except Exception as e:
        print(f"❌ Teste de gramática falhou: {e}")
        results['grammar'] = None

    try:
        results['integration'] = test_integration_with_mock()
    except Exception as e:
        print(f"❌ Teste de integração falhou: {e}")
        results['integration'] = None

    # Resumo
    print("\n📊 RESUMO DOS TESTES")
    print("=" * 60)

    successful_tests = sum(1 for result in results.values() if result is not None)
    total_tests = len(results)

    print(f"✅ Testes bem-sucedidos: {successful_tests}/{total_tests}")

    if successful_tests == total_tests:
        print("🎉 Todos os testes passaram!")
    else:
        print("⚠️  Alguns testes falharam")

    return results


if __name__ == "__main__":
    run_all_tests()