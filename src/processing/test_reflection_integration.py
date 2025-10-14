"""
Sistema de Teste e Validação da Integração QuaternionReflectionLayer + DCFTokenAnalysis
"""

import time
import torch
from src.processing.token_analysis import DCFTokenAnalysis


def test_integration_comprehensive():
    """
    Teste abrangente da integração QuaternionReflectionLayer + DCFTokenAnalysis
    """
    print("🧪 TESTE DE INTEGRAÇÃO COMPREENSIVO")
    print("=" * 60)

    # 1. Inicializar sistema com diferentes modos
    test_modes = ['fast', 'analogical', 'adaptive']

    for mode in test_modes:
        print(f"\n🎯 Testando modo: {mode.upper()}")
        print("-" * 40)

        dcf_system = DCFTokenAnalysis(
            vocab_size=1000,
            hidden_size=256,
            reasoning_mode=mode
        )

        # 2. Testar com diferentes complexidades
        test_cases = [
            ([1, 2, 3, 4, 5], "Sequência simples"),
            ([42, 17, 89, 156, 203, 87, 12, 45], "Sequência complexa"),
            ([10, 10, 10, 10, 10], "Sequência repetitiva")
        ]

        for token_ids, description in test_cases:
            print(f"   🔍 {description}: {token_ids}")

            result = dcf_system.analyze_tokens(token_ids)

            print(f"      Modo executado: {result['reasoning_mode']}")
            print(f"      FCI: {result['fci_score']:.3f}")
            print(f"      Coerência semântica: {result['semantic_coherence']:.3f}")
            print(f"      Método: {result['processing_details']['method']}")

    # 3. Relatório de performance final
    print(f"\n📊 RELATÓRIO FINAL DE PERFORMANCE:")
    report = dcf_system.get_performance_report()
    for key, value in report.items():
        print(f"   {key}: {value}")


def benchmark_integration():
    """
    Benchmark de performance comparativo
    """
    print("\n🚀 BENCHMARK DE PERFORMANCE")
    print("=" * 50)

    dcf_adaptive = DCFTokenAnalysis(reasoning_mode='adaptive')

    # Teste de carga com 100 operações
    start_time = time.time()

    for i in range(100):
        token_ids = list(range(i % 50 + 1))  # Sequências variadas
        dcf_adaptive.analyze_tokens(token_ids)

    total_time = time.time() - start_time

    report = dcf_adaptive.get_performance_report()

    print(f"⏱️  Tempo total: {total_time:.2f}s")
    print(f"📈 Operações por segundo: {100/total_time:.1f} ops/s")
    print(f"🎯 Eficiência: {report['efficiency_gain']}")
    print(f"🔀 Razão Fast/Kuramoto: {report['fast_reasoning_ratio']:.1%} / {report['kuramoto_fallback_ratio']:.1%}")


if __name__ == "__main__":
    test_integration_comprehensive()
    benchmark_integration()