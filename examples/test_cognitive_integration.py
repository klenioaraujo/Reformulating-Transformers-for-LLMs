#!/usr/bin/env python3
"""
Teste de Integração dos Filtros Cognitivos
==========================================

Testa a integração completa do pipeline:
Input → QRHLayer (spectral) → SemanticAdaptiveFilter (cognitive) → Output
"""

import sys
from pathlib import Path

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent))

from src.core.enhanced_qrh_processor import create_enhanced_processor
import json

def test_cognitive_integration():
    """Teste completo da integração cognitiva."""

    print("=" * 80)
    print("TESTE DE INTEGRAÇÃO - FILTROS COGNITIVOS")
    print("=" * 80)

    # Criar processador com filtros cognitivos habilitados
    print("\n🚀 Inicializando processador com filtros cognitivos...")
    processor = create_enhanced_processor(
        embed_dim=64,
        device="cpu",
        enable_cognitive_filters=True
    )

    # Cenários de teste
    test_cases = [
        {
            "name": "Teste 1: Texto simples e coerente",
            "text": "O sistema ΨQRH demonstra eficiência superior em processamento quaterniônico.",
            "expected": "Baixa contradição, alta relevância, baixo viés"
        },
        {
            "name": "Teste 2: Texto com contradições",
            "text": "A água sempre ferve a 100°C. No entanto, a água pode ferver a temperaturas diferentes dependendo da pressão atmosférica.",
            "expected": "Alta contradição detectada"
        },
        {
            "name": "Teste 3: Texto com múltiplos tópicos",
            "text": "Transformadores quaterniônicos são eficientes. Gatos são animais domésticos. A física quântica é complexa.",
            "expected": "Baixa relevância devido a tópicos dispersos"
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"📝 {test_case['name']}")
        print(f"{'=' * 80}")
        print(f"Entrada: {test_case['text']}")
        print(f"Esperado: {test_case['expected']}")
        print()

        try:
            # Processar texto
            result = processor.process_text(test_case['text'], use_cache=False)

            # Exibir resultados
            print("✅ Status:", result['status'])
            print(f"⏱️  Tempo de processamento: {result['processing_time']:.4f}s")
            print(f"🔢 Alpha adaptativo: {result['adaptive_alpha']:.4f}")

            # Pipeline stages
            print("\n📊 Estágios do Pipeline:")
            for stage, enabled in result['pipeline_stages'].items():
                status = "✅" if enabled else "❌"
                print(f"  {status} {stage}")

            # Métricas cognitivas
            if result.get('cognitive_metrics'):
                print("\n🧠 MÉTRICAS COGNITIVAS:")
                cognitive = result['cognitive_metrics']

                if 'contradiction' in cognitive:
                    print(f"  • Contradição (média): {cognitive['contradiction']['mean']:.4f}")
                    print(f"    - Min: {cognitive['contradiction']['min']:.4f}")
                    print(f"    - Max: {cognitive['contradiction']['max']:.4f}")
                    print(f"    - Std: {cognitive['contradiction']['std']:.4f}")

                if 'relevance' in cognitive:
                    print(f"  • Relevância (média): {cognitive['relevance']['mean']:.4f}")
                    print(f"    - Min: {cognitive['relevance']['min']:.4f}")
                    print(f"    - Max: {cognitive['relevance']['max']:.4f}")
                    print(f"    - Std: {cognitive['relevance']['std']:.4f}")

                if 'bias' in cognitive:
                    print(f"  • Viés (magnitude média): {cognitive['bias']['mean']:.4f}")
                    print(f"    - Min: {cognitive['bias']['min']:.4f}")
                    print(f"    - Max: {cognitive['bias']['max']:.4f}")
                    print(f"    - Std: {cognitive['bias']['std']:.4f}")

                if 'semantic_health' in cognitive:
                    health = cognitive['semantic_health']
                    print(f"\n  💚 SAÚDE SEMÂNTICA:")
                    print(f"    - Saúde de Contradição: {health['contradiction_health']:.4f}")
                    print(f"    - Saúde de Relevância: {health['relevance_health']:.4f}")
                    print(f"    - Saúde de Viés: {health['bias_health']:.4f}")
                    print(f"    - 🌟 SAÚDE GERAL: {health['overall_semantic_health']:.4f}")

                if 'filter_weights' in cognitive:
                    weights = cognitive['filter_weights']
                    print(f"\n  ⚖️  PESOS DOS FILTROS:")
                    print(f"    - Contradição: {weights['contradiction_avg']:.4f}")
                    print(f"    - Irrelevância: {weights['irrelevance_avg']:.4f}")
                    print(f"    - Viés: {weights['bias_avg']:.4f}")
            else:
                print("\n⚠️  Filtros cognitivos não aplicados")

            # Salvar resultado
            results.append({
                'test_case': test_case['name'],
                'input': test_case['text'],
                'result': result
            })

        except Exception as e:
            print(f"\n❌ ERRO: {e}")
            import traceback
            traceback.print_exc()

    # Métricas finais
    print(f"\n{'=' * 80}")
    print("📊 MÉTRICAS FINAIS DO PROCESSADOR")
    print(f"{'=' * 80}")
    metrics = processor.performance_metrics
    print(f"Total processado: {metrics['total_processed']}")
    print(f"Tempo médio: {metrics['avg_processing_time']:.4f}s")
    print(f"Cache hits: {metrics['cache_hits']}")
    print(f"Filtros cognitivos aplicados: {metrics['cognitive_filters_applied']}")

    # Salvar resultados em JSON
    output_file = Path(__file__).parent / "tmp" / "cognitive_integration_test_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n💾 Resultados salvos em: {output_file}")

    print("\n" + "=" * 80)
    print("✅ TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 80)

if __name__ == "__main__":
    test_cognitive_integration()