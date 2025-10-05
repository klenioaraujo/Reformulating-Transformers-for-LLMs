#!/usr/bin/env python3
"""
Demonstração dos Filtros Cognitivos ΨQRH
========================================

Este script demonstra o funcionamento dos filtros cognitivos integrados
ao pipeline ΨQRH, mostrando como eles detectam e filtram:
- Contradições semânticas
- Irrelevâncias
- Vieses cognitivos
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.enhanced_qrh_processor import create_enhanced_processor

def print_header(title):
    """Imprime cabeçalho formatado."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_metrics(result):
    """Imprime métricas de forma formatada."""
    if not result.get('cognitive_metrics'):
        print("⚠️  Filtros cognitivos não aplicados")
        return

    cognitive = result['cognitive_metrics']

    print("\n🧠 MÉTRICAS COGNITIVAS:")

    # Contradição
    if 'contradiction' in cognitive:
        c = cognitive['contradiction']
        print(f"\n  📌 Contradição:")
        print(f"     Média: {c['mean']:.4f} {'🔴 ALTA' if c['mean'] > 0.5 else '🟢 BAIXA'}")

    # Relevância
    if 'relevance' in cognitive:
        r = cognitive['relevance']
        print(f"\n  🎯 Relevância:")
        print(f"     Média: {r['mean']:.4f} {'🟢 ALTA' if r['mean'] > 0.6 else '🔴 BAIXA'}")

    # Viés
    if 'bias' in cognitive:
        b = cognitive['bias']
        print(f"\n  ⚖️  Viés:")
        print(f"     Magnitude: {b['mean']:.4f} {'🔴 ALTO' if b['mean'] > 1.0 else '🟢 BAIXO'}")

    # Saúde Semântica
    if 'semantic_health' in cognitive:
        health = cognitive['semantic_health']
        overall = health['overall_semantic_health']

        print(f"\n  💚 SAÚDE SEMÂNTICA GERAL: {overall:.4f}")

        if overall >= 0.8:
            status = "🌟 EXCELENTE"
        elif overall >= 0.6:
            status = "✅ BOA"
        elif overall >= 0.4:
            status = "⚠️  REGULAR"
        else:
            status = "❌ BAIXA"

        print(f"     Status: {status}")
        print(f"\n     Detalhes:")
        print(f"     - Saúde de Contradição: {health['contradiction_health']:.4f}")
        print(f"     - Saúde de Relevância: {health['relevance_health']:.4f}")
        print(f"     - Saúde de Viés: {health['bias_health']:.4f}")

    # Pesos dos Filtros
    if 'filter_weights' in cognitive:
        weights = cognitive['filter_weights']
        print(f"\n  🎛️  ATIVAÇÃO DOS FILTROS:")
        print(f"     - Contradição: {weights['contradiction_avg']:.2%}")
        print(f"     - Irrelevância: {weights['irrelevance_avg']:.2%}")
        print(f"     - Viés: {weights['bias_avg']:.2%}")

def demo():
    """Executa demonstração completa."""

    print_header("DEMONSTRAÇÃO - FILTROS COGNITIVOS ΨQRH")

    print("\n🚀 Inicializando processador com filtros cognitivos...")
    processor = create_enhanced_processor(
        embed_dim=64,
        device="cpu",
        enable_cognitive_filters=True
    )

    # Cenários de demonstração
    demos = [
        {
            "title": "1️⃣  TEXTO COERENTE E RELEVANTE",
            "text": "O processamento quaterniônico oferece vantagens computacionais significativas para transformadores neurais.",
            "explanation": "Este texto é coerente, sem contradições, e mantém relevância no tópico."
        },
        {
            "title": "2️⃣  TEXTO COM CONTRADIÇÃO APARENTE",
            "text": "A IA é completamente determinística e previsível. No entanto, redes neurais exibem comportamento emergente imprevisível.",
            "explanation": "Contém afirmações contraditórias sobre previsibilidade de IA."
        },
        {
            "title": "3️⃣  TEXTO COM MÚLTIPLOS TÓPICOS DISPERSOS",
            "text": "Quaternions são úteis em rotações 3D. Gatos dormem muito. Pizza é deliciosa.",
            "explanation": "Tópicos completamente não relacionados, baixa relevância entre si."
        },
        {
            "title": "4️⃣  TEXTO TÉCNICO FOCADO",
            "text": "A transformada de Fourier quaterniônica permite análise espectral em domínios multidimensionais complexos.",
            "explanation": "Texto técnico focado em um único tópico bem definido."
        },
        {
            "title": "5️⃣  TEXTO COM VIÉS COGNITIVO",
            "text": "Obviamente, todos concordam que esta é a única solução correta possível para o problema.",
            "explanation": "Apresenta viés de confirmação e generalização excessiva."
        }
    ]

    for demo_case in demos:
        print_header(demo_case['title'])
        print(f"\n📝 Texto:")
        print(f'   "{demo_case["text"]}"')
        print(f"\n💡 Análise Esperada:")
        print(f"   {demo_case['explanation']}")

        # Processar
        result = processor.process_text(demo_case['text'], use_cache=False)

        # Mostrar resultados
        print(f"\n⏱️  Tempo de processamento: {result['processing_time']:.4f}s")
        print(f"🔢 Alpha adaptativo: {result['adaptive_alpha']:.4f}")

        print_metrics(result)

        input("\n⏎  Pressione ENTER para continuar...")

    # Estatísticas finais
    print_header("ESTATÍSTICAS FINAIS")
    metrics = processor.performance_metrics
    print(f"\n📊 Total processado: {metrics['total_processed']} textos")
    print(f"⏱️  Tempo médio: {metrics['avg_processing_time']:.4f}s")
    print(f"🧠 Filtros cognitivos aplicados: {metrics['cognitive_filters_applied']} vezes")
    print(f"💾 Cache hits: {metrics['cache_hits']}")

    print("\n" + "=" * 80)
    print("✅ Demonstração concluída!")
    print("=" * 80)
    print("\n📚 Para mais informações, consulte:")
    print("   - COGNITIVE_INTEGRATION_SUMMARY.md")
    print("   - configs/cognitive_filters_config.yaml")
    print("   - src/cognitive/semantic_adaptive_filters.py")
    print()

if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstração interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()