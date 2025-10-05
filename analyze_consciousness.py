#!/usr/bin/env python3
"""
Analisador de Consciência para arquivos .Ψcws
============================================

Script para análise das métricas de consciência dos arquivos .Ψcws
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

def analyze_consciousness():
    """Analisa consciência de todos arquivos .Ψcws"""

    try:
        from src.conscience.psicws_native_reader import get_native_reader

        reader = get_native_reader()
        files = reader.list_available()

        print(f"📊 Analisando {len(files)} arquivos .Ψcws...")
        print("=" * 50)

        if not files:
            print("⚠️ Nenhum arquivo .Ψcws encontrado")
            return

        total_complexity = 0
        total_coherence = 0
        total_adaptability = 0
        total_integration = 0

        for i, f in enumerate(files):
            print(f"\n📄 Arquivo {i+1}: {f['original_name']}")
            print(f"   Hash: {f['hash']}")
            print(f"   Tamanho: {f['size_kb']} KB")

            summary = reader.get_consciousness_summary(f['hash'])
            if summary:
                complexity = summary['complexity']
                coherence = summary['coherence']
                adaptability = summary['adaptability']
                integration = summary['integration']

                print(f"   🧠 Métricas de Consciência:")
                print(f"      Complexity: {complexity:.4f}")
                print(f"      Coherence: {coherence:.4f}")
                print(f"      Adaptability: {adaptability:.4f}")
                print(f"      Integration: {integration:.4f}")
                print(f"   📊 Tipo: {summary['file_type']}")
                print(f"   🌊 Frequência: {summary['frequency_range']} Hz")

                total_complexity += complexity
                total_coherence += coherence
                total_adaptability += adaptability
                total_integration += integration

        # Médias
        if files:
            count = len(files)
            avg_complexity = total_complexity / count
            avg_coherence = total_coherence / count
            avg_adaptability = total_adaptability / count
            avg_integration = total_integration / count

            print(f"\n🧮 ANÁLISE AGREGADA:")
            print("=" * 50)
            print(f"Total de arquivos: {count}")
            print(f"Complexity média: {avg_complexity:.4f}")
            print(f"Coherence média: {avg_coherence:.4f}")
            print(f"Adaptability média: {avg_adaptability:.4f}")
            print(f"Integration média: {avg_integration:.4f}")

            # Interpretação das métricas
            print(f"\n🔍 INTERPRETAÇÃO:")
            print("=" * 50)

            if avg_complexity > 0.5:
                print("✅ Alta entropia informacional detectada")
            else:
                print("📊 Entropia informacional moderada")

            if avg_coherence > 0.5:
                print("✅ Estrutura temporal consistente")
            else:
                print("⚠️ Estrutura temporal fragmentada")

            if avg_adaptability > 0.8:
                print("✅ Excelente diversidade espectral")
            elif avg_adaptability > 0.5:
                print("📊 Boa diversidade espectral")
            else:
                print("⚠️ Baixa diversidade espectral")

            if avg_integration > 0.8:
                print("✅ Alta correlação dimensional")
            elif avg_integration > 0.5:
                print("📊 Correlação dimensional moderada")
            else:
                print("⚠️ Baixa correlação dimensional")

    except Exception as e:
        print(f"❌ Erro durante análise: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_consciousness()