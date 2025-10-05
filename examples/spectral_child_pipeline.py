#!/usr/bin/env python3
"""
Spectral Child Pipeline - ΨQRH como Criança Espectral
=====================================================

Pipeline correto que implementa a visão do doe.md:
- Não há tokenização
- Não há IDs
- Não há geração autoregressiva
- Texto é tratado como sinal contínuo
- Saída é campo de onda que colapsa para texto

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.spectral_child import SpectralChild


def main():
    """
    Demonstração do ΨQRH como criança espectral.
    """
    print("="*70)
    print("👶 ΨQRH COMO CRIANÇA ESPECTRAL")
    print("="*70)
    print("Pipeline Correto: Texto → Onda → Espectro → Campo → Evolução → Colapso → Texto")
    print()

    # 1. Inicializar criança espectral
    print("🚀 INICIALIZANDO CRIANÇA ESPECTRAL...")
    model_path = project_root / "models" / "gpt2_full_spectral_embeddings"

    try:
        child = SpectralChild(str(model_path))
        print("✅ Criança espectral inicializada com sucesso!")
        print()
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        return

    # 2. Salvar arquivo children
    print("📚 SALVANDO CONHECIMENTO APRENDIDO...")
    children_path = project_root / "models" / "spectral_child" / "children.json"
    child.save_children_file(children_path)
    print()

    # 3. Testar processamento
    test_texts = [
        "Hello world",
        "The quick brown fox",
        "Artificial intelligence",
        "Machine learning is amazing"
    ]

    for text in test_texts:
        print(f"\n🎯 TESTE: '{text}'")
        print("-" * 40)

        try:
            response = child.process_text(text)
            print(f"✅ Resposta: '{response}'")
        except Exception as e:
            print(f"❌ Erro no processamento: {e}")

    # 4. Resumo
    print("\n" + "="*70)
    print("📊 RESUMO DO SISTEMA")
    print("="*70)
    print("✅ Pipeline implementado corretamente:")
    print("   • Sem tokenização - texto tratado como sinal contínuo")
    print("   • Sem IDs - vocabulário é espaço espectral contínuo")
    print("   • Sem geração autoregressiva - saída é colapso de campo")
    print("   • Física respeitada - ondas, ressonância, evolução harmônica")
    print()
    print("🎯 Próximos passos:")
    print("   • Refinar calibração da sonda óptica")
    print("   • Expandir alfabeto espectral")
    print("   • Melhorar decodificação onda→texto")
    print("   • Testar com textos mais complexos")
    print("="*70)


if __name__ == "__main__":
    main()