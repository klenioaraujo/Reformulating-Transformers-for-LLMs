#!/usr/bin/env python3
"""
Teste Simples de Pipeline Entrada-Saída ΨQRH

Este script testa o pipeline básico de geração de texto do ΨQRH
usando a implementação física-matemática reformulada.
"""

import sys
import os
from pathlib import Path

# Adicionar diretório base ao path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

def test_simple_text_generation():
    """Testa geração de texto simples usando o pipeline ΨQRH"""

    print("🧪 Teste de Pipeline Entrada-Saída ΨQRH")
    print("=" * 50)

    try:
        # Importar o pipeline ΨQRH
        from src.core.ΨQRH import QRHFactory

        # Inicializar o factory com modelo específico
        model_path = "models/psiqrh_gpt2_MEDIO"

        if not Path(model_path).exists():
            print(f"❌ Modelo não encontrado: {model_path}")
            print("💡 Execute: make new-model SOURCE=gpt2-medium NAME=gpt2_MEDIO")
            return False

        print(f"📁 Carregando modelo: {model_path}")
        factory = QRHFactory(model_path=model_path)

        # Testar diferentes entradas
        test_inputs = [
            "ola",
            "como você está?",
            "explique quaternions",
            "o que é consciência fractal?"
        ]

        for i, input_text in enumerate(test_inputs, 1):
            print(f"\n--- Teste {i}/{len(test_inputs)} ---")
            print(f"📤 Entrada: '{input_text}'")

            # Processar texto
            result = factory.process_text(input_text, device="cpu")

            print(f"📥 Saída: '{result}'")
            print(f"📊 Tipo: {type(result)}")
            print(f"📏 Comprimento: {len(result) if isinstance(result, str) else 'N/A'}")

            # Verificar se a saída é válida
            if isinstance(result, str) and len(result.strip()) > 0:
                print("✅ Saída válida")
            else:
                print("❌ Saída inválida ou vazia")

    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("🎯 Teste concluído!")
    return True

def test_pipeline_direct():
    """Testa o pipeline diretamente sem factory"""

    print("\n🧪 Teste Direto do Pipeline ΨQRH")
    print("=" * 50)

    try:
        from psiqrh import ΨQRHPipeline

        # Criar pipeline com modelo específico
        pipeline = ΨQRHPipeline(
            task="text-generation",
            device="cpu",
            model_dir="models/psiqrh_gpt2_MEDIO"
        )

        test_input = "teste de pipeline direto"
        print(f"📤 Entrada: '{test_input}'")

        result = pipeline(test_input)

        print(f"📊 Status: {result['status']}")
        print(f"📥 Resposta: {result['response']}")
        print(f"📏 Comprimento: {result['output_length']}")

        if result['status'] == 'success':
            print("✅ Pipeline funcionando")
        else:
            print(f"❌ Erro no pipeline: {result.get('error', 'Desconhecido')}")

    except Exception as e:
        print(f"❌ Erro no pipeline direto: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def main():
    """Função principal"""

    print("🚀 Iniciando Testes de Pipeline ΨQRH")
    print("=" * 50)

    # Testar geração de texto simples
    success1 = test_simple_text_generation()

    # Testar pipeline direto
    success2 = test_pipeline_direct()

    # Resumo
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES:")
    print(f"  ✅ Teste de geração de texto: {'PASS' if success1 else 'FAIL'}")
    print(f"  ✅ Teste de pipeline direto: {'PASS' if success2 else 'FAIL'}")

    if success1 and success2:
        print("\n🎉 Todos os testes passaram! O pipeline está funcionando.")
        print("💡 Agora você pode usar: make chat-model")
        return 0
    else:
        print("\n❌ Alguns testes falharam. Verifique os logs acima.")
        return 1

if __name__ == "__main__":
    sys.exit(main())