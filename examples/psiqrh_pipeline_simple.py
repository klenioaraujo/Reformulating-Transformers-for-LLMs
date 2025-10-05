#!/usr/bin/env python3
"""
Pipeline Simples de Entrada-Saída usando ΨQRH (Versão Simplificada)
Este script implementa um pipeline básico usando o framework ΨQRH existente
com o modelo ativo psiqrh_gpt2_MEDIO, sem modo interativo.
"""
import sys
import os
from pathlib import Path

# Adicionar diretório base ao path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

def setup_psiqrh_pipeline():
    """
    Configura o pipeline ΨQRH usando o modelo ativo
    """
    print("🚀 Inicializando Pipeline ΨQRH...")

    try:
        # Importar o pipeline principal do ΨQRH
        from psiqrh import ΨQRHPipeline

        # Usar o modelo ativo (psiqrh_gpt2_MEDIO)
        model_dir = "models/psiqrh_gpt2_MEDIO"

        if not Path(model_dir).exists():
            print(f"❌ Modelo não encontrado: {model_dir}")
            print("💡 Verifique se o modelo está no diretório correto")
            return None

        pipeline = ΨQRHPipeline(task="text-generation", model_dir=model_dir)
        print(f"✅ Pipeline ΨQRH carregado com sucesso!")
        print(f"   - Modelo: psiqrh_gpt2_MEDIO")
        print(f"   - Dispositivo: {pipeline.device}")

        return pipeline

    except Exception as e:
        print(f"❌ Erro ao carregar pipeline ΨQRH: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_text_input(pipeline, text_input):
    """
    Processa entrada de texto e gera saída usando ΨQRH
    """
    print(f"\n📥 Processando entrada: '{text_input}'")

    try:
        # Processar texto através do pipeline ΨQRH
        result = pipeline(text_input)

        if result['status'] == 'success':
            response = result['response']

            # Handle different response formats
            if isinstance(response, dict) and 'text_analysis' in response:
                output_text = response['text_analysis']
            elif isinstance(response, str):
                output_text = response
            else:
                output_text = str(response)

            print(f"📤 Saída: '{output_text[:200]}...'")
            print(f"📏 Comprimento: {len(output_text)} caracteres")

            return output_text

        else:
            print(f"❌ Erro no processamento: {result.get('error', 'Desconhecido')}")
            return None

    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Função principal do pipeline
    """
    print("🚀 INICIANDO PIPELINE ΨQRH SIMPLES")
    print("="*60)

    try:
        # Configurar pipeline
        pipeline = setup_psiqrh_pipeline()

        if not pipeline:
            print("❌ Não foi possível inicializar o pipeline ΨQRH")
            return 1

        # Testar algumas entradas
        test_inputs = [
            "O futuro da inteligência artificial",
            "A ciência nos permite",
            "Python é uma linguagem",
            "Quaternions são úteis para",
            "Explique consciência fractal"
        ]

        print(f"\n🧪 Testando {len(test_inputs)} entradas...")

        for i, text in enumerate(test_inputs, 1):
            print(f"\n--- Teste {i}/{len(test_inputs)} ---")
            output = process_text_input(pipeline, text)

            if output:
                print(f"✅ Processamento bem-sucedido")
            else:
                print(f"❌ Falha no processamento")

        print("\n✅ Pipeline ΨQRH executado com sucesso!")

    except Exception as e:
        print(f"\n❌ Erro no pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())