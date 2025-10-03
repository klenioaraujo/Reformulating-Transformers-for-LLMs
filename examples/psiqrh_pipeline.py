#!/usr/bin/env python3
"""
Pipeline Simples de Entrada-Saída usando ΨQRH
Este script implementa um pipeline básico usando o framework ΨQRH existente
com o modelo ativo psiqrh_gpt2_MEDIO.
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

            print(f"📤 Saída: '{output_text}'")
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

def analyze_pipeline_flow(pipeline, sample_texts):
    """
    Analisa o fluxo completo do pipeline para múltiplas entradas
    """
    print("\n" + "="*60)
    print("🔍 ANÁLISE DO FLUXO DO PIPELINE ΨQRH")
    print("="*60)

    for i, text in enumerate(sample_texts, 1):
        print(f"\n📋 Exemplo {i}:")
        print(f"   Entrada: '{text}'")

        # Processar entrada
        output = process_text_input(pipeline, text)

        if output:
            print(f"   ✅ Processamento bem-sucedido")
        else:
            print(f"   ❌ Falha no processamento")

def test_batch_processing(pipeline):
    """
    Testa processamento em lote
    """
    print("\n" + "="*60)
    print("🧪 TESTE DE PROCESSAMENTO EM LOTE")
    print("="*60)

    batch_texts = [
        "O céu é",
        "A tecnologia avança",
        "Aprendizado de máquina é",
        "Quaternions são úteis para"
    ]

    print(f"Processando lote com {len(batch_texts)} textos...")

    for text in batch_texts:
        output = process_text_input(pipeline, text)
        if output:
            print(f"\n📥 '{text}'")
            print(f"📤 '{output[:100]}...'")

def demonstrate_psiqrh_features(pipeline):
    """
    Demonstra características específicas do ΨQRH
    """
    print("\n" + "="*60)
    print("🔬 DEMONSTRAÇÃO DE CARACTERÍSTICAS ΨQRH")
    print("="*60)

    # Testar diferentes tipos de entrada
    test_cases = [
        "Explique quaternions",
        "O que é consciência fractal?",
        "Como funciona a atenção espectral?",
        "Simule um campo quaterniônico"
    ]

    for test_case in test_cases:
        print(f"\n🧪 Teste: '{test_case}'")
        output = process_text_input(pipeline, test_case)

        if output:
            print(f"   ✅ ΨQRH processou com sucesso")
        else:
            print(f"   ❌ ΨQRH não conseguiu processar")

def main():
    """
    Função principal do pipeline
    """
    print("🚀 INICIANDO PIPELINE ΨQRH")
    print("="*60)

    try:
        # Configurar pipeline
        pipeline = setup_psiqrh_pipeline()

        if not pipeline:
            print("❌ Não foi possível inicializar o pipeline ΨQRH")
            return 1

        # Textos de exemplo para teste
        sample_texts = [
            "O futuro da inteligência artificial",
            "A ciência nos permite",
            "Python é uma linguagem",
            "A matemática é a linguagem do universo"
        ]

        # Executar análises
        analyze_pipeline_flow(pipeline, sample_texts)
        test_batch_processing(pipeline)
        demonstrate_psiqrh_features(pipeline)

        # Teste interativo
        print("\n" + "="*60)
        print("💬 TESTE INTERATIVO ΨQRH")
        print("="*60)
        print("Digite 'quit' para sair")

        while True:
            user_input = input("\n📥 Digite um texto: ").strip()

            if user_input.lower() in ['quit', 'exit', 'sair']:
                break

            if user_input:
                output = process_text_input(pipeline, user_input)
                if output:
                    print(f"📤 ΨQRH: {output}")

        print("\n✅ Pipeline ΨQRH executado com sucesso!")

    except Exception as e:
        print(f"\n❌ Erro no pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())