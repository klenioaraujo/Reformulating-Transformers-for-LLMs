#!/usr/bin/env python3
"""
Pipeline Simples de Entrada-Saída usando GPT-2
Este script implementa um pipeline básico usando o modelo GPT-2
como referência para demonstrar o funcionamento de entrada-saída.
"""
import sys
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Adicionar o diretório pai ao path para importar módulos locais
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_gpt2_pipeline():
    """
    Configura o pipeline GPT-2 com modelo e tokenizer
    """
    print("🚀 Inicializando Pipeline GPT-2...")

    # Carregar modelo e tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Configurar tokenizer para padding
    tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Modelo {model_name} carregado com sucesso!")
    print(f"   - Vocabulário: {tokenizer.vocab_size} tokens")
    print(f"   - Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer

def process_text_input(model, tokenizer, text_input, max_length=50):
    """
    Processa entrada de texto e gera saída usando GPT-2
    """
    print(f"\n📥 Processando entrada: '{text_input}'")

    # Tokenizar entrada
    inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"   - Tokens de entrada: {input_ids.shape[1]}")
    print(f"   - IDs: {input_ids.tolist()[0]}")

    # Gerar saída
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decodificar saída
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

def analyze_pipeline_flow(model, tokenizer, sample_texts):
    """
    Analisa o fluxo completo do pipeline para múltiplas entradas
    """
    print("\n" + "="*60)
    print("🔍 ANÁLISE DO FLUXO DO PIPELINE")
    print("="*60)

    for i, text in enumerate(sample_texts, 1):
        print(f"\n📋 Exemplo {i}:")
        print(f"   Entrada: '{text}'")

        # Processar entrada
        output = process_text_input(model, tokenizer, text)

        print(f"   Saída: '{output}'")

        # Análise adicional
        inputs = tokenizer(text, return_tensors="pt")
        print(f"   - Comprimento entrada: {inputs['input_ids'].shape[1]} tokens")
        print(f"   - Comprimento saída: {len(tokenizer.encode(output))} tokens")

def test_batch_processing(model, tokenizer):
    """
    Testa processamento em lote
    """
    print("\n" + "="*60)
    print("🧪 TESTE DE PROCESSAMENTO EM LOTE")
    print("="*60)

    batch_texts = [
        "O céu é",
        "A tecnologia avança",
        "Aprendizado de máquina é"
    ]

    print(f"Processando lote com {len(batch_texts)} textos...")

    for text in batch_texts:
        output = process_text_input(model, tokenizer, text, max_length=30)
        print(f"\n📥 '{text}'")
        print(f"📤 '{output}'")

def demonstrate_tokenization(tokenizer):
    """
    Demonstra o processo de tokenização
    """
    print("\n" + "="*60)
    print("🔤 DEMONSTRAÇÃO DE TOKENIZAÇÃO")
    print("="*60)

    sample_text = "Transformers são incríveis para NLP!"

    print(f"Texto: '{sample_text}'")

    # Tokenização
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)

    print(f"Tokens: {tokens}")
    print(f"IDs: {token_ids}")

    # Decodificação
    decoded = tokenizer.decode(token_ids)
    print(f"Decodificado: '{decoded}'")

def main():
    """
    Função principal do pipeline
    """
    print("🚀 INICIANDO PIPELINE GPT-2")
    print("="*60)

    try:
        # Configurar pipeline
        model, tokenizer = setup_gpt2_pipeline()

        # Textos de exemplo para teste
        sample_texts = [
            "O futuro da inteligência artificial",
            "A ciência nos permite",
            "Python é uma linguagem",
            "A matemática é a"
        ]

        # Executar análises
        analyze_pipeline_flow(model, tokenizer, sample_texts)
        demonstrate_tokenization(tokenizer)
        test_batch_processing(model, tokenizer)

        # Teste interativo
        print("\n" + "="*60)
        print("💬 TESTE INTERATIVO")
        print("="*60)
        print("Digite 'quit' para sair")

        while True:
            user_input = input("\n📥 Digite um texto: ").strip()

            if user_input.lower() in ['quit', 'exit', 'sair']:
                break

            if user_input:
                output = process_text_input(model, tokenizer, user_input)
                print(f"📤 Resposta: {output}")

        print("\n✅ Pipeline executado com sucesso!")

    except Exception as e:
        print(f"\n❌ Erro no pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()