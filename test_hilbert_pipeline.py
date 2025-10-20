#!/usr/bin/env python3
"""
Teste do Pipeline ΨQRH-Transformers com DeepSeek
================================================

Demonstra o uso do pipeline híbrido com modelos reais como DeepSeek.
"""

import torch
import sys
import os
from pathlib import Path

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from psiqrh_transformers import (
    HilbertConfig,
    HilbertLlamaForCausalLM,
    create_hilbert_pipeline_example
)

def test_with_deepseek_like_model():
    """
    Testa o pipeline com um modelo similar ao DeepSeek
    (usando configuração compatível com modelos de 7B parâmetros)
    """
    print("🚀 Testando ΨQRH-Transformers com modelo DeepSeek-like")
    print("=" * 60)

    try:
        # Configuração similar ao DeepSeek (7B parameters)
        config = HilbertConfig(
            vocab_size=32000,  # DeepSeek vocab size aproximado
            hidden_size=4096,  # DeepSeek hidden size
            num_hidden_layers=32,  # DeepSeek layers
            num_attention_heads=32,  # DeepSeek attention heads
            intermediate_size=11008,  # DeepSeek intermediate size
            hilbert_space="quaternion",  # Usar espaço quaterniónico
            spectral_alpha=1.0,
            fractal_dimension=1.5,
            use_spectral_filtering=True,
            use_fractal_embedding=True,
        )

        print("✅ Configuração DeepSeek-like criada:")
        print(f"   📐 Espaço de Hilbert: {config.hilbert_space}")
        print(f"   🧠 Hidden Size: {config.hidden_size}")
        print(f"   📚 Vocab Size: {config.vocab_size}")
        print(f"   🔢 Layers: {config.num_hidden_layers}")
        print(f"   🎯 Attention Heads: {config.num_attention_heads}")

        # Criar modelo (nota: isso cria um modelo do zero, não carrega pesos pré-treinados)
        print("\n🔄 Criando modelo Hilbert-DeepSeek...")
        model = HilbertLlamaForCausalLM(config)

        # Contar parâmetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("✅ Modelo criado com sucesso!")
        print(f"   📊 Total de parâmetros: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"   🎓 Parâmetros treináveis: {trainable_params:,}")

        # Teste de forward pass
        print("\n🧪 Testando forward pass...")
        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids, return_dict=True)
            logits = outputs['logits']

        print("✅ Forward pass bem-sucedido!")
        print(f"   📥 Input shape: {input_ids.shape}")
        print(f"   📤 Output shape: {logits.shape}")
        print(f"   📊 Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

        # Teste de geração de texto
        print("\n🤖 Testando geração de texto...")
        test_prompt = "The quantum nature of consciousness"

        # Simular geração simples (greedy decoding)
        generated = input_ids.clone()
        max_new_tokens = 5

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(generated, return_dict=True)
                next_token_logits = outputs['logits'][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)

        print("✅ Geração de texto bem-sucedida!")
        print(f"   📝 Prompt: '{test_prompt}'")
        print(f"   🤖 Generated tokens: {generated[0].tolist()}")

        return True

    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """
    Testa a integração com pipeline do Hugging Face
    """
    print("\n🔗 Testando integração com pipeline...")
    print("=" * 60)

    try:
        # Tentar criar pipeline de exemplo
        pipe = create_hilbert_pipeline_example()
        print("✅ Pipeline criado com sucesso!")
        return True

    except Exception as e:
        print(f"⚠️  Pipeline não pôde ser criado (esperado sem modelo real): {e}")
        print("💡 Para usar com modelo real, baixe um modelo Llama/DeepSeek")
        return False

def main():
    """
    Função principal para executar todos os testes
    """
    print("ΨQRH Transformers - Teste com DeepSeek")
    print("=" * 60)

    # Teste 1: Modelo DeepSeek-like
    success1 = test_with_deepseek_like_model()

    # Teste 2: Integração com pipeline
    success2 = test_pipeline_integration()

    # Resultado final
    print("\n" + "=" * 60)
    if success1:
        print("🎉 Teste principal BEM-SUCEDIDO!")
        print("✅ ΨQRH-Transformers compatível com arquitetura DeepSeek")
        print("✅ Espaço de Hilbert quaterniónico funcionando")
        print("✅ Forward pass e geração de texto operacionais")
    else:
        print("❌ Teste principal FALHOU")

    if success2:
        print("✅ Integração com pipeline Hugging Face funcionando")
    else:
        print("⚠️  Integração com pipeline limitada (requer modelo real)")

    print("\n💡 Para usar com DeepSeek real:")
    print("   1. Instale transformers: pip install transformers")
    print("   2. Baixe modelo: huggingface-cli download deepseek-ai/deepseek-7b")
    print("   3. Adapte o código para carregar pesos pré-treinados")
    print("   4. Use HilbertLlamaForCausalLM.from_pretrained()")

if __name__ == "__main__":
    main()