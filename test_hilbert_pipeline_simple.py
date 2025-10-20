#!/usr/bin/env python3
"""
Teste Simples do Pipeline ΨQRH-Transformers com DeepSeek
=======================================================

Versão simplificada para evitar travamentos de memória.
"""

import torch
import sys
import os

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from psiqrh_transformers import (
    HilbertConfig,
    HilbertLlamaForCausalLM,
)

def test_minimal_deepseek():
    """
    Teste minimalista com configurações pequenas
    """
    print("🧪 Teste Minimalista ΨQRH-Transformers")
    print("=" * 50)

    try:
        # Configuração minimalista similar ao DeepSeek
        config = HilbertConfig(
            vocab_size=32000,
            hidden_size=256,  # Muito menor para teste
            num_attention_heads=8,
            num_hidden_layers=2,  # Apenas 2 camadas
            intermediate_size=1024,
            hilbert_space="complex",  # Começar com complexo (mais simples)
            spectral_alpha=1.0,
            fractal_dimension=1.5,
            use_spectral_filtering=False,  # Desabilitar para teste
            use_fractal_embedding=True,
        )

        print("✅ Configuração criada:")
        print(f"   📐 Espaço de Hilbert: {config.hilbert_space}")
        print(f"   🧠 Hidden Size: {config.hidden_size}")
        print(f"   📚 Vocab Size: {config.vocab_size}")
        print(f"   🔢 Layers: {config.num_hidden_layers}")

        # Criar modelo pequeno
        print("\n🔄 Criando modelo...")
        model = HilbertLlamaForCausalLM(config)

        # Contar parâmetros
        total_params = sum(p.numel() for p in model.parameters())
        print("✅ Modelo criado!")
        print(f"   📊 Parâmetros: {total_params:,} ({total_params/1e6:.1f}M)")

        # Teste de forward pass mínimo
        print("\n🧪 Testando forward pass...")
        batch_size, seq_len = 1, 4  # Muito pequeno
        input_ids = torch.randint(0, min(1000, config.vocab_size), (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids, return_dict=True)
            logits = outputs['logits']

        print("✅ Forward pass OK!")
        print(f"   📥 Input: {input_ids.shape}")
        print(f"   📤 Output: {logits.shape}")
        print(f"   📊 Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        # Teste de geração simples
        print("\n🤖 Testando geração...")
        generated = input_ids.clone()
        for _ in range(2):  # Apenas 2 tokens
            with torch.no_grad():
                outputs = model(generated, return_dict=True)
                next_token = torch.argmax(outputs['logits'][:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)

        print("✅ Geração OK!")
        print(f"   📝 Original: {input_ids[0].tolist()}")
        print(f"   🤖 Gerado: {generated[0].tolist()}")

        return True

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_spaces():
    """
    Testa diferentes espaços de Hilbert
    """
    print("\n🔬 Testando espaços de Hilbert diferentes")
    print("=" * 50)

    spaces = ["complex", "quaternion"]
    results = {}

    for space in spaces:
        print(f"\n🧪 Testando espaço: {space}")
        try:
            config = HilbertConfig(
                vocab_size=1000,
                hidden_size=128,
                num_attention_heads=4,
                num_hidden_layers=1,
                hilbert_space=space,
                use_spectral_filtering=False,
            )

            model = HilbertLlamaForCausalLM(config)
            input_ids = torch.randint(0, config.vocab_size, (1, 3))

            with torch.no_grad():
                outputs = model(input_ids, return_dict=True)

            print(f"   ✅ {space}: OK")
            results[space] = True

        except Exception as e:
            print(f"   ❌ {space}: {e}")
            results[space] = False

    return results

def main():
    """
    Função principal
    """
    print("ΨQRH Transformers - Teste Simples com DeepSeek")
    print("=" * 60)

    # Teste 1: Modelo minimalista
    success1 = test_minimal_deepseek()

    # Teste 2: Diferentes espaços
    results = test_different_spaces()

    # Resultado final
    print("\n" + "=" * 60)
    if success1:
        print("🎉 Teste principal: SUCESSO!")
        print("✅ ΨQRH-Transformers funcional")
    else:
        print("❌ Teste principal: FALHA")

    print("\n📊 Espaços de Hilbert testados:")
    for space, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {space}")

    successful_spaces = sum(results.values())
    print(f"\n📈 {successful_spaces}/{len(results)} espaços funcionais")

    if success1 and successful_spaces > 0:
        print("\n🚀 Pronto para integração com DeepSeek!")
        print("💡 Para usar com modelo real:")
        print("   1. Ajuste hidden_size para 4096")
        print("   2. Aumente num_hidden_layers para 32")
        print("   3. Use from_pretrained() com pesos do DeepSeek")
    else:
        print("\n⚠️  Revisar implementação antes de prosseguir")

if __name__ == "__main__":
    main()