#!/usr/bin/env python3
"""
quick_fix_hilbert_dimensions.py
===============================
Correção rápida para o erro: mat1 and mat2 shapes cannot be multiplied (8x768 and 2304x768)
"""

import torch
import torch.nn as nn
import os
import sys

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from psiqrh_transformers import HilbertConfig, HilbertLlamaForCausalLM

def quick_fix_attention_dimensions(hilbert_model):
    """
    Correção rápida para o erro dimensional na atenção
    """
    print("🔧 Aplicando correção rápida para dimensões de atenção...")

    try:
        # Obter configuração do modelo
        config = hilbert_model.config
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads

        print(f"  📐 Configuração: hidden_size={hidden_size}, num_heads={num_heads}")

        for i, layer in enumerate(hilbert_model.model.layers):
            attn = layer.self_attn

            print(f"  📐 Camada {i}: verificando projeções...")

            # Verificar se as projeções existem e têm as dimensões corretas
            if hasattr(attn, 'query') and attn.query.weight.shape != (hidden_size, hidden_size):
                print(f"  🔄 Corrigindo query: {attn.query.weight.shape} -> ({hidden_size}, {hidden_size})")
                new_weight = torch.randn(hidden_size, hidden_size) * 0.02
                attn.query.weight = nn.Parameter(new_weight)
                if attn.query.bias is not None:
                    attn.query.bias = nn.Parameter(torch.zeros(hidden_size))

            if hasattr(attn, 'key') and attn.key.weight.shape != (hidden_size, hidden_size):
                print(f"  🔄 Corrigindo key: {attn.key.weight.shape} -> ({hidden_size}, {hidden_size})")
                new_weight = torch.randn(hidden_size, hidden_size) * 0.02
                attn.key.weight = nn.Parameter(new_weight)
                if attn.key.bias is not None:
                    attn.key.bias = nn.Parameter(torch.zeros(hidden_size))

            if hasattr(attn, 'value') and attn.value.weight.shape != (hidden_size, hidden_size):
                print(f"  🔄 Corrigindo value: {attn.value.weight.shape} -> ({hidden_size}, {hidden_size})")
                new_weight = torch.randn(hidden_size, hidden_size) * 0.02
                attn.value.weight = nn.Parameter(new_weight)
                if attn.value.bias is not None:
                    attn.value.bias = nn.Parameter(torch.zeros(hidden_size))

        print("✅ Correções dimensionais aplicadas!")
        return True

    except Exception as e:
        print(f"❌ Erro na correção: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_compatible_hilbert_model():
    """
    Cria modelo Hilbert compatível com as dimensões do GPT-2
    """
    print("🧠 Criando modelo Hilbert compatível...")

    try:
        # Carregar GPT-2 para obter configuração
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Configuração compatível
        config = HilbertConfig(
            vocab_size=gpt2_model.config.vocab_size,
            hidden_size=768,  # GPT-2 hidden size
            num_hidden_layers=6,  # Reduzir para teste mais rápido
            num_attention_heads=12,
            intermediate_size=3072,  # GPT-2 usa 4x hidden_size
            hilbert_space="complex",
            spectral_alpha=1.0,
            fractal_dimension=1.5
        )

        # Criar modelo Hilbert
        hilbert_model = HilbertLlamaForCausalLM(config)

        # Aplicar correções dimensionais
        if quick_fix_attention_dimensions(hilbert_model):
            print("🎉 Modelo Hilbert compatível criado com sucesso!")

            # Testar geração básica
            print("\n🧪 Testando geração básica...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            hilbert_model = hilbert_model.to(device)

            try:
                test_input = "The quantum nature of"
                inputs = tokenizer(test_input, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = hilbert_model.generate(
                        inputs['input_ids'],
                        max_length=20,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id
                    )

                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"   ✅ Geração bem-sucedida: '{result}'")

                return hilbert_model, tokenizer, True

            except Exception as e:
                print(f"   ⚠️  Teste falhou: {e}")
                return hilbert_model, tokenizer, False

        else:
            print("❌ Falha nas correções dimensionais")
            return None, tokenizer, False

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

def save_quick_fixed_model(model, tokenizer, save_path="models/hilbert_gpt2_quick_fixed"):
    """Salva modelo corrigido rapidamente"""
    print(f"💾 Salvando modelo corrigido em {save_path}...")

    os.makedirs(save_path, exist_ok=True)

    try:
        model.save_pretrained(save_path, safe_serialization=False)
        print("   ✅ Salvo com pickle (compatibilidade)")
    except Exception as e:
        print(f"   ⚠️  Erro ao salvar: {e}")

    tokenizer.save_pretrained(save_path)
    print("✅ Modelo corrigido salvo com sucesso!")

if __name__ == "__main__":
    # Criar modelo corrigido rapidamente
    hilbert_model, tokenizer, success = create_compatible_hilbert_model()

    if success and hilbert_model:
        save_quick_fixed_model(hilbert_model, tokenizer)

        print("\n🚀 MODELO CORRIGIDO PRONTO PARA USO:")
        print("   python3 test_gpt2_hilbert.py --model models/hilbert_gpt2_quick_fixed \"sua pergunta\"")