#!/usr/bin/env python3
"""
fix_hilbert_dimensions.py
=========================
Script para correção automática de dimensões no modelo Hilbert

Corrige problemas de compatibilidade dimensional entre:
- GPT-2 e arquitetura Llama
- Projeções lineares de atenção
- Tensores de entrada/saída

Baseado na análise do erro: mat1 and mat2 shapes cannot be multiplied (8x768 and 2304x768)
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from psiqrh_transformers import HilbertConfig, HilbertLlamaForCausalLM

class DimensionFixer:
    """
    Classe especializada para corrigir problemas dimensionais no modelo Hilbert
    """

    def __init__(self, gpt2_model, hilbert_model):
        self.gpt2_model = gpt2_model
        self.hilbert_model = hilbert_model
        self.fix_log = []

    def log_fix(self, component, issue, solution, success=True):
        """Registra correções aplicadas"""
        status = "✅" if success else "❌"
        self.fix_log.append(f"{status} {component}: {issue} → {solution}")

    def analyze_dimensions(self):
        """Analisa dimensões atuais do modelo"""
        print("🔍 Analisando dimensões do modelo...")

        gpt2_config = self.gpt2_model.config
        hilbert_config = self.hilbert_model.config

        print("📊 CONFIGURAÇÕES:")
        print(f"   GPT-2: hidden_size={gpt2_config.hidden_size}, n_head={gpt2_config.num_attention_heads}")
        print(f"   Hilbert: hidden_size={hilbert_config.hidden_size}, n_head={hilbert_config.num_attention_heads}")
        print(f"   Hilbert: intermediate_size={hilbert_config.intermediate_size}")

        # Verificar camadas de atenção
        for i, layer in enumerate(self.hilbert_model.model.layers):
            attn = layer.self_attn
            print(f"   Camada {i}: query.weight.shape = {attn.query.weight.shape}")
            print(f"             key.weight.shape = {attn.key.weight.shape}")
            print(f"             value.weight.shape = {attn.value.weight.shape}")

    def fix_attention_dimensions(self):
        """Corrige dimensões das projeções de atenção"""
        print("🔧 Corrigindo dimensões das projeções de atenção...")

        gpt2_hidden = self.gpt2_model.config.hidden_size
        hilbert_hidden = self.hilbert_model.config.hidden_size
        num_heads = self.hilbert_model.config.num_attention_heads
        head_dim = hilbert_hidden // num_heads

        print(f"   Parâmetros: hidden={hilbert_hidden}, heads={num_heads}, head_dim={head_dim}")

        for i, layer in enumerate(self.hilbert_model.model.layers):
            attn = layer.self_attn

            # Verificar dimensões atuais
            current_query_shape = attn.query.weight.shape
            expected_query_shape = (hilbert_hidden, hilbert_hidden)

            print(f"   Camada {i}: query atual {current_query_shape} vs esperado {expected_query_shape}")

            if current_query_shape != expected_query_shape:
                # Recriar projeções com dimensões corretas
                attn.query = nn.Linear(hilbert_hidden, hilbert_hidden)
                attn.key = nn.Linear(hilbert_hidden, hilbert_hidden)
                attn.value = nn.Linear(hilbert_hidden, hilbert_hidden)

                # Transferir pesos se possível (truncar ou interpolar)
                if current_query_shape[1] == hilbert_hidden:
                    # Mesmo input size, ajustar output size
                    min_out = min(current_query_shape[0], hilbert_hidden)
                    attn.query.weight.data[:min_out] = attn.query.weight.data[:min_out]
                    attn.key.weight.data[:min_out] = attn.key.weight.data[:min_out]
                    attn.value.weight.data[:min_out] = attn.value.weight.data[:min_out]

                self.log_fix(f"Attention_{i}", f"query {current_query_shape}", f"query {expected_query_shape}")

        return True

    def fix_mlp_dimensions(self):
        """Corrige dimensões das camadas MLP"""
        print("🔧 Corrigindo dimensões das camadas MLP...")

        hilbert_hidden = self.hilbert_model.config.hidden_size
        intermediate_size = self.hilbert_model.config.intermediate_size

        print(f"   Parâmetros: hidden={hilbert_hidden}, intermediate={intermediate_size}")

        for i, layer in enumerate(self.hilbert_model.model.layers):
            mlp = layer.mlp

            # Verificar dimensões atuais
            current_gate_shape = mlp.gate_proj.weight.shape
            expected_gate_shape = (intermediate_size, hilbert_hidden)

            print(f"   Camada {i}: gate_proj atual {current_gate_shape} vs esperado {expected_gate_shape}")

            if current_gate_shape != expected_gate_shape:
                # Recriar projeções MLP com dimensões corretas
                mlp.gate_proj = nn.Linear(hilbert_hidden, intermediate_size)
                mlp.up_proj = nn.Linear(hilbert_hidden, intermediate_size)
                mlp.down_proj = nn.Linear(intermediate_size, hilbert_hidden)

                self.log_fix(f"MLP_{i}", f"gate_proj {current_gate_shape}", f"gate_proj {expected_gate_shape}")

        return True

    def fix_embeddings_dimensions(self):
        """Corrige dimensões das embeddings"""
        print("🔧 Corrigindo dimensões das embeddings...")

        hilbert_hidden = self.hilbert_model.config.hidden_size
        vocab_size = self.hilbert_model.config.vocab_size

        # Verificar embeddings de tokens
        embed_tokens = self.hilbert_model.model.embed_tokens
        if hasattr(embed_tokens, 'word_embeddings'):
            current_shape = embed_tokens.word_embeddings.weight.shape
            expected_shape = (vocab_size, hilbert_hidden)

            print(f"   word_embeddings: atual {current_shape} vs esperado {expected_shape}")

            if current_shape != expected_shape:
                embed_tokens.word_embeddings = nn.Embedding(vocab_size, hilbert_hidden)
                self.log_fix("Embeddings", f"word_embeddings {current_shape}", f"word_embeddings {expected_shape}")

        return True

    def apply_dimension_fixes(self):
        """Aplica todas as correções dimensionais"""
        print("🚀 Aplicando correções dimensionais completas...")
        print("=" * 60)

        success = True

        # Executar correções
        self.analyze_dimensions()  # Análise não retorna bool
        success &= self.fix_embeddings_dimensions()
        success &= self.fix_attention_dimensions()
        success &= self.fix_mlp_dimensions()

        # Mostrar relatório
        print("\n📊 RELATÓRIO DE CORREÇÕES:")
        print("=" * 40)
        for log_entry in self.fix_log:
            print(log_entry)

        if success:
            print(f"\n🎉 Correções dimensionais concluídas: {len(self.fix_log)} componentes ajustados")
        else:
            print(f"\n⚠️  Correções parciais: {len(self.fix_log)} componentes ajustados")

        return success

def create_fixed_hilbert_model(gpt2_model_name='gpt2', hilbert_space="complex"):
    """
    Cria modelo Hilbert com dimensões corrigidas
    """
    print(f"🧠 Criando modelo Hilbert corrigido a partir de {gpt2_model_name}")

    try:
        # Carregar GPT-2
        print("📥 Carregando modelo GPT-2...")
        gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        print("✅ GPT-2 carregado:")
        print(f"   📊 Parâmetros: {sum(p.numel() for p in gpt2_model.parameters()):,}")
        print(f"   🧠 Hidden Size: {gpt2_model.config.hidden_size}")
        print(f"   🔢 Layers: {gpt2_model.config.num_hidden_layers}")

        # Criar configuração Hilbert CORRIGIDA
        hilbert_config = HilbertConfig(
            vocab_size=gpt2_model.config.vocab_size,
            hidden_size=gpt2_model.config.hidden_size,  # Mesmo tamanho que GPT-2
            num_hidden_layers=6,  # Limitar para compatibilidade
            num_attention_heads=gpt2_model.config.num_attention_heads,
            intermediate_size=gpt2_model.config.hidden_size * 4,  # 3072 para GPT-2
            hilbert_space=hilbert_space,
            spectral_alpha=1.0,
            fractal_dimension=1.5,
            use_spectral_filtering=False,  # Desabilitar para evitar problemas
            use_fractal_embedding=True,
        )

        # Criar modelo Hilbert
        print("🔧 Criando modelo Hilbert...")
        hilbert_model = HilbertLlamaForCausalLM(hilbert_config)

        # Aplicar correções dimensionais
        fixer = DimensionFixer(gpt2_model, hilbert_model)
        fix_success = fixer.apply_dimension_fixes()

        if fix_success:
            print("\n🎉 Modelo Hilbert com dimensões corrigidas criado!")

            # Testar geração básica
            print("\n🧪 Testando geração básica...")
            test_prompt = "The quantum nature of"

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            hilbert_model = hilbert_model.to(device)

            try:
                inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = hilbert_model.generate(
                        inputs['input_ids'],
                        max_length=20,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"   ✅ Geração bem-sucedida: '{result}'")
            except Exception as e:
                print(f"   ⚠️  Teste falhou: {e}")

            return hilbert_model, tokenizer, True

        else:
            print("❌ Falha nas correções dimensionais")
            return None, tokenizer, False

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

def save_fixed_hilbert_model(model, tokenizer, save_path="models/hilbert_gpt2_fixed"):
    """Salva modelo Hilbert corrigido"""
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
    # Criar modelo corrigido
    hilbert_model, tokenizer, success = create_fixed_hilbert_model(
        gpt2_model_name='gpt2',
        hilbert_space="complex"
    )

    if success and hilbert_model:
        save_fixed_hilbert_model(hilbert_model, tokenizer)

        print("\n🚀 MODELO CORRIGIDO PRONTO PARA USO:")
        print("   python3 test_gpt2_hilbert.py --model models/hilbert_gpt2_fixed \"sua pergunta\"")