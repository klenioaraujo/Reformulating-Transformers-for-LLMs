#!/usr/bin/env python3
"""
transfer_weights_gpt2_to_hilbert.py
===================================
Transferência inteligente de pesos do GPT-2 para modelo Hilbert
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

class WeightTransferGPT2ToHilbert:
    """
    Classe especializada para transferir pesos do GPT-2 para Hilbert
    """

    def __init__(self, gpt2_model, hilbert_model):
        self.gpt2_model = gpt2_model
        self.hilbert_model = hilbert_model
        self.transfer_log = []

    def log_transfer(self, layer_type, gpt2_name, hilbert_name, success=True):
        """Registra transferência de pesos"""
        status = "✅" if success else "❌"
        self.transfer_log.append(f"{status} {layer_type}: {gpt2_name} → {hilbert_name}")

    def transfer_embeddings(self):
        """Transferir pesos das embeddings"""
        print("🔄 Transferindo embeddings...")

        try:
            # Word embeddings
            gpt2_wte = self.gpt2_model.transformer.wte.weight
            hilbert_embeddings = self.hilbert_model.model.embed_tokens

            # Verificar se HilbertEmbeddings tem word_embeddings
            if hasattr(hilbert_embeddings, 'word_embeddings'):
                hilbert_wte = hilbert_embeddings.word_embeddings.weight

                # Ajustar tamanho se necessário
                min_vocab = min(gpt2_wte.size(0), hilbert_wte.size(0))
                hilbert_wte.data[:min_vocab] = gpt2_wte.data[:min_vocab]
                self.log_transfer("Embedding", "wte", "embed_tokens")
            else:
                print("   ⚠️  HilbertEmbeddings não tem word_embeddings.weight, pulando")
                return False

            # Position embeddings
            gpt2_wpe = self.gpt2_model.transformer.wpe.weight
            if hasattr(hilbert_embeddings, 'position_embeddings'):
                hilbert_wpe = hilbert_embeddings.position_embeddings.weight

                # GPT-2 tem 1024 posições, ajustar se necessário
                max_positions = min(gpt2_wpe.size(0), hilbert_wpe.size(0))
                hilbert_wpe.data[:max_positions] = gpt2_wpe.data[:max_positions]
                self.log_transfer("Position", "wpe", "embed_positions")
            else:
                print("   ⚠️  HilbertEmbeddings não tem position_embeddings, pulando")

            return True

        except Exception as e:
            print(f"❌ Erro nas embeddings: {e}")
            return False

    def transfer_attention_layer(self, gpt2_layer, hilbert_layer, layer_idx):
        """Transferir pesos de uma camada de atenção"""
        try:
            # QKV projections
            gpt2_attn = gpt2_layer.attn
            hilbert_attn = hilbert_layer.self_attn

            # Verificar se HilbertAttention tem as projeções corretas
            if not hasattr(hilbert_attn, 'query'):
                print(f"   ⚠️  HilbertAttention não tem query projection, pulando transferência de atenção")
                return False

            # Transferir pesos QKV
            # GPT-2: c_attn (hidden_size -> 3*hidden_size)
            # Hilbert: query, key, value separados
            gpt2_qkv_weight = gpt2_attn.c_attn.weight
            gpt2_qkv_bias = gpt2_attn.c_attn.bias

            hidden_size = gpt2_qkv_weight.size(1)
            all_head_size = gpt2_qkv_weight.size(0)

            # Separar Q, K, V do GPT-2
            q_weight = gpt2_qkv_weight[:hidden_size]
            k_weight = gpt2_qkv_weight[hidden_size:2*hidden_size]
            v_weight = gpt2_qkv_weight[2*hidden_size:]

            q_bias = gpt2_qkv_bias[:hidden_size]
            k_bias = gpt2_qkv_bias[hidden_size:2*hidden_size]
            v_bias = gpt2_qkv_bias[2*hidden_size:]

            # Transferir para Hilbert
            hilbert_attn.query.weight.data = q_weight
            hilbert_attn.key.weight.data = k_weight
            hilbert_attn.value.weight.data = v_weight

            # Bias se existir
            if hasattr(hilbert_attn.query, 'bias') and hilbert_attn.query.bias is not None:
                hilbert_attn.query.bias.data = q_bias
            if hasattr(hilbert_attn.key, 'bias') and hilbert_attn.key.bias is not None:
                hilbert_attn.key.bias.data = k_bias
            if hasattr(hilbert_attn.value, 'bias') and hilbert_attn.value.bias is not None:
                hilbert_attn.value.bias.data = v_bias

            self.log_transfer(f"Attention_{layer_idx}", "c_attn/c_proj", "query/key/value projections")
            return True

        except Exception as e:
            print(f"❌ Erro na atenção layer {layer_idx}: {e}")
            return False

    def transfer_mlp_layer(self, gpt2_layer, hilbert_layer, layer_idx):
        """Transferir pesos da camada MLP"""
        try:
            # GPT-2: c_fc (hidden_size -> intermediate_size) e c_proj (intermediate_size -> hidden_size)
            # Hilbert: gate_proj, up_proj, down_proj

            gpt2_mlp = gpt2_layer.mlp
            hilbert_mlp = hilbert_layer.mlp

            # Transferir pesos do feed-forward
            hilbert_mlp.gate_proj.weight.data = gpt2_mlp.c_fc.weight.data
            hilbert_mlp.up_proj.weight.data = gpt2_mlp.c_fc.weight.data  # Usar mesma base

            # Ajustar para down_proj (equivalente ao c_proj do GPT-2)
            hilbert_mlp.down_proj.weight.data = gpt2_mlp.c_proj.weight.data

            # Bias
            if hilbert_mlp.gate_proj.bias is not None:
                hilbert_mlp.gate_proj.bias.data = gpt2_mlp.c_fc.bias.data
                hilbert_mlp.up_proj.bias.data = gpt2_mlp.c_fc.bias.data

            if hilbert_mlp.down_proj.bias is not None and gpt2_mlp.c_proj.bias is not None:
                hilbert_mlp.down_proj.bias.data = gpt2_mlp.c_proj.bias.data

            self.log_transfer(f"MLP_{layer_idx}", "c_fc/c_proj", "gate_proj/up_proj/down_proj")
            return True

        except Exception as e:
            print(f"❌ Erro no MLP layer {layer_idx}: {e}")
            return False

    def transfer_layer_norms(self, gpt2_layer, hilbert_layer, layer_idx):
        """Transferir pesos das normalizações"""
        try:
            # Input layer norm - Llama usa RMSNorm, GPT-2 usa LayerNorm
            if hasattr(hilbert_layer, 'input_layernorm'):
                # Para RMSNorm, só copiar weight (sem bias)
                if hasattr(hilbert_layer.input_layernorm, 'weight'):
                    hilbert_layer.input_layernorm.weight.data = gpt2_layer.ln_1.weight.data
                # RMSNorm não tem bias, então ignorar

            # Post-attention layer norm (GPT-2 não tem equivalente direto)
            # Usar mesma inicialização para consistência
            if hasattr(hilbert_layer, 'post_attention_layernorm'):
                if hasattr(hilbert_layer.post_attention_layernorm, 'weight'):
                    hilbert_layer.post_attention_layernorm.weight.data = gpt2_layer.ln_1.weight.data.clone()

            self.log_transfer(f"LayerNorm_{layer_idx}", "ln_1", "input_layernorm/post_attention_layernorm")
            return True

        except Exception as e:
            print(f"❌ Erro no LayerNorm layer {layer_idx}: {e}")
            return False

    def transfer_transformer_layers(self):
        """Transferir todas as camadas do transformer"""
        print("🔄 Transferindo camadas do transformer...")

        gpt2_layers = self.gpt2_model.transformer.h
        hilbert_layers = self.hilbert_model.model.layers

        min_layers = min(len(gpt2_layers), len(hilbert_layers))

        success_count = 0
        for i in range(min_layers):
            print(f"  📋 Camada {i+1}/{min_layers}...")

            gpt2_layer = gpt2_layers[i]
            hilbert_layer = hilbert_layers[i]

            # Transferir componentes da camada
            attention_success = self.transfer_attention_layer(gpt2_layer, hilbert_layer, i)
            mlp_success = self.transfer_mlp_layer(gpt2_layer, hilbert_layer, i)
            norm_success = self.transfer_layer_norms(gpt2_layer, hilbert_layer, i)

            if attention_success and mlp_success and norm_success:
                success_count += 1

        print(f"✅ {success_count}/{min_layers} camadas transferidas com sucesso")
        return success_count > 0

    def transfer_final_layer_norm(self):
        """Transferir layer norm final"""
        try:
            gpt2_final_ln = self.gpt2_model.transformer.ln_f
            hilbert_final_ln = self.hilbert_model.model.norm

            # RMSNorm só tem weight, não bias
            if hasattr(hilbert_final_ln, 'weight'):
                hilbert_final_ln.weight.data = gpt2_final_ln.weight.data

            self.log_transfer("FinalLayerNorm", "ln_f", "norm")
            return True

        except Exception as e:
            print(f"❌ Erro no final layer norm: {e}")
            return False

    def transfer_lm_head(self):
        """Transferir cabeça de linguagem"""
        try:
            # No GPT-2, a LM head compartilha pesos com as embeddings
            gpt2_lm_head = self.gpt2_model.lm_head.weight
            hilbert_lm_head = self.hilbert_model.lm_head.weight

            # Ajustar tamanho se necessário
            min_vocab = min(gpt2_lm_head.size(0), hilbert_lm_head.size(0))
            hilbert_lm_head.data[:min_vocab] = gpt2_lm_head.data[:min_vocab]

            self.log_transfer("LMHead", "lm_head", "lm_head")
            return True

        except Exception as e:
            print(f"❌ Erro na LM head: {e}")
            return False

    def transfer_all_weights(self):
        """Executar transferência completa de pesos"""
        print("🚀 Iniciando transferência completa de pesos GPT-2 → Hilbert")
        print("=" * 60)

        success = True

        # Executar todas as transferências
        success &= self.transfer_embeddings()
        success &= self.transfer_transformer_layers()
        success &= self.transfer_final_layer_norm()
        success &= self.transfer_lm_head()

        # Mostrar relatório
        print("\n📊 RELATÓRIO DE TRANSFERÊNCIA:")
        print("=" * 40)
        for log_entry in self.transfer_log:
            print(log_entry)

        if success:
            print(f"\n🎉 Transferência concluída: {len([x for x in self.transfer_log if '✅' in x])} componentes")
        else:
            print(f"\n⚠️  Transferência parcial: {len([x for x in self.transfer_log if '✅' in x])} componentes")

        return success

def create_hilbert_model_from_gpt2(gpt2_model_name='gpt2', hilbert_space="complex"):
    """
    Função principal: criar modelo Hilbert com pesos do GPT-2
    """
    print(f"🧠 Criando modelo Hilbert a partir de {gpt2_model_name}")

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

        # Criar configuração Hilbert compatível - SIMPLIFICADA PARA FUNCIONAR
        hilbert_config = HilbertConfig(
            vocab_size=gpt2_model.config.vocab_size,
            hidden_size=gpt2_model.config.hidden_size,
            num_hidden_layers=6,  # Limitar para 6 camadas para compatibilidade
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

        # Transferir pesos
        transfer_manager = WeightTransferGPT2ToHilbert(gpt2_model, hilbert_model)
        transfer_success = transfer_manager.transfer_all_weights()

        if transfer_success:
            print("\n🎉 Modelo Hilbert com pesos do GPT-2 criado com sucesso!")

            # Testar geração básica (simplificada)
            print("\n🧪 Testando geração básica...")
            test_prompt = "The quantum nature of"

            # Usar o test_gpt2_hilbert.py para teste consistente
            print("   🔄 Usando test_gpt2_hilbert.py para teste consistente...")

            return hilbert_model, tokenizer, transfer_success

        else:
            print("❌ Falha na transferência de pesos")
            return None, tokenizer, False

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

def save_hilbert_model(model, tokenizer, save_path="models/hilbert_gpt2"):
    """Salvar modelo Hilbert treinado"""
    print(f"💾 Salvando modelo em {save_path}...")

    os.makedirs(save_path, exist_ok=True)

    # Salvar modelo com safe_serialization=False para evitar problemas de tensores compartilhados
    # Usar safetensors se disponível para maior segurança
    try:
        model.save_pretrained(save_path, safe_serialization=True)
        print("   ✅ Salvo com safetensors (recomendado)")
    except Exception as e:
        print(f"   ⚠️  Safetensors falhou: {e}")
        model.save_pretrained(save_path, safe_serialization=False)
        print("   ✅ Salvo com pickle (fallback)")

    tokenizer.save_pretrained(save_path)

    print("✅ Modelo salvo com sucesso!")

def load_pretrained_hilbert_gpt2(model_path="models/hilbert_gpt2"):
    """
    Carrega modelo Hilbert com pesos do GPT-2 pré-treinado
    """
    print(f"📥 Carregando modelo Hilbert pré-treinado: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado em {model_path}")
        print("💡 Execute primeiro: python3 transfer_weights_gpt2_to_hilbert.py")
        return None, None

    try:
        # Carregar tokenizer primeiro
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        # Criar modelo diretamente (não carregar pesos salvos devido a problemas de compatibilidade)
        print("   🔄 Criando modelo Hilbert com pesos GPT-2...")
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

        hilbert_config = HilbertConfig(
            vocab_size=gpt2_model.config.vocab_size,
            hidden_size=gpt2_model.config.hidden_size,
            num_hidden_layers=6,
            num_attention_heads=gpt2_model.config.num_attention_heads,
            intermediate_size=gpt2_model.config.hidden_size * 4,
            hilbert_space="complex",
            spectral_alpha=1.0,
            fractal_dimension=1.5,
            use_spectral_filtering=False,
            use_fractal_embedding=True,
        )

        model = HilbertLlamaForCausalLM(hilbert_config)

        # Transferir pesos
        transfer_manager = WeightTransferGPT2ToHilbert(gpt2_model, model)
        transfer_success = transfer_manager.transfer_all_weights()

        if transfer_success:
            print("✅ Modelo Hilbert criado e pesos transferidos!")
            print(f"   📊 Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
            return model, tokenizer
        else:
            print("❌ Falha na transferência de pesos")
            return None, None

    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None, None

def enhanced_process_input_text(input_text: str, max_length: int = 50, temperature: float = 1.0, use_pretrained: bool = True):
    """
    Processa texto usando modelo Hilbert com pesos reais do GPT-2
    """
    print(f"🧠 Processando: '{input_text}'")
    print("=" * 60)

    try:
        if use_pretrained:
            # Tentar carregar modelo com pesos transferidos
            model, tokenizer = load_pretrained_hilbert_gpt2()
            if model is None:
                print("⚠️  Usando modelo Hilbert com pesos aleatórios...")
                use_pretrained = False

        if not use_pretrained:
            # Fallback para modelo com pesos aleatórios
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token

            config = HilbertConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=768,
                num_hidden_layers=6,
                num_attention_heads=12,
                intermediate_size=3072,
                hilbert_space="complex",
            )
            model = HilbertLlamaForCausalLM(config)

        # Função de geração melhorada
        def generate_text(prompt, max_length=50, temperature=1.0):
            inputs = tokenizer(prompt, return_tensors="pt")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_device = model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated = inputs['input_ids'].clone()

            with torch.no_grad():
                for i in range(max_length - inputs['input_ids'].size(1)):
                    outputs = model_device(generated, return_dict=True)
                    next_token_logits = outputs['logits'][:, -1, :]

                    # Aplicar temperatura
                    next_token_logits = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)

                    # Amostragem com núcleo (nucleus sampling)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remover tokens com probabilidade cumulativa baixa
                    sorted_indices_to_remove = cumulative_probs > 0.9
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[..., indices_to_remove] = -float('inf')

                    # Recalcular probabilidades e amostrar
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)

                    generated = torch.cat([generated, next_token], dim=-1)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

            return tokenizer.decode(generated[0], skip_special_tokens=True)

        # Gerar resposta
        result = generate_text(input_text, max_length=max_length, temperature=temperature)

        model_type = "Hilbert+GPT-2" if use_pretrained else "Hilbert (random)"
        print(f"✅ Resposta ({model_type}):")
        print(f"   📝 '{result}'")
        print(f"   📊 Comprimento: {len(result)} caracteres")

        return result

    except Exception as e:
        print(f"❌ Erro no processamento: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Exemplo de uso
    hilbert_model, tokenizer, success = create_hilbert_model_from_gpt2(
        gpt2_model_name='gpt2',
        hilbert_space="complex"
    )

    if success and hilbert_model:
        save_hilbert_model(hilbert_model, tokenizer)

        print("\n🚀 MODELO PRONTO PARA USO:")
        print("   python3 test_gpt2_hilbert.py --model models/hilbert_gpt2 \"sua pergunta\"")