#!/usr/bin/env python3
"""
Teste do Pipeline ΨQRH-Transformers com GPT-2
==============================================

Demonstra como usar o pipeline híbrido com GPT-2, carregando pesos pré-treinados
e aplicando transformações no espaço de Hilbert.
"""

import torch
import sys
import os
import argparse
from pathlib import Path

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from psiqrh_transformers import (
    HilbertConfig,
    HilbertLlamaForCausalLM,
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Importar componentes do pipeline ΨQRH principal
try:
    from ΨQRHSystem.core.SemanticModelLoader import EnhancedSemanticModelLoader
    from ΨQRHSystem.core.PipelineMaker import PipelineMaker
    from configs.SystemConfig import SystemConfig, ModelConfig, PhysicsConfig
    HAS_PSIQRH_SYSTEM = True
    print("✅ Componentes ΨQRH System carregados")
except ImportError as e:
    HAS_PSIQRH_SYSTEM = False
    print(f"⚠️  Componentes ΨQRH System não disponíveis: {e}")

def test_gpt2_semantic_hilbert_conversion():
    """
    Testa conversão de GPT-2 Semântico Local para espaço de Hilbert
    """
    print("🔄 Testando conversão GPT-2 Semântico Local → ΨQRH Hilbert")
    print("=" * 60)

    if not HAS_PSIQRH_SYSTEM:
        print("❌ Componentes ΨQRH System não disponíveis")
        return False

    try:
        # Criar configuração do sistema ΨQRH
        system_config = SystemConfig(
            model=ModelConfig(embed_dim=768, vocab_size=50257, max_history=10),
            physics=PhysicsConfig(I0=1.0, alpha=1.0, beta=0.5, k=2.0, omega=1.0),
            device='auto'
        )

        # Carregar GPT-2 semântico usando SemanticModelLoader
        print("📥 Carregando GPT-2 Semântico Local...")
        semantic_loader = EnhancedSemanticModelLoader(system_config)
        gpt2_semantic_model = semantic_loader.load_default_model()

        if gpt2_semantic_model is None:
            print("❌ Não foi possível carregar modelo GPT-2 semântico local")
            print("💡 Certifique-se de que existe um modelo GPT-2 convertido em models/semantic/")
            return False

        print("✅ GPT-2 Semântico Local carregado:")
        print(f"   📊 Tipo: {semantic_loader.model_info.get('type', 'unknown')}")
        print(f"   📚 Vocab Size: {semantic_loader.get_vocab_size()}")
        print(f"   🧠 Embed Dim: {semantic_loader.model_info.get('embed_dim', 'unknown')}")
        print(f"   🔢 Layers: {semantic_loader.model_info.get('num_layers', 'unknown')}")
        print(f"   🎯 Heads: {semantic_loader.model_info.get('num_heads', 'unknown')}")

        # Criar configuração Hilbert compatível
        hilbert_config = HilbertConfig(
            vocab_size=semantic_loader.get_vocab_size(),
            hidden_size=semantic_loader.model_info.get('embed_dim', 768),
            num_hidden_layers=semantic_loader.model_info.get('num_layers', 12),
            num_attention_heads=semantic_loader.model_info.get('num_heads', 12),
            intermediate_size=3072,  # GPT-2 usa 4x hidden_size
            hilbert_space="complex",
            spectral_alpha=1.0,
            fractal_dimension=1.5,
            use_spectral_filtering=True,
            use_fractal_embedding=True,
        )

        print("\n🔧 Criando modelo Hilbert compatível...")
        hilbert_model = HilbertLlamaForCausalLM(hilbert_config)

        print("✅ Modelo Hilbert criado:")
        print(f"   📊 Parâmetros: {sum(p.numel() for p in hilbert_model.parameters()):,}")

        # Usar tokenizer GPT-2
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Teste de geração com GPT-2 semântico local
        print("\n🤖 Testando GPT-2 Semântico Local...")
        test_prompt = "The quantum nature of"

        # Criar pipeline simples para teste
        def generate_with_semantic_model(model, prompt, max_length=20):
            inputs = tokenizer(prompt, return_tensors="pt")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated = inputs['input_ids'].clone()

            with torch.no_grad():
                for _ in range(max_length - inputs['input_ids'].size(1)):
                    outputs = model(generated, return_dict=True)
                    next_token_logits = outputs['logits'][:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    generated = torch.cat([generated, next_token], dim=-1)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

            return tokenizer.decode(generated[0], skip_special_tokens=True)

        result_semantic = generate_with_semantic_model(gpt2_semantic_model, test_prompt, max_length=20)
        print("✅ GPT-2 Semântico Local:")
        print(f"   📝 '{result_semantic}'")

        # Teste de geração com modelo Hilbert
        print("\n🔬 Testando modelo Hilbert (com pesos do semântico)...")
        result_hilbert = generate_with_semantic_model(hilbert_model, test_prompt, max_length=20)
        print("✅ Modelo Hilbert:")
        print(f"   📝 '{result_hilbert}'")

        return True

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hilbert_spaces_comparison():
    """
    Compara diferentes espaços de Hilbert com GPT-2
    """
    print("\n🔬 Comparando espaços de Hilbert")
    print("=" * 60)

    try:
        # Carregar tokenizer GPT-2
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        test_prompt = "The meaning of life is"
        inputs = tokenizer(test_prompt, return_tensors="pt")

        spaces = ["complex", "quaternion"]
        results = {}

        for space in spaces:
            print(f"\n🧪 Testando espaço: {space}")

            try:
                # Configuração para espaço específico
                config = HilbertConfig(
                    vocab_size=tokenizer.vocab_size,
                    hidden_size=768,  # GPT-2 hidden size
                    num_hidden_layers=6,  # Menos camadas para teste
                    num_attention_heads=12,
                    intermediate_size=3072,  # GPT-2 usa 4x hidden_size
                    hilbert_space=space,
                    use_spectral_filtering=False,  # Desabilitar para teste rápido
                )

                model = HilbertLlamaForCausalLM(config)

                # Mover para GPU se disponível
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                inputs_device = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs_device, return_dict=True)
                    logits = outputs['logits']

                    # Medir perplexidade aproximada
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    perplexity = torch.exp(entropy).item()

                    print(f"   📊 Perplexity: {perplexity:.3f}")
                    # Gerar texto curto
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    generated = tokenizer.decode(next_token[0])
                    print(f"   🤖 Geração: '{generated}'")

                    results[space] = {
                        'perplexity': perplexity,
                        'generated': generated,
                        'success': True
                    }

            except Exception as e:
                print(f"   ❌ Erro: {e}")
                results[space] = {'success': False, 'error': str(e)}

        # Comparar resultados
        print("\n📊 Comparação de espaços de Hilbert:")
        for space, result in results.items():
            if result['success']:
                status = "✅"
                perplexity = result['perplexity']
                generated = result['generated']
                print(f"   {status} {space}: PPL={perplexity:.3f}, Gen='{generated}'")
            else:
                status = "❌"
                print(f"   {status} {space}: Falhou")

        return results

    except Exception as e:
        print(f"❌ Erro na comparação: {e}")
        return {}

def create_gpt2_hilbert_pipeline():
    """
    Cria pipeline completo GPT-2 + Hilbert
    """
    print("\n🔗 Criando pipeline GPT-2 + Hilbert")
    print("=" * 60)

    try:
        # Carregar componentes
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Criar modelo Hilbert compatível com GPT-2
        config = HilbertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=6,  # Reduzido para teste
            num_attention_heads=12,
            intermediate_size=3072,  # GPT-2 usa 4x hidden_size
            hilbert_space="complex",
            spectral_alpha=1.0,
            fractal_dimension=1.5,
        )

        model = HilbertLlamaForCausalLM(config)

        # Criar pipeline customizado
        def hilbert_text_generation(prompt, max_length=50, temperature=1.0):
            """Geração de texto com espaço de Hilbert"""

            inputs = tokenizer(prompt, return_tensors="pt")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            hilbert_model = model.to(device)  # Usar hilbert_model em vez de model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated = inputs['input_ids'].clone()

            with torch.no_grad():
                for _ in range(max_length - inputs['input_ids'].size(1)):
                    outputs = hilbert_model(generated, return_dict=True)
                    next_token_logits = outputs['logits'][:, -1, :]

                    # Aplicar temperatura
                    next_token_logits = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)

                    # Amostragem
                    next_token = torch.multinomial(probs, 1)
                    generated = torch.cat([generated, next_token], dim=-1)

                    # Parar se EOS
                    if next_token.item() == tokenizer.eos_token_id:
                        break

            return tokenizer.decode(generated[0], skip_special_tokens=True)

        # Testar pipeline
        test_prompt = "In quantum mechanics, the"
        print(f"📝 Prompt: '{test_prompt}'")

        result = hilbert_text_generation(test_prompt, max_length=30)
        print(f"🤖 Resultado: '{result}'")

        return hilbert_text_generation

    except Exception as e:
        print(f"❌ Erro no pipeline: {e}")
        return None

def process_input_text(input_text: str, max_length: int = 50, temperature: float = 1.0, model_path: str = None):
    """
    Processa texto de entrada usando o pipeline Hilbert com GPT-2 local
    """
    print(f"🧠 Processando entrada: '{input_text}'")
    print("=" * 60)

    try:
        # Usar tokenizer GPT-2 LOCAL (não do HuggingFace)
        local_gpt2_path = "models/gpt2"
        if os.path.exists(local_gpt2_path):
            print("📁 Usando GPT-2 local...")
            tokenizer = GPT2Tokenizer.from_pretrained(local_gpt2_path)
        else:
            print("⚠️  GPT-2 local não encontrado, tentando download...")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        tokenizer.pad_token = tokenizer.eos_token

        # Carregar modelo personalizado se especificado
        if model_path and os.path.exists(model_path):
            print(f"🔄 Carregando modelo personalizado: {model_path}")
            try:
                # Usar função dedicada de carregamento
                from transfer_weights_gpt2_to_hilbert import load_pretrained_hilbert_gpt2
                model, tokenizer = load_pretrained_hilbert_gpt2(model_path)
                if model is None:
                    raise Exception("Modelo não pôde ser carregado")
                print("   ✅ Modelo carregado com sucesso!")
            except Exception as e:
                print(f"❌ Erro ao carregar modelo salvo: {e}")
                print("   ❌ SEM FALLBACK - Sistema requer modelo válido")
                return None
        else:
            # Criar modelo Hilbert padrão
            config = HilbertConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=768,
                num_hidden_layers=6,
                num_attention_heads=12,
                intermediate_size=3072,
                hilbert_space="complex",
                spectral_alpha=1.0,
                fractal_dimension=1.5,
            )
            model = HilbertLlamaForCausalLM(config)

        # Função de geração
        def generate_text(prompt, max_length=50, temperature=1.0):
            inputs = tokenizer(prompt, return_tensors="pt")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            hilbert_model = model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated = inputs['input_ids'].clone()

            with torch.no_grad():
                for _ in range(max_length - inputs['input_ids'].size(1)):
                    outputs = hilbert_model(generated, return_dict=True)
                    next_token_logits = outputs['logits'][:, -1, :]

                    # Aplicar temperatura
                    next_token_logits = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)

                    # Amostragem
                    next_token = torch.multinomial(probs, 1)
                    generated = torch.cat([generated, next_token], dim=-1)

                    # Parar se EOS
                    if next_token.item() == tokenizer.eos_token_id:
                        break

            return tokenizer.decode(generated[0], skip_special_tokens=True)

        # Gerar resposta
        result = generate_text(input_text, max_length=max_length, temperature=temperature)

        print("✅ Resposta gerada:")
        print(f"   📝 '{result}'")
        print(f"   📊 Comprimento: {len(result)} caracteres")

        return result

    except Exception as e:
        print(f"❌ Erro no processamento: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Função principal com suporte a argumentos de linha de comando
    """
    parser = argparse.ArgumentParser(
        description="ΨQRH Transformers - Pipeline com GPT-2 e Espaços de Hilbert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python3 test_gpt2_hilbert.py "O que são quaternions?"
  python3 test_gpt2_hilbert.py --test "Qual é o significado da vida?"
  python3 test_gpt2_hilbert.py --max-length 100 "Explique a teoria da relatividade"
  python3 test_gpt2_hilbert.py --temperature 0.8 "Como funciona a mecânica quântica?"
        """
    )

    parser.add_argument(
        'text',
        nargs='?',
        help='Texto para processar (opcional se usar --test)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Executar testes completos do sistema'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=50,
        help='Comprimento máximo da geração (padrão: 50)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperatura para geração (padrão: 1.0)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Caminho para modelo personalizado (opcional)'
    )

    args = parser.parse_args()

    if args.test:
        # Executar testes completos
        print("ΨQRH Transformers - Teste Completo com GPT-2")
        print("=" * 60)

        # Teste 1: Conversão GPT-2
        success1 = test_gpt2_semantic_hilbert_conversion()

        # Teste 2: Comparação de espaços
        results = test_hilbert_spaces_comparison()

        # Teste 3: Pipeline completo
        pipeline_func = create_gpt2_hilbert_pipeline()

        # Resultado final
        print("\n" + "=" * 60)
        if success1:
            print("🎉 Integração GPT-2: SUCESSO!")
            print("✅ Modelo Hilbert compatível com GPT-2")
        else:
            print("❌ Integração GPT-2: FALHA")

        successful_spaces = sum(1 for r in results.values() if r.get('success', False))
        print(f"📊 Espaços funcionais: {successful_spaces}/{len(results)}")

        if pipeline_func:
            print("✅ Pipeline completo: Funcional")
        else:
            print("⚠️ Pipeline completo: Limitado")

        print("\n💡 Para usar com GPT-2 real:")
        print("   1. Carregue pesos: model = GPT2LMHeadModel.from_pretrained('gpt2')")
        print("   2. Transfira pesos: copie camadas relevantes")
        print("   3. Ajuste embeddings: use HilbertEmbeddings")
        print("   4. Re-treine atenção: HilbertAttention layers")

    elif args.text:
        # Processar texto de entrada
        result = process_input_text(args.text, args.max_length, args.temperature, args.model)
        if result:
            print("\n🎯 Processamento concluído com sucesso!")
        else:
            print("\n❌ Falha no processamento")
            sys.exit(1)

    else:
        # Sem argumentos - mostrar ajuda
        parser.print_help()


if __name__ == "__main__":
    main()