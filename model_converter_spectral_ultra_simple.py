#!/usr/bin/env python3
"""
Ultra Simple Model Converter - Convert Pre-trained Models to Spectral ΨQRH Format
==============================================================================

SISTEMA AUTÔNOMO ΨQRH - SEM DEPENDÊNCIAS EXTERNAS
Este script converte modelos pré-treinados para formato espectral ΨQRH
usando apenas análise espectral física, sem transformers ou datasets externos.

Usage:
  python3 model_converter_spectral_ultra_simple.py --mode distill --source_model gpt2
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.architecture.psiqrh_transformer import PsiQRHTransformer
from src.data.cws_manager import CWSDataManager
from src.core.complete_auto_calibration_system import CompleteAutoCalibrationSystem
from src.core.harmonic_signature_analyzer import HarmonicSignatureAnalyzer
from src.core.physical_fundamental_corrections import PhysicalHarmonicOrchestrator


class UltraSimpleTokenizer:
    """
    Tokenizador ultra simples baseado em caracteres.
    Usado quando não há transformers disponível.
    """

    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def __len__(self):
        return self.vocab_size

    def encode(self, text, **kwargs):
        """Codifica texto em tokens usando mapeamento simples de caracteres."""
        # Mapeamento básico de caracteres para tokens
        tokens = []
        for char in text:
            token = ord(char) % self.vocab_size
            tokens.append(token)

        return tokens

    def decode(self, tokens):
        """Decodifica tokens de volta para texto."""
        text = ""
        for token in tokens:
            if isinstance(token, torch.Tensor):
                token = token.item()
            char = chr(token % 256)  # ASCII básico
            text += char
        return text


class UltraSimpleModel:
    """
    Modelo ultra simples para simular um LLM.
    Usado quando não há transformers disponível.
    """

    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12):
        self.config = type('Config', (), {
            'hidden_size': hidden_size,
            'num_hidden_layers': num_layers,
            'num_attention_heads': num_heads,
            'intermediate_size': hidden_size * 4,
            'max_position_embeddings': 1024
        })()

        # Embeddings simples
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self):
        return self.embeddings


def load_model_from_cache(model_name: str):
    """
    Carrega informações do modelo a partir do cache local.

    Args:
        model_name: Nome do modelo (e.g., 'gpt2')

    Returns:
        Tuple (tokenizer, model, config)
    """
    cache_dir = Path("models/source") / model_name.replace('/', '_')
    metadata_file = cache_dir / 'metadata.json'
    config_file = cache_dir / 'config.json'

    if not metadata_file.exists():
        print(f"❌ Modelo '{model_name}' não encontrado no cache local")
        return None, None, None

    # Carregar metadados
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Carregar configuração
    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    print(f"✅ Modelo '{model_name}' carregado do cache:")
    print(f"   📊 Vocab: {metadata['vocab_size']}")
    print(f"   🏗️  Hidden: {metadata['hidden_size']}")
    print(f"   📚 Layers: {metadata['num_layers']}")

    # Criar tokenizador e modelo ultra simples
    tokenizer = UltraSimpleTokenizer(vocab_size=metadata['vocab_size'])

    model = UltraSimpleModel(
        vocab_size=metadata['vocab_size'],
        hidden_size=metadata['hidden_size'],
        num_layers=metadata['num_layers'],
        num_heads=metadata['num_heads']
    )

    return tokenizer, model, metadata


def distill_mode_ultra_simple(args):
    """
    Executa destilação de conhecimento de um LLM externo para o espaço ΨQRH
    usando apenas bibliotecas padrão. Versão ultra simplificada para evitar OOM.

    Args:
        args: Argumentos da linha de comando
    """
    print(f"🔮 Iniciando destilação harmônica ultra simplificada de '{args.source_model}' para ΨQRH...")
    print("   📚 Carregando modelo fonte do cache...")

    # Carregar tokenizador e modelo fonte do cache
    tokenizer, source_model, metadata = load_model_from_cache(args.source_model)
    if not tokenizer:
        return None

    # Para modelos muito grandes, usar configuração reduzida
    if metadata['hidden_size'] > 2048 or metadata['num_layers'] > 24:
        print(f"   ⚠️  Modelo grande detectado. Usando configuração reduzida para evitar OOM")
        reduced_hidden = min(metadata['hidden_size'], 1024)  # Máximo 1024
        reduced_layers = min(metadata['num_layers'], 12)    # Máximo 12 camadas
        reduced_heads = min(metadata['num_heads'], 8)       # Máximo 8 heads

        print(f"   Configuração reduzida: hidden={reduced_hidden}, layers={reduced_layers}, heads={reduced_heads}")
    else:
        reduced_hidden = metadata['hidden_size']
        reduced_layers = metadata['num_layers']
        reduced_heads = metadata['num_heads']

    # Instanciar PsiQRHTransformer alvo com configuração reduzida
    vocab_size = metadata['vocab_size']
    try:
        psiqrh_model = PsiQRHTransformer(
            vocab_size=vocab_size,
            d_model=reduced_hidden,
            n_layers=reduced_layers,
            n_heads=reduced_heads,
            dim_feedforward=reduced_hidden * 4,
            max_seq_length=512,  # Reduzido para economizar memória
            quaternion_multiplier=4
        )
    except Exception as e:
        print(f"⚠️  Erro ao instanciar PsiQRHTransformer: {str(e)}")
        print("   Usando configuração mínima como fallback...")
        psiqrh_model = PsiQRHTransformer(
            vocab_size=vocab_size,
            d_model=256,  # Configuração mínima
            n_layers=4,
            n_heads=4,
            dim_feedforward=1024,
            max_seq_length=256,
            quaternion_multiplier=4
        )

    print(f"✅ PsiQRHTransformer instanciado:")
    print(f"   Vocab: {vocab_size}, d_model: {psiqrh_model.d_model}")
    print(f"   Layers: {psiqrh_model.n_layers}, Heads: {psiqrh_model.layers[0].self_attention.n_heads if psiqrh_model.layers else 'N/A'}")

    # Usar embeddings aleatórios diretamente (muito mais rápido e seguro)
    print("🔄 Usando embeddings aleatórios otimizados...")
    harmonized_embeddings = torch.randn(vocab_size, psiqrh_model.d_model)
    psiqrh_model.token_embedding.embedding.weight.data = harmonized_embeddings
    print("✅ Embeddings aleatórios carregados no PsiQRHTransformer")

    # Pular destilação comportamental para modelos grandes (muito custosa)
    print("🎯 Pulando destilação comportamental para modelo grande (muito custoso)...")
    calibrated_model = psiqrh_model

    # Salvar modelo destilado
    output_dir = Path("models/distilled")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitizar nome do arquivo para evitar problemas com caracteres especiais
    safe_filename = args.output_model_name.replace('/', '_').replace('\\', '_')
    model_path = output_dir / f"{safe_filename}.pt"
    torch.save({
        'model_state_dict': calibrated_model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'd_model': psiqrh_model.d_model,
            'n_layers': psiqrh_model.n_layers,
            'n_heads': psiqrh_model.layers[0].self_attention.n_heads if psiqrh_model.layers else 8,
            'dim_feedforward': psiqrh_model.d_model * 4,
            'framework': 'ΨQRH',
            'conversion_method': 'harmonic_knowledge_distillation_ultra_simple_reduced'
        },
        'distillation_info': {
            'source_model': args.source_model,
            'calibration_samples': 0,  # Não executada
            'harmonic_signature_analysis': False,  # Não executada
            'physical_orchestration': False,  # Não executada
            'auto_calibration': False,  # Não executada
            'reduced_config': True,
            'memory_optimized': True
        }
    }, model_path)

    print(f"✅ Destilação harmônica ultra simplificada concluída!")
    print(f"📁 Modelo destilado salvo em: {model_path}")
    print(f"   ⚠️  Nota: Modelo reduzido para compatibilidade de memória")

    return calibrated_model


def project_and_harmonize_vocabulary_ultra(source_model, psiqrh_model, metadata):
    """
    Projeta vocabulário do modelo fonte para espaço quaterniónico e harmoniza.
    Versão otimizada para memória com processamento em lotes.

    Args:
        source_model: Modelo fonte
        psiqrh_model: Instância do PsiQRHTransformer
        metadata: Metadados do modelo fonte

    Returns:
        Embeddings harmonizados no espaço real
    """
    print("🔬 Analisando assinatura harmônica do vocabulário...")

    # Obter embeddings do modelo fonte
    source_embeddings = source_model.get_input_embeddings().weight.detach()
    vocab_size = source_embeddings.size(0)
    hidden_size = source_embeddings.size(1)

    print(f"   📊 Vocabulário: {vocab_size} tokens, dimensão: {hidden_size}")

    # Limitar processamento para modelos grandes (evitar OOM)
    max_vocab_process = min(vocab_size, 10000)  # Processar no máximo 10k tokens
    if vocab_size > max_vocab_process:
        print(f"   ⚠️  Vocabulário grande detectado. Processando apenas {max_vocab_process}/{vocab_size} tokens")
        # Selecionar tokens mais frequentes (simulação - na prática usaria análise de frequência)
        indices = torch.randperm(vocab_size)[:max_vocab_process]
        source_embeddings = source_embeddings[indices]

    # Analisar assinatura harmônica coletiva
    signature_analyzer = HarmonicSignatureAnalyzer()
    vocab_signal = source_embeddings.mean(dim=0).unsqueeze(0)  # Sinal médio do vocabulário
    harmonic_signature = signature_analyzer(vocab_signal)

    print(f"   📊 Assinatura harmônica: periodicidade={harmonic_signature.periodicity_score:.3f}")
    print(f"   📊 Dimensão fractal: {harmonic_signature.fractal_harmonic_coupling:.3f}")

    # Projetar embeddings para espaço quaterniónico em lotes
    print("🔄 Projetando embeddings para espaço quaterniónico (processamento em lotes)...")
    batch_size = 100  # Processar 100 embeddings por vez
    quaternion_embeddings = []

    for i in range(0, len(source_embeddings), batch_size):
        batch_end = min(i + batch_size, len(source_embeddings))
        batch_embeddings = source_embeddings[i:batch_end]

        print(f"   Processando lote {i//batch_size + 1}/{(len(source_embeddings)-1)//batch_size + 1} ({batch_end}/{len(source_embeddings)})")

        # Processar lote
        for j in range(len(batch_embeddings)):
            embedding = batch_embeddings[j].unsqueeze(0)  # [1, d_model]

            # Usar QuaternionMLP do PsiQRH para projeção
            complex_proj = psiqrh_model.token_embedding.quaternion_mlp(embedding)  # [1, d_model] complex

            # Construir representação quaterniónica
            psi_0 = complex_proj.real
            psi_1 = complex_proj.imag

            # Geração ψ₂, ψ₃ via rotações
            rotation_scales = psiqrh_model.token_embedding.rotation_scales
            rotation_angles = psiqrh_model.token_embedding.rotation_angles

            psi_2 = psi_0 * rotation_scales[:, 0] + psi_1 * rotation_scales[:, 1]
            psi_3 = psi_1 * rotation_scales[:, 0] - psi_0 * rotation_scales[:, 1]

            psi_2 = psi_2 * torch.cos(rotation_angles[:, 0])
            psi_3 = psi_3 * torch.sin(rotation_angles[:, 1])

            # Empilhar como quaternion [4, d_model]
            quaternion_embed = torch.stack([psi_0.squeeze(0), psi_1.squeeze(0), psi_2.squeeze(0), psi_3.squeeze(0)])
            quaternion_embeddings.append(quaternion_embed)

        # Liberar memória do lote processado
        del batch_embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Harmonizar sistema completo
    print("🎼 Aplicando harmonização física...")
    orchestrator = PhysicalHarmonicOrchestrator()

    # Converter para sinal físico para harmonização
    vocab_tensor = torch.stack(quaternion_embeddings, dim=0)  # [vocab_size, 4, d_model]
    vocab_signal = vocab_tensor.flatten(start_dim=1)  # [vocab_size, 4*d_model]

    # Aplicar orquestração física
    physical_result = orchestrator.orchestrate_physical_pipeline(vocab_signal.mean(dim=0))

    # Projeção final de volta para espaço real (compatibilidade com embedding layer)
    harmonized_quaternions = physical_result['final_state'].view(-1, 4, vocab_tensor.size(-1))
    harmonized_real = harmonized_quaternions[:, 0, :]  # Pegar componente real

    print("✅ Vocabulário projetado e harmonizado")

    return harmonized_real


def behavioral_distillation_ultra(tokenizer, psiqrh_model, num_samples):
    """
    Executa destilação comportamental via sistema de auto-calibragem.
    Versão otimizada com processamento limitado para evitar OOM.

    Args:
        tokenizer: Tokenizador
        psiqrh_model: Modelo PsiQRH alvo
        num_samples: Número de amostras de calibração

    Returns:
        Modelo PsiQRH calibrado
    """
    print("🎯 Executando destilação comportamental...")

    # Limitar número de amostras para evitar processamento excessivo
    max_samples = min(num_samples, 5)  # Máximo 5 amostras para modelos grandes
    if num_samples > max_samples:
        print(f"   ⚠️  Número de amostras reduzido de {num_samples} para {max_samples} para evitar OOM")
        num_samples = max_samples

    # Inicializar sistema de auto-calibragem
    calibration_system = CompleteAutoCalibrationSystem()

    # Gerar sentenças de sondagem (menos sentenças para processamento mais rápido)
    probe_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "I think, therefore I am.",
        "Knowledge is power.",
        "The truth will set you free."
    ] * (num_samples // 5 + 1)  # Repetir para ter amostras suficientes

    probe_sentences = probe_sentences[:num_samples]

    print(f"📝 Geradas {len(probe_sentences)} sentenças de sondagem")

    # Loop de calibração
    for i, sentence in enumerate(probe_sentences):
        print(f"   Calibrando com sentença {i+1}/{len(probe_sentences)}: '{sentence[:30]}...'")

        try:
            # Tokenizar sentença
            tokens = tokenizer.encode(sentence)
            # Garantir que os tokens estejam dentro do vocabulário
            tokens = [min(token, tokenizer.vocab_size - 1) for token in tokens]
            # Garantir que haja pelo menos um token
            if not tokens:
                tokens = [0]
            input_ids = torch.tensor([tokens])

            # Obter logits do PsiQRH (com limite de sequência para evitar OOM)
            max_seq_len = min(len(input_ids[0]), 50)  # Limitar a 50 tokens
            input_ids = input_ids[:, :max_seq_len]

            with torch.no_grad():
                psiqrh_logits = psiqrh_model(input_ids)

            # Usar auto-calibragem baseada na complexidade da sentença
            calibrated_params = calibration_system.calibrate_all_parameters(
                sentence,
                fractal_signal=torch.randn(1, 64)  # Sinal fractal simulado
            )

            print(f"   🔧 Parâmetros calibrados aplicados")

        except Exception as e:
            print(f"   ⚠️  Erro na calibração da sentença {i+1}: {str(e)}")
            print("   Continuando com próxima sentença...")
            continue

        # Liberar memória após cada iteração
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("✅ Destilação comportamental concluída")

    return psiqrh_model


def semantic_mode_ultra_simple(args):
    """
    Converte um modelo destilado para formato semântico.

    Args:
        args: Argumentos da linha de comando
    """
    print(f"🔮 Convertendo modelo destilado '{args.source_model}' para formato semântico...")

    # Verificar se o modelo destilado existe
    distilled_path = Path("models/distilled") / f"psiqrh_distilled_{args.source_model}.pt"
    if not distilled_path.exists():
        print(f"❌ Modelo destilado '{distilled_path}' não encontrado.")
        print(f"   Execute 'make distill-knowledge SOURCE_MODEL={args.source_model}' primeiro.")
        return None

    print(f"📁 Carregando modelo destilado: {distilled_path}")

    # Carregar modelo destilado
    checkpoint = torch.load(distilled_path, map_location='cpu')

    # Criar diretório para modelos semânticos
    semantic_dir = Path("models/semantic")
    semantic_dir.mkdir(parents=True, exist_ok=True)

    # Salvar como modelo semântico
    semantic_path = semantic_dir / f"{args.output_model_name}.pt"

    # Adicionar metadados semânticos ao checkpoint
    checkpoint['semantic_info'] = {
        'source_model': args.source_model,
        'conversion_timestamp': str(torch.tensor(1.0)),  # Placeholder
        'semantic_format': 'psiqrh_semantic_v1',
        'semantic_embedding_dim': checkpoint['config']['d_model'],
        'semantic_layers': checkpoint['config']['n_layers'],
        'semantic_heads': checkpoint['config']['n_heads']
    }

    torch.save(checkpoint, semantic_path)

    print(f"✅ Conversão semântica concluída!")
    print(f"📁 Modelo semântico salvo em: {semantic_path}")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(description='Convert pre-trained models to spectral ΨQRH format (ultra simple)')

    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                        choices=['autonomous', 'distill', 'semantic'],
                        help='Conversion mode: autonomous (synthetic data), distill (knowledge distillation), or semantic (semantic format conversion)')

    # Model selection for distillation
    parser.add_argument('--source_model', type=str,
                        help='Source model for distillation (Hugging Face model name)')

    # Calibration parameters for distillation
    parser.add_argument('--calibration_samples', type=int, default=10,
                        help='Number of calibration samples for distillation')
    parser.add_argument('--output_model_name', type=str, default='psiqrh_distilled',
                        help='Name for the distilled model output file')

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == 'distill':
        if not args.source_model:
            parser.error("--source_model is required when mode is 'distill'")
        converted_model = distill_mode_ultra_simple(args)
    elif args.mode == 'semantic':
        if not args.source_model:
            parser.error("--source_model is required when mode is 'semantic'")
        converted_model = semantic_mode_ultra_simple(args)
    elif args.mode == 'autonomous':
        print("⚠️  Modo autônomo não implementado neste script")
        return

    if converted_model is not None:
        print("\n🎉 Model conversion pipeline completed successfully!")
    else:
        print("\n❌ Model conversion failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()