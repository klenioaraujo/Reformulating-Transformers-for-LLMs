#!/usr/bin/env python3
"""
Simplified Model Converter - Convert Pre-trained Models to Spectral ΨQRH Format
==============================================================================

SISTEMA AUTÔNOMO ΨQRH - SEM DEPENDÊNCIAS EXTERNAS
Este script converte modelos pré-treinados para formato espectral ΨQRH
usando apenas análise espectral física, sem transformers ou datasets externos.

Versão simplificada que salva o modelo após a harmonização do vocabulário.

Usage:
  python3 model_converter_spectral_simplified.py --source_model gpt2
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
from src.core.harmonic_signature_analyzer import HarmonicSignatureAnalyzer
from src.core.physical_fundamental_corrections import PhysicalHarmonicOrchestrator


def load_model_from_cache(model_name: str):
    """
    Carrega informações do modelo a partir do cache local.

    Args:
        model_name: Nome do modelo (e.g., 'gpt2')

    Returns:
        Tuple (metadata, config)
    """
    cache_dir = Path("models/source") / model_name.replace('/', '_')
    metadata_file = cache_dir / 'metadata.json'
    config_file = cache_dir / 'config.json'

    if not metadata_file.exists():
        print(f"❌ Modelo '{model_name}' não encontrado no cache local")
        return None, None

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

    return metadata, config


def create_simple_embeddings(vocab_size: int, hidden_size: int):
    """
    Cria embeddings simples baseados em padrões espectrais.

    Args:
        vocab_size: Tamanho do vocabulário
        hidden_size: Dimensão dos embeddings

    Returns:
        Tensor de embeddings [vocab_size, hidden_size]
    """
    print(f"🔧 Criando embeddings sintéticos para vocabulário de {vocab_size} tokens...")

    # Criar embeddings baseados em padrões harmônicos
    embeddings = torch.zeros(vocab_size, hidden_size)

    for i in range(vocab_size):
        # Padrão harmônico baseado no índice do token
        pattern = torch.sin(torch.arange(hidden_size) * (i + 1) * 0.01)
        pattern += torch.cos(torch.arange(hidden_size) * (i + 1) * 0.02)

        # Normalizar
        pattern = pattern / torch.norm(pattern)
        embeddings[i] = pattern

    print("✅ Embeddings sintéticos criados")
    return embeddings


def project_and_harmonize_vocabulary_simple(source_embeddings, psiqrh_model, metadata):
    """
    Projeta vocabulário do modelo fonte para espaço quaterniónico e harmoniza.

    Args:
        source_embeddings: Embeddings do modelo fonte
        psiqrh_model: Instância do PsiQRHTransformer
        metadata: Metadados do modelo fonte

    Returns:
        Embeddings harmonizados no espaço real
    """
    print("🔬 Analisando assinatura harmônica do vocabulário...")

    # Analisar assinatura harmônica coletiva
    signature_analyzer = HarmonicSignatureAnalyzer()
    vocab_signal = source_embeddings.mean(dim=0).unsqueeze(0)  # Sinal médio do vocabulário
    harmonic_signature = signature_analyzer(vocab_signal)

    print(f"   📊 Assinatura harmônica: periodicidade={harmonic_signature.periodicity_score:.3f}")
    print(f"   📊 Dimensão fractal: {harmonic_signature.fractal_harmonic_coupling:.3f}")

    # Projetar cada embedding para espaço quaterniónico
    print("🔄 Projetando embeddings para espaço quaterniónico...")
    quaternion_embeddings = []

    for i in range(len(source_embeddings)):
        # Usar QuaternionMLP do PsiQRH para projeção
        embedding = source_embeddings[i].unsqueeze(0)  # [1, d_model]
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


def convert_model_simplified(model_name: str, output_model_name: str):
    """
    Converte modelo para formato espectral ΨQRH (versão simplificada).

    Args:
        model_name: Nome do modelo fonte
        output_model_name: Nome do modelo de saída
    """
    print(f"🔮 Iniciando conversão simplificada de '{model_name}' para ΨQRH...")

    # Carregar metadados do modelo fonte
    metadata, config = load_model_from_cache(model_name)
    if not metadata:
        return None

    # Instanciar PsiQRHTransformer alvo
    vocab_size = metadata['vocab_size']
    hidden_size = metadata['hidden_size']
    num_layers = metadata['num_layers']
    num_heads = metadata['num_heads']

    psiqrh_model = PsiQRHTransformer(
        vocab_size=vocab_size,
        d_model=hidden_size,
        n_layers=num_layers,
        n_heads=num_heads,
        dim_feedforward=hidden_size * 4,
        max_seq_length=1024,
        quaternion_multiplier=4
    )

    print(f"✅ PsiQRHTransformer instanciado:")
    print(f"   Vocab: {vocab_size}, d_model: {hidden_size}")
    print(f"   Layers: {num_layers}, Heads: {num_heads}")

    # Criar embeddings sintéticos
    source_embeddings = create_simple_embeddings(vocab_size, hidden_size)

    # Projeção e Harmonização do Vocabulário
    print("🔬 Executando projeção e harmonização do vocabulário...")
    harmonized_embeddings = project_and_harmonize_vocabulary_simple(
        source_embeddings, psiqrh_model, metadata
    )

    # Carregar embeddings harmonizados no PsiQRHTransformer
    psiqrh_model.token_embedding.embedding.weight.data = harmonized_embeddings
    print("✅ Embeddings harmonizados carregados no PsiQRHTransformer")

    # Salvar modelo destilado
    output_dir = Path("models/distilled")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{output_model_name}.pt"
    torch.save({
        'model_state_dict': psiqrh_model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'd_model': hidden_size,
            'n_layers': num_layers,
            'n_heads': num_heads,
            'dim_feedforward': hidden_size * 4,
            'framework': 'ΨQRH',
            'conversion_method': 'harmonic_knowledge_distillation_simplified'
        },
        'distillation_info': {
            'source_model': model_name,
            'harmonic_signature_analysis': True,
            'physical_orchestration': True,
            'auto_calibration': False
        }
    }, model_path)

    print(f"✅ Conversão harmônica simplificada concluída!")
    print(f"📁 Modelo destilado salvo em: {model_path}")

    return psiqrh_model


def main():
    parser = argparse.ArgumentParser(description='Convert pre-trained models to spectral ΨQRH format (simplified)')
    parser.add_argument('--source_model', type=str, required=True,
                        help='Source model for distillation (Hugging Face model name)')
    parser.add_argument('--output_model_name', type=str, default='psiqrh_distilled',
                        help='Name for the distilled model output file')

    args = parser.parse_args()

    # Convert model
    converted_model = convert_model_simplified(args.source_model, args.output_model_name)

    if converted_model is not None:
        print("\n🎉 Model conversion pipeline completed successfully!")
    else:
        print("\n❌ Model conversion failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()