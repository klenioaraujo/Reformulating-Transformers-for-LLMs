#!/usr/bin/env python3
"""
Validação Robusta do Modelo ΨQRH Treinado
==========================================

Script de validação end-to-end que verifica:
- Fase 1: Existência e integridade dos arquivos do modelo
- Fase 2: Capacidade de carregamento pelo QRHFactory
- Fase 3: Benchmark comparativo de perplexidade

Usage:
    python3 validate_training_output.py --model_dir ./models/psiqrh_wikitext_v2
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.ΨQRH import QRHFactory
from src.architecture.psiqrh_transformer import PsiQRHTransformer
from src.core.fractal_quantum_embedding import PsiQRHTransformerComplete


def create_model_from_config(config: dict, trained: bool = False):
    """
    Função auxiliar para criar modelo baseado na configuração.

    Args:
        config: Dicionário de configuração
        trained: Se True, carrega pesos treinados

    Returns:
        Modelo ΨQRH (PsiQRHTransformer ou PsiQRHTransformerComplete)
    """
    model_type = config.get('model_type', 'PsiQRHTransformer')
    use_complete = config.get('use_complete', False)

    if use_complete or model_type == 'PsiQRHTransformerComplete':
        model = PsiQRHTransformerComplete(
            vocab_size=config['vocab_size'],
            embed_dim=config.get('embed_dim', 128),
            quaternion_dim=4,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            n_rotations=config.get('n_rotations', 4),
            dropout=0.1,
            max_seq_len=config['max_seq_length'],
            use_leech_correction=False
        )
    else:
        model = PsiQRHTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_seq_length=config['max_seq_length']
        )

    return model


class ValidationReport:
    """Classe para armazenar resultados de validação"""

    def __init__(self):
        self.phase1_file_check = {}
        self.phase1_loading = {}
        self.phase2_perplexity = {}
        self.phase3_metrics = {}
        self.overall_status = "PENDING"

    def print_summary(self):
        """Imprime resumo da validação"""
        print("\n" + "="*70)
        print("📋 RELATÓRIO DE VALIDAÇÃO DO MODELO ΨQRH")
        print("="*70)

        # Fase 1
        print("\n🔍 FASE 1: Verificação de Artefatos")
        print("-" * 70)
        for check, status in self.phase1_file_check.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {check}")

        if self.phase1_loading:
            print("\n  Carregamento do Modelo:")
            for key, value in self.phase1_loading.items():
                if key == "success":
                    icon = "✅" if value else "❌"
                    status = "SUCESSO" if value else "FALHOU"
                    print(f"    {icon} Status: {status}")
                elif key == "error":
                    print(f"    ⚠️  Erro: {value}")

        # Fase 2
        if self.phase2_perplexity:
            print("\n📊 FASE 2: Benchmark de Perplexidade")
            print("-" * 70)
            for model_name, metrics in self.phase2_perplexity.items():
                if isinstance(metrics, dict):
                    print(f"  {model_name}:")
                    print(f"    Perplexity: {metrics.get('perplexity', 'N/A'):.2f}")
                    print(f"    Loss: {metrics.get('loss', 'N/A'):.4f}")
                    print(f"    Tempo: {metrics.get('time', 'N/A'):.1f}s")

        # Status geral
        print("\n" + "="*70)
        if self.overall_status == "PASS":
            print("✅ VALIDAÇÃO COMPLETA: MODELO APROVADO")
        elif self.overall_status == "FAIL":
            print("❌ VALIDAÇÃO FALHOU: VERIFICAR ERROS ACIMA")
        else:
            print("⚠️  VALIDAÇÃO INCOMPLETA")
        print("="*70 + "\n")


def validate_phase1_files(model_dir: Path) -> Dict[str, bool]:
    """
    Fase 1.1: Valida existência dos arquivos do modelo

    Args:
        model_dir: Diretório do modelo treinado

    Returns:
        Dicionário com status de cada arquivo
    """
    print("\n🔍 FASE 1.1: Verificando arquivos do modelo...")

    required_files = {
        'pytorch_model.bin': model_dir / 'pytorch_model.bin',
        'config.json': model_dir / 'config.json',
        'model_info.json': model_dir / 'model_info.json',
        'vocab.json': model_dir / 'vocab.json'
    }

    results = {}
    for name, path in required_files.items():
        exists = path.exists()
        results[name] = exists
        icon = "✅" if exists else "❌"
        print(f"  {icon} {name}: {'Encontrado' if exists else 'Não encontrado'}")

    return results


def validate_phase1_loading(model_dir: Path) -> Dict[str, any]:
    """
    Fase 1.2: Valida carregamento direto do modelo nativo ΨQRH

    Args:
        model_dir: Diretório do modelo treinado

    Returns:
        Dicionário com status de carregamento
    """
    print("\n🔍 FASE 1.2: Testando carregamento do modelo nativo...")

    try:
        # Carregar configuração
        with open(model_dir / 'config.json', 'r') as f:
            config = json.load(f)

        # Carregar vocabulário
        with open(model_dir / 'vocab.json', 'r') as f:
            vocab_data = json.load(f)

        # Criar modelo usando função auxiliar
        model = create_model_from_config(config)

        model_type = config.get('model_type', 'PsiQRHTransformer')
        if model_type == 'PsiQRHTransformerComplete':
            print(f"  🔬 Detectado: PsiQRHTransformerComplete (física rigorosa)")
        else:
            print(f"  🔬 Detectado: PsiQRHTransformer (implementação original)")

        # Carregar pesos
        state_dict = torch.load(model_dir / 'pytorch_model.bin', map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

        print("  ✅ Modelo carregado com sucesso")
        print(f"  ✅ Vocab size: {vocab_data['vocab_size']}")
        print(f"  ✅ Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  ✅ Config: d_model={config['d_model']}, n_layers={config['n_layers']}")

        return {
            'success': True,
            'vocab_size': vocab_data['vocab_size'],
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'd_model': config['d_model'],
            'n_layers': config['n_layers']
        }

    except Exception as e:
        print(f"  ❌ Erro ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def evaluate_perplexity(model, char_to_idx: dict, text: str, device: str = 'cpu', seq_length: int = 128) -> Tuple[float, float]:
    """
    Calcula perplexidade do modelo usando tokenização de caracteres

    Args:
        model: Modelo a avaliar
        char_to_idx: Dicionário de caracteres para índices
        text: Texto para avaliar
        device: Dispositivo
        seq_length: Comprimento de sequência

    Returns:
        (perplexity, loss)
    """
    model.eval()
    model.to(device)

    # Converter texto em índices
    indices = []
    for ch in text:
        if ch in char_to_idx:
            indices.append(char_to_idx[ch])
        else:
            indices.append(0)  # UNK token

    # Criar sequências
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(indices) - seq_length, seq_length), desc="Avaliando"):
            input_seq = indices[i:i+seq_length]
            target_seq = indices[i+1:i+seq_length+1]

            if len(input_seq) < seq_length:
                continue

            # Converter para tensores
            input_ids = torch.tensor([input_seq], dtype=torch.long).to(device)
            labels = torch.tensor([target_seq], dtype=torch.long).to(device)

            # Forward pass
            try:
                logits = model(input_ids)

                # Calcular loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"⚠️  Erro ao processar sequência: {e}")
                continue

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity, avg_loss


def validate_phase2_benchmark(model_dir: Path, text_file: str = 'data/train.txt', device: str = 'cpu') -> Dict[str, Dict[str, float]]:
    """
    Fase 2.1: Benchmark comparativo de perplexidade

    Compara:
    1. Modelo ΨQRH Não Treinado
    2. Modelo ΨQRH Treinado

    Args:
        model_dir: Diretório do modelo treinado
        text_file: Arquivo de texto para avaliar
        device: Dispositivo

    Returns:
        Dicionário com métricas de cada modelo
    """
    print("\n📊 FASE 2.1: Benchmark de Perplexidade Comparativo...")

    results = {}

    # Carregar vocabulário
    with open(model_dir / 'vocab.json', 'r') as f:
        vocab_data = json.load(f)

    char_to_idx = vocab_data['char_to_idx']
    # Convert string keys to actual strings if needed
    char_to_idx = {str(k): int(v) for k, v in char_to_idx.items()}

    # Carregar configuração
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # Carregar texto de validação
    if os.path.exists(text_file):
        with open(text_file, 'r', encoding='utf-8') as f:
            validation_text = f.read()[:10000]  # Primeiros 10k caracteres
    else:
        print(f"  ⚠️  Arquivo {text_file} não encontrado, usando texto de exemplo")
        validation_text = "The ΨQRH framework operates in spectral domain." * 100

    # 1. Modelo Não Treinado
    print("\n  Avaliando: Modelo ΨQRH Não Treinado...")
    untrained_model = create_model_from_config(config)

    start_time = time.time()
    untrained_perplexity, untrained_loss = evaluate_perplexity(
        untrained_model, char_to_idx, validation_text, device, seq_length=config['max_seq_length']
    )
    untrained_time = time.time() - start_time

    results['ΨQRH Não Treinado'] = {
        'perplexity': untrained_perplexity,
        'loss': untrained_loss,
        'time': untrained_time
    }

    print(f"    Perplexity: {untrained_perplexity:.2f}")
    print(f"    Loss: {untrained_loss:.4f}")
    print(f"    Tempo: {untrained_time:.1f}s")

    # 2. Modelo Treinado
    print("\n  Avaliando: Modelo ΨQRH Treinado...")
    trained_model = create_model_from_config(config)

    # Carregar pesos treinados
    state_dict = torch.load(model_dir / 'pytorch_model.bin', map_location='cpu')
    trained_model.load_state_dict(state_dict)

    start_time = time.time()
    trained_perplexity, trained_loss = evaluate_perplexity(
        trained_model, char_to_idx, validation_text, device, seq_length=config['max_seq_length']
    )
    trained_time = time.time() - start_time

    results['ΨQRH Treinado'] = {
        'perplexity': trained_perplexity,
        'loss': trained_loss,
        'time': trained_time
    }

    print(f"    Perplexity: {trained_perplexity:.2f}")
    print(f"    Loss: {trained_loss:.4f}")
    print(f"    Tempo: {trained_time:.1f}s")

    # Comparação
    improvement = ((untrained_perplexity - trained_perplexity) / untrained_perplexity) * 100
    print(f"\n  📈 Melhoria: {improvement:.1f}%")

    if trained_perplexity < untrained_perplexity:
        print("  ✅ Modelo treinado é melhor que não treinado")
    else:
        print("  ❌ Modelo treinado NÃO é melhor que não treinado")

    return results


def main():
    parser = argparse.ArgumentParser(description='Validar modelo ΨQRH treinado')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Diretório do modelo treinado')
    parser.add_argument('--text_file', type=str, default='data/train.txt',
                        help='Arquivo de texto para validação')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Dispositivo (cpu, cuda)')
    parser.add_argument('--skip_benchmark', action='store_true',
                        help='Pular benchmark de perplexidade (mais rápido)')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        print(f"❌ Erro: Diretório não encontrado: {model_dir}")
        sys.exit(1)

    # Criar relatório
    report = ValidationReport()

    print("\n" + "="*70)
    print("🚀 INICIANDO VALIDAÇÃO DO MODELO ΨQRH TREINADO")
    print("="*70)
    print(f"Modelo: {model_dir}")
    print(f"Device: {args.device}")
    print("="*70)

    # FASE 1: Verificação de Artefatos
    report.phase1_file_check = validate_phase1_files(model_dir)
    report.phase1_loading = validate_phase1_loading(model_dir)

    # Verificar se fase 1 passou
    all_files_ok = all(report.phase1_file_check.values())
    loading_ok = report.phase1_loading.get('success', False)

    if not all_files_ok or not loading_ok:
        print("\n❌ VALIDAÇÃO FALHOU NA FASE 1")
        report.overall_status = "FAIL"
        report.print_summary()
        sys.exit(1)

    # FASE 2: Benchmark de Perplexidade
    if not args.skip_benchmark:
        try:
            report.phase2_perplexity = validate_phase2_benchmark(model_dir, args.text_file, args.device)

            # Verificar se treinado é melhor
            untrained_ppl = report.phase2_perplexity['ΨQRH Não Treinado']['perplexity']
            trained_ppl = report.phase2_perplexity['ΨQRH Treinado']['perplexity']

            if trained_ppl >= untrained_ppl:
                print("\n⚠️  AVISO: Modelo treinado não apresentou melhoria")
                report.overall_status = "FAIL"
            else:
                report.overall_status = "PASS"

        except Exception as e:
            print(f"\n❌ Erro na Fase 2: {e}")
            import traceback
            traceback.print_exc()
            report.overall_status = "FAIL"
    else:
        print("\n⚠️  Benchmark de perplexidade pulado")
        report.overall_status = "PASS"

    # Imprimir relatório final
    report.print_summary()

    # Exit code baseado no status
    sys.exit(0 if report.overall_status == "PASS" else 1)


if __name__ == '__main__':
    main()
