#!/usr/bin/env python3
"""
Treinamento Nativo do ΨQRH Transformer
=======================================

Script de treinamento usando apenas a arquitetura ΨQRH nativa,
sem dependências do Hugging Face. Opera diretamente no domínio espectral.

Usage:
    python3 train_psiqrh_native.py --epochs 3 --batch_size 8
"""

import argparse
import os
import sys
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.architecture.psiqrh_transformer import PsiQRHTransformer
from src.core.fractal_quantum_embedding import PsiQRHTransformerComplete


class SimpleTextDataset(Dataset):
    """
    Dataset simples que converte texto em índices de tokens.
    Usa vocabulário baseado em caracteres (muito mais simples que BPE).
    """

    def __init__(self, text_file: str, seq_length: int = 256):
        self.seq_length = seq_length

        # Ler texto
        print(f"📖 Carregando texto de {text_file}...")
        with open(text_file, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # Criar vocabulário baseado em caracteres
        chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)

        print(f"✅ Vocabulário: {self.vocab_size} caracteres únicos")
        print(f"✅ Texto: {len(self.text)} caracteres")

        # Converter texto para índices
        self.data = [self.char_to_idx[ch] for ch in self.text]

        # Calcular número de sequências
        self.num_sequences = len(self.data) // seq_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length

        # Input: sequência atual
        input_seq = self.data[start:end]

        # Target: próximo caractere para cada posição
        target_seq = self.data[start+1:end+1]

        # Pad se necessário
        if len(input_seq) < self.seq_length:
            input_seq += [0] * (self.seq_length - len(input_seq))
        if len(target_seq) < self.seq_length:
            target_seq += [0] * (self.seq_length - len(target_seq))

        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(target_seq, dtype=torch.long)
        }

    def save_vocab(self, output_dir: Path):
        """Salva vocabulário para uso posterior"""
        vocab_file = output_dir / 'vocab.json'
        with open(vocab_file, 'w') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size
            }, f, indent=2)
        print(f"✅ Vocabulário salvo em {vocab_file}")


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Treina por uma época"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Época {epoch}")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(input_ids)

        # Calculate loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """Avalia o modelo"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validação"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


def save_model(model, output_dir, config, dataset, n_heads, max_seq_length):
    """Salva modelo e configuração"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Salvar state_dict
    model_path = output_path / 'pytorch_model.bin'
    torch.save(model.state_dict(), model_path)
    print(f"✅ Modelo salvo em {model_path}")

    # Salvar configuração
    config_path = output_path / 'config.json'
    try:
        model_info = model.get_model_info()
    except:
        # Fallback se get_model_info() falhar
        model_info = {
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "n_layers": model.n_layers,
            "n_heads": n_heads,
            "max_seq_length": max_seq_length,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "architecture": "ΨQRH Transformer"
        }
    model_info.update(config)

    with open(config_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"✅ Configuração salva em {config_path}")

    # Salvar vocabulário
    dataset.save_vocab(output_path)

    # Salvar model_info para QRHFactory
    model_info_data = {
        'model_type': 'PsiQRHTransformer',
        'framework': 'ΨQRH',
        'version': '1.0.0',
        'architecture': 'Native ΨQRH Spectral Transformer',
        'checkpoint_path': str(model_path),
        'config_path': str(config_path)
    }
    model_info_data.update(model_info)

    info_path = output_path / 'model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info_data, f, indent=2)
    print(f"✅ Model info salvo em {info_path}")


def main():
    parser = argparse.ArgumentParser(description='Treinar PsiQRHTransformer nativo')

    # Dados
    parser.add_argument('--text_file', type=str, default='data/train.txt',
                        help='Arquivo de texto para treinamento')
    parser.add_argument('--secure_asset', type=str,
                        help='Ativo seguro .Ψcws para treinamento (substitui text_file)')
    parser.add_argument('--secure_key', type=str,
                        help='Chave para ativo seguro (requer --secure_asset)')
    parser.add_argument('--seq_length', type=int, default=256,
                        help='Comprimento de sequência')

    # Treinamento
    parser.add_argument('--epochs', type=int, default=3,
                        help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Tamanho do batch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')

    # Modelo
    parser.add_argument('--d_model', type=int, default=256,
                        help='Dimensão do modelo')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Número de camadas')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Número de attention heads')
    parser.add_argument('--use_complete', action='store_true',
                        help='Usar PsiQRHTransformerComplete (nova implementação física)')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Dimensão do embedding fractal (para --use_complete)')
    parser.add_argument('--n_rotations', type=int, default=4,
                        help='Número de rotações SO(4) (para --use_complete)')

    # Output
    parser.add_argument('--output_dir', type=str, default='./models/psiqrh_native_v1',
                        help='Diretório de saída')
    parser.add_argument('--device', type=str, default='auto',
                        help='Dispositivo (auto, cpu, cuda)')

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print("🚀 TREINAMENTO NATIVO ΨQRH TRANSFORMER")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Validar configuração de dados seguros
    if args.secure_asset:
        if not args.secure_key:
            print("❌ ERRO: --secure_key é obrigatório quando --secure_asset é usado")
            sys.exit(1)

        print(f"🛡️  Ativo Seguro: {args.secure_asset}")
        print(f"🔑 Chave: {'*' * len(args.secure_key)}")

        # Validar ativo seguro
        try:
            from scripts.secure_training_integration import SecureTrainingSystem
            training_system = SecureTrainingSystem()

            if not training_system.validate_training_asset(args.secure_asset, args.secure_key):
                print("❌ Ativo seguro não validado. Treinamento cancelado.")
                sys.exit(1)

            print("✅ Ativo seguro validado com sucesso")

        except ImportError:
            print("⚠️  Sistema de segurança não disponível. Continuando sem validação...")

        # Usar arquivo seguro
        text_file = f"data/secure_assets/Ψcws/{args.secure_asset}.Ψcws"

    else:
        print(f"📄 Arquivo: {args.text_file}")
        text_file = args.text_file

    print(f"Épocas: {args.epochs}")
    print(f"Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    # Criar dataset
    if not os.path.exists(text_file):
        print(f"❌ Arquivo não encontrado: {text_file}")

        if args.secure_asset:
            print(f"💡 Execute 'make new-secure-asset' para criar o ativo seguro primeiro")
            sys.exit(1)

        print(f"\n💡 Criando arquivo de exemplo...")

        # Criar diretório se não existir
        os.makedirs(os.path.dirname(text_file) or '.', exist_ok=True)

        # Criar texto de exemplo
        example_text = """The ΨQRH framework represents a paradigm shift in transformer architectures.
By operating in the spectral domain with quaternionic representations, it achieves
superior parameter efficiency and energy conservation. The fractal consciousness
metrics enable real-time analysis of model behavior and adaptation.""" * 100

        with open(text_file, 'w') as f:
            f.write(example_text)

        print(f"✅ Arquivo de exemplo criado: {text_file}")

    dataset = SimpleTextDataset(text_file, args.seq_length)

    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Criar modelo
    if args.use_complete:
        print(f"\n🏗️  Criando PsiQRHTransformerComplete (nova implementação física)...")
        model = PsiQRHTransformerComplete(
            vocab_size=dataset.vocab_size,
            embed_dim=args.embed_dim,
            quaternion_dim=4,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_rotations=args.n_rotations,
            dropout=0.1,
            max_seq_len=args.seq_length,
            use_leech_correction=False,
            padilha_config=None
        )
        model_type = "PsiQRHTransformerComplete"
        print(f"✅ Implementação: Fractal Quantum Embedding + Spectral Attention + SO(4)")
    else:
        print(f"\n🏗️  Criando PsiQRHTransformer nativo (implementação original)...")
        model = PsiQRHTransformer(
            vocab_size=dataset.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_length=args.seq_length
        )
        model_type = "PsiQRHTransformer"

    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modelo criado: {num_params:,} parâmetros")
    print(f"✅ Tipo: {model_type}")
    print(f"✅ Vocabulário: {dataset.vocab_size} caracteres")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    print(f"\n🎯 Iniciando treinamento...\n")

    best_val_loss = float('inf')
    training_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_perplexity = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        print(f"\n📊 Época {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_perplexity:.2f}")
        print(f"  Tempo: {epoch_time:.1f}s")

        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'time': epoch_time
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✨ Melhor modelo!")

    # Salvar modelo final
    print(f"\n💾 Salvando modelo...")
    config = {
        'training_history': training_history,
        'best_val_loss': best_val_loss,
        'best_val_perplexity': torch.exp(torch.tensor(best_val_loss)).item(),
        'model_type': model_type,
        'use_complete': args.use_complete,
        'embed_dim': args.embed_dim if args.use_complete else None,
        'n_rotations': args.n_rotations if args.use_complete else None
    }

    save_model(model, args.output_dir, config, dataset, args.n_heads, args.seq_length)

    print(f"\n{'='*60}")
    print("✅ TREINAMENTO CONCLUÍDO!")
    print(f"{'='*60}")
    print(f"Melhor Val Loss: {best_val_loss:.4f}")
    print(f"Modelo salvo em: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
