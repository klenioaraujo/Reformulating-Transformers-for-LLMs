#!/usr/bin/env python3
"""
Train PureSpectralTransformer on WikiText-103 Benchmark
========================================================

Script de treinamento integrado que treina um PureSpectralTransformer
no benchmark WikiText-103 e salva o modelo em formato compatível com QRHFactory.

Usage:
    python3 train_model.py --output_dir ./models/psiqrh_wikitext_v2 --epochs 3

Requisitos:
    pip install datasets transformers torch
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_spectral import PureSpectralTransformer


class WikiTextDataset(torch.utils.data.Dataset):
    """Dataset wrapper para WikiText-103"""

    def __init__(self, split: str, tokenizer, max_length: int = 512):
        """
        Args:
            split: 'train', 'validation', ou 'test'
            tokenizer: Tokenizer do Hugging Face
            max_length: Comprimento máximo de sequência
        """
        print(f"📚 Carregando WikiText-103 split '{split}'...")
        self.dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Filtrar textos vazios
        self.dataset = self.dataset.filter(lambda x: len(x['text'].strip()) > 0)
        print(f"✅ {len(self.dataset)} exemplos carregados")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']

        # Tokenizar
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Para language modeling: labels = input_ids shifted
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Treina por uma época"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Época {epoch}")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        logits = model(input_ids)

        # Calcular loss (CrossEntropy)
        # Flatten para compatibilidade com CrossEntropyLoss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=model.tokenizer.pad_token_id if hasattr(model, 'tokenizer') else -100
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Tracking
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """Avalia o modelo no dataset de validação"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validação"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids)

            # Calcular loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=model.tokenizer.pad_token_id if hasattr(model, 'tokenizer') else -100
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


def save_model(model, tokenizer, output_dir, config):
    """
    Salva modelo em formato compatível com QRHFactory

    Args:
        model: Modelo treinado
        tokenizer: Tokenizer usado
        output_dir: Diretório de saída
        config: Configuração do treinamento
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Salvar model state_dict
    model_path = output_path / 'pytorch_model.bin'
    torch.save(model.state_dict(), model_path)
    print(f"✅ Modelo salvo em {model_path}")

    # Salvar tokenizer
    tokenizer.save_pretrained(output_path)
    print(f"✅ Tokenizer salvo em {output_path}")

    # Salvar configuração
    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Configuração salva em {config_path}")

    # Criar arquivo de modelo info para QRHFactory
    model_info = {
        'model_type': 'PureSpectralTransformer',
        'framework': 'ΨQRH',
        'version': '1.0.0',
        'vocab_size': model.vocab_size,
        'spectral_dim': model.spectral_dim,
        'n_layers': model.n_layers,
        'trained_on': 'WikiText-103',
        'checkpoint_path': str(model_path),
        'tokenizer_path': str(output_path)
    }

    info_path = output_path / 'model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"✅ Model info salvo em {info_path}")


def main():
    parser = argparse.ArgumentParser(description='Treinar PureSpectralTransformer em WikiText-103')

    # Argumentos de treinamento
    parser.add_argument('--output_dir', type=str, default='./models/psiqrh_wikitext_v2',
                        help='Diretório para salvar modelo treinado')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Número de épocas de treinamento')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Tamanho do batch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Comprimento máximo de sequência')

    # Argumentos do modelo
    parser.add_argument('--spectral_dim', type=int, default=256,
                        help='Dimensão espectral')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Número de camadas')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Número de attention heads')

    # Argumentos de infraestrutura
    parser.add_argument('--device', type=str, default='auto',
                        help='Dispositivo (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Número de workers para DataLoader')

    args = parser.parse_args()

    # Configurar device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print("🚀 TREINAMENTO PURESPECTRAL TRANSFORMER")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*60}\n")

    # Carregar tokenizer
    print("📝 Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"✅ Tokenizer carregado: vocab_size={vocab_size}")

    # Criar datasets
    train_dataset = WikiTextDataset('train', tokenizer, args.max_seq_length)
    val_dataset = WikiTextDataset('validation', tokenizer, args.max_seq_length)

    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Criar modelo
    print(f"\n🏗️  Criando PureSpectralTransformer...")
    model = PureSpectralTransformer(
        vocab_size=vocab_size,
        spectral_dim=args.spectral_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_length=args.max_seq_length
    )
    model.to(device)

    # Armazenar tokenizer no modelo para salvar depois
    model.tokenizer = tokenizer

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modelo criado com {num_params:,} parâmetros")

    # Criar optimizer e scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Loop de treinamento
    print(f"\n🎯 Iniciando treinamento por {args.epochs} épocas...\n")

    best_val_loss = float('inf')
    training_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Treinar
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)

        # Validar
        val_loss, val_perplexity = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # Log
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

        # Salvar melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✨ Novo melhor modelo! (val_loss={val_loss:.4f})")

    # Salvar modelo final
    print(f"\n💾 Salvando modelo final...")
    config = {
        'vocab_size': vocab_size,
        'spectral_dim': args.spectral_dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'max_seq_length': args.max_seq_length,
        'training_history': training_history,
        'best_val_loss': best_val_loss,
        'best_val_perplexity': torch.exp(torch.tensor(best_val_loss)).item()
    }

    save_model(model, tokenizer, args.output_dir, config)

    print(f"\n{'='*60}")
    print("✅ TREINAMENTO CONCLUÍDO!")
    print(f"{'='*60}")
    print(f"Melhor Val Loss: {best_val_loss:.4f}")
    print(f"Melhor Val Perplexity: {config['best_val_perplexity']:.2f}")
    print(f"Modelo salvo em: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
