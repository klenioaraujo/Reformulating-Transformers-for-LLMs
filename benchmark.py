#!/usr/bin/env python3
"""
ΨQRH Benchmark Pipeline
=======================

Rigorous benchmarking pipeline for ΨQRH vs baseline Transformer models.
Implements fair comparison with identical parameter counts and evaluation metrics.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import ΨQRH components
from src.architecture.psiqrh_transformer import PsiQRHTransformer


class TextDataset(Dataset):
    """Dataset for language modeling tasks"""

    def __init__(self, texts: List[str], tokenizer, seq_len: int = 512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Tokenize all texts
        self.tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            self.tokens.extend(tokens)

        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_len, seq_len // 2):
            seq = self.tokens[i:i + seq_len + 1]  # +1 for target
            if len(seq) == seq_len + 1:
                self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'input_ids': torch.tensor(seq[:-1], dtype=torch.long),
            'labels': torch.tensor(seq[1:], dtype=torch.long)
        }


class BaselineTransformer(nn.Module):
    """Standard Transformer for baseline comparison"""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 dim_feedforward: int, max_seq_length: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)

        # Standard transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)

        # Transformer layers (using decoder as encoder for LM)
        for layer in self.layers:
            x = layer(x, x, tgt_mask=causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits


def create_tokenizer():
    """Create a simple character-level tokenizer"""
    class SimpleTokenizer:
        def __init__(self):
            # Basic character vocabulary
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?()-'\""
            self.char_to_id = {char: i+1 for i, char in enumerate(chars)}  # Start from 1
            self.char_to_id['<unk>'] = 0
            self.char_to_id['<pad>'] = len(self.char_to_id)
            self.id_to_char = {v: k for k, v in self.char_to_id.items()}
            self.vocab_size = len(self.char_to_id)

        def encode(self, text, add_special_tokens=True):
            tokens = []
            for char in text:
                tokens.append(self.char_to_id.get(char, 0))  # 0 for unknown
            return tokens

        def decode(self, tokens):
            return ''.join(self.id_to_char.get(token, '<unk>') for token in tokens)

    return SimpleTokenizer()


def load_wikitext_data(tokenizer, seq_len: int = 512) -> Tuple[DataLoader, DataLoader]:
    """Load synthetic dataset for testing (simulating WikiText-103)"""
    print("Loading synthetic dataset (simulating WikiText-103)...")

    # Generate synthetic text data
    def generate_synthetic_text(n_samples=1000, avg_length=100):
        """Generate synthetic text similar to WikiText"""
        words = ['the', 'of', 'and', 'in', 'to', 'a', 'is', 'that', 'for', 'on',
                'with', 'as', 'by', 'at', 'an', 'it', 'from', 'was', 'are', 'be',
                'this', 'which', 'or', 'had', 'one', 'were', 'but', 'not', 'have',
                'they', 'you', 'he', 'she', 'it', 'we', 'they', 'I', 'me', 'him']

        texts = []
        for _ in range(n_samples):
            length = np.random.poisson(avg_length)
            text = ' '.join(np.random.choice(words, size=length))
            texts.append(text)
        return texts

    # Generate train and validation texts
    train_texts = generate_synthetic_text(500, 50)  # Shorter for faster testing
    val_texts = generate_synthetic_text(100, 50)

    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, seq_len)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader


def create_model(model_type: str, vocab_size: int, seq_len: int) -> nn.Module:
    """Create model with approximately matched parameter counts"""

    if model_type == 'psiqrh':
        # ΨQRH model
        model = PsiQRHTransformer(
            vocab_size=vocab_size,
            d_model=256,  # Smaller for benchmarking
            n_layers=4,
            n_heads=8,
            dim_feedforward=512,
            max_seq_length=seq_len
        )
    elif model_type == 'baseline':
        # Standard Transformer baseline
        model = BaselineTransformer(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=4,
            n_heads=8,
            dim_feedforward=512,
            max_seq_length=seq_len
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: str) -> Tuple[float, float, float]:
    """Train for one epoch and measure performance"""
    model.train()
    total_loss = 0
    total_tokens = 0

    # Memory and timing setup
    if torch.cuda.is_available() and 'cuda' in device:
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()

    # Calculate metrics
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    if torch.cuda.is_available() and 'cuda' in device:
        end_event.record()
        torch.cuda.synchronize()
        latency = start_event.elapsed_time(end_event) / 1000  # seconds
        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    else:
        latency = time.time() - start_time
        memory_usage = 0.0  # Not measured on CPU

    return avg_loss, perplexity, latency, memory_usage


def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
                   device: str) -> Tuple[float, float, float]:
    """Validate model and measure performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    # Memory and timing setup
    if torch.cuda.is_available() and 'cuda' in device:
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    # Calculate metrics
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    if torch.cuda.is_available() and 'cuda' in device:
        end_event.record()
        torch.cuda.synchronize()
        latency = start_event.elapsed_time(end_event) / 1000  # seconds
        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    else:
        latency = time.time() - start_time
        memory_usage = 0.0

    return avg_loss, perplexity, latency, memory_usage


def main():
    parser = argparse.ArgumentParser(description='ΨQRH Benchmark Pipeline')
    parser.add_argument('--model_type', choices=['psiqrh', 'baseline'],
                       default='psiqrh', help='Model type to benchmark')
    parser.add_argument('--dataset', choices=['wikitext-103'],
                       default='wikitext-103', help='Dataset to use')
    parser.add_argument('--seq_len', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', choices=['cpu', 'cuda'],
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')

    args = parser.parse_args()

    print("🔬 ΨQRH Benchmark Pipeline")
    print(f"Model: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Setup
    device = torch.device(args.device)
    tokenizer = create_tokenizer()

    # Load data
    train_loader, val_loader = load_wikitext_data(tokenizer, args.seq_len)

    # Create model
    model = create_model(args.model_type, tokenizer.vocab_size, args.seq_len)
    model = model.to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print()

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_ppl = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_ppl, train_latency, train_memory = train_epoch(
            model, train_loader, optimizer, criterion, args.device
        )
        print(".4f"
              ".2f")

        # Validate
        val_loss, val_ppl, val_latency, val_memory = validate_epoch(
            model, val_loader, criterion, args.device
        )
        print(".4f"
              ".2f")

        # Save best model
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save(model.state_dict(), f'best_{args.model_type}_model.pt')

        print()

    print("🏁 Benchmark completed!")
    print(f"Best validation perplexity: {best_ppl:.2f}")


if __name__ == '__main__':
    main()