#!/usr/bin/env python3
"""
ΨQRH Contrastive Training Script
================================

Trains the ΨQRH pipeline using contrastive learning to distinguish between
correct and incorrect quantum representations of characters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import argparse
import logging

# Import ΨQRH components
from psiqrh import ΨQRHPipeline
from src.core.losses import QuantumContrastiveLoss


class ContrastiveTrainer:
    """Trainer class for contrastive learning in ΨQRH."""

    def __init__(self, pipeline: ΨQRHPipeline, device: str = 'cpu',
                 learning_rate: float = 1e-4, margin: float = 1.0):
        """
        Initialize the contrastive trainer.

        Args:
            pipeline: The ΨQRH pipeline to train
            device: Device to use for training
            learning_rate: Learning rate for optimization
            margin: Margin for contrastive loss
        """
        self.pipeline = pipeline
        self.device = device
        self.margin = margin

        # Get learnable parameters (only inverse_projector)
        learnable_params = []
        if hasattr(self.pipeline, 'inverse_projector') and self.pipeline.inverse_projector:
            learnable_params.extend(list(self.pipeline.inverse_projector.parameters()))

        if learnable_params:
            self.optimizer = optim.AdamW(learnable_params, lr=learning_rate, weight_decay=1e-5)
            print(f"🎓 Optimizer initialized with {len(learnable_params)} learnable parameters")
        else:
            self.optimizer = None
            print("⚠️  No learnable parameters found")

        # Loss function
        self.contrastive_loss = QuantumContrastiveLoss(margin=margin)

        # Training statistics
        self.training_stats = {
            'epoch': 0,
            'total_loss': 0.0,
            'best_loss': float('inf'),
            'learning_rate': learning_rate
        }

    def encode_text_to_quantum(self, text: str) -> torch.Tensor:
        """Encode text to quantum state."""
        if hasattr(self.pipeline, '_text_to_fractal_signal'):
            fractal_signal = self.pipeline._text_to_fractal_signal(text, self.pipeline.config['embed_dim'])
            psi = self.pipeline._signal_to_quaternions(fractal_signal, self.pipeline.config['embed_dim'])
            return psi
        else:
            # Fallback
            char_values = torch.tensor([ord(c) / 127.0 for c in text], dtype=torch.float32)
            embed_dim = self.pipeline.config['embed_dim']
            if len(char_values) < embed_dim:
                padding = torch.zeros(embed_dim - len(char_values))
                char_values = torch.cat([char_values, padding])
            else:
                char_values = char_values[:embed_dim]

            psi = torch.zeros(1, 1, embed_dim, 4, dtype=torch.float32, device=self.device)
            psi[0, 0, :, 0] = char_values
            psi[0, 0, :, 1] = torch.sin(char_values)
            psi[0, 0, :, 2] = torch.cos(char_values)
            psi[0, 0, :, 3] = torch.sin(char_values * 2)
            return psi

    def train_epoch(self, dataloader: DataLoader, log_every: int = 10) -> float:
        """
        Train for one epoch using contrastive learning.

        Args:
            dataloader: DataLoader with training data
            log_every: Log progress every N batches

        Returns:
            Average contrastive loss for the epoch
        """
        if not self.optimizer:
            print("⚠️  No optimizer available - skipping training")
            return 0.0

        # Set to training mode
        if hasattr(self.pipeline, 'inverse_projector') and self.pipeline.inverse_projector:
            self.pipeline.inverse_projector.train()

        epoch_loss = 0.0
        num_batches = 0

        print("🔄 Training with contrastive loss: Learning quantum semantic distinctions")

        for batch_idx, batch in enumerate(dataloader):
            batch_loss = 0.0
            valid_samples = 0

            # Get batch data
            contexts = batch.get('context', batch.get('combined', []))
            targets = batch.get('target', batch.get('combined', []))

            for context_text, target_text in zip(contexts, targets):
                try:
                    if not context_text or not target_text:
                        continue

                    # Encode to quantum states
                    psi_context = self.encode_text_to_quantum(context_text)
                    psi_positive = self.encode_text_to_quantum(target_text)

                    # Create negative example (perturbed version)
                    psi_negative = psi_positive + torch.randn_like(psi_positive) * 0.1

                    # Ensure proper shapes - flatten to [embed_dim, 4]
                    if psi_context.dim() == 4:  # [1, seq_len, embed_dim, 4]
                        psi_context = psi_context.view(-1, psi_context.shape[-2], 4)  # [seq_len*embed_dim, 4]
                        psi_context = psi_context.mean(dim=0, keepdim=True)  # [1, embed_dim, 4]
                    if psi_positive.dim() == 4:
                        psi_positive = psi_positive.view(-1, psi_positive.shape[-2], 4)
                        psi_positive = psi_positive.mean(dim=0, keepdim=True)
                    if psi_negative.dim() == 4:
                        psi_negative = psi_negative.view(-1, psi_negative.shape[-2], 4)
                        psi_negative = psi_negative.mean(dim=0, keepdim=True)

                    # Move to device
                    psi_context = psi_context.to(self.device)
                    psi_positive = psi_positive.to(self.device)
                    psi_negative = psi_negative.to(self.device)

                    # Compute contrastive loss
                    loss = self.contrastive_loss(psi_context, psi_positive, psi_negative)
                    batch_loss += loss.item()
                    valid_samples += 1

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                except Exception as e:
                    print(f"❌ Training error: {e}")
                    continue

            if valid_samples > 0:
                epoch_loss += batch_loss / valid_samples
                num_batches += 1

            # Log progress
            if (batch_idx + 1) % log_every == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"   📊 Batch {batch_idx + 1}/{len(dataloader)} - Contrastive Loss: {avg_loss:.6f}")

        # Calculate final epoch loss
        final_epoch_loss = epoch_loss / max(num_batches, 1)

        # Update training statistics
        self.training_stats['epoch'] += 1
        self.training_stats['total_loss'] = final_epoch_loss

        return final_epoch_loss

    def save_checkpoint(self, checkpoint_path: Path):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.training_stats['epoch'],
            'loss': self.training_stats['total_loss'],
            'training_stats': self.training_stats,
            'model_state_dict': {},
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        }

        # Save inverse projector state
        if hasattr(self.pipeline, 'inverse_projector'):
            checkpoint['model_state_dict']['inverse_projector'] = self.pipeline.inverse_projector.state_dict()

        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")


def create_training_data():
    """Create synthetic training data for contrastive learning."""
    contexts = [
        "quantum", "physics", "mechanics", "wave", "function",
        "uncertainty", "principle", "schrodinger", "equation", "superposition",
        "entanglement", "theory", "computing", "algorithm", "state"
    ]

    targets = [
        "mechanics", "quantum", "physics", "function", "wave",
        "principle", "uncertainty", "equation", "schrodinger", "state",
        "theory", "entanglement", "algorithm", "computing", "superposition"
    ]

    training_pairs = []
    for i, (context, target) in enumerate(zip(contexts, targets)):
        training_pairs.append({
            'id': i,
            'context': context,
            'target': target,
            'combined': f"{context} {target}"
        })

    return training_pairs


class ContrastiveDataset(Dataset):
    """Dataset for contrastive training."""

    def __init__(self, data_path: str = None):
        if data_path and Path(data_path).exists():
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = create_training_data()
            # Save synthetic data
            os.makedirs('data', exist_ok=True)
            with open('data/contrastive_training_data.json', 'w') as f:
                json.dump(self.data, f, indent=2)
            print("💾 Saved synthetic contrastive training data")

        print(f"📚 Loaded {len(self.data)} contrastive training pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="ΨQRH Contrastive Training")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Contrastive loss margin')
    parser.add_argument('--device', type=str, default='cpu', help='Device')

    args = parser.parse_args()

    print("🚀 Starting ΨQRH Contrastive Training")
    print("=" * 50)

    # Create pipeline
    print("🔧 Initializing ΨQRH Pipeline...")
    pipeline = ΨQRHPipeline(
        task="text-generation",
        device=args.device,
        enable_auto_calibration=False,
        audit_mode=False
    )

    # Create trainer
    trainer = ContrastiveTrainer(
        pipeline,
        device=args.device,
        learning_rate=args.learning_rate,
        margin=args.margin
    )

    # Load dataset
    dataset = ContrastiveDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    checkpoint_dir = Path('models/checkpoints')
    best_loss = float('inf')

    print(f"🎯 Starting contrastive training for {args.epochs} epochs...")
    print(f"   📊 Dataset size: {len(dataset)}")
    print(f"   📊 Batch size: {args.batch_size}")
    print(f"   📊 Learning rate: {args.learning_rate}")
    print(f"   📊 Margin: {args.margin}")
    print()

    for epoch in range(1, args.epochs + 1):
        print(f"🎯 Epoch {epoch}/{args.epochs} - Contrastive Training")
        print("-" * 60)

        # Train epoch
        epoch_loss = trainer.train_epoch(dataloader, log_every=5)

        print(".6f")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"contrastive_epoch_{epoch}.pt"
        trainer.save_checkpoint(checkpoint_path)

        # Update best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = checkpoint_dir / "contrastive_best.pt"
            trainer.save_checkpoint(best_path)
            print(".6f")

        print()

    print("✅ Contrastive training completed!")
    print(".6f")
    print(f"💾 Checkpoints saved in: {checkpoint_dir}")


if __name__ == "__main__":
    main()