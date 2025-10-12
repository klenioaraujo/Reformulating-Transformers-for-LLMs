#!/usr/bin/env python3
"""
ΨQRH Inverse Cognitive Projector Supervised Training Script
===========================================================

Trains the InverseCognitiveProjector in a supervised manner to learn the inverse
mapping from quantum states back to their original representations.

This script focuses exclusively on teaching the projector to be the perfect inverse
of the QuantumEmbedding layer.
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


class CharacterVocabularyDataset(Dataset):
    """Dataset for training the inverse projector over character vocabulary."""

    def __init__(self, vocab_size: int = 256):
        """
        Initialize the character vocabulary dataset.

        Args:
            vocab_size: Size of the character vocabulary (default: 256 for ASCII)
        """
        self.vocab_size = vocab_size
        self.characters = [chr(i) for i in range(vocab_size)]

    def __len__(self) -> int:
        return self.vocab_size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'char': self.characters[idx],
            'char_id': idx
        }


class InverseProjectorTrainer:
    """Trainer class for the Inverse Cognitive Projector."""

    def __init__(self, pipeline: ΨQRHPipeline, device: str = 'cpu',
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        """
        Initialize the inverse projector trainer.

        Args:
            pipeline: The ΨQRH pipeline containing the quantum embedding and inverse projector
            device: Device to use for training
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
        """
        self.pipeline = pipeline
        self.device = device

        # Freeze all parameters except inverse_projector
        self._freeze_non_inverse_parameters()

        # Get learnable parameters (only inverse_projector)
        learnable_params = self._get_learnable_parameters()

        if learnable_params:
            self.optimizer = optim.AdamW(learnable_params, lr=learning_rate, weight_decay=weight_decay)
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=100, T_mult=2
            )
            print(f"🎓 Optimizer initialized with {len(learnable_params)} learnable parameters (inverse_projector only)")
        else:
            raise ValueError("No learnable parameters found in inverse_projector")

        # Loss function
        self.mse_loss = nn.MSELoss()

        # Training statistics
        self.training_stats = {
            'epoch': 0,
            'total_loss': 0.0,
            'num_batches': 0,
            'best_loss': float('inf'),
            'learning_rate': learning_rate
        }

    def _freeze_non_inverse_parameters(self):
        """Freeze all parameters except those in inverse_projector."""
        print("🔒 Freezing all parameters except inverse_projector...")

        frozen_count = 0
        trainable_count = 0

        # List of components that have parameters
        components_to_check = [
            ('quantum_embedding', self.pipeline.quantum_embedding),
            ('context_funnel', getattr(self.pipeline, 'context_funnel', None)),
            ('inverse_projector', getattr(self.pipeline, 'inverse_projector', None)),
            ('dcf_analyzer.kuramoto_layer', getattr(getattr(self.pipeline, 'dcf_analyzer', None), 'kuramoto_layer', None)),
        ]

        for comp_name, component in components_to_check:
            if component is not None:
                for param_name, param in component.named_parameters():
                    full_name = f"{comp_name}.{param_name}"
                    if 'inverse_projector' in comp_name:
                        param.requires_grad = True
                        trainable_count += 1
                    else:
                        param.requires_grad = False
                        frozen_count += 1

        print(f"   ✅ Frozen: {frozen_count} parameters")
        print(f"   🎯 Trainable: {trainable_count} parameters (inverse_projector only)")

    def _get_learnable_parameters(self) -> List[torch.nn.Parameter]:
        """Get learnable parameters (only inverse_projector)."""
        learnable_params = []

        # Get parameters from inverse_projector only
        if hasattr(self.pipeline, 'inverse_projector') and self.pipeline.inverse_projector is not None:
            for param_name, param in self.pipeline.inverse_projector.named_parameters():
                if param.requires_grad:
                    learnable_params.append(param)

        return learnable_params

    def train_epoch(self, dataloader: DataLoader, log_every: int = 10) -> float:
        """
        Train one epoch using reconstruction loss.

        Args:
            dataloader: DataLoader containing character data
            log_every: Log progress every N batches

        Returns:
            Average loss for the epoch
        """
        # Set training mode for components that support it
        if hasattr(self.pipeline, 'inverse_projector') and self.pipeline.inverse_projector is not None:
            self.pipeline.inverse_projector.train()

        epoch_loss = 0.0
        num_batches = 0

        print(f"🎯 Training inverse projector with reconstruction loss...")

        for batch_idx, batch in enumerate(dataloader):
            try:
                batch_loss = 0.0
                valid_samples = 0

                # Process all characters in the batch together
                char_ids = torch.tensor(batch['char_id'], dtype=torch.long, device=self.device).unsqueeze(1)  # [batch_size, 1]

                # Generate ideal quantum states using the quantum embedding
                with torch.no_grad():  # No gradients for embedding during training
                    psi_ideal_quat = self.pipeline.quantum_embedding(char_ids)  # [batch_size, 1, embed_dim//4, 4]
                    # Flatten to [batch_size, embed_dim] for inverse projector input
                    psi_ideal = psi_ideal_quat.flatten(start_dim=1)  # [batch_size, embed_dim]

                # Pass through inverse projector to get reconstruction
                psi_reconstructed, confidence = self.pipeline.inverse_projector(
                    psi_ideal,  # [batch_size, embed_dim]
                    quantum_vocab=self.pipeline.quantum_vocab_representations,
                    return_confidence=True
                )

                # Compute reconstruction loss for the batch
                # psi_reconstructed is [batch_size, embed_dim, 4], psi_ideal is [batch_size, embed_dim]
                # Flatten psi_reconstructed to match dimensions
                if psi_reconstructed.dim() == 3 and psi_reconstructed.shape[-1] == 4:
                    psi_reconstructed_flat = psi_reconstructed.flatten(start_dim=1)  # [batch_size, embed_dim*4]
                    # For fair comparison, we need to compare flattened versions
                    psi_ideal_expanded = psi_ideal_quat.flatten(start_dim=1)  # [batch_size, embed_dim]
                    loss = self.mse_loss(psi_reconstructed_flat, psi_ideal_expanded)
                else:
                    loss = self.mse_loss(psi_reconstructed, psi_ideal)

                batch_loss = loss.item()
                valid_samples = len(batch['char'])

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if valid_samples > 0:
                    epoch_loss += batch_loss
                    num_batches += 1

                # Log progress
                if (batch_idx + 1) % log_every == 0:
                    avg_loss = epoch_loss / max(num_batches, 1)
                    print(f"   📊 Batch {batch_idx + 1}/{len(dataloader)} - Reconstruction Loss: {avg_loss:.8f}")

            except Exception as e:
                print(f"❌ Training error for batch {batch_idx}: {e}")
                continue

        # Update learning rate
        if self.scheduler:
            self.scheduler.step()

        # Calculate final epoch loss
        final_epoch_loss = epoch_loss / max(num_batches, 1)

        # Update training statistics
        self.training_stats['epoch'] += 1
        self.training_stats['total_loss'] = final_epoch_loss
        self.training_stats['num_batches'] = num_batches

        if self.scheduler:
            self.training_stats['learning_rate'] = self.scheduler.get_last_lr()[0]

        return final_epoch_loss

    def save_checkpoint(self, checkpoint_path: Path, epoch: int, loss: float):
        """
        Save a training checkpoint.

        Args:
            checkpoint_path: Path to save the checkpoint
            epoch: Current epoch number
            loss: Current loss value
        """
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'training_stats': self.training_stats,
            'inverse_projector_state_dict': self.pipeline.inverse_projector.state_dict(),
        }

        # Save checkpoint
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

            # Update best loss
            if loss < self.training_stats['best_loss']:
                self.training_stats['best_loss'] = loss
                best_path = checkpoint_path.parent / "inverse_projector_best.pt"
                torch.save(checkpoint, best_path)
                print(f"🏆 Best model saved: {best_path}")

        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging for training."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"inverse_projector_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Inverse projector training log started: {log_file}")

    return logger


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="ΨQRH Inverse Cognitive Projector Training Script")
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='models/inverse_projector',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/inverse_projector_training',
                        help='Directory to save training logs')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training (cpu/cuda)')
    parser.add_argument('--model-path', type=str, default='models/checkpoints/best_model.pt',
                        help='Path to the trained model checkpoint')

    args = parser.parse_args()

    print("🔧 Starting ΨQRH Inverse Cognitive Projector Training")
    print("=" * 60)

    # Setup logging
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir)

    # Create pipeline and load trained weights
    print("🔧 Initializing ΨQRH Pipeline...")
    pipeline = ΨQRHPipeline(
        task="text-generation",
        device=args.device,
        enable_auto_calibration=False,
        audit_mode=False
    )

    # Load trained model weights if available
    if os.path.exists(args.model_path):
        print(f"📁 Loading trained model weights from: {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location=args.device)
            if 'model_state_dict' in checkpoint:
                # Load quantum embedding weights
                if 'quantum_embedding' in checkpoint['model_state_dict']:
                    pipeline.quantum_embedding.load_state_dict(
                        checkpoint['model_state_dict']['quantum_embedding']
                    )
                    print("✅ Loaded trained quantum embedding weights")
                else:
                    print("⚠️  No quantum embedding weights found in checkpoint")
            else:
                print("⚠️  No model_state_dict found in checkpoint")
        except Exception as e:
            print(f"⚠️  Could not load model weights: {e}")
    else:
        print(f"⚠️  Model checkpoint not found: {args.model_path}")
        print("   Training with randomly initialized weights...")

    # Create trainer
    trainer = InverseProjectorTrainer(pipeline, device=args.device, learning_rate=args.learning_rate)

    # Create dataset and dataloader
    dataset = CharacterVocabularyDataset(vocab_size=256)  # ASCII characters
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    checkpoint_dir = Path(args.checkpoint_dir)

    print(f"🎯 Starting inverse projector training for {args.epochs} epochs...")
    print(f"   📊 Dataset size: {len(dataset)} characters")
    print(f"   📦 Batch size: {args.batch_size}")
    print(f"   📊 Learning rate: {args.learning_rate}")
    print(f"   💾 Checkpoints: {checkpoint_dir}")
    print()

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"🎯 Epoch {epoch}/{args.epochs} - Inverse Projector Reconstruction Training")
        print("-" * 80)

        # Train epoch
        epoch_loss = trainer.train_epoch(dataloader, log_every=10)

        # Log epoch results
        logger.info(f"Epoch {epoch}: Reconstruction Loss = {epoch_loss:.8f}")

        print(".8f")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        trainer.save_checkpoint(checkpoint_path, epoch, epoch_loss)

        # Update best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(".8f")

        print()

    print("✅ Inverse projector training completed!")
    print(".8f")
    print(f"💾 Final checkpoint saved in: {checkpoint_dir}")
    print(f"📊 Training logs saved in: {log_dir}")


if __name__ == "__main__":
    main()