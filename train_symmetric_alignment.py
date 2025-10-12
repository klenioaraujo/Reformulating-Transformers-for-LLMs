#!/usr/bin/env python3
"""
ΨQRH Symmetric Alignment Training Script
========================================

Trains the InverseCognitiveProjector to be the perfect inverse of the QuantumEmbedding layer,
restoring symmetry to the quantum pipeline and enabling coherent text generation.

This script implements the final phase of the ΨQRH training pipeline: symmetric alignment
between encoder and decoder in the quantum domain.
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

# Anomaly detection disabled after fixing gradient issues
# torch.autograd.set_detect_anomaly(True)

# Import ΨQRH components
from psiqrh import ΨQRHPipeline


class SymmetricAlignmentDataset(Dataset):
    """Dataset for symmetric alignment training between encoder and decoder."""

    def __init__(self, vocab_size: int = 256):
        """
        Initialize the symmetric alignment dataset.

        Args:
            vocab_size: Size of the character vocabulary (default: 256 for ASCII)
        """
        self.vocab_size = vocab_size
        self.characters = [chr(i) for i in range(vocab_size)]

    def __len__(self) -> int:
        return self.vocab_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(idx, dtype=torch.long)


class SymmetricAlignmentTrainer:
    """Trainer class for symmetric alignment between QuantumEmbedding and InverseCognitiveProjector."""

    def __init__(self, pipeline: ΨQRHPipeline, device: str = 'cpu',
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        """
        Initialize the symmetric alignment trainer.

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

        # Loss function for symmetric alignment
        self.mse_loss = nn.MSELoss()

        # Training statistics
        self.training_stats = {
            'epoch': 0,
            'total_loss': 0.0,
            'num_batches': 0,
            'best_loss': float('inf'),
            'learning_rate': learning_rate,
            'convergence_epoch': None,
            'final_reconstruction_error': None
        }

    def _freeze_non_inverse_parameters(self):
        """Freeze all parameters except those in inverse_projector."""
        print("🔒 Freezing all parameters except inverse_projector for symmetric alignment...")

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

        print(f"   ✅ Frozen: {frozen_count} parameters (encoder remains fixed)")
        print(f"   🎯 Trainable: {trainable_count} parameters (decoder learns inverse mapping)")
        print(f"   🔄 Symmetric Alignment: Encoder → Decoder inverse function learning")

    def _get_learnable_parameters(self) -> List[torch.nn.Parameter]:
        """Get learnable parameters (only inverse_projector)."""
        learnable_params = []

        # Get parameters from inverse_projector only
        if hasattr(self.pipeline, 'inverse_projector') and self.pipeline.inverse_projector is not None:
            for param_name, param in self.pipeline.inverse_projector.named_parameters():
                if param.requires_grad:
                    learnable_params.append(param)

        return learnable_params

    def train_epoch(self, dataloader: DataLoader, epoch: int, num_epochs: int, log_every: int = 10) -> float:
        """
        Train one epoch using symmetric alignment reconstruction loss with proper mini-batch processing.

        Args:
            dataloader: DataLoader containing character data
            log_every: Log progress every N batches

        Returns:
            Average reconstruction loss for the epoch
        """
        # Set training mode for inverse_projector only
        if hasattr(self.pipeline, 'inverse_projector') and self.pipeline.inverse_projector is not None:
            self.pipeline.inverse_projector.train()

        epoch_loss = 0.0
        num_batches = 0

        print(f"🔄 Training symmetric alignment: QuantumEmbedding → InverseCognitiveProjector")

        for batch_idx, batch_ids_tensor in enumerate(dataloader):
            # 1. Limpe os gradientes da iteração anterior
            self.optimizer.zero_grad()

            batch_ids = batch_ids_tensor

            # --- Forward Pass ---
            # 2. Gere os estados ideais para o batch (com gradientes congelados)
            with torch.no_grad():
                Ψ_ideal_batch = self.pipeline.quantum_embedding(batch_ids.unsqueeze(1))

            # Flatten for inverse projector input
            Ψ_ideal_flat = Ψ_ideal_batch.flatten(start_dim=1).detach()

            # 3. Execute o decodificador (com gradientes ativos)
            Ψ_reconstruido_batch, _ = self.pipeline.inverse_projector(Ψ_ideal_flat.clone(), quantum_vocab=self.pipeline.quantum_vocab_representations.clone().detach(), return_confidence=True)

            # --- Cálculo da Perda e Retropropagação ---
            # 4. Calcule a perda para este batch específico
            if Ψ_reconstruido_batch.dim() == 3 and Ψ_reconstruido_batch.shape[-1] == 4:
                Ψ_reconstruido_flat = Ψ_reconstruido_batch.flatten(start_dim=1)
                reconstruction_loss = self.mse_loss(Ψ_reconstruido_flat, Ψ_ideal_flat)
            else:
                reconstruction_loss = self.mse_loss(Ψ_reconstruido_batch, Ψ_ideal_flat)

            # 5. Retropropague a perda (cria e libera o grafo APENAS para este batch)
            reconstruction_loss.backward()

            # 6. Atualize os pesos do projetor
            self.optimizer.step()

            # Update epoch statistics
            epoch_loss += reconstruction_loss.item()
            num_batches += 1

            # Log do progresso do batch (opcional, mas útil)
            if batch_idx % log_every == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Batch Loss: {reconstruction_loss.item():.8f}")

        # Update learning rate
        if self.scheduler:
            self.scheduler.step()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/{num_epochs} concluída. Perda de Reconstrução Média: {avg_epoch_loss:.8f}")

        # Update training statistics
        self.training_stats['epoch'] += 1
        self.training_stats['total_loss'] = avg_epoch_loss
        self.training_stats['num_batches'] = num_batches

        if self.scheduler:
            self.training_stats['learning_rate'] = self.scheduler.get_last_lr()[0]

        # Check for convergence
        if avg_epoch_loss < 1e-6 and self.training_stats['convergence_epoch'] is None:
            self.training_stats['convergence_epoch'] = epoch
            print(f"🎯 CONVERGENCE ACHIEVED at epoch {epoch}: Loss < 1e-6")

        return avg_epoch_loss

    def validate_alignment(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the quality of symmetric alignment.

        Args:
            dataloader: Validation dataloader

        Returns:
            Dictionary with validation metrics
        """
        self.pipeline.inverse_projector.eval()

        total_loss = 0.0
        total_confidence = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                char_ids = batch.unsqueeze(1)

                # Generate target
                psi_target_batch = self.pipeline.quantum_embedding(char_ids)
                psi_target = psi_target_batch.flatten(start_dim=1).detach()

                # Reconstruct
                psi_reconstructed, confidence = self.pipeline.inverse_projector(
                    psi_target.clone(),
                    quantum_vocab=self.pipeline.quantum_vocab_representations.clone().detach(),
                    return_confidence=True
                )

                # Compute loss
                if psi_reconstructed.dim() == 3 and psi_reconstructed.shape[-1] == 4:
                    psi_reconstructed_flat = psi_reconstructed.flatten(start_dim=1)
                    loss = self.mse_loss(psi_reconstructed_flat, psi_target)
                else:
                    loss = self.mse_loss(psi_reconstructed, psi_target)

                total_loss += loss.item()
                total_confidence += confidence.mean().item()
                num_samples += len(batch)

        avg_loss = total_loss / max(num_samples, 1)
        avg_confidence = total_confidence / max(num_samples, 1)

        return {
            'reconstruction_loss': avg_loss,
            'average_confidence': avg_confidence,
            'convergence_achieved': avg_loss < 1e-6,
            'high_confidence': avg_confidence > 0.9
        }

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
            print(f"💾 Symmetric alignment checkpoint saved: {checkpoint_path}")

            # Update best loss
            if loss < self.training_stats['best_loss']:
                self.training_stats['best_loss'] = loss
                best_path = checkpoint_path.parent / "psiqrh_fully_aligned.pt"
                torch.save(checkpoint, best_path)
                print(f"🏆 Best symmetric alignment saved: {best_path}")

        except Exception as e:
            print(f"❌ Error saving symmetric alignment checkpoint: {e}")


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging for symmetric alignment training."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"symmetric_alignment_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Symmetric alignment training log started: {log_file}")

    return logger


def main():
    """Main symmetric alignment training function."""
    parser = argparse.ArgumentParser(description="ΨQRH Symmetric Alignment Training Script")
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='models/symmetric_alignment',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/symmetric_alignment_training',
                        help='Directory to save training logs')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training (cpu/cuda)')
    parser.add_argument('--model-path', type=str, default='models/checkpoints/best_model.pt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--validate-every', type=int, default=10,
                        help='Validate alignment every N epochs')

    args = parser.parse_args()

    print("🔄 Starting ΨQRH Symmetric Alignment Training")
    print("=" * 60)
    print("🎯 Goal: Train InverseCognitiveProjector to be the perfect inverse of QuantumEmbedding")
    print("🔄 Approach: Minimize MSE(Ψ_reconstructed, Ψ_target) where Ψ_target = QuantumEmbedding(char)")
    print("✅ Success Criteria: Reconstruction Loss < 1e-6, Confidence > 0.9")
    print()

    # Setup logging
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir)

    # Create pipeline and load trained weights
    print("🔧 Initializing ΨQRH Pipeline with trained QuantumEmbedding...")
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
    trainer = SymmetricAlignmentTrainer(pipeline, device=args.device, learning_rate=args.learning_rate)

    # Create dataset and dataloader
    dataset = SymmetricAlignmentDataset(vocab_size=256)  # ASCII characters
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    checkpoint_dir = Path(args.checkpoint_dir)

    print(f"🎯 Starting symmetric alignment training for {args.epochs} epochs...")
    print(f"   📊 Dataset size: {len(dataset)} characters")
    print(f"   📦 Batch size: {args.batch_size}")
    print(f"   📊 Learning rate: {args.learning_rate}")
    print(f"   💾 Checkpoints: {checkpoint_dir}")
    print(f"   🔍 Validation every: {args.validate_every} epochs")
    print()

    best_loss = float('inf')
    convergence_achieved = False

    for epoch in range(1, args.epochs + 1):
        print(f"🔄 Epoch {epoch}/{args.epochs} - Symmetric Alignment Training")
        print("-" * 80)

        # Train epoch
        epoch_loss = trainer.train_epoch(dataloader, epoch, args.epochs, log_every=10)

        # Log epoch results
        logger.info(f"Epoch {epoch}: Symmetric Reconstruction Loss = {epoch_loss:.8f}")

        print(".8f")

        # Validation
        if epoch % args.validate_every == 0:
            print(f"🔍 Validating symmetric alignment at epoch {epoch}...")
            validation_metrics = trainer.validate_alignment(dataloader)
            print(f"   📊 Validation - Loss: {validation_metrics['reconstruction_loss']:.8f}, Confidence: {validation_metrics['average_confidence']:.4f}")

            if validation_metrics['convergence_achieved'] and not convergence_achieved:
                convergence_achieved = True
                print(f"🎯 SYMMETRIC ALIGNMENT CONVERGENCE ACHIEVED at epoch {epoch}!")
                print("   ✅ Reconstruction Loss < 1e-6")
                if validation_metrics['high_confidence']:
                    print("   ✅ High Confidence > 0.9")
                    print("   🎉 FULL SUCCESS: Symmetric alignment complete!")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        trainer.save_checkpoint(checkpoint_path, epoch, epoch_loss)

        # Update best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(".8f")

        print()

    print("✅ Symmetric alignment training completed!")
    print(".8f")

    if convergence_achieved:
        print("🎯 SUCCESS: Symmetric alignment achieved!")
        print("   🔄 Encoder ↔ Decoder symmetry restored")
        print("   📝 Coherent text generation now possible")
    else:
        print("⚠️  WARNING: Symmetric alignment did not fully converge")
        print("   📊 Consider increasing epochs or adjusting learning rate")

    print(f"💾 Final checkpoint saved in: {checkpoint_dir}")
    print(f"📊 Training logs saved in: {log_dir}")

    # Final validation
    print("\n🔍 Final Validation:")
    final_metrics = trainer.validate_alignment(dataloader)
    print(f"   📊 Final Reconstruction Loss: {final_metrics['reconstruction_loss']:.8f}")
    print(f"   🎯 Final Average Confidence: {final_metrics['average_confidence']:.4f}")
    print(f"   ✅ Convergence Achieved: {final_metrics['convergence_achieved']}")
    print(f"   ✅ High Confidence: {final_metrics['high_confidence']}")


if __name__ == "__main__":
    main()