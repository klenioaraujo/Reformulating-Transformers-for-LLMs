#!/usr/bin/env python3
"""
ΨQRH Pipeline Supervised Training Script
========================================

Trains the learnable components of the ΨQRH pipeline (ContextFunnel, InverseCognitiveProjector, etc.)
using supervised learning to map between quantum states and linguistic elements.

This script orchestrates the complete training process with proper loss calculation,
optimization, and checkpointing.
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
from src.core.losses import QuantumContrastiveLoss, QuantumEmbedding, SpectralCoherenceLoss


class TrainingPairsDataset(Dataset):
    """Dataset for training pairs of (context, target) text."""

    def __init__(self, data_path: str, tokenizer_config: Optional[Dict] = None):
        """
        Initialize the training dataset.

        Args:
            data_path: Path to the JSON file containing training pairs
            tokenizer_config: Configuration for the tokenizer
        """
        self.data_path = Path(data_path)
        self.tokenizer_config = tokenizer_config or {
            'embed_dim': 64,
            'spectral_params_dim': 8,
            'learnable': True
        }

        # Load training data
        self.training_pairs = self._load_training_data()

        print(f"📚 Loaded {len(self.training_pairs)} training pairs")

    def _load_training_data(self) -> List[Dict[str, str]]:
        """Load training pairs from JSON file."""
        if not self.data_path.exists():
            print(f"⚠️  Training data file not found: {self.data_path}")
            print("   Creating synthetic training data...")

            # Create synthetic training data for demonstration
            synthetic_data = self._create_synthetic_training_data()
            self._save_synthetic_data(synthetic_data)
            return synthetic_data

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'training_pairs' in data:
                return data['training_pairs']
            else:
                print("⚠️  Unexpected data format, using synthetic data")
                return self._create_synthetic_training_data()

        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            print("   Using synthetic training data")
            return self._create_synthetic_training_data()

    def _create_synthetic_training_data(self) -> List[Dict[str, str]]:
        """Create synthetic training data for demonstration."""
        contexts = [
            "quantum mechanics",
            "wave function",
            "uncertainty principle",
            "schrodinger equation",
            "superposition state",
            "entanglement theory",
            "quantum computing",
            "physics principles",
            "mathematical model",
            "scientific theory"
        ]

        targets = [
            "explains physical phenomena",
            "describes quantum systems",
            "limits measurement precision",
            "governs quantum evolution",
            "allows multiple states",
            "connects distant particles",
            "enables new computation",
            "govern natural laws",
            "represents complex systems",
            "explains natural phenomena"
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

    def _save_synthetic_data(self, data: List[Dict[str, str]]):
        """Save synthetic training data to file."""
        try:
            os.makedirs(self.data_path.parent, exist_ok=True)
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved synthetic training data to: {self.data_path}")
        except Exception as e:
            print(f"⚠️  Could not save synthetic data: {e}")

    def __len__(self) -> int:
        return len(self.training_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.training_pairs[idx]


class ΨQRHTrainer:
    """Trainer class for the ΨQRH pipeline."""

    def __init__(self, pipeline: ΨQRHPipeline, device: str = 'cpu',
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        """
        Initialize the trainer.

        Args:
            pipeline: The ΨQRH pipeline to train
            device: Device to use for training
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
        """
        self.pipeline = pipeline
        self.device = device

        # Get learnable parameters
        learnable_params = self._get_learnable_parameters()

        if learnable_params:
            try:
                self.optimizer = optim.AdamW(learnable_params, lr=learning_rate, weight_decay=weight_decay)
                self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=100, T_mult=2
                )
                print(f"🎓 Optimizer initialized with {len(learnable_params)} learnable parameters")
            except Exception as e:
                print(f"⚠️  Failed to initialize optimizer: {e}")
                print("   This might be due to missing dependencies (sympy). Using SGD as fallback...")
                try:
                    self.optimizer = optim.SGD(learnable_params, lr=learning_rate, weight_decay=weight_decay)
                    self.scheduler = None  # No scheduler for SGD
                    print("✅ Fallback optimizer (SGD) initialized successfully")
                except Exception as e2:
                    print(f"❌ Even fallback optimizer failed: {e2}")
                    self.optimizer = None
                    self.scheduler = None
        else:
            self.optimizer = None
            self.scheduler = None
            print("⚠️  No learnable parameters found - training will be skipped")

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.contrastive_loss = QuantumContrastiveLoss(margin=1.0)
        self.spectral_loss = SpectralCoherenceLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # Training statistics
        self.training_stats = {
            'epoch': 0,
            'total_loss': 0.0,
            'num_batches': 0,
            'best_loss': float('inf'),
            'learning_rate': learning_rate
        }

    def _get_learnable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all learnable parameters from the pipeline and quantum embedding."""
        learnable_params = []

        # Quantum embedding parameters (from pipeline)
        if hasattr(self.pipeline, 'quantum_embedding'):
            learnable_params.extend(list(self.pipeline.quantum_embedding.parameters()))

        # Context Funnel parameters
        if hasattr(self.pipeline, 'context_funnel'):
            learnable_params.extend(list(self.pipeline.context_funnel.parameters()))

        # Inverse Cognitive Projector parameters
        if hasattr(self.pipeline, 'inverse_projector'):
            learnable_params.extend(list(self.pipeline.inverse_projector.parameters()))

        # Any other learnable components
        if hasattr(self.pipeline, 'dcf_analyzer') and self.pipeline.dcf_analyzer:
            # Add DCF learnable parameters if available
            if hasattr(self.pipeline.dcf_analyzer, 'kuramoto_layer'):
                learnable_params.extend(list(self.pipeline.dcf_analyzer.kuramoto_layer.parameters()))

        return learnable_params

    def encode_target_to_quantum_state(self, target_text: str) -> torch.Tensor:
        """
        Encode target text to its ideal quantum state representation.

        Args:
            target_text: The target text to encode

        Returns:
            Quantum state tensor representing the target
        """
        # Use the pipeline's text-to-quantum encoding
        if hasattr(self.pipeline, '_text_to_fractal_signal'):
            # Convert text to fractal signal
            fractal_signal = self.pipeline._text_to_fractal_signal(target_text, self.pipeline.config['embed_dim'])

            # Convert to quaternion representation
            psi_target = self.pipeline._signal_to_quaternions(fractal_signal, self.pipeline.config['embed_dim'])

            return psi_target
        else:
            # Fallback: create a simple encoded representation
            # Use character values as a basic encoding
            char_values = torch.tensor([ord(c) / 127.0 for c in target_text], dtype=torch.float32)

            # Pad or truncate to embed_dim
            embed_dim = self.pipeline.config['embed_dim']
            if len(char_values) < embed_dim:
                padding = torch.zeros(embed_dim - len(char_values))
                char_values = torch.cat([char_values, padding])
            else:
                char_values = char_values[:embed_dim]

            # Create quaternion representation [1, seq_len=1, embed_dim, 4]
            psi_target = torch.zeros(1, 1, embed_dim, 4, dtype=torch.float32, device=self.device)
            psi_target[0, 0, :, 0] = char_values  # Real part
            psi_target[0, 0, :, 1] = torch.sin(char_values)  # i component
            psi_target[0, 0, :, 2] = torch.cos(char_values)  # j component
            psi_target[0, 0, :, 3] = torch.sin(char_values * 2)  # k component

            return psi_target

    def compute_training_loss(self, predicted_psi: torch.Tensor, target_text: str) -> torch.Tensor:
        """
        Compute the training loss between predicted and target quantum states.

        Args:
            predicted_psi: Predicted quantum state from the pipeline
            target_text: Target text to encode

        Returns:
            MSE loss between predicted and target states
        """
        # Encode target text to quantum state
        target_psi = self.encode_target_to_quantum_state(target_text)
        target_psi = target_psi.to(self.device)

        # Ensure shapes match for loss computation
        # predicted_psi shape: [batch, seq_len, embed_dim, 4]
        # target_psi shape: [1, 1, embed_dim, 4]

        # For now, use mean pooling to compare overall representations
        pred_mean = predicted_psi.mean(dim=[0, 1])  # [embed_dim, 4]
        target_mean = target_psi.mean(dim=[0, 1])    # [embed_dim, 4]

        # Compute MSE loss
        loss = self.mse_loss(pred_mean, target_mean)

        return loss

    def train_epoch_contrastive(self, dataloader: DataLoader, log_every: int = 10) -> float:
        """
        Train for one epoch using contrastive learning in quantum space.

        Args:
            dataloader: DataLoader with training data (context, target pairs)
            log_every: Log progress every N batches

        Returns:
            Average contrastive loss for the epoch
        """
        if not self.optimizer:
            print("⚠️  No optimizer available - skipping training")
            return 0.0

        # Set learnable components to train mode
        if hasattr(self.pipeline, 'quantum_embedding'):
            self.pipeline.quantum_embedding.train()
        if hasattr(self.pipeline, 'context_funnel') and self.pipeline.context_funnel:
            self.pipeline.context_funnel.train()
        if hasattr(self.pipeline, 'inverse_projector') and self.pipeline.inverse_projector:
            self.pipeline.inverse_projector.train()
        if hasattr(self.pipeline, 'dcf_analyzer') and self.pipeline.dcf_analyzer:
            if hasattr(self.pipeline.dcf_analyzer, 'kuramoto_layer'):
                self.pipeline.dcf_analyzer.kuramoto_layer.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Get context and target texts
            context_texts = batch['context']
            target_texts = batch['target']

            batch_loss = 0.0
            valid_samples = 0

            for context_text, target_text in zip(context_texts, target_texts):
                try:
                    # Use first character of context and target for training
                    context_char = context_text[0] if context_text else ' '
                    target_char = target_text[0] if target_text else ' '

                    # Convert characters to IDs
                    context_id = ord(context_char) if ord(context_char) < 256 else ord(' ')
                    target_id = ord(target_char) if ord(target_char) < 256 else ord(' ')

                    context_ids = torch.tensor([[context_id]], dtype=torch.long, device=self.device)  # [1, 1]
                    target_ids = torch.tensor([[target_id]], dtype=torch.long, device=self.device)   # [1, 1]

                    # Get quantum embeddings
                    psi_context_full = self.pipeline.quantum_embedding(context_ids)  # [1, 1, embed_dim//4, 4]
                    psi_target_full = self.pipeline.quantum_embedding(target_ids)    # [1, 1, embed_dim//4, 4]

                    # Extract the quantum states [1, embed_dim//4, 4] -> [embed_dim, 4]
                    psi_context = psi_context_full[:, 0, :, :].view(psi_context_full.shape[0], -1, 4)  # [1, embed_dim, 4]
                    psi_pos = psi_target_full[:, 0, :, :].view(psi_target_full.shape[0], -1, 4)       # [1, embed_dim, 4]

                    # Generate negative example (random character)
                    neg_char_id = torch.randint(32, 127, (1,), device=self.device)  # Printable ASCII
                    neg_ids = torch.tensor([[neg_char_id]], dtype=torch.long, device=self.device)  # [1, 1]
                    psi_negative = self.pipeline.quantum_embedding(neg_ids)  # [1, 1, embed_dim//4, 4]
                    psi_neg = psi_negative[:, 0, :, :].view(psi_negative.shape[0], -1, 4)  # [1, embed_dim, 4]

                    # Compute contrastive loss
                    loss = self.contrastive_loss(psi_context, psi_pos, psi_neg)
                    batch_loss += loss.item()
                    valid_samples += 1

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                except Exception as e:
                    print(f"❌ Training error for context '{context_text[:20]}...' -> target '{target_text[:20]}...': {e}")
                    continue

            if valid_samples > 0:
                epoch_loss += batch_loss / valid_samples
                num_batches += 1

            # Log progress
            if (batch_idx + 1) % log_every == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"   📊 Batch {batch_idx + 1}/{len(dataloader)} - Contrastive Loss: {avg_loss:.6f}")

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

    def train_epoch_spectral(self, dataloader: DataLoader, log_every: int = 10) -> float:
        """
        Train for one epoch using spectral coherence loss.

        Args:
            dataloader: DataLoader with training data
            log_every: Log progress every N batches

        Returns:
            Average spectral coherence loss for the epoch
        """
        if not self.optimizer:
            print("⚠️  No optimizer available - skipping training")
            return 0.0

        # Set learnable components to train mode
        if hasattr(self.pipeline, 'quantum_embedding'):
            self.pipeline.quantum_embedding.train()
        if hasattr(self.pipeline, 'context_funnel') and self.pipeline.context_funnel:
            self.pipeline.context_funnel.train()
        if hasattr(self.pipeline, 'inverse_projector') and self.pipeline.inverse_projector:
            self.pipeline.inverse_projector.train()
        if hasattr(self.pipeline, 'dcf_analyzer') and self.pipeline.dcf_analyzer:
            if hasattr(self.pipeline.dcf_analyzer, 'kuramoto_layer'):
                self.pipeline.dcf_analyzer.kuramoto_layer.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            batch_loss = 0.0
            valid_samples = 0

            # Get the batch data
            contexts = batch.get('context', batch.get('combined', []))
            targets = batch.get('target', batch.get('combined', []))

            for input_text, reference_text in zip(contexts, targets):
                try:

                    if not input_text or not reference_text:
                        continue

                    # Generate Ψ_output from input text
                    psi_output = self.encode_target_to_quantum_state(input_text)

                    # Generate Ψ_reference from reference text
                    psi_reference = self.encode_target_to_quantum_state(reference_text)

                    # Ensure same batch size (add batch dimension if needed)
                    if psi_output.dim() == 3:  # [seq_len, embed_dim, 4]
                        psi_output = psi_output.unsqueeze(0)  # [1, seq_len, embed_dim, 4]
                    if psi_reference.dim() == 3:  # [seq_len, embed_dim, 4]
                        psi_reference = psi_reference.unsqueeze(0)  # [1, seq_len, embed_dim, 4]

                    # Move to device
                    psi_output = psi_output.to(self.device)
                    psi_reference = psi_reference.to(self.device)

                    # Compute spectral coherence loss
                    loss = self.spectral_loss(psi_output, psi_reference)
                    batch_loss += loss.item()
                    valid_samples += 1

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                except Exception as e:
                    print(f"❌ Spectral training error: {e}")
                    continue

            if valid_samples > 0:
                epoch_loss += batch_loss / valid_samples
                num_batches += 1

            # Log progress
            if (batch_idx + 1) % log_every == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"   📊 Batch {batch_idx + 1}/{len(dataloader)} - Spectral Coherence Loss: {avg_loss:.6f}")

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

    def get_logits_from_pipeline(self, input_text: str) -> torch.Tensor:
        """
        Execute pipeline and return raw logits from InverseCognitiveProjector.

        Args:
            input_text: Input text to process

        Returns:
            Logits tensor [vocab_size]
        """
        # Use first character only for training (as per original plan)
        if not input_text:
            input_text = " "

        first_char = input_text[0]
        input_id = ord(first_char) if ord(first_char) < 256 else ord(' ')

        # Map to valid vocabulary range (0 to vocab_size-1)
        if hasattr(self.pipeline, 'quantum_embedding') and hasattr(self.pipeline.quantum_embedding, 'vocab_size'):
            vocab_size = self.pipeline.quantum_embedding.vocab_size
            input_id = input_id % vocab_size

        # Convert to tensor [1, 1]
        input_tensor = torch.tensor([[input_id]], dtype=torch.long, device=self.device)

        # Get quantum embedding
        with torch.no_grad():  # No gradients for embedding during forward pass
            quantum_embed = self.pipeline.quantum_embedding(input_tensor)  # [1, 1, embed_dim//4, 4]

        # Reshape for inverse projector input
        batch_size, seq_len, embed_quart, quat_dim = quantum_embed.shape
        embed_dim = embed_quart * quat_dim  # Full embedding dimension

        # Flatten quaternion dimensions: [1, 1, embed_dim]
        quantum_flat = quantum_embed.view(batch_size, seq_len, embed_dim)

        # Get raw logits directly from vocab_projection layer
        # Process through quantum processor layers first
        x = quantum_flat.squeeze(0).squeeze(0)  # [embed_dim]
        for layer in self.pipeline.inverse_projector.quantum_processor:
            x = layer(x)

        # Get logits from vocab projection
        logits = self.pipeline.inverse_projector.vocab_projection(x)  # [vocab_size]

        return logits

    def text_to_token_ids(self, text: str) -> torch.Tensor:
        """
        Convert text to token IDs for loss computation (first character only).

        Args:
            text: Text to convert

        Returns:
            Token ID tensor [1]
        """
        if not text:
            text = " "

        first_char = text[0]
        token_id = ord(first_char) if ord(first_char) < 256 else ord(' ')

        # Get vocabulary size from inverse projector
        if hasattr(self.pipeline, 'inverse_projector') and self.pipeline.inverse_projector:
            vocab_size = self.pipeline.inverse_projector.vocab_size
            # Map ASCII token to vocabulary range (0 to vocab_size-1)
            # Use modulo to map any ASCII value to valid range
            token_id = token_id % vocab_size

        return torch.tensor([token_id], dtype=torch.long, device=self.device)

    def train_epoch_semantic(self, dataloader: DataLoader, log_every: int = 10) -> float:
        """
        Train for one epoch using supervised semantic learning.

        Args:
            dataloader: DataLoader with training data (context, target pairs)
            log_every: Log progress every N batches

        Returns:
            Average semantic loss for the epoch
        """
        if not self.optimizer:
            print("⚠️  No optimizer available - skipping training")
            return 0.0

        # Set learnable components to train mode
        if hasattr(self.pipeline, 'quantum_embedding'):
            self.pipeline.quantum_embedding.train()
        if hasattr(self.pipeline, 'context_funnel') and self.pipeline.context_funnel:
            self.pipeline.context_funnel.train()
        if hasattr(self.pipeline, 'inverse_projector') and self.pipeline.inverse_projector:
            self.pipeline.inverse_projector.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            batch_loss = 0.0
            valid_samples = 0

            # Get context and target texts
            context_texts = batch.get('context', [])
            target_texts = batch.get('target', [])

            for context_text, target_text in zip(context_texts, target_texts):
                try:
                    # Get logits from pipeline with input text
                    logits = self.get_logits_from_pipeline(context_text)  # [vocab_size]

                    # Convert target text to token IDs (first character only)
                    target_ids = self.text_to_token_ids(target_text)  # [1]

                    # Reshape for loss: [1, vocab_size] and [1]
                    logits_reshaped = logits.unsqueeze(0)  # [1, vocab_size]
                    target_ids_flat = target_ids  # [1]

                    # Compute cross-entropy loss
                    loss = self.cross_entropy_loss(logits_reshaped, target_ids_flat)
                    batch_loss += loss.item()
                    valid_samples += 1

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                except Exception as e:
                    print(f"❌ Semantic training error for '{context_text[:20]}' -> '{target_text[:20]}': {e}")
                    continue

            if valid_samples > 0:
                epoch_loss += batch_loss / valid_samples
                num_batches += 1

            # Log progress
            if (batch_idx + 1) % log_every == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"   📊 Batch {batch_idx + 1}/{len(dataloader)} - Semantic Loss: {avg_loss:.6f}")

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
            'model_state_dict': {},
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }

        # Save learnable component states
        if hasattr(self.pipeline, 'quantum_embedding'):
            checkpoint['model_state_dict']['quantum_embedding'] = self.pipeline.quantum_embedding.state_dict()

        if hasattr(self.pipeline, 'context_funnel'):
            checkpoint['model_state_dict']['context_funnel'] = self.pipeline.context_funnel.state_dict()

        if hasattr(self.pipeline, 'inverse_projector'):
            checkpoint['model_state_dict']['inverse_projector'] = self.pipeline.inverse_projector.state_dict()

        # Save checkpoint
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

            # Update best loss
            if loss < self.training_stats['best_loss']:
                self.training_stats['best_loss'] = loss
                best_path = checkpoint_path.parent / "best_model.pt"
                torch.save(checkpoint, best_path)
                print(f"🏆 Best model saved: {best_path}")

        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging for training."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Training log started: {log_file}")

    return logger


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="ΨQRH Pipeline Training Script")
    parser.add_argument('--data-path', type=str, default='data/training_pairs.json',
                       help='Path to training data JSON file')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/training',
                       help='Directory to save training logs')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for training (cpu/cuda)')

    args = parser.parse_args()

    print("🚀 Starting ΨQRH Pipeline Training")
    print("=" * 50)

    # Setup logging
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir)

    # Create pipeline
    print("🔧 Initializing ΨQRH Pipeline...")
    pipeline = ΨQRHPipeline(
        task="text-generation",
        device=args.device,
        enable_auto_calibration=False,  # Disable for controlled training
        audit_mode=False  # Disable audit for faster training
    )

    # Create trainer
    trainer = ΨQRHTrainer(pipeline, device=args.device, learning_rate=args.learning_rate)

    # Load dataset
    dataset = TrainingPairsDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    checkpoint_dir = Path(args.checkpoint_dir)

    print(f"🎯 Starting training for {args.epochs} epochs...")
    print(f"   📊 Dataset size: {len(dataset)}")
    print(f"   📊 Batch size: {args.batch_size}")
    print(f"   📊 Learning rate: {args.learning_rate}")
    print(f"   💾 Checkpoints: {checkpoint_dir}")
    print()

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"🎯 Epoch {epoch}/{args.epochs} - Semantic Supervised Training")
        print("-" * 70)

        # Train epoch using supervised semantic learning
        epoch_loss = trainer.train_epoch_semantic(dataloader, log_every=5)

        # Log epoch results
        logger.info(f"Epoch {epoch}: Semantic Loss = {epoch_loss:.6f}")

        print(".6f")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        trainer.save_checkpoint(checkpoint_path, epoch, epoch_loss)

        # Update best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(".6f")

        print()

    print("✅ Training completed!")
    print(".6f")
    print(f"💾 Final checkpoint saved in: {checkpoint_dir}")
    print(f"📊 Training logs saved in: {log_dir}")


if __name__ == "__main__":
    main()