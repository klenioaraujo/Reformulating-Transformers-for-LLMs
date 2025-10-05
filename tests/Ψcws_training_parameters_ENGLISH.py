#!/usr/bin/env python3
"""
ΨCWS Training Parameters - Parameters for ΨCWS Model Training
========================================================================

This file defines training parameters for the ΨCWS system that converts:
TEXT → SPECTRUM → OUTPUT SPECTRUM → INPUT SPECTRUM → TEXT CONVERSION

The system uses:
- Open-source models as base
- Encryption layer (7 layers)
- Scientific mask to ensure pattern
- Spectral conversion

Parameters optimized for efficient training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class ΨCWSTrainingConfig:
    """Complete configuration for ΨCWS training."""

    # ===== GENERAL PARAMETERS =====
    model_name: str = "ΨCWS-Spectral-Transformer"
    version: str = "1.0"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== MODEL DIMENSIONS =====
    vocab_size: int = 50000  # Vocabulary size
    embedding_dim: int = 512  # Embedding dimension
    hidden_dim: int = 1024  # Hidden dimension
    num_layers: int = 6  # Number of layers
    num_heads: int = 8  # Number of attention heads
    dropout: float = 0.1  # Dropout rate

    # ===== SPECTRAL PARAMETERS =====
    spectral_dim: int = 256  # Spectral dimension
    fft_bins: int = 128  # Number of FFT bins
    window_size: int = 64  # Window size
    hop_length: int = 32  # Hop length for STFT

    # ===== TRAINING PARAMETERS =====
    batch_size: int = 32  # Batch size
    learning_rate: float = 1e-4  # Learning rate
    warmup_steps: int = 1000  # Warmup steps
    max_epochs: int = 100  # Maximum epochs
    gradient_clip: float = 1.0  # Gradient clipping
    weight_decay: float = 1e-6  # Weight decay

    # ===== ENCRYPTION PARAMETERS =====
    encryption_layers: int = 7  # 7 encryption layers
    encryption_key_size: int = 32  # Key size (bytes)
    salt_size: int = 16  # Salt size

    # ===== SCIENTIFIC MASK PARAMETERS =====
    scientific_mask_enabled: bool = True
    mask_pattern: str = "fractal_gaussian"  # Mask pattern
    mask_entropy_threshold: float = 0.8  # Entropy threshold

    # ===== TEXT-SPECTRUM CONVERSION PARAMETERS =====
    text_encoding: str = "utf-8"  # Text encoding
    max_sequence_length: int = 1024  # Maximum sequence length
    spectral_compression_ratio: float = 0.5  # Spectral compression ratio

    # ===== VALIDATION PARAMETERS =====
    validation_frequency: int = 1000  # Validation frequency
    early_stopping_patience: int = 10  # Early stopping patience
    checkpoint_frequency: int = 5000  # Checkpoint frequency

    # ===== ADVANCED PARAMETERS =====
    use_mixed_precision: bool = True  # Mixed precision
    gradient_accumulation_steps: int = 4  # Gradient accumulation steps
    use_spectral_norm: bool = True  # Spectral normalization
    use_layer_norm: bool = True  # Layer normalization

    # ===== OPTIMIZATION PARAMETERS =====
    optimizer: str = "AdamW"  # Optimizer
    scheduler: str = "cosine"  # Scheduler
    beta1: float = 0.9  # Beta1 for Adam
    beta2: float = 0.999  # Beta2 for Adam
    epsilon: float = 1e-8  # Epsilon for Adam

    # ===== MEMORY PARAMETERS =====
    memory_efficient: bool = True  # Memory efficient mode
    gradient_checkpointing: bool = True  # Gradient checkpointing

    # ===== LOGGING PARAMETERS =====
    log_frequency: int = 100  # Logging frequency
    save_frequency: int = 1000  # Save frequency

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@dataclass
class ΨCWSSpectralConfig:
    """Specific configuration for spectral processing."""

    # Spectral transformations
    use_stft: bool = True  # Use Short-Time Fourier Transform
    use_mel_scale: bool = True  # Use Mel scale
    use_mfcc: bool = False  # Use MFCC

    # STFT parameters
    sample_rate: int = 16000  # Sample rate
    n_fft: int = 1024  # Number of FFT points
    win_length: int = 1024  # Window length
    hop_length: int = 256  # Hop length
    window: str = "hann"  # Window type

    # Mel parameters
    n_mels: int = 80  # Number of Mel bands
    fmin: float = 80.0  # Minimum frequency
    fmax: float = 8000.0  # Maximum frequency

    # Spectral normalization
    spectral_norm_type: str = "per_channel"  # Normalization type
    spectral_mean: float = 0.0  # Mean for normalization
    spectral_std: float = 1.0  # Standard deviation for normalization

    # Spectral compression
    compression_method: str = "log"  # Compression method
    compression_factor: float = 0.3  # Compression factor

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@dataclass
class ΨCWSEncryptionConfig:
    """Configuration for encryption system."""

    # Encryption layers
    layer_1: str = "AES-256-GCM"
    layer_2: str = "ChaCha20-Poly1305"
    layer_3: str = "Fernet"
    layer_4: str = "XOR-Custom"
    layer_5: str = "Transposition"
    layer_6: str = "HMAC-AES"
    layer_7: str = "Obfuscation"

    # Security parameters
    key_derivation_iterations: int = 1000000  # Key derivation iterations
    salt: str = "PSIQRH_SECURE_SALT_v1.0"  # Salt for derivation
    hmac_algorithm: str = "SHA256"  # HMAC algorithm

    # Anti-violation policy
    max_access_attempts: int = 3  # Maximum access attempts
    violation_threshold: float = 0.9  # Violation detection threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@dataclass
class ΨCWSModelArchitecture:
    """ΨCWS model architecture."""

    # Encoder (Text → Spectrum)
    encoder_layers: int = 6
    encoder_heads: int = 8
    encoder_dim: int = 512
    encoder_ff_dim: int = 2048

    # Decoder (Spectrum → Text)
    decoder_layers: int = 6
    decoder_heads: int = 8
    decoder_dim: int = 512
    decoder_ff_dim: int = 2048

    # Spectral Layer
    spectral_encoder_layers: int = 4
    spectral_decoder_layers: int = 4
    spectral_attention_heads: int = 4

    # Residual Connections
    use_residual_connections: bool = True
    residual_scaling: float = 0.1

    # Normalization
    use_layer_norm: bool = True
    use_spectral_norm: bool = True
    norm_epsilon: float = 1e-5

    # Activations
    activation_function: str = "gelu"
    output_activation: str = "softmax"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


class ΨCWSTrainingParameters:
    """Main class for managing ΨCWS training parameters."""

    def __init__(self):
        self.training_config = ΨCWSTrainingConfig()
        self.spectral_config = ΨCWSSpectralConfig()
        self.encryption_config = ΨCWSEncryptionConfig()
        self.model_architecture = ΨCWSModelArchitecture()

    def get_all_parameters(self) -> Dict[str, Any]:
        """Return all parameters in a dictionary."""
        return {
            "training": self.training_config.to_dict(),
            "spectral": self.spectral_config.to_dict(),
            "encryption": self.encryption_config.to_dict(),
            "architecture": self.model_architecture.to_dict()
        }

    def get_training_hyperparameters(self) -> Dict[str, Any]:
        """Return specific training hyperparameters."""
        return {
            "batch_size": self.training_config.batch_size,
            "learning_rate": self.training_config.learning_rate,
            "warmup_steps": self.training_config.warmup_steps,
            "max_epochs": self.training_config.max_epochs,
            "gradient_clip": self.training_config.gradient_clip,
            "weight_decay": self.training_config.weight_decay,
            "optimizer": self.training_config.optimizer,
            "scheduler": self.training_config.scheduler
        }

    def get_model_parameters(self) -> Dict[str, Any]:
        """Return specific model parameters."""
        return {
            "vocab_size": self.training_config.vocab_size,
            "embedding_dim": self.training_config.embedding_dim,
            "hidden_dim": self.training_config.hidden_dim,
            "num_layers": self.training_config.num_layers,
            "num_heads": self.training_config.num_heads,
            "dropout": self.training_config.dropout,
            "spectral_dim": self.training_config.spectral_dim
        }

    def get_spectral_parameters(self) -> Dict[str, Any]:
        """Return specific spectral processing parameters."""
        return {
            "fft_bins": self.training_config.fft_bins,
            "window_size": self.training_config.window_size,
            "hop_length": self.training_config.hop_length,
            "sample_rate": self.spectral_config.sample_rate,
            "n_fft": self.spectral_config.n_fft,
            "n_mels": self.spectral_config.n_mels,
            "compression_method": self.spectral_config.compression_method
        }

    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """Validate all parameters and return status and error list."""
        errors = []

        # Training validations
        if self.training_config.batch_size <= 0:
            errors.append("batch_size must be > 0")

        if self.training_config.learning_rate <= 0:
            errors.append("learning_rate must be > 0")

        if self.training_config.embedding_dim % self.training_config.num_heads != 0:
            errors.append("embedding_dim must be divisible by num_heads")

        # Spectral validations
        if self.spectral_config.n_fft <= 0:
            errors.append("n_fft must be > 0")

        if self.spectral_config.hop_length <= 0:
            errors.append("hop_length must be > 0")

        if self.spectral_config.fmin >= self.spectral_config.fmax:
            errors.append("fmin must be < fmax")

        return len(errors) == 0, errors

    def optimize_for_hardware(self, device_type: str = "auto") -> None:
        """Optimize parameters for specific hardware."""

        if device_type == "gpu" or (device_type == "auto" and torch.cuda.is_available()):
            # GPU optimizations
            self.training_config.batch_size = 64
            self.training_config.use_mixed_precision = True
            self.training_config.gradient_accumulation_steps = 2

        elif device_type == "cpu":
            # CPU optimizations
            self.training_config.batch_size = 16
            self.training_config.use_mixed_precision = False
            self.training_config.gradient_accumulation_steps = 8

        elif device_type == "tpu":
            # TPU optimizations
            self.training_config.batch_size = 128
            self.training_config.use_mixed_precision = True
            self.training_config.gradient_accumulation_steps = 1


# Utility functions for training
def create_optimizer(model: nn.Module, config: ΨCWSTrainingConfig):
    """Create optimizer based on configuration."""

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]

    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "adam":
        return torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def create_scheduler(optimizer, config: ΨCWSTrainingConfig, total_steps: int):
    """Create scheduler based on configuration."""

    if config.scheduler.lower() == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=total_steps)

    elif config.scheduler.lower() == "linear":
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    elif config.scheduler.lower() == "exponential":
        from torch.optim.lr_scheduler import ExponentialLR
        return ExponentialLR(optimizer, gamma=0.95)

    else:
        # Default scheduler: no scheduler
        return None


# Predefined configurations
def get_preset_config(preset_name: str) -> ΨCWSTrainingParameters:
    """Return predefined configuration."""

    params = ΨCWSTrainingParameters()

    if preset_name == "small":
        # Small configuration for quick testing
        params.training_config.batch_size = 8
        params.training_config.embedding_dim = 256
        params.training_config.hidden_dim = 512
        params.training_config.num_layers = 4
        params.training_config.num_heads = 4
        params.training_config.max_epochs = 10

    elif preset_name == "medium":
        # Medium configuration for development
        params.training_config.batch_size = 16
        params.training_config.embedding_dim = 384
        params.training_config.hidden_dim = 768
        params.training_config.num_layers = 6
        params.training_config.num_heads = 6
        params.training_config.max_epochs = 50

    elif preset_name == "large":
        # Large configuration for production
        params.training_config.batch_size = 32
        params.training_config.embedding_dim = 512
        params.training_config.hidden_dim = 1024
        params.training_config.num_layers = 8
        params.training_config.num_heads = 8
        params.training_config.max_epochs = 100

    elif preset_name == "spectral_focus":
        # Configuration focused on spectral processing
        params.training_config.spectral_dim = 512
        params.training_config.fft_bins = 256
        params.spectral_config.n_mels = 128
        params.spectral_config.use_mfcc = True

    else:
        raise ValueError(f"Preset not found: {preset_name}")

    return params


if __name__ == "__main__":
    # Usage example
    params = ΨCWSTrainingParameters()

    print("🔧 ΨCWS Training Parameters")
    print("=" * 50)

    all_params = params.get_all_parameters()

    for category, config in all_params.items():
        print(f"\n📋 {category.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    # Validation
    is_valid, errors = params.validate_parameters()
    if is_valid:
        print("\n✅ All parameters are valid!")
    else:
        print(f"\n❌ Errors found: {errors}")

    # Training hyperparameters
    hyperparams = params.get_training_hyperparameters()
    print(f"\n🎯 Training Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")