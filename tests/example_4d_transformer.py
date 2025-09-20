#!/usr/bin/env python3
"""
Example: Using 4D Unitary Layer in Transformer Architecture
==========================================================

This script demonstrates how to integrate the 4D Unitary Layer into a complete
transformer architecture for sequence processing tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# Import our 4D Unitary Layer components
import sys
import os

# Add parent directory to path to find modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from ΨQRH import (
    QRHLayer,
    NegentropyTransformerBlock,
    GateController,
    SpectralFilter
)


class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformers"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class TransformerWith4DUnitary(nn.Module):
    """
    Complete transformer with 4D Unitary Layer integration

    This example shows how to replace standard attention with 4D processing
    while maintaining compatibility with existing transformer workflows.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 1024,
                 use_4d_unitary: bool = True,
                 qrh_embed_dim: int = 128):
        super().__init__()

        self.d_model = d_model
        self.use_4d_unitary = use_4d_unitary

        # Standard transformer components
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # Choose between standard and 4D unitary blocks
        if use_4d_unitary:
            self.layers = nn.ModuleList([
                NegentropyTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    qrh_embed_dim=qrh_embed_dim,
                    enable_gate=True
                ) for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(num_layers)
            ])

        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer

        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_mask: Source mask [batch_size, src_len]
            tgt_mask: Target mask [batch_size, tgt_len]

        Returns:
            Output logits [batch_size, tgt_len, vocab_size]
        """

        # Embed source and target
        src_embed = self.token_embedding(src)  # [batch, src_len, d_model]
        tgt_embed = self.token_embedding(tgt)  # [batch, tgt_len, d_model]

        # Add positional encoding
        src_embed = self.positional_encoding(src_embed.transpose(0, 1)).transpose(0, 1)
        tgt_embed = self.positional_encoding(tgt_embed.transpose(0, 1)).transpose(0, 1)

        # Apply dropout
        src_embed = self.dropout(src_embed)
        tgt_embed = self.dropout(tgt_embed)

        # Process through layers
        for layer in self.layers:
            if self.use_4d_unitary:
                # 4D Unitary processing (encoder-decoder style)
                tgt_embed = layer(tgt_embed, src_mask)
            else:
                # Standard transformer processing
                tgt_embed = layer(tgt_embed, src_embed, tgt_mask=tgt_mask, memory_mask=src_mask)

        # Final normalization and projection
        output = self.norm(tgt_embed)
        logits = self.output_projection(output)

        return logits

    def get_4d_statistics(self) -> dict:
        """Get statistics from 4D unitary layers"""
        if not self.use_4d_unitary:
            return {"status": "4D unitary layers not enabled"}

        stats = {
            "num_layers": len(self.layers),
            "gate_decisions": [],
            "spectral_filter_params": [],
            "rotation_params": []
        }

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'gate_controller') and layer.gate_controller:
                # This would require tracking gate decisions during forward pass
                stats["gate_decisions"].append(f"Layer {i}: gate enabled")

            if hasattr(layer, 'qrh_layer'):
                qrh = layer.qrh_layer
                stats["spectral_filter_params"].append({
                    "layer": i,
                    "alpha": qrh.spectral_filter.alpha,
                    "use_stable_activation": qrh.spectral_filter.use_stable_activation
                })

                if hasattr(qrh, 'theta_left'):
                    stats["rotation_params"].append({
                        "layer": i,
                        "theta_left": qrh.theta_left.item(),
                        "theta_right": qrh.theta_right.item()
                    })

        return stats


class SequenceClassifierWith4D(nn.Module):
    """
    Example: Sequence classification using 4D Unitary Layer

    This demonstrates using the 4D layer for classification tasks
    where we want to capture complex patterns in the input sequence.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 num_classes: int = 2,
                 qrh_embed_dim: int = 64,
                 max_seq_len: int = 512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # 4D Unitary processing layer
        from qrh_layer import QRHConfig
        config = QRHConfig(
            embed_dim=qrh_embed_dim,
            alpha=1.0,
            use_learned_rotation=True
        )
        self.qrh_layer = QRHLayer(config)

        # Projection layers
        self.input_proj = nn.Linear(d_model, 4 * qrh_embed_dim)
        self.output_proj = nn.Linear(4 * qrh_embed_dim, d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence classification

        Args:
            input_ids: Input token ids [batch_size, seq_len]

        Returns:
            Classification logits [batch_size, num_classes]
        """

        # Embedding and positional encoding
        x = self.embedding(input_ids)  # [batch, seq, d_model]
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)

        # Project to 4D space
        x_4d = self.input_proj(x)  # [batch, seq, 4*qrh_embed_dim]

        # Apply 4D unitary transformation
        x_4d_processed = self.qrh_layer(x_4d)

        # Project back to d_model space
        x_processed = self.output_proj(x_4d_processed)  # [batch, seq, d_model]

        # Global average pooling
        x_pooled = torch.mean(x_processed, dim=1)  # [batch, d_model]

        # Classification
        logits = self.classifier(x_pooled)  # [batch, num_classes]

        return logits


def demonstrate_basic_usage():
    """Demonstrate basic usage of 4D Unitary Layer"""
    print("=" * 60)
    print("4D UNITARY LAYER BASIC USAGE DEMONSTRATION")
    print("=" * 60)

    # Create a simple 4D layer
    from qrh_layer import QRHConfig
    config = QRHConfig(embed_dim=32, use_learned_rotation=True)
    layer = QRHLayer(config)

    # Sample input
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, 4 * 32)  # 4 * embed_dim

    print(f"Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = layer(x)

    print(f"Output shape: {output.shape}")
    print(f"Energy preservation ratio: {torch.norm(output) / torch.norm(x):.4f}")

    # Demonstrate gate controller
    gate = GateController()
    receipts = gate.calculate_receipts(x, output, {
        'theta_left': layer.theta_left,
        'omega_left': layer.omega_left,
        'phi_left': layer.phi_left,
        'theta_right': layer.theta_right,
        'omega_right': layer.omega_right,
        'phi_right': layer.phi_right
    })

    print("Gate receipts:")
    for key, value in receipts.items():
        print(f"{key}: {value:.6f}")

    decision = gate.decide_gate(receipts)
    print(f"\nGate decision: {decision}")


def demonstrate_transformer_integration():
    """Demonstrate transformer integration"""
    print("\n" + "=" * 60)
    print("TRANSFORMER INTEGRATION DEMONSTRATION")
    print("=" * 60)

    # Model parameters
    vocab_size = 1000
    d_model = 128
    seq_len = 32

    # Create models
    standard_transformer = TransformerWith4DUnitary(
        vocab_size=vocab_size,
        d_model=d_model,
        use_4d_unitary=False
    )

    unitary_transformer = TransformerWith4DUnitary(
        vocab_size=vocab_size,
        d_model=d_model,
        use_4d_unitary=True,
        qrh_embed_dim=32
    )

    # Sample data
    src = torch.randint(0, vocab_size, (2, seq_len))
    tgt = torch.randint(0, vocab_size, (2, seq_len))

    print(f"Input shapes: src={src.shape}, tgt={tgt.shape}")

    # Forward passes
    with torch.no_grad():
        standard_out = standard_transformer(src, tgt)
        unitary_out = unitary_transformer(src, tgt)

    print(f"Standard output shape: {standard_out.shape}")
    print(f"4D Unitary output shape: {unitary_out.shape}")

    # Show 4D statistics
    stats = unitary_transformer.get_4d_statistics()
    print("4D Layer Statistics:")
    print(f"Number of layers: {stats['num_layers']}")
    if stats['spectral_filter_params']:
        print(f"Spectral alpha values: {[p['alpha'] for p in stats['spectral_filter_params']]}")


def demonstrate_sequence_classification():
    """Demonstrate sequence classification with 4D layer"""
    print("\n" + "=" * 60)
    print("SEQUENCE CLASSIFICATION DEMONSTRATION")
    print("=" * 60)

    # Model setup
    vocab_size = 500
    num_classes = 3
    seq_len = 64

    classifier = SequenceClassifierWith4D(
        vocab_size=vocab_size,
        d_model=128,
        num_classes=num_classes,
        qrh_embed_dim=32
    )

    # Sample input
    input_ids = torch.randint(0, vocab_size, (4, seq_len))

    print(f"Input shape: {input_ids.shape}")

    # Forward pass
    with torch.no_grad():
        logits = classifier(input_ids)

    print(f"Output logits shape: {logits.shape}")
    print(f"Predicted classes: {torch.argmax(logits, dim=1)}")

    # Show probability distributions
    probs = F.softmax(logits, dim=1)
    print("Class probabilities:")
    for i in range(min(4, logits.shape[0])):
        print(f"  Sample {i}: {probs[i].numpy()}")


def main():
    """Main demonstration function"""
    print("4D UNITARY LAYER COMPREHENSIVE DEMONSTRATION")
    print("This example shows various ways to use the 4D Unitary Layer")
    print("in transformer architectures and sequence processing tasks.\n")

    try:
        # Run demonstrations
        demonstrate_basic_usage()
        demonstrate_transformer_integration()
        demonstrate_sequence_classification()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("The 4D Unitary Layer provides:")
        print("• SO(4) rotations with two independent quaternions")
        print("• Numerically stable spectral filtering")
        print("• Gate mechanism for flow control")
        print("• Seamless transformer integration")
        print("• Mixed precision support")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Make sure PyTorch is installed and all dependencies are available.")


if __name__ == "__main__":
    main()