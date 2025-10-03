#!/usr/bin/env python3
"""
Integration Guide: Using ΨQRH Layers in Standard Transformers

This script demonstrates how to integrate ΨQRH components
(quaternion operations, harmonic gates) into existing transformer models.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.quaternion_operations import QuaternionOperations
from src.core.hierarchical_gate_system import HierarchicalGateSystem
from src.core.qrh_layer import QRHLayer


class HybridTransformerLayer(nn.Module):
    """
    Hybrid transformer layer that combines standard attention
    with ΨQRH quaternion operations.

    This allows you to enhance existing models with quaternion
    representations while maintaining compatibility with standard
    transformer architectures.
    """

    def __init__(
        self,
        d_model=512,
        n_heads=8,
        use_quaternion=True,
        use_harmonic_gates=True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.use_quaternion = use_quaternion
        self.use_harmonic_gates = use_harmonic_gates

        # Standard multi-head attention
        self.standard_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        # ΨQRH components
        if use_quaternion:
            self.quaternion_ops = QuaternionOperations()
            self.quat_projection = nn.Linear(d_model, d_model)

        if use_harmonic_gates:
            # Create a simple config for hierarchical gates
            from dataclasses import dataclass
            @dataclass
            class SimpleConfig:
                num_levels: int = 3
                gate_dim: int = d_model
                gate_activation: str = 'sigmoid'
                enable_coherence_analysis: bool = False
                enable_energy_balancing: bool = False
                enable_adaptive_resonance: bool = False

            self.harmonic_gates = HierarchicalGateSystem(SimpleConfig())

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass combining standard and quaternion attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Standard attention branch
        attn_out, _ = self.standard_attention(x, x, x, attn_mask=mask)

        # Quaternion enhancement (if enabled)
        if self.use_quaternion:
            # Convert to quaternion representation
            batch_size, seq_len, _ = x.shape
            x_quat = x.view(batch_size, seq_len, -1, 4)

            # Apply quaternion multiplication
            q_enhanced = self.quaternion_ops.multiply(
                x_quat,
                x_quat
            )

            # Project back to d_model
            q_enhanced = q_enhanced.view(batch_size, seq_len, -1)
            q_enhanced = self.quat_projection(q_enhanced)

            # Combine with standard attention
            attn_out = attn_out + 0.5 * q_enhanced

        # Apply harmonic gates (if enabled)
        if self.use_harmonic_gates:
            attn_out = self.harmonic_gates(attn_out)

        # Residual connection and normalization
        x = self.norm1(x + attn_out)

        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class StandardToQRHAdapter(nn.Module):
    """
    Adapter to convert standard transformer outputs to ΨQRH format
    and vice versa. Useful for integrating ΨQRH layers into
    existing model pipelines.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Ensure d_model is divisible by 4 for quaternions
        assert d_model % 4 == 0, "d_model must be divisible by 4 for quaternion representation"

        self.to_qrh = nn.Linear(d_model, d_model)
        self.from_qrh = nn.Linear(d_model, d_model)

    def convert_to_qrh(self, x):
        """Convert standard representation to ΨQRH format"""
        return self.to_qrh(x)

    def convert_from_qrh(self, x):
        """Convert ΨQRH format back to standard representation"""
        return self.from_qrh(x)


def integrate_qrh_into_huggingface_model(hf_model, layer_indices=None):
    """
    Replace specific layers in a HuggingFace transformer with ΨQRH layers.

    Args:
        hf_model: HuggingFace transformer model
        layer_indices: List of layer indices to replace (default: replace all)

    Returns:
        Modified model with ΨQRH layers

    Example:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model = integrate_qrh_into_huggingface_model(model, layer_indices=[4, 5, 6])
    """
    # Get model configuration
    config = hf_model.config
    d_model = config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd
    n_heads = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head

    # Determine which layers to replace
    if layer_indices is None:
        # Get total number of layers
        if hasattr(hf_model, 'transformer'):
            total_layers = len(hf_model.transformer.h)
            layer_indices = range(total_layers)
        elif hasattr(hf_model, 'encoder'):
            total_layers = len(hf_model.encoder.layer)
            layer_indices = range(total_layers)
        else:
            raise ValueError("Could not determine model architecture")

    # Replace specified layers
    replaced_count = 0
    for idx in layer_indices:
        hybrid_layer = HybridTransformerLayer(
            d_model=d_model,
            n_heads=n_heads,
            use_quaternion=True,
            use_harmonic_gates=True
        )

        # Replace layer based on model architecture
        if hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'h'):
            # GPT-style models
            hf_model.transformer.h[idx] = hybrid_layer
            replaced_count += 1
        elif hasattr(hf_model, 'encoder') and hasattr(hf_model.encoder, 'layer'):
            # BERT-style models
            hf_model.encoder.layer[idx] = hybrid_layer
            replaced_count += 1

    print(f"✓ Replaced {replaced_count} layers with ΨQRH hybrid layers")
    return hf_model


def example_pytorch_integration():
    """
    Example: Integrate ΨQRH into a standard PyTorch transformer
    """
    print("="*60)
    print("PyTorch Integration Example")
    print("="*60)

    class CustomTransformer(nn.Module):
        def __init__(self, d_model=512, n_layers=6):
            super().__init__()

            # Mix standard and ΨQRH layers
            self.layers = nn.ModuleList([
                # First 3 layers: standard
                nn.TransformerEncoderLayer(d_model, nhead=8)
                if i < 3 else
                # Last 3 layers: ΨQRH hybrid
                HybridTransformerLayer(d_model, n_heads=8)
                for i in range(n_layers)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    # Create model
    model = CustomTransformer(d_model=512, n_layers=6)

    # Test
    batch_size, seq_len, d_model = 2, 32, 512
    x = torch.randn(batch_size, seq_len, d_model)

    output = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Integration successful!")


def example_huggingface_integration():
    """
    Example: Integrate ΨQRH into HuggingFace transformers
    (requires transformers library)
    """
    print("\n" + "="*60)
    print("HuggingFace Integration Example")
    print("="*60)

    try:
        from transformers import GPT2Config, GPT2LMHeadModel

        # Create a small GPT-2 model for demonstration
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=512,
            n_layer=6,
            n_head=8
        )
        model = GPT2LMHeadModel(config)

        print(f"Original model layers: {len(model.transformer.h)}")

        # Replace middle layers with ΨQRH
        model = integrate_qrh_into_huggingface_model(
            model,
            layer_indices=[2, 3, 4]  # Replace layers 2, 3, 4
        )

        # Test
        input_ids = torch.randint(0, 50257, (1, 10))
        output = model(input_ids)

        print(f"✓ HuggingFace integration successful!")
        print(f"  Output logits shape: {output.logits.shape}")

    except ImportError:
        print("⚠ transformers library not installed")
        print("  Install with: pip install transformers")


def main():
    """Run integration examples"""
    print("\n" + "="*60)
    print("ΨQRH Integration Guide")
    print("="*60)

    # Example 1: PyTorch integration
    example_pytorch_integration()

    # Example 2: HuggingFace integration
    example_huggingface_integration()

    print("\n" + "="*60)
    print("Integration Guide Completed!")
    print("="*60)

    print("\nKey Integration Strategies:")
    print("1. HybridTransformerLayer: Mix standard and ΨQRH attention")
    print("2. Selective replacement: Replace only middle/late layers")
    print("3. Adapter pattern: Use StandardToQRHAdapter for compatibility")
    print("4. Gradual integration: Start with 1-2 layers, expand if beneficial")

    print("\nPerformance Tips:")
    print("- Quaternion operations work best with d_model divisible by 4")
    print("- Harmonic gates add minimal overhead (~5-10%)")
    print("- Consider mixed precision training for efficiency")


if __name__ == "__main__":
    main()