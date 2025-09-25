#!/usr/bin/env python3
"""
ΨQRH Core Layer - Technical Usage Example

This example demonstrates the basic usage of the ΨQRH architecture
for reformulating attention mechanisms in transformers using
quaternion-based spectral processing.
"""

import torch
import yaml
from src.core.ΨQRH import QRHFactory
from src.core.qrh_layer import QRHConfig
from src.core.negentropy_transformer_block import NegentropyTransformerBlock

def main():
    """Main demonstration of ΨQRH usage."""

    print("ΨQRH Core Layer - Technical Example")
    print("=" * 40)

    # Configuration for ΨQRH layer
    config = QRHConfig(
        embed_dim=64,              # Base embedding dimension
        alpha=1.0,                 # Spectral filter strength
        theta_left=0.1,           # Left quaternion rotation angle
        omega_left=0.05,          # Left quaternion omega parameter
        phi_left=0.02,            # Left quaternion phi parameter
        theta_right=0.08,         # Right quaternion rotation angle
        omega_right=0.03,         # Right quaternion omega parameter
        phi_right=0.015,          # Right quaternion phi parameter
        use_learned_rotation=True, # Enable learnable quaternion parameters
        use_windowing=True,       # Enable spectral windowing
        window_type='hann',       # Window type for spectral processing
        device='cpu'              # Computation device
    )

    # Initialize ΨQRH layer
    from src.core.qrh_layer import QRHLayer
    qrh_layer = QRHLayer(config)

    # Create sample input tensor
    # Shape: [batch_size, sequence_length, 4 * embed_dim]
    # Factor of 4 accounts for quaternion representation (w, x, y, z)
    batch_size, seq_len = 2, 32
    input_tensor = torch.randn(batch_size, seq_len, 4 * config.embed_dim)

    print(f"Input shape: {input_tensor.shape}")

    # Forward pass through ΨQRH layer
    with torch.no_grad():
        output = qrh_layer(input_tensor)

    print(f"Output shape: {output.shape}")
    print(f"✅ ΨQRH layer processed successfully")

    # Demonstrate Negentropy Transformer Block
    print("\nNegentropy Transformer Block Example:")
    print("-" * 40)

    # Standard transformer dimensions
    d_model = 512
    transformer_input = torch.randn(batch_size, seq_len, d_model)

    # Initialize Negentropy Transformer Block
    negentropy_block = NegentropyTransformerBlock(
        d_model=d_model,
        nhead=8,
        qrh_embed_dim=64,
        alpha=1.0,
        use_learned_rotation=True,
        enable_gate=True
    )

    # Forward pass
    with torch.no_grad():
        block_output, metrics = negentropy_block(transformer_input)

    print(f"Transformer input shape: {transformer_input.shape}")
    print(f"Transformer output shape: {block_output.shape}")
    print(f"✅ Negentropy Transformer Block processed successfully")

    # Display key metrics
    if metrics:
        print(f"Processing metrics: {list(metrics.keys())}")

    print("\n🎉 ΨQRH demonstration completed successfully!")

if __name__ == "__main__":
    main()