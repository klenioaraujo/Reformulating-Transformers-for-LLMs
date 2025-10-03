#!/usr/bin/env python3
"""
Rotational Quaternion 4x Test
=============================

Test script to evaluate the efficiency of rotational quaternion operations
compared to standard quaternion operations in the ΨQRH transformer.

This test implements a 4x rotational quaternion approach where quaternions
are used to represent rotations in 4D space, potentially offering better
efficiency than standard quaternion operations.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def create_rotational_quaternion_layer(d_model: int, out_features: int) -> nn.Module:
    """
    Create a rotational quaternion layer that uses 4x quaternion rotations.

    This implementation uses quaternion rotations to transform the input,
    potentially reducing parameter count while maintaining expressivity.
    """

    class RotationalQuaternionLayer(nn.Module):
        def __init__(self, d_model: int, out_features: int):
            super().__init__()
            self.d_model = d_model
            self.out_features = out_features

            # ULTRA-OPTIMIZED: Single rotation quaternion per output feature
            # Total parameters: out_features * 4 + out_features * d_model
            # This is much simpler and more efficient

            # One rotation quaternion per output feature
            self.rotation_quaternions = nn.Parameter(
                torch.randn(out_features, 4) * 0.01
            )

            # Linear combination weights: d_model -> out_features
            # This replaces the heavy matrix multiplication
            self.linear_weights = nn.Parameter(
                torch.randn(out_features, d_model) * 0.01
            )

            # Learnable scaling factors
            self.scales = nn.Parameter(torch.ones(out_features))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Apply ultra-optimized rotational quaternion transformation.

            Args:
                x: Input tensor of shape [batch_size, seq_len, d_model * 4]

            Returns:
                Output tensor of shape [batch_size, seq_len, out_features * 4]
            """
            batch_size, seq_len, _ = x.shape

            # Reshape input to work with quaternions
            # x shape: [batch_size, seq_len, d_model * 4]
            # Reshape to: [batch_size, seq_len, d_model, 4]
            x_reshaped = x.view(batch_size, seq_len, self.d_model, 4)

            # Step 1: Linear combination across d_model dimension
            # x_reshaped: [batch, seq, d_model, 4]
            # linear_weights: [out_features, d_model]
            # Output: [batch, seq, out_features, 4]
            combined = torch.einsum('bsdi,od->bsoi', x_reshaped, self.linear_weights)

            # Step 2: Apply rotation quaternions
            # Normalize rotation quaternions
            rotation_norms = torch.norm(self.rotation_quaternions, dim=-1, keepdim=True) + 1e-8
            rotation_q_norm = self.rotation_quaternions / rotation_norms  # [out_features, 4]

            # Apply element-wise rotation (simplified quaternion product)
            # combined: [batch, seq, out_features, 4]
            # rotation_q_norm: [out_features, 4]
            rotated = combined * rotation_q_norm.unsqueeze(0).unsqueeze(0)

            # Step 3: Apply scaling
            rotated_scaled = rotated * self.scales.view(1, 1, -1, 1)

            # Flatten output: [batch, seq, out_features * 4]
            output = rotated_scaled.reshape(batch_size, seq_len, self.out_features * 4)

            return output

    return RotationalQuaternionLayer(d_model, out_features)


def create_rotational_psiqrh_transformer(vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                                        dim_feedforward: int = 512, max_seq_length: int = 1024):
    """
    Create a ΨQRH transformer that uses rotational quaternion operations.
    """

    class RotationalPsiQRHTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers, n_heads, dim_feedforward, max_seq_length):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.n_layers = n_layers

            # Token embedding
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.quaternion_projection = nn.Linear(d_model, d_model * 4)

            # Use rotational layers instead of standard quaternion layers
            self.rotational_layers = nn.ModuleList([
                create_rotational_quaternion_layer(d_model, d_model)
                for _ in range(n_layers)
            ])

            # Output projection
            self.output_projection = nn.Linear(d_model * 4, vocab_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Embed tokens
            x = self.token_embedding(x)
            x = self.quaternion_projection(x)

            # Apply rotational layers
            for layer in self.rotational_layers:
                x = layer(x) + x  # Residual connection

            # Project back to vocabulary
            output = self.output_projection(x)

            return output

    return RotationalPsiQRHTransformer(vocab_size, d_model, n_layers, n_heads, dim_feedforward, max_seq_length)


def test_rotational_quaternion_efficiency():
    """
    Test the efficiency of rotational quaternion operations.
    """
    print("🧪 ROTATIONAL QUATERNION 4X EFFICIENCY TEST")
    print("=" * 60)

    # Test configurations
    configs = [
        {"d_model": 64, "n_layers": 2, "n_heads": 4},
        {"d_model": 128, "n_layers": 4, "n_heads": 4},
        {"d_model": 256, "n_layers": 6, "n_heads": 8}
    ]

    vocab_size = 5000
    batch_size = 8
    seq_len = 64

    for config in configs:
        d_model = config["d_model"]
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]

        print(f"\n📊 Testing Configuration: d_model={d_model}, layers={n_layers}, heads={n_heads}")
        print("-" * 50)

        # Create models
        try:
            from src.architecture.psiqrh_transformer import PsiQRHTransformer
            standard_psiqrh = PsiQRHTransformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads
            )

            rotational_psiqrh = create_rotational_psiqrh_transformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads
            )

            # Count parameters
            standard_params = sum(p.numel() for p in standard_psiqrh.parameters())
            rotational_params = sum(p.numel() for p in rotational_psiqrh.parameters())

            # Test memory usage
            input_tensor = torch.randint(0, vocab_size - 100, (batch_size, seq_len))

            # Standard ΨQRH memory
            import psutil
            import os
            process = psutil.Process(os.getpid())

            memory_before = process.memory_info().rss / (1024 ** 2)
            with torch.no_grad():
                standard_output = standard_psiqrh(input_tensor)
            memory_after_standard = process.memory_info().rss / (1024 ** 2)
            standard_memory = memory_after_standard - memory_before

            # Rotational ΨQRH memory
            memory_before = process.memory_info().rss / (1024 ** 2)
            with torch.no_grad():
                rotational_output = rotational_psiqrh(input_tensor)
            memory_after_rotational = process.memory_info().rss / (1024 ** 2)
            rotational_memory = memory_after_rotational - memory_before

            # Calculate efficiency metrics
            param_ratio = rotational_params / standard_params
            memory_ratio = rotational_memory / standard_memory

            param_change_percent = (1 - param_ratio) * 100
            memory_change_percent = (1 - memory_ratio) * 100

            print(f"Standard ΨQRH:")
            print(f"  Parameters: {standard_params:,}")
            print(f"  Memory: {standard_memory:.2f} MB")

            print(f"Rotational ΨQRH:")
            print(f"  Parameters: {rotational_params:,}")
            print(f"  Memory: {rotational_memory:.2f} MB")

            print(f"\n📈 Efficiency Comparison:")
            if param_change_percent > 0:
                print(f"  ✅ Parameter Efficiency: {param_change_percent:.2f}% REDUCTION")
            else:
                print(f"  ❌ Parameter Inefficiency: {-param_change_percent:.2f}% INCREASE")

            if memory_change_percent > 0:
                print(f"  ✅ Memory Efficiency: {memory_change_percent:.2f}% REDUCTION")
            else:
                print(f"  ❌ Memory Inefficiency: {-memory_change_percent:.2f}% INCREASE")

            print(f"  Ratio: {param_ratio:.2f}x parameters, {memory_ratio:.2f}x memory")

            # Cleanup
            del standard_psiqrh, rotational_psiqrh, standard_output, rotational_output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"❌ Error testing configuration: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_rotational_quaternion_efficiency()