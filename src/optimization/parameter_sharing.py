"""
Parameter Sharing for ΨQRH Transformer

Implements cross-layer parameter sharing for attention weights
to reduce memory usage while maintaining performance.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List


class SharedAttentionWeights:
    """Shared attention weights across transformer layers"""

    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = (d_model * 4) // n_heads

        # Shared attention projections otimizadas
        # Usar projeções lineares regulares com reshape para eficiência
        self.q_proj = nn.Linear(d_model * 4, d_model * 4)
        self.k_proj = nn.Linear(d_model * 4, d_model * 4)
        self.v_proj = nn.Linear(d_model * 4, d_model * 4)
        self.out_proj = nn.Linear(d_model * 4, d_model * 4)

    def forward(self, x: torch.Tensor, layer_idx: int = 0):
        """Forward pass with shared weights"""
        batch_size, seq_len, _ = x.shape

        # Project to query, key, value
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        actual_head_dim = Q.size(-1) // self.n_heads
        Q = Q.view(batch_size, seq_len, self.n_heads, actual_head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, actual_head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, actual_head_dim)

        # Apply attention (simplified for demonstration)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (actual_head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Combine heads
        attention_output = attention_output.reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.out_proj(attention_output)

        return output


class SharedPsiQRHTransformer(nn.Module):
    """ΨQRH Transformer with shared attention weights"""

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dim_feedforward: int = 1024,
                 share_attention: bool = True,
                 share_ffn: bool = False):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.share_attention = share_attention
        self.share_ffn = share_ffn

        # Token embedding
        from src.architecture.psiqrh_transformer import QuaternionTokenEmbedding
        self.token_embedding = QuaternionTokenEmbedding(vocab_size, d_model)

        # Positional encoding
        from src.architecture.psiqrh_transformer import SpectralPositionalEncoding
        self.positional_encoding = SpectralPositionalEncoding(d_model)

        # Shared attention weights
        if share_attention:
            self.shared_attention = SharedAttentionWeights(d_model, n_heads)
        else:
            self.shared_attention = None

        # Transformer layers
        self.layers = nn.ModuleList([
            self._create_layer(i) for i in range(n_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model * 4, vocab_size)

    def _create_layer(self, layer_idx: int):
        """Create a transformer layer with optional weight sharing"""
        from src.architecture.psiqrh_transformer import (
            QuaternionLayerNorm,
            PsiQRHFeedForward
        )

        return SharedPsiQRHLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            shared_attention=self.shared_attention,
            layer_idx=layer_idx,
            share_attention=self.share_attention,
            share_ffn=self.share_ffn
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with energy conservation"""
        # Import energy normalization
        from src.core.utils import energy_normalize

        # Token embedding
        x = self.token_embedding(x)
        x_embedded = x  # Save for energy reference

        # Positional encoding
        x = self.positional_encoding(x)

        # Process through layers with energy conservation
        for i, layer in enumerate(self.layers):
            x_before_layer = x
            x = layer(x)
            x = energy_normalize(x_before_layer, x)

        # Output projection
        output = self.output_projection(x)

        # Final energy normalization
        output = energy_normalize(x_embedded, output)

        return output


class SharedPsiQRHLayer(nn.Module):
    """Single ΨQRH layer with shared weights"""

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dim_feedforward: int,
                 shared_attention: SharedAttentionWeights,
                 layer_idx: int,
                 share_attention: bool = True,
                 share_ffn: bool = False):
        super().__init__()

        self.d_model = d_model
        self.layer_idx = layer_idx
        self.share_attention = share_attention
        self.share_ffn = share_ffn

        # Layer normalization
        from src.architecture.psiqrh_transformer import QuaternionLayerNorm
        self.norm1 = QuaternionLayerNorm(d_model)
        self.norm2 = QuaternionLayerNorm(d_model)

        # Attention mechanism
        if share_attention and shared_attention is not None:
            self.attention = shared_attention
        else:
            from src.architecture.psiqrh_transformer import PsiQRHAttention
            self.attention = PsiQRHAttention(d_model, n_heads)

        # Feed-forward network
        if share_ffn:
            # Shared FFN would be implemented here
            from src.architecture.psiqrh_transformer import PsiQRHFeedForward
            self.ffn = PsiQRHFeedForward(d_model, dim_feedforward)
        else:
            from src.architecture.psiqrh_transformer import PsiQRHFeedForward
            self.ffn = PsiQRHFeedForward(d_model, dim_feedforward)

        # Layer scaling
        self.layer_scale_attention = nn.Parameter(torch.ones(d_model * 4))
        self.layer_scale_ffn = nn.Parameter(torch.ones(d_model * 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections"""
        # Import energy normalization
        from src.core.utils import energy_normalize

        # Self-attention with residual
        residual = x
        x = self.norm1(x)

        if self.share_attention:
            attention_out = self.attention.forward(x, self.layer_idx)
        else:
            attention_out = self.attention(x, x, x)

        attention_out = energy_normalize(x, attention_out)
        x = residual + self.layer_scale_attention * attention_out

        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        ffn_out = energy_normalize(x, ffn_out)
        x = residual + self.layer_scale_ffn * ffn_out

        return x


def test_parameter_sharing():
    """Test parameter sharing functionality"""
    print("=== Testing ΨQRH Parameter Sharing ===")

    # Create models with and without sharing
    standard_model = SharedPsiQRHTransformer(
        vocab_size=1000,
        d_model=256,
        n_layers=6,
        n_heads=8,
        share_attention=False
    )

    shared_model = SharedPsiQRHTransformer(
        vocab_size=1000,
        d_model=256,
        n_layers=6,
        n_heads=8,
        share_attention=True
    )

    # Calculate parameters
    standard_params = sum(p.numel() for p in standard_model.parameters())
    shared_params = sum(p.numel() for p in shared_model.parameters())

    print(f"Standard model parameters: {standard_params:,}")
    print(f"Shared model parameters: {shared_params:,}")
    print(f"Parameter reduction: {(1 - shared_params/standard_params)*100:.1f}%")

    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 64))

    with torch.no_grad():
        standard_output = standard_model(input_ids)
        shared_output = shared_model(input_ids)

    print(f"Standard output shape: {standard_output.shape}")
    print(f"Shared output shape: {shared_output.shape}")

    # Test energy conservation
    input_embeddings = standard_model.token_embedding(input_ids)
    input_energy = torch.sum(input_embeddings**2).item()

    standard_energy = torch.sum(standard_output**2).item()
    shared_energy = torch.sum(shared_output**2).item()

    standard_ratio = standard_energy / input_energy
    shared_ratio = shared_energy / input_energy

    print(f"Standard energy ratio: {standard_ratio:.6f}")
    print(f"Shared energy ratio: {shared_ratio:.6f}")
    print(f"Standard energy preserved: {'✅ YES' if abs(standard_ratio - 1.0) < 0.05 else '❌ NO'}")
    print(f"Shared energy preserved: {'✅ YES' if abs(shared_ratio - 1.0) < 0.05 else '❌ NO'}")

    return shared_model


if __name__ == "__main__":
    test_parameter_sharing()