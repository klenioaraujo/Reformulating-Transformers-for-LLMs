"""
ΨQRH Spectral Attention Mechanism
==================================

Implements the spectral attention mechanism for ΨQRH transformers,
combining quaternion operations with spectral filtering in the frequency domain.

This module provides:
- SpectralAttention: Core spectral attention implementation
- QuaternionSpectralAttention: Quaternion-based spectral attention
- SpectralFilter: Frequency domain filtering with fractal-informed parameters

Mathematical Foundation:
- Spectral Attention: Attention(Q,K,V) = F⁻¹{F(k) · F{Ψ(Q) ⊗ Ψ(K) ⊗ Ψ(V)}}
- Quaternion Operations: Hamilton product with SO(4) rotations
- Spectral Filtering: F(k) = exp(iα · arctan(ln|k| + ε))

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
Project: https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from .quaternion_operations import (
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_norm,
    quaternion_normalize,
    OptimizedQuaternionOperations
)


class SpectralFilter(nn.Module):
    """
    Spectral filter for frequency domain processing.

    Implements logarithmic phase filtering with fractal-informed parameters:
    F(k) = exp(iα · arctan(ln|k| + ε))

    Where:
    - α: Filtering parameter (adaptive based on fractal dimension)
    - k: Frequency domain variable
    - ε: Numerical stability constant
    """

    def __init__(self, alpha: float = 1.0, use_windowing: bool = True,
                 window_type: str = 'hann', fractal_dimension: Optional[float] = None):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.use_windowing = use_windowing
        self.window_type = window_type
        self.fractal_dimension = fractal_dimension

        # Numerical stability
        self.epsilon = 1e-10

        # Adaptive alpha based on fractal dimension
        if fractal_dimension is not None:
            self._adapt_alpha_to_fractal(fractal_dimension)

    def _adapt_alpha_to_fractal(self, fractal_dim: float):
        """Adapt spectral filter alpha based on fractal dimension."""
        # Map fractal dimension to alpha parameter
        # Higher fractal dimension = more complex signal = stronger filtering
        alpha_fractal = 0.1 + 2.9 * (fractal_dim - 1.0) / 2.0  # Map [1,3] to [0.1,3.0]
        alpha_fractal = torch.clamp(torch.tensor(alpha_fractal), 0.1, 3.0)

        with torch.no_grad():
            self.alpha.data = alpha_fractal

    def forward(self, k_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral filtering in frequency domain.

        Args:
            k_magnitude: Frequency magnitudes [..., freq_bins]

        Returns:
            Complex filter response [..., freq_bins]
        """
        # Logarithmic phase filter: exp(iα · arctan(ln|k| + ε))
        log_k = torch.log(k_magnitude + self.epsilon)
        phase = self.alpha * torch.arctan(log_k)

        # Return complex filter response
        return torch.exp(1j * phase)

    def apply_window(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply windowing function to reduce spectral leakage."""
        if not self.use_windowing:
            return x

        if self.window_type == 'hann':
            window = torch.hann_window(seq_len, device=x.device)
        elif self.window_type == 'hamming':
            window = torch.hamming_window(seq_len, device=x.device)
        elif self.window_type == 'blackman':
            # Approximate Blackman window
            n = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            window = 0.42 - 0.5 * torch.cos(2 * math.pi * n / (seq_len - 1)) + \
                     0.08 * torch.cos(4 * math.pi * n / (seq_len - 1))
        else:
            return x  # No windowing

        # Apply window along sequence dimension
        return x * window.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


class QuaternionSpectralAttention(nn.Module):
    """
    Quaternion-based spectral attention mechanism.

    Implements: Attention(Q,K,V) = F⁻¹{F(k) · F{Ψ(Q) ⊗ Ψ(K) ⊗ Ψ(V)}}

    Where Ψ represents quaternion embeddings and ⊗ is the Hamilton product.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, alpha: float = 1.0,
                 dropout: float = 0.1, fractal_dimension: Optional[float] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.quaternion_dim = 4 * embed_dim  # 4 components per quaternion

        # Spectral filter for frequency domain processing
        self.spectral_filter = SpectralFilter(alpha=alpha, fractal_dimension=fractal_dimension)

        # Quaternion operations
        self.quaternion_ops = OptimizedQuaternionOperations()

        # Linear projections for Q, K, V (adapted for quaternion space)
        self.q_proj = nn.Linear(self.quaternion_dim, self.quaternion_dim)
        self.k_proj = nn.Linear(self.quaternion_dim, self.quaternion_dim)
        self.v_proj = nn.Linear(self.quaternion_dim, self.quaternion_dim)

        # Output projection
        self.out_proj = nn.Linear(self.quaternion_dim, self.quaternion_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _prepare_quaternion_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare quaternion states from input tensor.

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim * 4]

        Returns:
            Quaternion states [batch_size, seq_len, embed_dim, 4]
        """
        batch_size, seq_len, _ = x.shape

        # Reshape to quaternion format: [B, T, D, 4]
        x_quaternion = x.view(batch_size, seq_len, self.embed_dim, 4)

        return x_quaternion

    def _apply_spectral_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral attention mechanism.

        Args:
            Q, K, V: Quaternion tensors [batch_size, seq_len, embed_dim, 4]

        Returns:
            Attention output [batch_size, seq_len, embed_dim, 4]
        """
        batch_size, seq_len, embed_dim, _ = Q.shape

        # Reshape for multi-head attention: [B, T, H, D_head, 4]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim, 4)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim, 4)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim, 4)

        # Spectral attention computation
        attention_outputs = []

        for head in range(self.num_heads):
            Q_head = Q[:, :, head, :, :]  # [B, T, D_head, 4]
            K_head = K[:, :, head, :, :]  # [B, T, D_head, 4]
            V_head = V[:, :, head, :, :]  # [B, T, D_head, 4]

            # Compute quaternion attention scores
            # Q_head ⊗ K_head* (conjugate for similarity)
            K_conj = quaternion_conjugate(K_head)
            scores = self._quaternion_similarity(Q_head, K_conj)  # [B, T, T, D_head]

            # Apply scaling
            scores = scores * self.scale

            # Spectral filtering in frequency domain
            scores_fft = torch.fft.fft2(scores, dim=(-2, -1))  # [B, T, T, D_head]

            # Get frequency magnitudes for filtering
            freq_magnitude = torch.abs(scores_fft)

            # Apply spectral filter
            filter_response = self.spectral_filter(freq_magnitude)
            scores_filtered_fft = scores_fft * filter_response

            # Inverse FFT back to time domain
            scores_filtered = torch.fft.ifft2(scores_filtered_fft, dim=(-2, -1)).real

            # Apply softmax along sequence dimension
            attention_weights = F.softmax(scores_filtered, dim=-2)  # [B, T, T, D_head]

            # Apply dropout
            attention_weights = self.dropout(attention_weights)

            # Apply attention to values: weighted sum of V
            # attention_weights @ V_head
            context = torch.einsum('bijd,bjdd->bidd', attention_weights, V_head)  # [B, T, D_head, 4]

            attention_outputs.append(context)

        # Concatenate heads: [B, T, H * D_head, 4] = [B, T, D, 4]
        output = torch.cat(attention_outputs, dim=-2)

        return output

    def _quaternion_similarity(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Compute quaternion similarity scores.

        Args:
            q1, q2: Quaternion tensors [batch_size, seq_len, head_dim, 4]

        Returns:
            Similarity scores [batch_size, seq_len, seq_len, head_dim]
        """
        batch_size, seq_len1, head_dim, _ = q1.shape
        _, seq_len2, _, _ = q2.shape

        # Compute quaternion products for all pairs
        similarity_scores = []

        for i in range(seq_len1):
            for j in range(seq_len2):
                # Hamilton product: q1[i] ⊗ q2[j]
                product_ij = quaternion_multiply(
                    q1[:, i:i+1, :, :].expand(-1, seq_len2, -1, -1),
                    q2[:, j:j+1, :, :].expand(-1, seq_len1, -1, -1).transpose(1, 0)
                )

                # Take real part of the product as similarity score
                score_ij = product_ij.real.mean(dim=-1)  # [B, seq_len, head_dim]
                similarity_scores.append(score_ij)

        # Stack all scores: [B, T, T, D_head]
        scores = torch.stack(similarity_scores, dim=2).view(batch_size, seq_len1, seq_len2, head_dim)

        return scores

    def forward(self, query: torch.Tensor, key: torch.Tensor = None,
                value: torch.Tensor = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of quaternion spectral attention.

        Args:
            query: Query tensor [batch_size, seq_len, embed_dim * 4]
            key: Key tensor [batch_size, seq_len, embed_dim * 4] (optional)
            value: Value tensor [batch_size, seq_len, embed_dim * 4] (optional)
            attn_mask: Attention mask (optional)

        Returns:
            Attention output [batch_size, seq_len, embed_dim * 4]
        """
        # Set key and value to query if not provided (self-attention)
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, seq_len, _ = query.shape

        # Linear projections
        Q = self.q_proj(query)  # [B, T, 4*D]
        K = self.k_proj(key)    # [B, T, 4*D]
        V = self.v_proj(value)  # [B, T, 4*D]

        # Prepare quaternion states
        Q_quat = self._prepare_quaternion_states(Q)  # [B, T, D, 4]
        K_quat = self._prepare_quaternion_states(K)  # [B, T, D, 4]
        V_quat = self._prepare_quaternion_states(V)  # [B, T, D, 4]

        # Apply spectral attention
        attention_output = self._apply_spectral_attention(Q_quat, K_quat, V_quat)  # [B, T, D, 4]

        # Reshape back to linear format
        attention_flat = attention_output.view(batch_size, seq_len, self.quaternion_dim)  # [B, T, 4*D]

        # Output projection
        output = self.out_proj(attention_flat)  # [B, T, 4*D]

        return output


class SpectralAttention(nn.Module):
    """
    Standard spectral attention mechanism (non-quaternion version).

    Implements spectral filtering in frequency domain for improved
    attention computation with O(n log n) complexity.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, alpha: float = 1.0,
                 dropout: float = 0.1, fractal_dimension: Optional[float] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Spectral filter
        self.spectral_filter = SpectralFilter(alpha=alpha, fractal_dimension=fractal_dimension)

        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor = None,
                value: torch.Tensor = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of spectral attention.

        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim] (optional)
            value: Value tensor [batch_size, seq_len, embed_dim] (optional)
            attn_mask: Attention mask (optional)

        Returns:
            Attention output [batch_size, seq_len, embed_dim]
        """
        # Set key and value to query if not provided
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, seq_len, _ = query.shape

        # Linear projections
        Q = self.q_proj(query)  # [B, T, D]
        K = self.k_proj(key)    # [B, T, D]
        V = self.v_proj(value)  # [B, T, D]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D_head]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D_head]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D_head]

        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        # Apply spectral filtering in frequency domain
        scores_fft = torch.fft.fft2(scores, dim=(-2, -1))  # [B, H, T, T]

        # Get frequency magnitudes
        freq_magnitude = torch.abs(scores_fft)

        # Apply spectral filter
        filter_response = self.spectral_filter(freq_magnitude)
        scores_filtered_fft = scores_fft * filter_response

        # Inverse FFT
        scores_filtered = torch.fft.ifft2(scores_filtered_fft, dim=(-2, -1)).real

        # Apply attention mask if provided
        if attn_mask is not None:
            scores_filtered = scores_filtered.masked_fill(attn_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        # Apply softmax
        attention_weights = F.softmax(scores_filtered, dim=-1)  # [B, H, T, T]

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [B, H, T, D_head]

        # Concatenate heads and put back in sequence dimension
        context = context.transpose(1, 2).contiguous()  # [B, T, H, D_head]
        context = context.view(batch_size, seq_len, self.embed_dim)  # [B, T, D]

        # Output projection
        output = self.out_proj(context)  # [B, T, D]

        return output


# Utility functions for spectral attention
def create_spectral_attention_layer(embed_dim: int, num_heads: int = 8,
                                   use_quaternion: bool = True, alpha: float = 1.0,
                                   fractal_dimension: Optional[float] = None) -> nn.Module:
    """
    Factory function to create spectral attention layers.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        use_quaternion: Whether to use quaternion-based attention
        alpha: Spectral filtering parameter
        fractal_dimension: Fractal dimension for adaptive filtering

    Returns:
        Spectral attention layer
    """
    if use_quaternion:
        return QuaternionSpectralAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            alpha=alpha,
            fractal_dimension=fractal_dimension
        )
    else:
        return SpectralAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            alpha=alpha,
            fractal_dimension=fractal_dimension
        )


def adapt_spectral_filter_to_fractal_dimension(filter_layer: SpectralFilter,
                                              fractal_dim: float) -> None:
    """
    Adapt spectral filter parameters based on fractal dimension.

    Args:
        filter_layer: Spectral filter to adapt
        fractal_dim: Computed fractal dimension
    """
    filter_layer._adapt_alpha_to_fractal(fractal_dim)


# Export main classes
__all__ = [
    'SpectralFilter',
    'SpectralAttention',
    'QuaternionSpectralAttention',
    'create_spectral_attention_layer',
    'adapt_spectral_filter_to_fractal_dimension'
]