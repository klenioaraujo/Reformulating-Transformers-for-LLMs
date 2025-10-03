"""
ΨQRH Transformer Architecture

Main transformer implementation integrating quaternionic operations,
spectral analysis, and fractal consciousness metrics.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
Project: https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple

from ..core.quaternion_operations import (
    QuaternionLinear,
    QuaternionLayerNorm,
    SpectralActivation,
    AdaptiveSpectralDropout,
    RealTimeFractalAnalyzer
)


class QuaternionTokenEmbedding(nn.Module):
    """Token embedding with quaternion representation"""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Standard embedding + quaternion projection
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.quaternion_projection = nn.Linear(d_model, 4 * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard embedding
        embedded = self.embedding(x)

        # Project to quaternion space
        quaternion_embedded = self.quaternion_projection(embedded)

        return quaternion_embedded


class SpectralPositionalEncoding(nn.Module):
    """Positional encoding using spectral decomposition"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Learnable frequency components
        self.frequencies = nn.Parameter(
            torch.randn(d_model // 4) * 2 * math.pi
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Generate spectral positional encoding
        positions = torch.arange(seq_len, device=x.device).float()

        # Apply frequency modulation
        spectral_encoding = torch.zeros_like(x)
        for i, freq in enumerate(self.frequencies):
            phase = positions * freq
            spectral_encoding[:, :, i*4:(i+1)*4] = torch.stack([
                torch.cos(phase), torch.sin(phase),
                torch.cos(phase * 1.5), torch.sin(phase * 1.5)
            ], dim=-1)

        return x + spectral_encoding


class AdaptiveSpectralFilter(nn.Module):
    """Adaptive spectral filter for ΨQRH attention"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Learnable filter parameters - match the head dimension
        # head_dim = (d_model * 4) // n_heads, but we need to handle variable n_heads
        # Use a larger dimension that can broadcast to common head dimensions
        self.alpha = nn.Parameter(torch.ones(256))  # head_dim
        self.beta = nn.Parameter(torch.zeros(256))  # head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply logarithmic phase filter
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        # Adaptive filtering - ensure proper broadcasting
        # Get the actual size of the last dimension
        last_dim = x.size(-1)

        # Slice or repeat parameters to match the input size
        if last_dim <= self.alpha.size(0):
            alpha_slice = self.alpha[:last_dim]
            beta_slice = self.beta[:last_dim]
        else:
            # Repeat parameters if input is larger
            repeat_factor = (last_dim + self.alpha.size(0) - 1) // self.alpha.size(0)
            alpha_slice = self.alpha.repeat(repeat_factor)[:last_dim]
            beta_slice = self.beta.repeat(repeat_factor)[:last_dim]

        # Expand to match input dimensions
        alpha_expanded = alpha_slice.view(1, 1, 1, -1)
        beta_expanded = beta_slice.view(1, 1, 1, -1)

        filtered_magnitude = magnitude * torch.sigmoid(alpha_expanded)
        filtered_phase = phase + beta_expanded

        # Reconstruct complex tensor
        filtered_x = filtered_magnitude * torch.exp(1j * filtered_phase)

        # Preserve Parseval by normalizing energy
        input_energy = torch.sum(torch.abs(x)**2)
        output_energy = torch.sum(torch.abs(filtered_x)**2)

        # Avoid division by zero
        if output_energy > 1e-8:
            scale = torch.sqrt(input_energy / output_energy)
            filtered_x = filtered_x * scale

        return filtered_x

    def update_alpha(self, new_alpha: torch.Tensor):
        """Update alpha parameter based on fractal analysis"""
        with torch.no_grad():
            self.alpha.data = 0.9 * self.alpha.data + 0.1 * new_alpha


class PsiQRHAttention(nn.Module):
    """Attention mechanism using ΨQRH spectral operations"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = (d_model * 4) // n_heads  # Quaternion expands by 4

        # ΨQRH-based projections - quaternion already expands by 4
        self.q_proj = QuaternionLinear(d_model, d_model)
        self.k_proj = QuaternionLinear(d_model, d_model)
        self.v_proj = QuaternionLinear(d_model, d_model)

        # Spectral filtering
        self.spectral_filter = AdaptiveSpectralFilter(d_model * 4)

        # Single output projection to combine heads and maintain quaternion dimensions
        self.out_proj = QuaternionLinear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape

        # Project to quaternion space
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head attention
        actual_head_dim = Q.size(-1) // self.n_heads
        Q = Q.view(batch_size, seq_len, self.n_heads, actual_head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, actual_head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, actual_head_dim)

        # Apply spectral attention
        attention_output = self._spectral_attention(Q, K, V)

        # Combine heads and maintain quaternion dimensions
        attention_output = attention_output.reshape(batch_size, seq_len, -1)

        # Ensure output has correct dimensions (d_model * 4)
        if attention_output.size(-1) != self.d_model * 4:
            # If dimensions don't match, use a temporary linear layer to adjust
            if not hasattr(self, '_dim_adjuster'):
                self._dim_adjuster = nn.Linear(attention_output.size(-1), self.d_model * 4).to(attention_output.device)
            attention_output = self._dim_adjuster(attention_output)

        return self.out_proj(attention_output)

    def _spectral_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Spectral-based attention using ΨQRH principles"""

        # Convert to frequency domain
        Q_fft = torch.fft.fft(Q, dim=1)
        K_fft = torch.fft.fft(K, dim=1)
        V_fft = torch.fft.fft(V, dim=1)

        # Apply spectral correlation
        correlation = Q_fft * K_fft.conj()

        # Apply adaptive spectral filter
        filtered_correlation = self.spectral_filter(correlation)

        # Combine with value
        attention_weights = torch.fft.ifft(filtered_correlation, dim=1).real
        attention_output = attention_weights * V

        return attention_output


class PsiQRHFeedForward(nn.Module):
    """Feed-forward network with ΨQRH spectral processing"""

    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()

        # Quaternion-based linear layers
        self.linear1 = QuaternionLinear(d_model, dim_feedforward)
        self.linear2 = QuaternionLinear(dim_feedforward, d_model)

        # Spectral activation
        self.activation = SpectralActivation()

        # Adaptive dropout
        self.dropout = AdaptiveSpectralDropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First linear transformation
        x = self.linear1(x)

        # Spectral activation
        x = self.activation(x)

        # Adaptive dropout
        x = self.dropout(x)

        # Second linear transformation
        x = self.linear2(x)

        return x


class PsiQRHTransformerBlock(nn.Module):
    """Complete ΨQRH transformer block"""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, fractal_analysis_freq: int):
        super().__init__()

        # ΨQRH attention
        self.self_attention = PsiQRHAttention(d_model, n_heads)
        self.attention_norm = QuaternionLayerNorm(d_model)

        # ΨQRH feed-forward
        self.feed_forward = PsiQRHFeedForward(d_model, dim_feedforward)
        self.ffn_norm = QuaternionLayerNorm(d_model)

        # Fractal analysis
        self.fractal_analyzer = RealTimeFractalAnalyzer()

        # Layer scaling - adjusted for quaternion dimensions
        self.layer_scale_attention = nn.Parameter(torch.ones(d_model * 4))
        self.layer_scale_ffn = nn.Parameter(torch.ones(d_model * 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Import energy normalization from BKP implementation
        from ..core.utils import energy_normalize

        # Self-attention with residual and energy conservation
        residual = x
        x = self.attention_norm(x)
        attention_out = self.self_attention(x, x, x)

        # Apply energy normalization to preserve energy conservation
        attention_out = energy_normalize(x, attention_out)
        x = residual + self.layer_scale_attention * attention_out

        # Feed-forward with residual and energy conservation
        residual = x
        x = self.ffn_norm(x)
        ffn_out = self.feed_forward(x)

        # Apply energy normalization to preserve energy conservation
        ffn_out = energy_normalize(x, ffn_out)
        x = residual + self.layer_scale_ffn * ffn_out

        # Real-time fractal analysis
        fractal_metrics = self.fractal_analyzer.analyze(x)
        self._adapt_parameters(fractal_metrics)

        return x

    def _adapt_parameters(self, fractal_metrics: Dict):
        """Adapt parameters based on fractal analysis"""
        # Update spectral filter parameters
        new_alpha = self._map_fractal_to_alpha(fractal_metrics['dimension'])
        self.self_attention.spectral_filter.update_alpha(new_alpha)

    def _map_fractal_to_alpha(self, dimension: torch.Tensor) -> torch.Tensor:
        """Map fractal dimension to spectral filter alpha parameter"""
        # Higher dimension = more complex signal = stronger filtering
        return torch.sigmoid(dimension - 1.5)  # Map to [0, 1] range


class PsiQRHTransformer(nn.Module):
    """Complete ΨQRH-based transformer architecture"""

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dim_feedforward: int = 2048,
                 max_seq_length: int = 1024,
                 fractal_analysis_freq: int = 1000,
                 quaternion_multiplier: int = 4):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.quaternion_multiplier = quaternion_multiplier

        # ΨQRH-based components
        self.token_embedding = QuaternionTokenEmbedding(vocab_size, d_model)
        self.positional_encoding = SpectralPositionalEncoding(d_model, max_seq_length)

        # ΨQRH transformer blocks
        self.layers = nn.ModuleList([
            PsiQRHTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                fractal_analysis_freq=fractal_analysis_freq
            ) for _ in range(n_layers)
        ])

        # Adaptive fractal controller (placeholder - would be implemented in optimization module)
        # self.fractal_controller = AdaptiveFractalController(
        #     window_size=fractal_analysis_freq
        # )

        # Output projection (from quaternion space back to vocabulary)
        # Use regular linear to go from quaternion space (d_model * quaternion_multiplier) to vocab_size
        self.output_projection = nn.Linear(d_model * quaternion_multiplier, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Import energy normalization from BKP implementation
        from ..core.utils import energy_normalize

        # Embed tokens as quaternions
        x = self.token_embedding(x)
        x_embedded = x  # Save embedded representation for energy reference

        # Apply spectral positional encoding
        x = self.positional_encoding(x)

        # Process through ΨQRH layers with energy conservation
        for i, layer in enumerate(self.layers):
            x_before_layer = x
            x = layer(x)

            # Apply energy conservation after each layer (from BKP)
            x = energy_normalize(x_before_layer, x)

            # Adaptive fractal analysis and parameter adjustment (placeholder)
            # if i % self.fractal_analysis_freq == 0:
            #     self.fractal_controller.update_parameters(x, layer)

        # Project back from quaternion space
        output = self.output_projection(x)

        # Final energy normalization to maintain overall energy conservation
        output = energy_normalize(x_embedded, output)

        return output

    def get_model_info(self) -> Dict:
        """Get information about the model architecture"""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.layers[0].self_attention.n_heads if self.layers else 0,
            "dim_feedforward": self.layers[0].feed_forward.linear1.out_features if self.layers else 0,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "architecture": "ΨQRH Transformer"
        }