import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple

from ..core.quaternion_operations import QuaternionOperations


class QuaternionTokenEmbedding(nn.Module):
    """Token embedding with quaternion representation"""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Direct quaternion embedding (4× d_model for quaternion components)
        self.embedding = nn.Embedding(vocab_size, 4 * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direct quaternion embedding
        embedded = self.embedding(x)
        return embedded


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
        self.alpha = nn.Parameter(torch.ones(d_model // 8))  # head_dim
        self.beta = nn.Parameter(torch.zeros(d_model // 8))  # head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply logarithmic phase filter
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        # Adaptive filtering - ensure proper broadcasting
        # x shape: [batch_size, seq_len, n_heads, head_dim]
        alpha_expanded = self.alpha.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        beta_expanded = self.beta.unsqueeze(0).unsqueeze(0).unsqueeze(0)

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
        self.head_dim = d_model // n_heads

        # ΨQRH-based projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Spectral filtering
        self.spectral_filter = AdaptiveSpectralFilter(d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape

        # Project to query, key, value
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply spectral attention
        attention_output = self._spectral_attention(Q, K, V)

        # Combine heads and project
        attention_output = attention_output.reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attention_output)

    def _spectral_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Spectral-based attention using ΨQRH principles"""

        # Convert to frequency domain with orthonormal FFT
        Q_fft = torch.fft.fft(Q, dim=1, norm="ortho")
        K_fft = torch.fft.fft(K, dim=1, norm="ortho")
        V_fft = torch.fft.fft(V, dim=1, norm="ortho")

        # Apply spectral correlation
        correlation = Q_fft * K_fft.conj()

        # Apply adaptive spectral filter
        filtered_correlation = self.spectral_filter(correlation)

        # Combine with value
        attention_weights = torch.fft.ifft(filtered_correlation, dim=1, norm="ortho").real
        attention_output = attention_weights * V

        return attention_output


class PsiQRHFeedForward(nn.Module):
    """Feed-forward network with ΨQRH spectral processing"""

    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()

        # Linear layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Spectral activation
        self.activation = nn.GELU()

        # Adaptive dropout
        self.dropout = nn.Dropout(0.1)

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
    """Complete ΨQRH transformer block with energy preservation"""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int):
        super().__init__()

        # ΨQRH attention
        self.self_attention = PsiQRHAttention(4 * d_model, n_heads)  # 4× for quaternion
        self.attention_norm = nn.LayerNorm(4 * d_model)

        # ΨQRH feed-forward
        self.feed_forward = PsiQRHFeedForward(4 * d_model, dim_feedforward)
        self.ffn_norm = nn.LayerNorm(4 * d_model)

        # Layer scaling
        self.layer_scale_attention = nn.Parameter(torch.ones(4 * d_model))
        self.layer_scale_ffn = nn.Parameter(torch.ones(4 * d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.attention_norm(x)
        x = self.self_attention(x, x, x)

        # Apply energy normalization after attention
        from ..core.utils import energy_normalize
        x = energy_normalize(residual, x)

        x = residual + self.layer_scale_attention * x

        # Feed-forward with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)

        # Apply energy normalization after feed-forward
        x = energy_normalize(residual, x)

        x = residual + self.layer_scale_ffn * x

        return x


class PsiQRHTransformer(nn.Module):
    """Complete ΨQRH-based transformer architecture with energy preservation"""

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dim_feedforward: int = 2048,
                 max_seq_length: int = 1024):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # ΨQRH-based components
        self.token_embedding = QuaternionTokenEmbedding(vocab_size, d_model)
        self.positional_encoding = SpectralPositionalEncoding(d_model, max_seq_length)

        # ΨQRH transformer blocks with energy preservation
        self.layers = nn.ModuleList([
            PsiQRHTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward
            ) for _ in range(n_layers)
        ])

        # Output projection (from quaternion space back to vocabulary)
        self.output_projection = nn.Linear(4 * d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original input for energy preservation
        input_ids = x

        # Embed tokens as quaternions
        x = self.token_embedding(x)

        # Apply spectral positional encoding
        x = self.positional_encoding(x)

        # Process through ΨQRH layers
        for layer in self.layers:
            x = layer(x)

        # Project back from quaternion space
        x = self.output_projection(x)

        # Apply global energy normalization
        from ..core.utils import energy_normalize
        x = energy_normalize(self.token_embedding(input_ids), x)

        return x

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