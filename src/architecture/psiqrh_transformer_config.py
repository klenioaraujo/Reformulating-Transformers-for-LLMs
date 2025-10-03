import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Optional, Dict, Any

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

        # Generate position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)

        # Apply spectral encoding
        encoding = torch.sin(positions * self.frequencies.unsqueeze(0))

        # Expand to match input dimensions
        encoding = encoding.repeat(1, 4)

        return x + encoding


class EnergyNormalizer(nn.Module):
    """Energy normalization for ΨQRH stability"""

    def __init__(self, target_ratio: float = 1.0):
        super().__init__()
        self.target_ratio = target_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate energy
        energy = torch.mean(x ** 2)

        # Normalize to target ratio
        scale = torch.sqrt(self.target_ratio / (energy + 1e-8))

        return x * scale


class PsiQRHAttention(nn.Module):
    """Attention mechanism using ΨQRH spectral operations"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = (d_model * 4) // n_heads  # Quaternion expands by 4

        # ΨQRH-based projections
        self.q_proj = QuaternionLinear(d_model, d_model)
        self.k_proj = QuaternionLinear(d_model, d_model)
        self.v_proj = QuaternionLinear(d_model, d_model)

        # Spectral filtering
        self.spectral_filter = AdaptiveSpectralFilter(d_model * 4)

        # Intermediate projection to reduce from n_heads * head_dim to d_model * 4
        # QuaternionLinear expands by 4, so we need to divide by 4
        self.intermediate_proj = QuaternionLinear(d_model * 4 * n_heads // 2, d_model)

        # Final projection to reduce from d_model * 4 to d_model * 4
        self.final_proj = QuaternionLinear(d_model * 4, d_model)

        # Output projection
        self.out_proj = QuaternionLinear(d_model * 4, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape

        # Project to quaternion space
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head - quaternion already expands by 4
        # Calculate actual head_dim based on the projected dimension
        actual_head_dim = Q.size(-1) // self.n_heads
        Q = Q.view(batch_size, seq_len, self.n_heads, actual_head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, actual_head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, actual_head_dim)

        # Apply spectral attention
        attention_output = self._spectral_attention(Q, K, V)

        # Combine heads and project
        # We have n_heads * head_dim elements per position
        # Need to reduce to d_model * 4 elements per position (quaternion expanded)
        attention_output = attention_output.reshape(batch_size, seq_len, -1)
        attention_output = self.intermediate_proj(attention_output)
        attention_output = self.final_proj(attention_output)
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
        self.d_model = d_model

        # ΨQRH-based feed-forward
        self.ff1 = QuaternionLinear(d_model, dim_feedforward)
        self.ff2 = QuaternionLinear(dim_feedforward, d_model)

        # Spectral activation
        self.activation = SpectralActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(self.activation(self.ff1(x)))


class PsiQRHTransformerBlock(nn.Module):
    """Single ΨQRH transformer block"""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int,
                 fractal_analysis_freq: int = 1000):
        super().__init__()
        self.d_model = d_model

        # ΨQRH components
        self.self_attention = PsiQRHAttention(d_model, n_heads)
        self.feed_forward = PsiQRHFeedForward(d_model, dim_feedforward)

        # Layer normalization
        self.norm1 = QuaternionLayerNorm(d_model)
        self.norm2 = QuaternionLayerNorm(d_model)

        # Fractal analysis
        self.fractal_analyzer = RealTimeFractalAnalyzer(
            window_size=fractal_analysis_freq
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_output = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        # Fractal analysis
        self.fractal_analyzer.analyze(x)

        return x


class PsiQRHTransformer(nn.Module):
    """Complete ΨQRH transformer with configuration support"""

    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None):
        super().__init__()

        # Load configuration
        if config_path:
            with open(config_path) as f:
                config = json.load(f)
        elif config_dict:
            config = config_dict
        else:
            raise ValueError("Configuration not provided")

        self.config = config

        # Extract parameters
        self.vocab_size = config["vocab_size"]
        self.d_model = config["d_model"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.dim_feedforward = config["dim_feedforward"]
        self.max_seq_len = config["max_seq_len"]

        # ΨQRH-based components with conditional initialization
        if config["quaternion"]["use_quaternion_embeddings"]:
            self.token_embedding = QuaternionTokenEmbedding(
                self.vocab_size, self.d_model
            )
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)

        if config["spectral"]["use_spectral_positional_encoding"]:
            self.positional_encoding = SpectralPositionalEncoding(
                self.d_model, self.max_seq_len
            )
        else:
            # Standard sinusoidal positional encoding
            self.positional_encoding = None

        if config["energy_control"]["enable_energy_normalization"]:
            self.energy_normalizer = EnergyNormalizer(
                target_ratio=config["energy_control"]["target_ratio"]
            )
        else:
            self.energy_normalizer = None

        # ΨQRH transformer blocks
        self.layers = nn.ModuleList([
            PsiQRHTransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dim_feedforward=self.dim_feedforward,
                fractal_analysis_freq=config["fractal_analysis"]["window_size"]
            ) for _ in range(self.n_layers)
        ])

        # Output projection (from quaternion space back to vocabulary)
        self.output_projection = QuaternionLinear(self.d_model, self.vocab_size)

        # Adaptive spectral dropout
        if config["optimization"]["use_adaptive_spectral_dropout"]:
            self.spectral_dropout = AdaptiveSpectralDropout(
                p=config["optimization"]["dropout_p"],
                adaptive_threshold=config["optimization"]["adaptive_threshold"]
            )
        else:
            self.spectral_dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token embedding
        x = self.token_embedding(x)

        # Positional encoding
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        # Energy normalization
        if self.energy_normalizer is not None:
            x = self.energy_normalizer(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Adaptive spectral dropout
        if self.spectral_dropout is not None:
            x = self.spectral_dropout(x)

        # Output projection
        output = self.output_projection(x)

        return output

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for validation"""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "dim_feedforward": self.dim_feedforward,
            "max_seq_len": self.max_seq_len,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "config": self.config
        }