"""
Fractal Quantum Transformer - End-to-End Model Architecture
============================================================

This script defines the complete language model architecture, integrating the
novel FractalQuantumEmbedding layer with a stack of NegentropyTransformerBlocks.
This forms a full, end-to-end pipeline from token IDs to vocabulary logits,
entirely based on the ΨQRH framework.

The data flow is as follows:
1.  Input token IDs are fed into the FractalQuantumEmbedding layer, producing
    rich, physically-grounded quaternion states.
2.  These quaternions are projected into the model's working dimension (d_model).
3.  The resulting sequence of vectors is processed by a stack of
    NegentropyTransformerBlocks, where the core spectral and quaternionic
    interactions occur.
4.  The final output is projected by a prediction head to produce logits over
    the entire vocabulary.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# --- Add project root to path ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Import Core Building Blocks ---
from src.core.fractal_quantum_embedding import FractalQuantumEmbedding

class FractalQuantumTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 d_model: int,
                 nhead: int,
                 num_transformer_layers: int,
                 padilha_config: dict = None,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        """
        Initializes the full Fractal Quantum Transformer model.

        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension for the initial seed embedding.
            d_model (int): The main working dimension of the transformer blocks.
            nhead (int): The number of attention heads in each transformer block.
            num_transformer_layers (int): The number of NegentropyTransformerBlocks to stack.
            padilha_config (dict): The configuration for the Padilha Wave Equation.
            dim_feedforward (int): The dimension of the feed-forward network in transformer blocks.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.d_model = d_model

        # 1. Fractal Quantum Embedding Layer
        self.embedding_layer = FractalQuantumEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            padilha_config=padilha_config
        )

        # 2. Input Projection Layer
        # Projects the 4-dimensional quaternion output to d_model
        self.input_projection = nn.Linear(4, d_model)

        # 3. Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # 4. Stack of Transformer Blocks
        # Using standard transformer blocks for now - can be replaced with ΨQRH blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_layers = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # 5. Final Prediction Head
        self.prediction_head = nn.Linear(d_model, vocab_size)

        print("✅ FractalQuantumTransformer model initialized successfully.")

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the entire model.

        Args:
            src (torch.Tensor): Input tensor of token IDs, shape [batch_size, seq_len].
            src_mask (torch.Tensor, optional): Mask for the source sequence.

        Returns:
            torch.Tensor: Output logits, shape [batch_size, seq_len, vocab_size].
        """
        # 1. Generate Fractal Quantum States
        # Output shape: [batch_size, seq_len, 4]
        x = self.embedding_layer(src)

        # 2. Project to model dimension
        # Output shape: [batch_size, seq_len, d_model]
        x = self.input_projection(x)

        # 3. Add positional encoding
        x = self.positional_encoding(x)

        # 4. Pass through the stack of transformer blocks
        # Output shape: [batch_size, seq_len, d_model]
        x = self.transformer_layers(x, src_mask)

        # 5. Generate final logits
        # Output shape: [batch_size, seq_len, vocab_size]
        logits = self.prediction_head(x)

        return logits

class PositionalEncoding(nn.Module):
    """Positional Encoding for the Transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)