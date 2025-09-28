import torch
import torch.nn as nn
import time
from typing import Optional, Tuple, Dict, Any, Literal
from torch.amp import autocast
from .qrh_layer import QRHLayer, QRHConfig

class NegentropyTransformerBlock(nn.Module):
    """
    Transformer block with integrated 4D Unitary (U) Layer.
    Combines standard attention, 4D quaternion processing, and a gate mechanism
    based on numerical "receipts" for flow control.
    """

    def __init__(self,
                 d_model: int,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 qrh_embed_dim: int = 64,
                 alpha: float = 1.0,
                 use_learned_rotation: bool = True,
                 enable_gate: bool = True,
                 qrh_normalization_type: Optional[Literal['layer_norm', 'unit_projection']] = None,
                 init_layer_scale: float = 1e-4):
        super().__init__()
        self.d_model = d_model
        self.enable_gate = enable_gate

        # Projections to map d_model -> 4 * qrh_embed_dim and vice-versa
        self.input_proj = nn.Linear(d_model, 4 * qrh_embed_dim)
        self.output_proj = nn.Linear(4 * qrh_embed_dim, d_model)

        # Standard transformer components
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 4D Quaternion layer
        config = QRHConfig(
            embed_dim=qrh_embed_dim,
            alpha=alpha,
            use_learned_rotation=use_learned_rotation,
            normalization_type=qrh_normalization_type
        )
        self.qrh_layer = QRHLayer(config)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Gate mechanism (simplified for now)
        self.gate_controller = None
        if enable_gate:
            # Simple gate mechanism - can be expanded later
            self.gate_weights = nn.Parameter(torch.ones(d_model))

        # Layer scaling parameters for residual connections
        self.layer_scale_qrh = nn.Parameter(init_layer_scale * torch.ones(d_model))
        self.layer_scale_ffn = nn.Parameter(init_layer_scale * torch.ones(d_model))

    def forward(self,
                x: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the negentropy transformer block.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            src_mask: Optional attention mask
            src_key_padding_mask: Optional key padding mask

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Store original input for residual connections
        residual = x

        # 1. Self-attention
        x = self.norm1(x)
        attn_output, _ = self.self_attn(x, x, x,
                                      attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)
        x = residual + self.dropout1(attn_output)

        # 2. QRH Layer (4D quaternion processing)
        residual = x
        x = self.norm2(x)

        # Project to QRH space
        x_qrh = self.input_proj(x)

        # Apply QRH transformation
        x_qrh = self.qrh_layer(x_qrh)

        # Project back to model space
        x_qrh = self.output_proj(x_qrh)

        # Residual connection with layer scaling
        x = residual + self.layer_scale_qrh * self.dropout2(x_qrh)

        # 3. Feed-forward network
        residual = x
        x = self.norm3(x)
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))

        # Final residual connection with layer scaling
        x = residual + self.layer_scale_ffn * self.dropout2(x)

        # 4. Apply gate mechanism if enabled
        if self.enable_gate and self.gate_controller is not None:
            # Simple gate mechanism
            x = x * torch.sigmoid(self.gate_weights)

        return x

    def extra_repr(self) -> str:
        """Extra representation string for debugging."""
        return f'd_model={self.d_model}, enable_gate={self.enable_gate}'