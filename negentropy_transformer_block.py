import torch
import torch.nn as nn
from typing import Optional
from torch.amp import autocast

from qrh_layer import QRHLayer
from gate_controller import GateController

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
                 enable_gate: bool = True):
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
        from qrh_layer import QRHConfig
        config = QRHConfig(
            embed_dim=qrh_embed_dim,
            alpha=alpha,
            use_learned_rotation=use_learned_rotation
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

        # Gate mechanism
        if enable_gate:
            self.gate_controller = GateController()
        else:
            self.gate_controller = None

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer block with 4D integration and mixed precision.

        Args:
            src: Input tensor [batch_size, seq_len, d_model]
            src_mask: Optional attention mask

        Returns:
            Processed tensor [batch_size, seq_len, d_model]
        """
        with autocast('cuda', enabled=torch.cuda.is_available()):
            # Self-attention with residual connection
            src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            # Project to 4D quaternion space
            qrh_input = self.input_proj(src)  # [batch, seq, 4 * qrh_embed_dim]

            # 4D quaternion processing
            qrh_output = self.qrh_layer(qrh_input)

            # Apply gate mechanism if enabled
            if self.gate_controller is not None:
                # Calculate receipts
                rotation_params = {
                    'theta_left': self.qrh_layer.theta_left,
                    'omega_left': self.qrh_layer.omega_left,
                    'phi_left': self.qrh_layer.phi_left,
                    'theta_right': self.qrh_layer.theta_right,
                    'omega_right': self.qrh_layer.omega_right,
                    'phi_right': self.qrh_layer.phi_right
                }

                receipts = self.gate_controller.calculate_receipts(
                    qrh_input, qrh_output, rotation_params
                )

                # Make gate decision
                gate_decision = self.gate_controller.decide_gate(receipts)

                # Apply gate policy
                gated_output = self.gate_controller.apply_gate_policy(
                    gate_decision, qrh_input, qrh_output
                )
            else:
                gated_output = qrh_output

            # Project back to d_model space
            projected_output = self.output_proj(gated_output)

            # Residual connection and normalization
            src = src + self.dropout(projected_output)
            src = self.norm2(src)

            # Feed-forward network
            src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm3(src)

            return src
