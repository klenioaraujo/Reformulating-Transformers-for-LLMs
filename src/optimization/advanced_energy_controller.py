"""
Advanced energy controller for ΨQRH Transformer
"""

import torch
import torch.nn as nn


class AdvancedEnergyController(nn.Module):
    """Layer-wise energy controller"""

    def __init__(self, d_model, n_layers):
        super().__init__()
        self.scalers = nn.ParameterList([
            nn.Parameter(torch.ones(d_model)) for _ in range(n_layers)
        ])

    def forward(self, x, layer_idx):
        """
        Apply adaptive scaling by layer

        Args:
            x: Input tensor
            layer_idx: Index of the current layer

        Returns:
            Scaled output tensor
        """
        scale = self.scalers[layer_idx].view(1, 1, -1)
        return x * scale