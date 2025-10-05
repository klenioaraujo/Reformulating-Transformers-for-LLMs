"""
Energy normalization utilities for ΨQRH Transformer
"""

import torch
import torch.nn as nn


def energy_preserve(x_input, x_output, epsilon=1e-8):
    """
    Normalizes x_output to have the same energy as x_input.
    Preserves signal direction, only scales magnitude.

    Args:
        x_input: Input tensor with reference energy
        x_output: Output tensor to be normalized
        epsilon: Small value to avoid division by zero

    Returns:
        Normalized output tensor with same energy as input
    """
    if x_input.shape != x_output.shape:
        raise ValueError(f"Shape mismatch: {x_input.shape} vs {x_output.shape}")

    input_energy = torch.sum(x_input**2, dim=-1, keepdim=True)
    output_energy = torch.sum(x_output**2, dim=-1, keepdim=True)

    scale = torch.sqrt(input_energy / (output_energy + epsilon))
    return x_output * scale