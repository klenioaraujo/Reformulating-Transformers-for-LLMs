"""
Spectral normalization utilities for ΨQRH Transformer
"""

import torch
import torch.nn as nn


def normalize_spectral_magnitude(x_fft, epsilon=1e-8):
    """
    Normalizes magnitude to 1.0, preserving phase (unitarity)

    Args:
        x_fft: Complex tensor in frequency domain
        epsilon: Small value to avoid division by zero

    Returns:
        Complex tensor with normalized magnitude
    """
    magnitude = torch.abs(x_fft)
    phase = torch.angle(x_fft)

    # Avoid division by zero
    normalized_magnitude = torch.where(magnitude > epsilon, 1.0, 0.0)

    return normalized_magnitude * torch.exp(1j * phase)


def spectral_operation_with_unitarity(x, spectral_op, epsilon=1e-8):
    """
    Wrapper for spectral operations that ensures unitarity

    Args:
        x: Input tensor
        spectral_op: Spectral operation function
        epsilon: Small value for numerical stability

    Returns:
        Result of spectral operation with preserved unitarity
    """
    # Transform to frequency domain with orthonormal FFT
    x_fft = torch.fft.fft(x, norm="ortho")

    # Apply spectral operation
    result_fft = spectral_op(x_fft)

    # Normalize magnitude to preserve unitarity
    result_fft_normalized = normalize_spectral_magnitude(result_fft, epsilon)

    # Transform back to time domain
    result = torch.fft.ifft(result_fft_normalized, norm="ortho").real

    return result


class SpectralUnitarityPreserver(nn.Module):
    """Module for preserving unitarity in spectral operations"""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_fft):
        """
        Apply unitarity-preserving normalization

        Args:
            x_fft: Complex tensor in frequency domain

        Returns:
            Complex tensor with unit magnitude
        """
        return normalize_spectral_magnitude(x_fft, self.epsilon)