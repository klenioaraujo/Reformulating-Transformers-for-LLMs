"""
Quaternion Operations for ΨQRH Transformer

Implements quaternion algebra and linear operations for efficient 4D representation.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
Project: https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product of two quaternions.

    Args:
        q1: First quaternion tensor of shape [..., 4]
        q2: Second quaternion tensor of shape [..., 4]

    Returns:
        Hamilton product of shape [..., 4]
    """
    a1, b1, c1, d1 = torch.unbind(q1, dim=-1)
    a2, b2, c2, d2 = torch.unbind(q2, dim=-1)

    # Hamilton product
    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    return torch.stack([a, b, c, d], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Compute conjugate of quaternion.

    Args:
        q: Quaternion tensor of shape [..., 4]

    Returns:
        Conjugate quaternion of shape [..., 4]
    """
    a, b, c, d = torch.unbind(q, dim=-1)
    return torch.stack([a, -b, -c, -d], dim=-1)


def quaternion_norm(q: torch.Tensor) -> torch.Tensor:
    """
    Compute norm of quaternion.

    Args:
        q: Quaternion tensor of shape [..., 4]

    Returns:
        Norm tensor of shape [...]
    """
    return torch.sqrt(torch.sum(q ** 2, dim=-1))


def quaternion_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize quaternion to unit quaternion.

    Args:
        q: Quaternion tensor of shape [..., 4]
        eps: Small value to avoid division by zero

    Returns:
        Normalized quaternion of shape [..., 4]
    """
    norm = quaternion_norm(q).unsqueeze(-1)
    return q / (norm + eps)


class QuaternionLinear(nn.Module):
    """
    Linear layer for quaternion-valued inputs.

    Implements quaternion linear transformation with proper weight initialization
    and energy preservation.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quaternion weights: maintain same feature dimension
        self.weight = nn.Parameter(torch.Tensor(out_features * 4, in_features * 4))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features * 4))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using quaternion-aware initialization."""
        # Xavier uniform initialization adapted for quaternions
        gain = nn.init.calculate_gain('linear')
        std = gain * math.sqrt(2.0 / (self.in_features * 4 + self.out_features * 4))

        with torch.no_grad():
            self.weight.data.uniform_(-std, std)
            if self.bias is not None:
                self.bias.data.uniform_(-std, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quaternion linear layer.

        Args:
            x: Input tensor of shape [..., in_features * 4]

        Returns:
            Output tensor of shape [..., out_features * 4]
        """
        # Reshape for matrix multiplication
        original_shape = x.shape

        # Flatten all but last dimension
        x_flat = x.reshape(-1, self.in_features * 4)

        # Apply linear transformation
        # Weight shape: [out_features, in_features * 4]
        output = F.linear(x_flat, self.weight, self.bias)

        # Reshape back to original batch dimensions
        output_shape = list(original_shape[:-1]) + [self.out_features * 4]
        return output.reshape(*output_shape)


class QuaternionLayerNorm(nn.Module):
    """
    Layer normalization for quaternion-valued inputs.

    Normalizes across all 4 components of quaternion representation.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters for each quaternion component
        self.weight = nn.Parameter(torch.ones(normalized_shape * 4))
        self.bias = nn.Parameter(torch.zeros(normalized_shape * 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quaternion layer normalization.

        Args:
            x: Input tensor of shape [..., normalized_shape * 4]

        Returns:
            Normalized tensor of shape [..., normalized_shape * 4]
        """
        # Compute mean and variance across all 4 components
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learned affine transformation
        return self.weight.unsqueeze(0).unsqueeze(0) * x_normalized + self.bias.unsqueeze(0).unsqueeze(0)


class QuaternionOperations:
    """
    Quaternion operations for ΨQRH transformer.
    """

    @staticmethod
    def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Hamilton product of two quaternions.

        Args:
            q1: First quaternion tensor of shape [..., 4]
            q2: Second quaternion tensor of shape [..., 4]

        Returns:
            Hamilton product of shape [..., 4]
        """
        w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
        w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    @staticmethod
    def create_unit_quaternion_batch(thetas: torch.Tensor, omegas: torch.Tensor, phis: torch.Tensor) -> torch.Tensor:
        """
        Create unit quaternions from rotation angles.

        Args:
            thetas: Tensor of theta angles
            omegas: Tensor of omega angles
            phis: Tensor of phi angles

        Returns:
            Unit quaternions of shape [..., 4]
        """
        # Create quaternions from rotation angles
        w = torch.cos(thetas / 2) * torch.cos(omegas / 2) * torch.cos(phis / 2)
        x = torch.sin(thetas / 2) * torch.cos(omegas / 2) * torch.cos(phis / 2)
        y = torch.cos(thetas / 2) * torch.sin(omegas / 2) * torch.cos(phis / 2)
        z = torch.cos(thetas / 2) * torch.cos(omegas / 2) * torch.sin(phis / 2)

        quaternions = torch.stack([w, x, y, z], dim=-1)

        # Normalize to unit quaternions
        norm = torch.norm(quaternions, p=2, dim=-1, keepdim=True)
        return quaternions / (norm + 1e-8)


class SpectralActivation(nn.Module):
    """
    Spectral domain activation function.

    Applies activation in frequency domain for better energy preservation.
    """

    def __init__(self, activation_type: str = "gelu"):
        super().__init__()
        self.activation_type = activation_type

        if activation_type == "gelu":
            self.activation = nn.GELU()
        elif activation_type == "relu":
            self.activation = nn.ReLU()
        elif activation_type == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        # Apply activation in frequency domain
        x_fft = torch.fft.fft(x, dim=-1, norm="ortho")

        # Apply activation to magnitude while preserving phase
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # Apply activation to magnitude
        activated_magnitude = self.activation(magnitude)

        # Reconstruct complex tensor
        activated_fft = activated_magnitude * torch.exp(1j * phase)

        # Transform back to time domain
        return torch.fft.ifft(activated_fft, dim=-1, norm="ortho").real


class AdaptiveSpectralDropout(nn.Module):
    """
    Adaptive dropout in spectral domain.

    Drops frequency components based on their energy contribution.
    """

    def __init__(self, p: float = 0.1, adaptive_threshold: float = 0.01):
        super().__init__()
        self.p = p
        self.adaptive_threshold = adaptive_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive spectral dropout.

        Args:
            x: Input tensor

        Returns:
            Tensor with spectral dropout applied
        """
        if not self.training or self.p == 0:
            return x

        # Transform to frequency domain
        x_fft = torch.fft.fft(x, dim=-1, norm="ortho")

        # Compute energy per frequency component
        energy = torch.abs(x_fft) ** 2
        total_energy = torch.sum(energy, dim=-1, keepdim=True)
        energy_ratio = energy / (total_energy + 1e-8)

        # Create dropout mask based on energy contribution
        # Low-energy components are more likely to be dropped
        dropout_probs = torch.ones_like(energy_ratio) * self.p

        # Increase dropout probability for low-energy components
        low_energy_mask = energy_ratio < self.adaptive_threshold
        dropout_probs[low_energy_mask] = self.p * 2.0

        # Generate dropout mask
        dropout_mask = torch.bernoulli(1 - dropout_probs)

        # Apply dropout
        x_fft_dropped = x_fft * dropout_mask

        # Transform back to time domain
        return torch.fft.ifft(x_fft_dropped, dim=-1, norm="ortho").real


class RealTimeFractalAnalyzer(nn.Module):
    """
    Real-time fractal dimension analyzer.

    Computes fractal dimension and other metrics for adaptive parameter adjustment.
    """

    def __init__(self, window_size: int = 1000):
        super().__init__()
        self.window_size = window_size

    def analyze(self, x: torch.Tensor) -> dict:
        """
        Analyze fractal properties of input tensor.

        Args:
            x: Input tensor

        Returns:
            Dictionary with fractal metrics
        """
        with torch.no_grad():
            # Compute approximate fractal dimension using box counting
            dimension = self._compute_fractal_dimension(x)

            # Compute spectral properties
            spectral_entropy = self._compute_spectral_entropy(x)

            # Compute energy distribution
            energy_metrics = self._compute_energy_metrics(x)

            return {
                'dimension': dimension,
                'spectral_entropy': spectral_entropy,
                'energy_metrics': energy_metrics
            }

    def _compute_fractal_dimension(self, x: torch.Tensor) -> torch.Tensor:
        """Compute approximate fractal dimension using box counting."""
        # Simplified box counting for real-time computation
        x_flat = x.view(-1)

        # Compute variance as proxy for fractal dimension
        variance = torch.var(x_flat)

        # Map variance to approximate dimension (1D to 2D range)
        dimension = 1.0 + torch.sigmoid(variance)

        return dimension

    def _compute_spectral_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spectral entropy of input."""
        x_fft = torch.fft.fft(x, dim=-1, norm="ortho")
        power_spectrum = torch.abs(x_fft) ** 2

        # Normalize to probability distribution
        prob_dist = power_spectrum / (torch.sum(power_spectrum, dim=-1, keepdim=True) + 1e-8)

        # Compute entropy
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8), dim=-1)

        return entropy.mean()

    def _compute_energy_metrics(self, x: torch.Tensor) -> dict:
        """Compute energy distribution metrics."""
        energy = torch.sum(x ** 2, dim=-1)

        return {
            'total_energy': torch.sum(energy),
            'mean_energy': torch.mean(energy),
            'energy_variance': torch.var(energy)
        }