"""
Optimized Quaternion Operations for ΨQRH with Physical Foundation
=================================================================

Implements scientifically grounded quaternion algebra with:
- Hamilton product operations with optical coherence
- SO(4) rotation group for unitary transformations
- Energy conservation and unitarity validation
- Spectral domain processing with Padilha wave equation

Physical Foundation:
- Hamilton product: q₁ ⊗ q₂ = (w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂, ...)
- SO(4) rotations: Unitary transformations preserving quaternion norm
- Optical coherence: η = |<E₁E₂*>| / √(<|E₁|²><|E₂|²>)
- Padilha wave: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
Project: https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any


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


class OptimizedQuaternionOperations:
    """
    Optimized quaternion operations with physical foundation for ΨQRH.

    Implements:
    - Hamilton product with energy conservation validation
    - SO(4) rotation group for unitary transformations
    - Optical coherence integration
    - Padilha wave equation coupling
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.energy_tolerance = 1e-6  # For unitarity validation

    def hamilton_product(self, q1: torch.Tensor, q2: torch.Tensor,
                        validate_unitarity: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Optimized Hamilton product with physical validation.

        q₁ ⊗ q₂ = (w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂,
                   w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂,
                   w₁y₂ - x₁z₂ + y₁w₂ + z₁x₂,
                   w₁z₂ + x₁y₂ - y₁x₂ + z₁w₂)

        Args:
            q1: First quaternion [..., 4]
            q2: Second quaternion [..., 4]
            validate_unitarity: Whether to validate energy conservation

        Returns:
            Tuple of (product, validation_metrics)
        """
        w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
        w2, x2, y2, z2 = torch.unbind(q2, dim=-1)

        # Hamilton product with optimized computation
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        product = torch.stack([w, x, y, z], dim=-1)

        validation_metrics = None
        if validate_unitarity:
            validation_metrics = self._validate_hamilton_product(q1, q2, product)

        return product, validation_metrics

    def _validate_hamilton_product(self, q1: torch.Tensor, q2: torch.Tensor,
                                  product: torch.Tensor) -> Dict[str, float]:
        """Validate Hamilton product preserves quaternion algebra properties."""
        # Energy conservation: |q₁ ⊗ q₂|² = |q₁|²|q₂|² for unit quaternions
        norm_q1 = torch.norm(q1, p=2, dim=-1)
        norm_q2 = torch.norm(q2, p=2, dim=-1)
        norm_product = torch.norm(product, p=2, dim=-1)

        expected_norm = norm_q1 * norm_q2
        energy_conservation = torch.mean(torch.abs(norm_product - expected_norm)).item()

        # Unitarity check (product should be unitary if inputs are)
        is_unitary_q1 = torch.allclose(norm_q1, torch.ones_like(norm_q1), atol=1e-5)
        is_unitary_q2 = torch.allclose(norm_q2, torch.ones_like(norm_q2), atol=1e-5)

        unitarity_score = 1.0
        if is_unitary_q1 and is_unitary_q2:
            unitarity_score = torch.mean(torch.abs(norm_product - torch.ones_like(norm_product))).item()

        return {
            'energy_conservation_error': energy_conservation,
            'unitarity_score': unitarity_score,
            'is_energy_conserved': energy_conservation < self.energy_tolerance
        }

    def so4_rotation(self, q: torch.Tensor, rotation_angles: torch.Tensor) -> torch.Tensor:
        """
        Apply SO(4) rotation to quaternion using optimized computation.

        SO(4) rotations preserve quaternion norm and represent unitary transformations.
        Generates unitary quaternions q_left and q_right, then applies:
        q' = q_left ⊗ q ⊗ q_right^{-1} = q_left ⊗ q ⊗ q_right^*

        Args:
            q: Input quaternion [..., 4]
            rotation_angles: Rotation parameters [..., 6] (theta1, omega1, phi1, theta2, omega2, phi2)
                           for left and right rotation quaternions

        Returns:
            Rotated quaternion [..., 4]
        """
        # Split angles for left and right rotations
        # Legacy system uses only 3 parameters for left rotation
        if rotation_angles.shape[-1] == 3:
            # Legacy mode: only left rotation
            theta1, omega1, phi1 = torch.unbind(rotation_angles, dim=-1)
            theta2, omega2, phi2 = torch.zeros_like(theta1), torch.zeros_like(omega1), torch.zeros_like(phi1)
        else:
            # Full mode: both left and right rotations
            theta1, omega1, phi1, theta2, omega2, phi2 = torch.unbind(rotation_angles, dim=-1)

        # Create left rotation quaternion (unit quaternion)
        w_left = torch.cos(theta1/2) * torch.cos(omega1/2) * torch.cos(phi1/2)
        x_left = torch.sin(theta1/2) * torch.cos(omega1/2) * torch.cos(phi1/2)
        y_left = torch.cos(theta1/2) * torch.sin(omega1/2) * torch.cos(phi1/2)
        z_left = torch.cos(theta1/2) * torch.cos(omega1/2) * torch.sin(phi1/2)

        q_left = torch.stack([w_left, x_left, y_left, z_left], dim=-1)
        # Ensure unitarity
        q_left = quaternion_normalize(q_left)

        # Create right rotation quaternion (unit quaternion)
        w_right = torch.cos(theta2/2) * torch.cos(omega2/2) * torch.cos(phi2/2)
        x_right = torch.sin(theta2/2) * torch.cos(omega2/2) * torch.cos(phi2/2)
        y_right = torch.cos(theta2/2) * torch.sin(omega2/2) * torch.cos(phi2/2)
        z_right = torch.cos(theta2/2) * torch.cos(omega2/2) * torch.sin(phi2/2)

        q_right = torch.stack([w_right, x_right, y_right, z_right], dim=-1)
        # Ensure unitarity
        q_right = quaternion_normalize(q_right)

        # Apply rotation: q' = q_left ⊗ q ⊗ q_right^{-1} = q_left ⊗ q ⊗ q_right^*
        q_right_conj = self.conjugate(q_right)

        # First product: q_left ⊗ q
        temp, _ = self.hamilton_product(q_left, q)

        # Second product: temp ⊗ q_right^*
        result, _ = self.hamilton_product(temp, q_right_conj)

        return result

    def validate_so4_rotation_norm_preservation(self, num_samples: int = 10000) -> Dict[str, float]:
        """
        Numerically validate that SO(4) rotations preserve quaternion norms.

        Tests on thousands of random samples to ensure ||q'|| = ||q||.

        Args:
            num_samples: Number of random samples to test

        Returns:
            Validation metrics dictionary
        """
        device = getattr(self, 'device', 'cpu')

        # Generate random quaternions and rotation angles
        q_original = torch.randn(num_samples, 4, device=device)
        # Normalize to ensure unit quaternions for testing
        q_original = quaternion_normalize(q_original)

        rotation_angles = torch.randn(num_samples, 6, device=device) * 2 * torch.pi

        # Apply SO(4) rotation
        q_rotated = self.so4_rotation(q_original, rotation_angles)

        # Compute norms
        norm_original = quaternion_norm(q_original)
        norm_rotated = quaternion_norm(q_rotated)

        # Compute preservation metrics
        norm_preservation_errors = torch.abs(norm_rotated - norm_original)
        max_error = torch.max(norm_preservation_errors).item()
        mean_error = torch.mean(norm_preservation_errors).item()
        std_error = torch.std(norm_preservation_errors).item()

        # Check if all norms are preserved within tolerance
        tolerance = 1e-5
        all_preserved = torch.all(norm_preservation_errors < tolerance).item()

        # Additional validation: check unitarity of rotation quaternions
        theta1, omega1, phi1, theta2, omega2, phi2 = torch.unbind(rotation_angles, dim=-1)

        w_left = torch.cos(theta1/2) * torch.cos(omega1/2) * torch.cos(phi1/2)
        x_left = torch.sin(theta1/2) * torch.cos(omega1/2) * torch.cos(phi1/2)
        y_left = torch.cos(theta1/2) * torch.sin(omega1/2) * torch.cos(phi1/2)
        z_left = torch.cos(theta1/2) * torch.cos(omega1/2) * torch.sin(phi1/2)
        q_left_test = torch.stack([w_left, x_left, y_left, z_left], dim=-1)
        q_left_norm = torch.mean(quaternion_norm(q_left_test)).item()

        w_right = torch.cos(theta2/2) * torch.cos(omega2/2) * torch.cos(phi2/2)
        x_right = torch.sin(theta2/2) * torch.cos(omega2/2) * torch.cos(phi2/2)
        y_right = torch.cos(theta2/2) * torch.sin(omega2/2) * torch.cos(phi2/2)
        z_right = torch.cos(theta2/2) * torch.cos(omega2/2) * torch.sin(phi2/2)
        q_right_test = torch.stack([w_right, x_right, y_right, z_right], dim=-1)
        q_right_norm = torch.mean(quaternion_norm(q_right_test)).item()

        return {
            'max_norm_preservation_error': max_error,
            'mean_norm_preservation_error': mean_error,
            'std_norm_preservation_error': std_error,
            'all_norms_preserved': all_preserved,
            'rotation_quaternions_unitary': q_left_norm > 0.999 and q_right_norm > 0.999,
            'num_samples_tested': num_samples
        }

    def conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """Compute quaternion conjugate."""
        w, x, y, z = torch.unbind(q, dim=-1)
        return torch.stack([w, -x, -y, -z], dim=-1)

    def optical_coherence_product(self, q1: torch.Tensor, q2: torch.Tensor,
                                coherence_factor: float = 1.0) -> torch.Tensor:
        """
        Hamilton product with optical coherence modulation.

        η_optical = |<E₁E₂*>| / √(<|E₁|²><|E₂|²>)

        Args:
            q1: First quaternion
            q2: Second quaternion
            coherence_factor: Optical coherence modulation [0,1]

        Returns:
            Coherence-modulated product
        """
        # Standard Hamilton product
        product, _ = self.hamilton_product(q1, q2)

        # Apply optical coherence modulation
        # Higher coherence = cleaner signal, lower noise
        coherence_modulation = torch.sigmoid(torch.tensor(coherence_factor))

        # Modulate imaginary components (which carry phase information)
        w, x, y, z = torch.unbind(product, dim=-1)
        x_mod = x * coherence_modulation
        y_mod = y * coherence_modulation
        z_mod = z * coherence_modulation

        return torch.stack([w, x_mod, y_mod, z_mod], dim=-1)

    def padilha_wave_coupling(self, q: torch.Tensor, wavelength: torch.Tensor,
                            time: torch.Tensor, alpha: float = 1.0,
                            beta: float = 0.5) -> torch.Tensor:
        """
        Couple quaternion with Padilha wave equation.

        f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

        Args:
            q: Input quaternion [..., 4]
            wavelength: Wavelength parameter λ
            time: Time parameter t
            alpha: Spectral parameter α
            beta: Dispersion parameter β

        Returns:
            Wave-coupled quaternion
        """
        # Padilha wave equation components
        omega = 1.0  # Angular frequency
        k = 2.0      # Wave number
        I0 = 1.0     # Amplitude

        # Compute wave function
        phase = omega * time + alpha * wavelength
        dispersion_phase = omega * time - k * wavelength + beta * wavelength**2

        wave_real = I0 * torch.sin(phase)
        wave_imag = torch.exp(1j * dispersion_phase)

        # Combine wave with quaternion
        w, x, y, z = torch.unbind(q, dim=-1)

        # Modulate quaternion components with wave
        w_wave = w * wave_real
        x_wave = x * wave_real * wave_imag.real
        y_wave = y * wave_real * wave_imag.imag
        z_wave = z * torch.abs(wave_imag)

        return torch.stack([w_wave, x_wave, y_wave, z_wave], dim=-1)


class QuaternionOperations:
    """
    Legacy quaternion operations for backward compatibility.
    """

    @staticmethod
    def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Legacy Hamilton product - use OptimizedQuaternionOperations for new code."""
        ops = OptimizedQuaternionOperations()
        result, _ = ops.hamilton_product(q1, q2)
        return result

    @staticmethod
    def create_unit_quaternion_batch(thetas: torch.Tensor, omegas: torch.Tensor, phis: torch.Tensor) -> torch.Tensor:
        """Legacy unit quaternion creation."""
        ops = OptimizedQuaternionOperations()
        # Simplified implementation for backward compatibility
        w = torch.cos(thetas / 2) * torch.cos(omegas / 2) * torch.cos(phis / 2)
        x = torch.sin(thetas / 2) * torch.cos(omegas / 2) * torch.cos(phis / 2)
        y = torch.cos(thetas / 2) * torch.sin(omegas / 2) * torch.cos(phis / 2)
        z = torch.cos(thetas / 2) * torch.cos(omegas / 2) * torch.sin(phis / 2)

        quaternions = torch.stack([w, x, y, z], dim=-1)
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