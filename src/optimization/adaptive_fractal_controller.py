"""
Adaptive Fractal Controller for ΨQRH Transformer

Implements real-time fractal analysis and parameter adaptation based on
input data complexity and fractal dimension.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FractalControllerConfig:
    """Configuration for adaptive fractal controller"""
    window_size: int = 1000
    update_frequency: int = 100
    alpha_range: Tuple[float, float] = (0.1, 2.0)
    beta_range: Tuple[float, float] = (-1.0, 1.0)
    dimension_threshold: float = 1.5
    learning_rate: float = 0.01
    momentum: float = 0.9


class AdaptiveFractalController(nn.Module):
    """
    Adaptive fractal controller that analyzes input data complexity
    and adjusts ΨQRH parameters in real-time.
    """

    def __init__(self, config: FractalControllerConfig):
        super().__init__()
        self.config = config

        # Fractal analysis history
        self.fractal_history = []
        self.parameter_history = []

        # Current parameters
        self.current_alpha = nn.Parameter(torch.tensor(1.0))
        self.current_beta = nn.Parameter(torch.tensor(0.0))

        # Momentum buffers for smooth parameter updates
        self.alpha_momentum = 0.0
        self.beta_momentum = 0.0

        # Statistical tracking
        self.dimension_stats = {
            'mean': 1.0,
            'std': 0.1,
            'min': 1.0,
            'max': 2.0
        }

    def analyze_fractal_dimension(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute fractal dimension and related metrics using box counting method.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Dictionary containing fractal metrics
        """
        with torch.no_grad():
            batch_size, seq_len, d_model = x.shape

            # Reshape for analysis
            x_flat = x.view(batch_size * seq_len, d_model)

            # Compute multiple fractal dimension estimates
            dimensions = []

            # Method 1: Variance-based estimation
            variance = torch.var(x_flat, dim=0).mean().item()
            dim_variance = 1.0 + np.tanh(variance)  # Map to [1, 2] range
            dimensions.append(dim_variance)

            # Method 2: Spectral slope estimation
            x_fft = torch.fft.fft(x_flat, dim=1, norm="ortho")
            power_spectrum = torch.abs(x_fft) ** 2

            # Compute spectral slope (approximate)
            freqs = torch.arange(1, d_model // 2 + 1, device=x.device)
            log_freqs = torch.log(freqs.float())
            log_power = torch.log(power_spectrum[:, :d_model//2].mean(dim=0))

            # Simple linear fit for spectral slope
            if len(log_freqs) > 1:
                # Manual linear regression
                x_mean = log_freqs.mean()
                y_mean = log_power.mean()
                numerator = torch.sum((log_freqs - x_mean) * (log_power - y_mean))
                denominator = torch.sum((log_freqs - x_mean) ** 2)
                if denominator > 1e-8:
                    slope = (numerator / denominator).item()
                    dim_spectral = 2.0 - slope / 2.0  # Map slope to dimension
                    dimensions.append(max(1.0, min(2.0, dim_spectral)))

            # Method 3: Entropy-based estimation
            entropy = self._compute_entropy(x_flat)
            dim_entropy = 1.0 + entropy / np.log(d_model)  # Normalize entropy
            dimensions.append(dim_entropy)

            # Final dimension estimate (average of methods)
            fractal_dimension = np.mean(dimensions)

            # Additional metrics
            metrics = {
                'fractal_dimension': fractal_dimension,
                'dimension_variance': dim_variance,
                'dimension_spectral': dimensions[1] if len(dimensions) > 1 else fractal_dimension,
                'dimension_entropy': dim_entropy,
                'complexity_score': self._compute_complexity_score(x),
                'energy_distribution': self._compute_energy_distribution(x)
            }

            return metrics

    def _compute_entropy(self, x: torch.Tensor) -> float:
        """Compute Shannon entropy of input tensor"""
        # Normalize to probability distribution
        x_abs = torch.abs(x)
        prob_dist = x_abs / (torch.sum(x_abs, dim=-1, keepdim=True) + 1e-8)

        # Compute entropy
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8), dim=-1)
        return entropy.mean().item()

    def _compute_complexity_score(self, x: torch.Tensor) -> float:
        """Compute overall complexity score"""
        # Combine multiple complexity measures
        variance = torch.var(x).item()
        entropy = self._compute_entropy(x)

        # Spectral complexity
        x_fft = torch.fft.fft(x, dim=-1, norm="ortho")
        spectral_flatness = torch.exp(torch.mean(torch.log(torch.abs(x_fft) + 1e-8))).item()

        # Normalized complexity score [0, 1]
        complexity = (variance + entropy + (1 - spectral_flatness)) / 3.0
        return min(1.0, max(0.0, complexity))

    def _compute_energy_distribution(self, x: torch.Tensor) -> Dict[str, float]:
        """Compute energy distribution metrics"""
        energy = torch.sum(x ** 2, dim=-1)

        return {
            'total_energy': torch.sum(energy).item(),
            'mean_energy': torch.mean(energy).item(),
            'energy_variance': torch.var(energy).item(),
            'energy_skewness': self._compute_skewness(energy).item()
        }

    def _compute_skewness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute skewness of tensor"""
        mean = torch.mean(x)
        std = torch.std(x)
        skewness = torch.mean(((x - mean) / std) ** 3)
        return skewness

    def update_parameters(self, x: torch.Tensor, layer: nn.Module) -> None:
        """
        Update ΨQRH parameters based on fractal analysis of input data.

        Args:
            x: Input tensor
            layer: ΨQRH layer to update
        """
        if not self.training:
            return

        # Analyze current input
        fractal_metrics = self.analyze_fractal_dimension(x)
        dimension = fractal_metrics['fractal_dimension']
        complexity = fractal_metrics['complexity_score']

        # Update dimension statistics
        self._update_dimension_stats(dimension)

        # Map fractal dimension to ΨQRH parameters
        target_alpha = self._map_dimension_to_alpha(dimension, complexity)
        target_beta = self._map_dimension_to_beta(dimension, complexity)

        # Apply momentum-based updates
        self._update_with_momentum(target_alpha, target_beta)

        # Apply updates to layer parameters
        self._apply_parameter_updates(layer)

        # Store history
        self.fractal_history.append(fractal_metrics)
        self.parameter_history.append({
            'alpha': self.current_alpha.item(),
            'beta': self.current_beta.item(),
            'dimension': dimension
        })

        # Limit history size
        if len(self.fractal_history) > self.config.window_size:
            self.fractal_history.pop(0)
            self.parameter_history.pop(0)

    def _update_dimension_stats(self, dimension: float) -> None:
        """Update running statistics for fractal dimension"""
        # Exponential moving average
        alpha = 0.1  # Smoothing factor

        self.dimension_stats['mean'] = (
            alpha * dimension + (1 - alpha) * self.dimension_stats['mean']
        )

        # Update min/max
        self.dimension_stats['min'] = min(self.dimension_stats['min'], dimension)
        self.dimension_stats['max'] = max(self.dimension_stats['max'], dimension)

        # Simple variance estimation
        diff = dimension - self.dimension_stats['mean']
        self.dimension_stats['std'] = np.sqrt(
            alpha * diff**2 + (1 - alpha) * self.dimension_stats['std']**2
        )

    def _map_dimension_to_alpha(self, dimension: float, complexity: float) -> float:
        """
        Map fractal dimension to spectral filter alpha parameter.

        Higher dimension = more complex signal = stronger filtering
        """
        # Normalize dimension to [0, 1] range based on stats
        norm_dim = (dimension - self.dimension_stats['min']) / \
                  (self.dimension_stats['max'] - self.dimension_stats['min'] + 1e-8)

        # Combine dimension and complexity
        combined_score = 0.7 * norm_dim + 0.3 * complexity

        # Map to alpha range with sigmoid-like function
        alpha_min, alpha_max = self.config.alpha_range
        target_alpha = alpha_min + (alpha_max - alpha_min) * combined_score

        return target_alpha

    def _map_dimension_to_beta(self, dimension: float, complexity: float) -> float:
        """
        Map fractal dimension to spectral filter beta parameter.

        Controls phase modulation based on signal complexity
        """
        # Higher dimension = more phase modulation
        norm_dim = (dimension - 1.0)  # Center around 1.0

        # Combine with complexity
        beta_score = 0.5 * norm_dim + 0.5 * complexity

        # Map to beta range
        beta_min, beta_max = self.config.beta_range
        target_beta = beta_min + (beta_max - beta_min) * torch.sigmoid(torch.tensor(beta_score)).item()

        return target_beta

    def _update_with_momentum(self, target_alpha: float, target_beta: float) -> None:
        """Update parameters with momentum"""
        # Alpha update
        alpha_diff = target_alpha - self.current_alpha.item()
        self.alpha_momentum = (
            self.config.momentum * self.alpha_momentum +
            self.config.learning_rate * alpha_diff
        )

        # Beta update
        beta_diff = target_beta - self.current_beta.item()
        self.beta_momentum = (
            self.config.momentum * self.beta_momentum +
            self.config.learning_rate * beta_diff
        )

        # Apply updates
        with torch.no_grad():
            self.current_alpha.data += torch.tensor(self.alpha_momentum)
            self.current_beta.data += torch.tensor(self.beta_momentum)

            # Clamp to valid ranges
            alpha_min, alpha_max = self.config.alpha_range
            beta_min, beta_max = self.config.beta_range

            self.current_alpha.data.clamp_(alpha_min, alpha_max)
            self.current_beta.data.clamp_(beta_min, beta_max)

    def _apply_parameter_updates(self, layer: nn.Module) -> None:
        """Apply updated parameters to ΨQRH layer"""
        # Update spectral filter parameters if available
        if hasattr(layer, 'spectral_filter') and hasattr(layer.spectral_filter, 'update_alpha'):
            layer.spectral_filter.update_alpha(
                torch.tensor(self.current_alpha.item())
            )

        # Update other layer parameters as needed
        # This can be extended based on specific layer architecture

    def get_controller_status(self) -> Dict:
        """Get current status of the fractal controller"""
        return {
            'current_alpha': self.current_alpha.item(),
            'current_beta': self.current_beta.item(),
            'dimension_stats': self.dimension_stats,
            'history_size': len(self.fractal_history),
            'recent_dimension': self.fractal_history[-1]['fractal_dimension'] if self.fractal_history else 0.0
        }

    def reset_history(self) -> None:
        """Reset fractal analysis history"""
        self.fractal_history.clear()
        self.parameter_history.clear()

        # Reset statistics
        self.dimension_stats = {
            'mean': 1.0,
            'std': 0.1,
            'min': 1.0,
            'max': 2.0
        }

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Forward pass for analysis only (no parameter updates)

        Args:
            x: Input tensor

        Returns:
            Fractal analysis results
        """
        return self.analyze_fractal_dimension(x)


class MultiScaleFractalController(nn.Module):
    """
    Multi-scale fractal controller for handling different temporal scales
    """

    def __init__(self, config: FractalControllerConfig, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales

        # Controllers for different scales
        self.scale_controllers = nn.ModuleList([
            AdaptiveFractalController(config) for _ in range(num_scales)
        ])

        # Scale weights (learnable)
        self.scale_weights = nn.Parameter(torch.ones(num_scales))

    def forward(self, x: torch.Tensor) -> Dict:
        """Analyze multi-scale fractal properties"""
        batch_size, seq_len, d_model = x.shape

        results = {}

        # Analyze at different temporal scales
        for i, controller in enumerate(self.scale_controllers):
            scale_factor = 2 ** i

            if seq_len >= scale_factor:
                # Downsample for this scale
                x_downsampled = self._downsample_temporal(x, scale_factor)
                scale_results = controller(x_downsampled)
                results[f'scale_{scale_factor}'] = scale_results

        # Combine results using learned weights
        combined_dimension = 0.0
        total_weight = 0.0

        for i, (scale_key, scale_result) in enumerate(results.items()):
            weight = torch.softmax(self.scale_weights, dim=0)[i]
            combined_dimension += weight * scale_result['fractal_dimension']
            total_weight += weight

        if total_weight > 0:
            combined_dimension /= total_weight

        results['combined_fractal_dimension'] = combined_dimension
        results['scale_weights'] = torch.softmax(self.scale_weights, dim=0).detach().cpu().numpy()

        return results

    def _downsample_temporal(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Downsample along temporal dimension"""
        batch_size, seq_len, d_model = x.shape

        # Simple average pooling
        if seq_len % factor == 0:
            x_downsampled = x.view(batch_size, seq_len // factor, factor, d_model)
            x_downsampled = torch.mean(x_downsampled, dim=2)
        else:
            # Use adaptive pooling for uneven lengths
            x_downsampled = nn.AdaptiveAvgPool1d(seq_len // factor)(x.transpose(1, 2))
            x_downsampled = x_downsampled.transpose(1, 2)

        return x_downsampled