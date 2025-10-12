#!/usr/bin/env python3
"""
Physical Parameter Calibrator for ΨQRH Pipeline
===============================================

Auto-calibrates physical parameters for the Padilha Wave Equation:
f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

Parameters calibrated:
- I₀ (Amplitude): Based on signal energy normalization
- ω (Angular frequency): Based on dominant spectral frequency
- k (Wave number): Based on fractal dimension
- α (Spectral parameter): Based on fractal coupling
- β (Nonlinear parameter): Based on fractal complexity
"""

import torch
import math
import numpy as np
from typing import Dict, Any, Tuple


class PhysicalParameterCalibrator:
    """
    Calibrates physical parameters for the Padilha Wave Equation
    """

    def __init__(self):
        """Initialize the physical parameter calibrator"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def calibrate_amplitude(self, fractal_signal: torch.Tensor) -> float:
        """
        Calibrate I₀ (Amplitude) based on signal energy normalization

        I₀ = torch.norm(fractal_signal) / len(fractal_signal)
        """
        signal_norm = torch.norm(fractal_signal).item()
        signal_length = len(fractal_signal)

        I0 = signal_norm / signal_length

        # Clamp to physical range
        I0 = max(0.1, min(I0, 5.0))

        return I0

    def calibrate_angular_frequency(self, fractal_signal: torch.Tensor) -> float:
        """
        Calibrate ω (Angular frequency) based on dominant spectral frequency

        ω = 2π × dominant_frequency
        """
        # Compute FFT
        spectrum = torch.fft.fft(fractal_signal)
        power_spectrum = torch.abs(spectrum) ** 2

        # Find dominant frequency index
        dominant_freq_idx = torch.argmax(power_spectrum).item()

        # Get frequency value
        freqs = torch.fft.fftfreq(len(fractal_signal))
        dominant_freq = torch.abs(freqs[dominant_freq_idx]).item()

        # Convert to angular frequency
        omega = 2 * math.pi * dominant_freq

        # Clamp to physical range
        omega = max(0.1, min(omega, 10.0))

        return omega

    def calibrate_wave_number(self, D_fractal: float) -> float:
        """
        Calibrate k (Wave number) based on fractal dimension

        k = 2.0 × D_fractal (proportional to complexity)
        """
        k = 2.0 * D_fractal

        # Clamp to physical range
        k = max(0.5, min(k, 5.0))

        return k

    def calibrate_spectral_parameters(self, D_fractal: float, text: str) -> Tuple[float, float]:
        """
        Calibrate α and β (spectral parameters) based on fractal dimension

        α(D) = α₀(1 + λ(D - D_euclidean)/D_euclidean)
        β = D / 2 (simplified)
        """
        D_euclidean = 1.0
        lambda_coupling = 0.8
        alpha_0 = 1.0

        alpha = alpha_0 * (1.0 + lambda_coupling * (D_fractal - D_euclidean) / D_euclidean)
        beta = D_fractal / 2.0

        # Clamp to physical ranges
        alpha = max(0.1, min(alpha, 3.0))
        beta = max(0.5, min(beta, 1.5))

        return alpha, beta

    def calibrate_all(self, D_fractal: float, text: str, fractal_signal: torch.Tensor) -> Dict[str, float]:
        """
        Calibrate all physical parameters

        Args:
            D_fractal: Fractal dimension
            text: Input text
            fractal_signal: Fractal signal tensor

        Returns:
            Dict with calibrated physical parameters
        """
        # Calibrate spectral parameters (α, β)
        alpha, beta = self.calibrate_spectral_parameters(D_fractal, text)

        # Calibrate wave equation parameters (I₀, ω, k)
        I0 = self.calibrate_amplitude(fractal_signal)
        omega = self.calibrate_angular_frequency(fractal_signal)
        k = self.calibrate_wave_number(D_fractal)

        return {
            'alpha': alpha,
            'beta': beta,
            'I0': I0,
            'omega': omega,
            'k': k
        }

    def validate_physical_consistency(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate physical consistency of calibrated parameters

        Args:
            params: Calibrated parameters

        Returns:
            Validation results
        """
        alpha = params['alpha']
        beta = params['beta']
        I0 = params['I0']
        omega = params['omega']
        k = params['k']

        # Check parameter ranges
        range_checks = {
            'alpha_range': 0.1 <= alpha <= 3.0,
            'beta_range': 0.5 <= beta <= 1.5,
            'I0_range': 0.1 <= I0 <= 5.0,
            'omega_range': 0.1 <= omega <= 10.0,
            'k_range': 0.5 <= k <= 5.0
        }

        # Physical consistency checks
        physical_checks = {
            'wave_dispersion': k > omega / (2 * math.pi),  # k > ω/(2π) for meaningful dispersion
            'nonlinear_coupling': beta > 0,  # β > 0 for nonlinear effects
            'amplitude_physical': I0 > 0  # I₀ > 0 for non-zero amplitude
        }

        all_checks_pass = all(range_checks.values()) and all(physical_checks.values())

        return {
            'range_checks': range_checks,
            'physical_checks': physical_checks,
            'all_checks_pass': all_checks_pass
        }