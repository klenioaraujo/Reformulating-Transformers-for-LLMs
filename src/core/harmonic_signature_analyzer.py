#!/usr/bin/env python3
"""
Harmonic Signature Analyzer for ΨQRH Pipeline
=============================================

Analyzes the harmonic properties of signals to extract a "Harmonic Signature"
that guides the dynamic reconfiguration of transformation algorithms.

This implements the first phase of the "Harmonic Orchestrator" system:
extracting the musical/mathematical essence of the signal to inform adaptive processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math


@dataclass
class HarmonicSignature:
    """
    Complete harmonic signature of a signal, containing all properties
    needed to orchestrate adaptive transformations.
    """
    # Periodicity vs Chaos metrics
    periodicity_score: float  # 0.0 = pure chaos, 1.0 = perfect periodicity
    chaos_entropy: float      # Shannon entropy of the signal

    # Spectral density characteristics
    spectral_density: float   # How concentrated vs spread the energy is
    energy_concentration: float  # Gini coefficient of spectral energy

    # Harmonic relationships
    harmonic_ratio: float     # Strength of harmonic relationships (f, 2f, 3f, etc.)
    fundamental_freq: float   # Estimated fundamental frequency
    harmonic_peaks: List[float]  # List of harmonic peak frequencies

    # Energy distribution patterns
    energy_distribution: torch.Tensor  # Full spectral energy distribution
    dominant_bands: List[Tuple[float, float]]  # (freq_start, freq_end) of dominant bands

    # Phase coherence
    phase_coherence: float    # How coherent the phases are across frequencies
    phase_locking: float      # Kuramoto order parameter for phase relationships

    # Fractal-harmonic coupling
    fractal_harmonic_coupling: float  # How fractal dimension relates to harmonic structure


class HarmonicSignatureAnalyzer(nn.Module):
    """
    Analyzes signals to extract their harmonic signatures for adaptive processing.

    This analyzer goes beyond traditional spectral analysis to understand the
    "musical" or "harmonic" essence of the signal, enabling truly adaptive
    algorithm reconfiguration.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize the Harmonic Signature Analyzer.

        Args:
            device: Computing device for tensor operations
        """
        super(HarmonicSignatureAnalyzer, self).__init__()
        self.device = device

        # Analysis parameters
        self.min_freq_ratio = 0.01  # Minimum frequency ratio for analysis
        self.max_harmonics = 10     # Maximum number of harmonics to analyze
        self.entropy_bins = 50      # Number of bins for entropy calculation

        print("🎼 Harmonic Signature Analyzer initialized")
        print("   🎵 Will analyze: periodicity, spectral density, harmonic relationships")
        print("   🎶 Will extract: energy patterns, phase coherence, fractal-harmonic coupling")

    def forward(self, signal: torch.Tensor) -> HarmonicSignature:
        """
        Analyze a signal to extract its complete harmonic signature.

        Args:
            signal: Input signal tensor [seq_len] or [batch, seq_len]

        Returns:
            Complete harmonic signature
        """
        # Handle batch dimension
        if signal.dim() == 2:
            # Take mean across batch for analysis
            signal = signal.mean(dim=0)

        # Ensure we have a 1D signal
        if signal.dim() > 1:
            signal = signal.flatten()

        # Compute FFT for spectral analysis
        spectrum = torch.fft.fft(signal)
        freqs = torch.fft.fftfreq(len(signal), device=self.device)
        magnitudes = torch.abs(spectrum)
        phases = torch.angle(spectrum)

        # Extract all harmonic properties
        periodicity_score = self._analyze_periodicity(signal, spectrum)
        chaos_entropy = self._compute_chaos_entropy(signal)
        spectral_density, energy_concentration = self._analyze_spectral_density(magnitudes)
        harmonic_ratio, fundamental_freq, harmonic_peaks = self._analyze_harmonic_relationships(magnitudes, freqs)
        dominant_bands = self._identify_dominant_bands(magnitudes, freqs)
        phase_coherence, phase_locking = self._analyze_phase_coherence(phases)
        fractal_harmonic_coupling = self._compute_fractal_harmonic_coupling(signal, harmonic_ratio)

        return HarmonicSignature(
            periodicity_score=periodicity_score,
            chaos_entropy=chaos_entropy,
            spectral_density=spectral_density,
            energy_concentration=energy_concentration,
            harmonic_ratio=harmonic_ratio,
            fundamental_freq=fundamental_freq,
            harmonic_peaks=harmonic_peaks,
            energy_distribution=magnitudes.real.float() if magnitudes.is_complex() else magnitudes.float(),
            dominant_bands=dominant_bands,
            phase_coherence=phase_coherence,
            phase_locking=phase_locking,
            fractal_harmonic_coupling=fractal_harmonic_coupling
        )

    def _analyze_periodicity(self, signal: torch.Tensor, spectrum: torch.Tensor) -> float:
        """
        Analyze how periodic vs chaotic the signal is.

        Returns score from 0.0 (pure chaos) to 1.0 (perfect periodicity)
        """
        # Ensure signal is real for autocorrelation
        if signal.is_complex():
            signal = signal.real.float()

        # Method 1: Autocorrelation peak strength
        # Simple autocorrelation using correlation
        signal_np = signal.detach().cpu().numpy()
        autocorr = np.correlate(signal_np, signal_np, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Positive lags only
        autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize

        # Find peaks in autocorrelation (excluding lag 0)
        if len(autocorr) > 1:
            max_autocorr = float(np.max(autocorr[1:]))
        else:
            max_autocorr = 0.0

        # Method 2: Spectral peak prominence
        magnitudes = torch.abs(spectrum)
        spectral_peakiness = self._compute_spectral_peakiness(magnitudes)

        # Combine metrics
        periodicity_score = (max_autocorr + spectral_peakiness) / 2.0
        return min(1.0, max(0.0, periodicity_score))

    def _compute_spectral_peakiness(self, magnitudes: torch.Tensor) -> float:
        """Compute how peaky/concentrated the spectrum is"""
        # Normalize magnitudes
        magnitudes = magnitudes / (torch.sum(magnitudes) + 1e-10)

        # Compute Gini coefficient (measure of inequality/concentration)
        sorted_mags = torch.sort(magnitudes, descending=True)[0]
        n = len(sorted_mags)
        gini = (2 * torch.sum(torch.arange(1, n+1, device=self.device) * sorted_mags) /
                (n * torch.sum(sorted_mags))) - (n + 1) / n

        # Convert to peakiness score (higher Gini = more concentrated = peakier)
        return gini.item()

    def _compute_chaos_entropy(self, signal: torch.Tensor) -> float:
        """Compute Shannon entropy as a measure of chaos"""
        # Discretize signal into bins
        # Ensure signal is real to avoid ComplexWarning
        if signal.is_complex():
            signal_np = signal.real.detach().cpu().numpy()
        else:
            signal_np = signal.detach().cpu().numpy()

        hist, _ = np.histogram(signal_np, bins=self.entropy_bins, density=True)

        # Remove zero probabilities
        hist = hist[hist > 0]

        # Compute Shannon entropy
        entropy = -np.sum(hist * np.log2(hist))

        # Normalize by maximum possible entropy
        max_entropy = np.log2(self.entropy_bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(1.0, normalized_entropy)

    def _analyze_spectral_density(self, magnitudes: torch.Tensor) -> Tuple[float, float]:
        """
        Analyze spectral density characteristics.

        Returns:
            spectral_density: How dense/concentrated the spectrum is
            energy_concentration: Gini coefficient of energy distribution
        """
        # Normalize magnitudes
        magnitudes = magnitudes / (torch.sum(magnitudes) + 1e-10)

        # Spectral density: ratio of significant frequencies to total
        threshold = torch.mean(magnitudes) + torch.std(magnitudes)
        significant_freqs = torch.sum(magnitudes > threshold).item()
        total_freqs = len(magnitudes)
        spectral_density = significant_freqs / total_freqs

        # Energy concentration: Gini coefficient
        sorted_mags = torch.sort(magnitudes, descending=False)[0]  # Sort ascending for proper Gini
        n = len(sorted_mags)
        if n > 0 and torch.sum(sorted_mags) > 0:
            gini = (2 * torch.sum(torch.arange(1, n+1, device=self.device) * sorted_mags) /
                    (n * torch.sum(sorted_mags))) - (n + 1) / n
            gini = max(0.0, min(1.0, gini.item()))  # Clamp to [0, 1]
        else:
            gini = 0.0

        return spectral_density, gini

    def _analyze_harmonic_relationships(self, magnitudes: torch.Tensor, freqs: torch.Tensor) -> Tuple[float, float, List[float]]:
        """
        Analyze harmonic relationships in the spectrum.

        Returns:
            harmonic_ratio: Strength of harmonic relationships
            fundamental_freq: Estimated fundamental frequency
            harmonic_peaks: List of harmonic peak frequencies
        """
        # Ensure magnitudes are real
        magnitudes = magnitudes.real.float() if magnitudes.is_complex() else magnitudes.float()

        # Find spectral peaks
        peaks = self._find_spectral_peaks(magnitudes)

        if len(peaks) < 2:
            return 0.0, 0.0, []

        # Sort peaks by magnitude
        peak_magnitudes = magnitudes[peaks]
        sorted_indices = torch.argsort(peak_magnitudes, descending=True)
        peaks = peaks[sorted_indices]

        # Assume strongest peak is fundamental
        fundamental_idx = peaks[0]
        fundamental_freq = abs(freqs[fundamental_idx].item())

        # Look for harmonics (integer multiples)
        harmonics_found = [fundamental_freq]
        total_harmonic_strength = magnitudes[fundamental_idx].item()

        for harmonic_num in range(2, self.max_harmonics + 1):
            expected_freq = fundamental_freq * harmonic_num

            # Find closest peak to expected harmonic frequency
            freq_diffs = torch.abs(freqs - expected_freq)
            closest_idx = torch.argmin(freq_diffs)

            # Check if it's close enough (within 10% of fundamental)
            freq_tolerance = fundamental_freq * 0.1
            if freq_diffs[closest_idx] < freq_tolerance:
                harmonics_found.append(abs(freqs[closest_idx].item()))
                total_harmonic_strength += magnitudes[closest_idx].item()

        # Compute harmonic ratio
        total_energy = torch.sum(magnitudes).item()
        harmonic_ratio = total_harmonic_strength / total_energy if total_energy > 0 else 0.0

        return harmonic_ratio, fundamental_freq, harmonics_found

    def _find_spectral_peaks(self, magnitudes: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """Find peaks in the magnitude spectrum"""
        # Ensure magnitudes are real
        magnitudes = magnitudes.real.float() if magnitudes.is_complex() else magnitudes.float()

        # Simple peak detection: local maxima above threshold
        threshold_value = threshold * torch.max(magnitudes)

        peaks = []
        for i in range(1, len(magnitudes) - 1):
            if (magnitudes[i] > magnitudes[i-1] and
                magnitudes[i] > magnitudes[i+1] and
                magnitudes[i] > threshold_value):
                peaks.append(i)

        return torch.tensor(peaks, device=self.device, dtype=torch.long)

    def _identify_dominant_bands(self, magnitudes: torch.Tensor, freqs: torch.Tensor,
                                num_bands: int = 3) -> List[Tuple[float, float]]:
        """Identify the most energetic frequency bands"""
        # Ensure magnitudes are real
        magnitudes = magnitudes.real.float() if magnitudes.is_complex() else magnitudes.float()

        # Sort frequencies by magnitude
        sorted_indices = torch.argsort(magnitudes, descending=True)
        top_indices = sorted_indices[:num_bands*2]  # Get more points to form bands

        # Group into bands (simple clustering by proximity)
        bands = []
        used_indices = set()

        for i in range(len(top_indices)):
            if top_indices[i].item() in used_indices:
                continue

            center_freq = abs(freqs[top_indices[i]].item())
            band_start = center_freq * 0.8  # 20% below
            band_end = center_freq * 1.2    # 20% above

            bands.append((band_start, band_end))
            used_indices.add(top_indices[i].item())

            # Mark nearby frequencies as used
            for j in range(len(top_indices)):
                if abs(freqs[top_indices[j]].item() - center_freq) < center_freq * 0.3:
                    used_indices.add(top_indices[j].item())

        return bands[:num_bands]

    def _analyze_phase_coherence(self, phases: torch.Tensor) -> Tuple[float, float]:
        """
        Analyze phase coherence across the spectrum.

        Returns:
            phase_coherence: Overall phase coherence
            phase_locking: Kuramoto order parameter
        """
        # Phase coherence: how consistent phases are
        phase_std = torch.std(phases).item()
        max_phase_std = np.pi  # Maximum possible phase variation
        phase_coherence = 1.0 - (phase_std / max_phase_std)

        # Phase locking: Kuramoto order parameter
        # Treat each frequency as an oscillator
        complex_phases = torch.exp(1j * phases)
        kuramoto_order = torch.abs(torch.mean(complex_phases)).item()

        return phase_coherence, kuramoto_order

    def _compute_fractal_harmonic_coupling(self, signal: torch.Tensor, harmonic_ratio: float) -> float:
        """
        Compute coupling between fractal dimension and harmonic structure.

        This measures how the fractal complexity relates to harmonic organization.
        """
        # Estimate fractal dimension using simple power-law fitting
        # (This is a simplified version - could be more sophisticated)
        signal_np = signal.detach().cpu().numpy()

        # Simple Hurst exponent estimation (simplified)
        # Higher Hurst = more persistent/fractal behavior
        diffs = np.diff(signal_np)
        hurst = self._estimate_hurst_exponent(signal_np)

        # Coupling: how harmonic structure enhances/suppresses fractal properties
        # High harmonic ratio + high fractal dimension = strong coupling
        coupling = harmonic_ratio * hurst

        # Ensure real value
        if hasattr(coupling, 'real'):
            coupling = coupling.real
        coupling = float(coupling)

        return min(1.0, max(0.0, coupling))

    def _estimate_hurst_exponent(self, signal: np.ndarray, max_lag: int = 20) -> float:
        """Simplified Hurst exponent estimation"""
        if len(signal) < 10:  # Very short signals
            return 0.5  # Random walk default

        # Compute rescaled range for different lags
        lags = range(2, min(max_lag, len(signal)//4))
        rs_values = []

        for lag in lags:
            # Divide signal into chunks of size lag
            chunks = [signal[i:i+lag] for i in range(0, len(signal)-lag+1, lag)]
            if len(chunks) < 2:
                continue

            rs_chunk = []
            for chunk in chunks:
                if len(chunk) > 1:
                    # R/S statistic
                    mean = np.mean(chunk)
                    cumulative = np.cumsum(chunk - mean)
                    r = np.max(cumulative) - np.min(cumulative)
                    s = np.std(chunk)
                    if s > 1e-10:  # Avoid division by very small numbers
                        rs_value = r / s
                        if rs_value > 0:  # Ensure positive values for log
                            rs_chunk.append(rs_value)

            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))

        if len(rs_values) < 2:
            return 0.5

        # Filter out any remaining invalid values
        rs_values = np.array(rs_values)
        valid_mask = (rs_values > 0) & np.isfinite(rs_values)
        rs_values = rs_values[valid_mask]
        lags_array = np.array(list(lags)[:len(rs_values)])

        if len(rs_values) < 2:
            return 0.5

        # Fit power law: RS ~ lag^H
        log_lags = np.log(lags_array)
        log_rs = np.log(rs_values)

        # Ensure no NaN or inf values
        if not np.all(np.isfinite(log_lags)) or not np.all(np.isfinite(log_rs)):
            return 0.5

        # Linear regression
        slope = np.polyfit(log_lags, log_rs, 1)[0]
        hurst = slope

        # Clamp to reasonable range
        return max(0.1, min(0.9, hurst))

    def get_signature_summary(self, signature: HarmonicSignature) -> Dict[str, Any]:
        """Get a human-readable summary of the harmonic signature"""
        return {
            'periodicity': f"{signature.periodicity_score:.2f} ({'periodic' if signature.periodicity_score > 0.7 else 'chaotic'})",
            'chaos_level': f"{signature.chaos_entropy:.2f} ({'ordered' if signature.chaos_entropy < 0.3 else 'chaotic'})",
            'spectral_density': f"{signature.spectral_density:.2f} ({'concentrated' if signature.spectral_density < 0.3 else 'spread'})",
            'harmonic_strength': f"{signature.harmonic_ratio:.2f} ({'harmonic' if signature.harmonic_ratio > 0.5 else 'inharmonic'})",
            'fundamental_freq': f"{signature.fundamental_freq:.3f}",
            'num_harmonics': len(signature.harmonic_peaks),
            'phase_coherence': f"{signature.phase_coherence:.2f}",
            'fractal_coupling': f"{signature.fractal_harmonic_coupling:.2f}",
            'dominant_bands': len(signature.dominant_bands)
        }


# Test function
if __name__ == "__main__":
    analyzer = HarmonicSignatureAnalyzer()

    # Test with different signals
    # 1. Pure sine wave (highly periodic)
    t = torch.linspace(0, 4*np.pi, 1000)
    sine_wave = torch.sin(2 * t)

    # 2. White noise (highly chaotic)
    noise = torch.randn(1000)

    # 3. Harmonic series (periodic with harmonics)
    harmonic_wave = torch.sin(t) + 0.5 * torch.sin(2*t) + 0.3 * torch.sin(3*t)

    signals = {
        'sine_wave': sine_wave,
        'noise': noise,
        'harmonic': harmonic_wave
    }

    for name, signal in signals.items():
        print(f"\n🎼 Analyzing {name}:")
        signature = analyzer(signal)
        summary = analyzer.get_signature_summary(signature)

        for key, value in summary.items():
            print(f"   {key}: {value}")