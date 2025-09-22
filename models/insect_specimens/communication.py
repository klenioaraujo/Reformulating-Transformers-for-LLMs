
import torch
import numpy as np
from typing import Optional, Dict, Any
from qrh_layer import QRHLayer, QRHConfig


def complex_wave_equation(alpha: float, beta: float, omega: float, amplitude: float = 1.0,
                         phase_shift: float = 0.0, harmonics: list = None,
                         dna_signature: dict = None, steps: int = 256, t: float = 0.0) -> np.ndarray:
    """
    Enhanced complex wave equation with preserved genetic identity.

    f(λ,t,GLS) = I₀ sin(ωt + αλ + φ) e^(i(ωt - kλ + βλ²)) × GLS_fractal(λ) × DNA_signature

    Args:
        alpha: Phase parameter from fractal dimension
        beta: Dispersion parameter from spectral distribution
        omega: Frequency parameter from spectral energy
        amplitude: Wave intensity
        phase_shift: Additional phase from spectral skewness
        harmonics: Spatial frequency components from fractal
        dna_signature: DNA-preserved spectral signature
        steps: Number of spatial samples
        t: Time parameter

    Returns:
        Complex waveform with preserved genetic identity
    """
    # Create spatial domain
    lambda_space = np.linspace(0, 1, steps)

    # Base wave equation components
    # f(λ,t) = I₀ sin(ωt + αλ + φ) e^(i(ωt - kλ + βλ²))
    phase_real = omega * t + alpha * lambda_space + phase_shift
    phase_complex = 1j * (omega * t - 2 * np.pi * lambda_space + beta * lambda_space**2)

    # Base waveform
    real_part = amplitude * np.sin(phase_real)
    complex_part = np.exp(phase_complex)
    base_wave = real_part * complex_part

    # Apply harmonic enhancement from fractal spatial frequencies
    if harmonics:
        harmonic_enhancement = np.ones_like(lambda_space, dtype=complex)
        for i, freq in enumerate(harmonics[:3]):  # Limit to 3 harmonics
            harmonic_weight = 0.3 / (i + 1)  # Decreasing weight for higher harmonics
            harmonic_phase = 2 * np.pi * freq * lambda_space
            harmonic_enhancement += harmonic_weight * np.exp(1j * harmonic_phase)

        base_wave = base_wave * harmonic_enhancement

    # Apply DNA signature modulation for genetic identity preservation
    if dna_signature:
        dna_modulation = _apply_dna_modulation(lambda_space, dna_signature)
        base_wave = base_wave * dna_modulation

    return base_wave


def _apply_dna_modulation(lambda_space: np.ndarray, dna_signature: dict) -> np.ndarray:
    """Apply DNA signature modulation to preserve genetic identity in the wave."""
    # Extract DNA spectral characteristics
    base_freq = dna_signature.get('base_frequency', 1.0)
    harmonic_ratios = dna_signature.get('harmonic_ratios', [1.0, 0.5])
    phase_pattern = dna_signature.get('phase_pattern', [0.0])

    # Create DNA-specific modulation
    dna_modulation = np.ones_like(lambda_space, dtype=complex)

    # Apply base frequency modulation
    base_modulation = 0.8 + 0.2 * np.cos(2 * np.pi * base_freq * lambda_space)

    # Apply harmonic ratio patterns
    harmonic_modulation = np.ones_like(lambda_space)
    for i, ratio in enumerate(harmonic_ratios[:3]):
        freq = base_freq * (i + 1) * ratio
        harmonic_modulation += 0.1 * np.sin(2 * np.pi * freq * lambda_space)

    # Apply phase patterns for genetic uniqueness
    phase_modulation = np.ones_like(lambda_space, dtype=complex)
    for i, phase in enumerate(phase_pattern[:4]):
        phase_freq = base_freq * (i + 1) * 0.5
        phase_component = np.exp(1j * (phase + 2 * np.pi * phase_freq * lambda_space))
        phase_modulation *= (0.9 + 0.1 * phase_component)

    # Combine all DNA modulations
    dna_modulation = base_modulation * harmonic_modulation * phase_modulation

    return dna_modulation


class PadilhaWave:
    """
    Enhanced PadilhaWave with GLS (Generalized Light Spectrum) integration.

    The wave carries the complete GLS visual signature, enabling genetic
    compatibility recognition through visual-spectral similarity analysis.
    """

    def __init__(self, emitter_signature: str, gls_layer: 'FractalGLS',
                 intensity: float = 1.0, steps: int = 256, time: float = 0.0):
        """
        Initialize PadilhaWave with GLS visual spectrum integration.

        Args:
            emitter_signature: String identifier for the emitter
            gls_layer: FractalGLS instance containing visual spectrum
            intensity: Wave amplitude
            steps: Number of spatial samples
            time: Time parameter for wave generation
        """
        # Store GLS visual spectrum and signature
        self.gls_visual = gls_layer
        self.signature = emitter_signature
        self.intensity = intensity
        self.steps = steps
        self.time = time

        # Generate waveform from GLS spectral features
        self.waveform = self.gls_to_waveform()

        # Store spectral parameters for analysis
        self.spectral_params = self.gls_visual.extract_spectral_features()

        # Create spatial domain
        self.lambda_space = np.linspace(0, 1, self.steps)

        # Legacy compatibility
        self.alpha = self.spectral_params['alpha']
        self.beta = self.spectral_params['beta']
        self.wave_shape = self.waveform

    def gls_to_waveform(self) -> np.ndarray:
        """
        Map GLS fractal points to wave parameters (α, β, ω) and generate waveform.
        This is the core innovation that preserves genetic identity through visual-spectral mapping.
        """
        # Extract spectral features from GLS
        spectral_params = self.gls_visual.extract_spectral_features()

        # Generate complex wave using extracted parameters
        waveform = complex_wave_equation(
            alpha=spectral_params['alpha'],
            beta=spectral_params['beta'],
            omega=spectral_params['omega'],
            amplitude=self.intensity * spectral_params['amplitude'],
            phase_shift=spectral_params['phase_shift'],
            harmonics=spectral_params['harmonics'],
            dna_signature=spectral_params['dna_signature'],
            steps=self.steps,
            t=self.time
        )

        return waveform

    def calculate_genetic_compatibility(self, other_wave: 'PadilhaWave') -> float:
        """
        Calculate genetic compatibility based on GLS visual-spectral similarity.

        This enables receivers to recognize genetic compatibility through
        visual-spectral analysis of the wave's GLS signature.

        Returns:
            Compatibility score (0.0 to 1.0)
        """
        if not isinstance(other_wave, PadilhaWave) or other_wave.gls_visual is None:
            return 0.0

        # Use GLS visual spectrum comparison for genetic compatibility
        visual_similarity = self.gls_visual.compare(other_wave.gls_visual)

        # Enhance with spectral parameter similarity
        spectral_similarity = self._compare_spectral_parameters(other_wave.spectral_params)

        # Combine visual and spectral similarities
        compatibility = 0.7 * visual_similarity + 0.3 * spectral_similarity

        return float(np.clip(compatibility, 0.0, 1.0))

    def _compare_spectral_parameters(self, other_params: dict) -> float:
        """Compare spectral parameters for enhanced compatibility assessment."""
        try:
            # Compare key wave parameters
            alpha_sim = np.exp(-abs(self.spectral_params['alpha'] - other_params['alpha']))
            beta_sim = np.exp(-abs(self.spectral_params['beta'] - other_params['beta']))
            omega_sim = np.exp(-abs(self.spectral_params['omega'] - other_params['omega']) / 10)

            # Compare fractal dimensions
            dim_diff = abs(self.spectral_params['fractal_dimension'] - other_params['fractal_dimension'])
            dim_sim = np.exp(-dim_diff)

            # Weighted average
            spectral_similarity = (0.3 * alpha_sim + 0.3 * beta_sim + 0.2 * omega_sim + 0.2 * dim_sim)

            return float(spectral_similarity)

        except (KeyError, TypeError):
            return 0.0

    def propagate(self, chaos_factor: float) -> np.ndarray:
        """
        Enhanced wave propagation through chaotic medium with GLS preservation.
        """
        # Apply chaos distortion while preserving GLS signature
        phase_distortion = np.exp(1j * chaos_factor * np.sin(np.pi * self.lambda_space)**3)

        # Reduce chaos impact on low-frequency components (genetic signature preservation)
        chaos_protection = 0.8 + 0.2 * np.exp(-self.spectral_params['fractal_dimension'] * chaos_factor)
        protected_chaos = phase_distortion * chaos_protection

        distorted_wave = self.waveform * protected_chaos

        return distorted_wave

    def extract_genetic_signature(self) -> dict:
        """Extract genetic signature for identity preservation and analysis."""
        return {
            'emitter_signature': self.signature,
            'gls_hash': self.gls_visual.spectrum_hash,
            'dna_signature': self.gls_visual.dna_signature,
            'fractal_dimension': self.gls_visual.fractal_dimension,
            'spectral_params': self.spectral_params,
            'visual_spectrum_shape': self.gls_visual.visual_spectrum.shape,
            'wave_energy': float(np.sum(np.abs(self.waveform)**2))
        }

    def get_wave_analysis(self) -> dict:
        """Get comprehensive wave analysis for debugging and visualization."""
        return {
            'signature': self.signature,
            'intensity': self.intensity,
            'steps': self.steps,
            'spectral_features': self.spectral_params,
            'wave_energy': float(np.sum(np.abs(self.waveform)**2)),
            'wave_complexity': float(np.std(np.abs(self.waveform))),
            'gls_dimension': self.gls_visual.fractal_dimension,
            'genetic_signature': self.extract_genetic_signature()
        }

    def __repr__(self) -> str:
        return (f"PadilhaWave(signature='{self.signature}', "
                f"gls_dim={self.gls_visual.fractal_dimension:.3f}, "
                f"energy={np.sum(np.abs(self.waveform)**2):.3f})")


# Legacy compatibility
class PadilhaWaveLegacy:
    """Legacy PadilhaWave implementation for backward compatibility."""
    def __init__(self, emitter_signature: tuple, intensity: float = 1.0, steps: int = 256):
        self.alpha, self.beta = emitter_signature
        self.intensity = intensity
        self.steps = steps

        self.lambda_space = np.linspace(0, 1, self.steps)
        real_part = np.sin(2 * np.pi * (self.alpha * self.lambda_space))
        imag_part = np.cos(2 * np.pi * (self.beta * self.lambda_space**2))

        self.wave_shape = self.intensity * (real_part + 1j * imag_part)

    def propagate(self, chaos_factor: float):
        phase_distortion = np.exp(1j * chaos_factor * np.sin(np.pi * self.lambda_space)**3)
        distorted_wave = self.wave_shape * phase_distortion
        return distorted_wave

class WaveAnalyzer:
    """
    Represents the agent's ability (via the ΨQRH model) to analyze a received wave.
    This class directly uses the QRHLayer for its analysis.
    """
    def __init__(self, embed_dim: int = 128, device: str = 'cpu'):
        # A minimal config for the QRHLayer used in analysis
        qrh_config = QRHConfig(
            embed_dim=embed_dim,
            alpha=1.5, # A baseline alpha for the analyzer
            use_learned_rotation=False,
            spatial_dims=None,
            device=device
        )
        self.qrh_analyzer = QRHLayer(qrh_config).to(device)
        self.device = device
        self.embed_dim = embed_dim

    def analyze_correlation(self, received_wave: np.ndarray, expected_signature: tuple) -> float:
        """
        Calculates the correlation between the received wave and an expected signature
        by processing both through the ΨQRH layer and comparing the outputs.
        """
        alpha_expected, beta_expected = expected_signature
        
        # 1. Create the "ideal" wave the receiver expects
        ideal_wave_generator = PadilhaWave(emitter_signature=expected_signature, steps=len(received_wave))
        ideal_wave = ideal_wave_generator.wave_shape

        # 2. Prepare tensors for the QRHLayer
        received_tensor = self._wave_to_tensor(received_wave)
        ideal_tensor = self._wave_to_tensor(ideal_wave)

        # 3. Process both waves through the actual QRHLayer
        with torch.no_grad():
            processed_received = self.qrh_analyzer(received_tensor)
            processed_ideal = self.qrh_analyzer(ideal_tensor)

        # 4. Compare the processed outputs
        similarity = torch.nn.functional.cosine_similarity(
            processed_received.flatten(), 
            processed_ideal.flatten(), 
            dim=0
        )
        
        return max(0, similarity.item())

    def _wave_to_tensor(self, wave: np.ndarray) -> torch.Tensor:
        """
        Converts a 1D complex wave into a tensor suitable for the QRHLayer.
        This version maps the wave to a sequence of quaternion embeddings.
        """
        seq_len = len(wave)
        # Reshape the wave to (seq_len, 1) and expand to (seq_len, 2)
        # where column 0 is real and column 1 is imag.
        complex_components = torch.from_numpy(np.vstack([np.real(wave), np.imag(wave)]).T).float()

        # The layer expects (batch, seq_len, 4 * embed_dim)
        # We will treat our wave as a sequence of length `seq_len`
        # and each point in the wave as a partial embedding.
        tensor = torch.zeros(1, seq_len, 4 * self.embed_dim, device=self.device)

        # Map real to the 'w' component of the first embedding vector
        tensor[0, :, 0] = complex_components[:, 0]
        # Map imag to the 'x' component of the first embedding vector
        tensor[0, :, 1] = complex_components[:, 1]
        
        return tensor
