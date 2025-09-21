
import torch
import numpy as np
from qrh_layer import QRHLayer, QRHConfig

class PadilhaWave:
    """
    Represents a communication wave based on the Padilha Wave Equation.
    The wave carries a unique emitter "signature" (alpha, beta).
    """
    def __init__(self, emitter_signature: tuple, intensity: float = 1.0, steps: int = 256):
        self.alpha, self.beta = emitter_signature
        self.intensity = intensity
        self.steps = steps
        
        # f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
        # We represent the wave's shape over a normalized spatial dimension λ
        self.lambda_space = np.linspace(0, 1, self.steps)
        # For simplicity, we model the core shape without the time component
        real_part = np.sin(2 * np.pi * (self.alpha * self.lambda_space))
        imag_part = np.cos(2 * np.pi * (self.beta * self.lambda_space**2)) # Using cos for imag part to create complex signal
        
        self.wave_shape = self.intensity * (real_part + 1j * imag_part)

    def propagate(self, chaos_factor: float):
        """
        Simulates the wave's propagation through a chaotic medium.
        The chaos factor introduces a non-linear phase distortion.
        """
        # Chaos non-linearly affects the wave's phase
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
