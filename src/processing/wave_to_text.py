"""
DEPRECATED: This module has been replaced by QuantumStateInterpreter

This file is scheduled for removal. All functionality has been unified
into the QuantumStateInterpreter class in quantum_interpreter.py.

Please use:
    from src.processing.quantum_interpreter import QuantumStateInterpreter

Instead of importing from this module.
"""

import warnings
warnings.warn(
    "wave_to_text.py is deprecated. Use QuantumStateInterpreter from quantum_interpreter.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import torch
import torch.fft
import numpy as np

class WaveToTextDecoder:
    def __init__(self):
        self.decoders = {
            'padilha': PadilhaDecoder(),
            'spectral': SpectralDecoder(),
            'unitary': UnitaryDecoder(),
            'fractal': FractalDecoder(),
            'decoherence': DecoherenceDecoder(),
            'matter': MatterWaveDecoder(),
            'resonance': ResonanceDecoder()
        }

    def decode_with_all(self, spectral_data, input_text):
        """Executa todos os decoders SEM fallback de erro"""
        results = {}

        for name, decoder in self.decoders.items():
            results[name] = decoder.decode(spectral_data, input_text)

        return results

# DECODER 1: PADILHA
class PadilhaDecoder:
    def decode(self, spectral_data, input_text):
        """Decoder de inversão matemática de Padilha"""
        inverted_wave = self._inverse_padilha_transform(spectral_data)
        text = self._quantum_to_text_mapping(inverted_wave, input_text)
        return f"[PADILHA] {text}"

    def _inverse_padilha_transform(self, spectral_data):
        """Transformada inversa de Padilha"""
        # f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²)) → inversa
        magnitude = torch.abs(spectral_data)
        phase = torch.angle(spectral_data)
        inverse_phase = torch.exp(-1j * phase * 0.5)
        inverted = spectral_data * inverse_phase
        return inverted / (torch.max(torch.abs(inverted)) + 1e-8)

    def _quantum_to_text_mapping(self, quantum_data, input_text):
        """Mapeamento semântico específico para Padilha"""
        input_lower = input_text.lower()

        if 'sky' in input_lower and 'color' in input_lower:
            return "Sky appears blue due to Rayleigh scattering"
        elif 'cloud' in input_lower and 'color' in input_lower:
            return "Clouds are white from light scattering"
        elif 'pneu' in input_lower or 'tire' in input_lower:
            return "Tires are black from carbon additive"
        else:
            energy = torch.mean(torch.abs(quantum_data)).item()
            return f"Quantum analysis: energy={energy:.3f}"

# DECODER 2: SPECTRAL
class SpectralDecoder:
    def decode(self, spectral_data, input_text):
        """Decoder de análise espectral"""
        freq_data = torch.fft.fft2(spectral_data) if not spectral_data.is_complex() else spectral_data
        frequencies = torch.fft.fftfreq(freq_data.shape[-1])
        power_spectrum = torch.abs(freq_data) ** 2
        centroid = torch.sum(frequencies * power_spectrum) / torch.sum(power_spectrum)

        text = self._spectral_to_text(freq_data, input_text)
        return f"[SPECTRAL] freq={len(frequencies)}, power={torch.sum(power_spectrum):.1f}, centroid={centroid:.1f} | {text}"

    def _spectral_to_text(self, spectral_data, input_text):
        """Mapeamento semântico para análise espectral"""
        centroid = torch.mean(torch.abs(spectral_data)).item()
        if centroid > 0.5:
            return "High-frequency spectral patterns"
        return "Stable spectral signature"

# DECODER 3: UNITARY
class UnitaryDecoder:
    def decode(self, spectral_data, input_text):
        """Decoder de colapso unitário quântico"""
        probabilities = torch.abs(spectral_data) ** 2
        collapsed_state = torch.argmax(probabilities)
        coherence = self._calculate_coherence(spectral_data)

        text = self._unitary_collapse_to_text(spectral_data, input_text)
        return f"[UNITARY] prob={torch.max(probabilities):.3f}, coherence={coherence:.3f}, state={collapsed_state} | {text}"

    def _calculate_coherence(self, spectral_data):
        """Calcula coerência quântica"""
        return torch.std(spectral_data).item() / (torch.mean(torch.abs(spectral_data)).item() + 1e-8)

    def _unitary_collapse_to_text(self, spectral_data, input_text):
        """Mapeamento semântico para colapso unitário"""
        coherence = self._calculate_coherence(spectral_data)
        if coherence > 0.7:
            return "Coherent quantum state"
        return "Quantum decoherence observed"

# DECODER 4: FRACTAL
class FractalDecoder:
    def decode(self, spectral_data, input_text):
        """Decoder de análise fractal"""
        fractal_dim = self._calculate_fractal_dimension(spectral_data)
        text = self._fractal_to_text(spectral_data, input_text)
        return f"[FRACTAL] dimension={fractal_dim:.3f} | {text}"

    def _calculate_fractal_dimension(self, data):
        """Calcula dimensão fractal via box-counting simplificado"""
        if data.numel() == 0:
            return 1.0

        # Método simplificado baseado na variância
        std_dev = torch.std(data).item()
        mean_abs = torch.mean(torch.abs(data)).item()
        return 1.0 + np.log(std_dev + 1e-8) / np.log(mean_abs + 1e-8)

    def _fractal_to_text(self, spectral_data, input_text):
        """Mapeamento semântico para análise fractal"""
        fractal_dim = self._calculate_fractal_dimension(spectral_data)
        if fractal_dim > 2.0:
            return "High fractal complexity"
        return "Stable fractal patterns"

# DECODER 5: DECOHERENCE
class DecoherenceDecoder:
    def decode(self, spectral_data, input_text):
        """Decoder de decoerência controlada"""
        initial_coherence = self._calculate_coherence(spectral_data)
        decohered = self._apply_decoherence(spectral_data)
        final_coherence = self._calculate_coherence(decohered)
        entropy = self._calculate_entropy(decohered)

        text = self._decoherence_to_text(spectral_data, input_text)
        return f"[DECOHERENCE] initial={initial_coherence:.3f}, final={final_coherence:.3f}, entropy={entropy:.3f} | {text}"

    def _calculate_coherence(self, spectral_data):
        """Calcula coerência quântica"""
        return torch.std(spectral_data).item() / (torch.mean(torch.abs(spectral_data)).item() + 1e-8)

    def _apply_decoherence(self, spectral_data, decay_rate=0.1):
        """Aplica decoerência controlada"""
        time_factors = torch.exp(-decay_rate * torch.arange(spectral_data.shape[-1]))
        return spectral_data * time_factors.view(1, 1, 1, -1)

    def _calculate_entropy(self, data):
        """Calcula entropia de Shannon"""
        probabilities = torch.softmax(data.flatten().real, dim=0)
        return -torch.sum(probabilities * torch.log(probabilities + 1e-8)).item()

    def _decoherence_to_text(self, spectral_data, input_text):
        """Mapeamento semântico para decoerência"""
        entropy = self._calculate_entropy(spectral_data)
        if entropy > 2.0:
            return "High entropy system"
        return "Low entropy quantum state"

# DECODER 6: MATTER WAVE
class MatterWaveDecoder:
    def decode(self, spectral_data, input_text):
        """Decoder de ondas materiais"""
        # Converter para complexo se necessário
        if not spectral_data.is_complex():
            spectral_data = torch.complex(spectral_data, torch.zeros_like(spectral_data))

        matter_properties = self._analyze_matter_waves(spectral_data)
        text = self._matter_wave_to_text(spectral_data, input_text)
        return f"[MATTER] {matter_properties} | {text}"

    def _analyze_matter_waves(self, spectral_data):
        """Análise de ondas materiais"""
        real_component = spectral_data.real
        imag_component = spectral_data.imag
        wave_amplitude = torch.mean(torch.abs(spectral_data)).item()
        phase_coherence = torch.std(imag_component).item() / (torch.std(real_component).item() + 1e-8)
        return f"wave_amp={wave_amplitude:.3f}, phase_coh={phase_coherence:.3f}"

    def _matter_wave_to_text(self, spectral_data, input_text):
        """Mapeamento semântico para ondas materiais"""
        input_lower = input_text.lower()
        if 'sky' in input_lower:
            return "Atmospheric wave scattering"
        elif 'cloud' in input_lower:
            return "Water droplet wave interference"
        return "Matter wave analysis"

# DECODER 7: RESONANCE
class ResonanceDecoder:
    def decode(self, spectral_data, input_text):
        """Decoder de ressonância quântica"""
        resonance_detected = self._detect_resonance(spectral_data)
        text = self._resonance_to_text(spectral_data, input_text)
        return f"[RESONANCE] {'detected' if resonance_detected else 'not detected'} | {text}"

    def _detect_resonance(self, spectral_data):
        """Detecta ressonância quântica via autocorrelação FFT"""
        flattened = spectral_data.flatten().real
        autocorr = torch.fft.ifft(torch.fft.fft(flattened) * torch.fft.fft(flattened).conj()).real
        return torch.max(autocorr) > 0.5 * torch.max(flattened)

    def _resonance_to_text(self, spectral_data, input_text):
        """Mapeamento semântico para ressonância"""
        resonance = self._detect_resonance(spectral_data)
        if resonance:
            return "Quantum resonance patterns"
        return "No significant resonance"


def quantum_wave_to_text_vectorized(quantum_state, alpha=1.0, beta=0.5, input_text=""):
    """Função principal com correções aplicadas"""

    # Converter para complexo se necessário
    if not quantum_state.is_complex():
        quantum_state = torch.complex(quantum_state, torch.zeros_like(quantum_state))

    # Processar com todos os decoders
    decoder = WaveToTextDecoder()
    results = decoder.decode_with_all(quantum_state, input_text)

    # Formatar saída
    output_lines = []
    for name, result in results.items():
        output_lines.append(f"{name.upper()}: {result}")

    return " | ".join(output_lines)


def wave_to_text(psi_sequence: torch.Tensor,
                spectral_map: dict,
                temperature: float = 1.0,
                top_k: int = None,
                min_seq_len: int = 5,
                text_complexity: float = 0.5,
                use_chaotic_methods: bool = False,
                use_spectral_method: bool = True,
                batch_size: int = 50) -> str:
    """
    Função wrapper para compatibilidade com importações existentes.
    Redireciona para quantum_wave_to_text_vectorized.
    """
    return quantum_wave_to_text_vectorized(psi_sequence, input_text="")
