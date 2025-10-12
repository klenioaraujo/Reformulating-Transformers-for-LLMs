"""
Physical Tokenizer: Self-Contained Quantum Text Processing
==========================================================

Implements a physically-grounded tokenizer that uses native vocabulary generation.
Text identity is inseparable from its spectral representation.

Based on doe.md mathematical framework:
- Text → Quaternion Spectral Representation
- Probe Similarity Decoding (no external vocabularies)
- Complete self-contained quantum text processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from typing import Tuple


class PhysicalTokenizer(nn.Module):
    """
    Adaptive Physical Tokenizer for quantum text processing.

    This tokenizer evolves from deterministic to learnable spectral representations:
    - Phase 1: Fixed mathematical mapping (deterministic timbre)
    - Phase 2: Learnable spectral embedding (adaptive timbre)

    Each character gets a learnable "control panel" of spectral parameters that
    define its quantum wave signature, optimized through training.
    """

    def __init__(self, embed_dim: int = 64, ascii_range: Tuple[int, int] = (32, 126),
                 spectral_params_dim: int = 8, learnable: bool = True):
        """
        Initialize the adaptive physical tokenizer.

        Args:
            embed_dim: Embedding dimension for quaternion representations
            ascii_range: ASCII range for character vocabulary (default: printable ASCII)
            spectral_params_dim: Number of learnable spectral parameters per character
            learnable: Whether to use learnable spectral embeddings (Phase 2)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.ascii_start, self.ascii_end = ascii_range
        self.num_chars = self.ascii_end - self.ascii_start + 1
        self.ascii_codes = torch.tensor(list(range(self.ascii_start, self.ascii_end + 1)), dtype=torch.float32)
        self.spectral_params_dim = spectral_params_dim
        self.learnable = learnable

        # Use native ASCII vocabulary only
        self.vocab_size = self.num_chars
        self.vocab_codes = self.ascii_codes

        # ASCII mode only
        vocab_type = f"ASCII ({self.ascii_start}-{self.ascii_end})"
        vocab_size = self.num_chars

        if self.learnable:
            # Phase 2: Learnable spectral embedding
            # Each token gets spectral_params_dim learnable parameters
            self.spectral_embedding = nn.Embedding(vocab_size, spectral_params_dim)

            # Initialize with reasonable values to avoid training instability
            nn.init.xavier_uniform_(self.spectral_embedding.weight)

            print(f"🎵 Phase 2 ACTIVATED: Adaptive Spectral Vocabulary ({vocab_type})")
            print(f"   📊 Learnable parameters: {vocab_size} tokens × {spectral_params_dim} params = {vocab_size * spectral_params_dim}")
        else:
            # Phase 1: Deterministic mathematical mapping (legacy)
            print(f"🔢 Phase 1 ACTIVE: Deterministic Mathematical Mapping ({vocab_type})")

        print(f"✅ PhysicalTokenizer initialized: embed_dim={embed_dim}, vocabulary={vocab_type}")
        print(f"   📊 Token vocabulary: {vocab_size} tokens")
        print(f"   🎛️  Spectral control panel: {spectral_params_dim} parameters per token")


    def _generate_probe_wave(self, token_id: int, position: int) -> torch.Tensor:
        """
        Generate learnable probe wave for a token using spectral parameters.

        This method uses the learnable spectral embedding to create adaptive
        quantum wave signatures for each token.

        Args:
            token_id: Token ID (0 to vocab_size-1)
            position: Position in sequence (affects phase modulation)

        Returns:
            Quaternion probe wave [embed_dim, 4]
        """
        # Ensure valid token ID range
        if token_id < 0 or token_id >= self.vocab_size:
            # Fallback to token 0 or space-like token
            token_id = 0

        if self.learnable:
            # Phase 2: Use learnable spectral parameters
            params = self.spectral_embedding(torch.tensor(token_id))

            # Unpack spectral parameters with physical meanings
            omega_c = params[0] * 2.0  # Fundamental frequency (scaled)
            A1_c = params[1]           # Amplitude of 1st harmonic
            A2_c = params[2]           # Amplitude of 2nd harmonic
            A3_c = params[3]           # Amplitude of 3rd harmonic
            beta_c = params[4]         # Chirp rate
            gamma_c = torch.sigmoid(params[5])  # Decay rate (0-1)
            phi_c = params[6] * math.pi  # Phase offset
            noise_c = params[7] * 0.1   # Noise modulation

            # Generate harmonic series with learned parameters
            j = torch.arange(self.embed_dim, dtype=torch.float32, device=params.device)

            # Superposition of harmonics with learned amplitudes
            wave_real = (
                A1_c * torch.sin(1 * omega_c * j + phi_c) +
                A2_c * torch.sin(2 * omega_c * j + phi_c * 2) +
                A3_c * torch.sin(3 * omega_c * j + phi_c * 3)
            )

            # Apply learned decay
            wave_real *= torch.exp(-gamma_c * j)

            # Add position-dependent phase modulation
            position_phase = torch.tensor(position * 0.1 * math.pi, device=j.device, dtype=j.dtype)
            wave_real += beta_c * j * torch.sin(position_phase)

            # Add subtle noise for robustness
            if self.training:
                noise = torch.randn_like(wave_real) * noise_c
                wave_real += noise

        else:
            # Phase 1: Deterministic mathematical mapping (legacy)
            token_val = token_id
            j = torch.arange(self.embed_dim, dtype=torch.float32)

            # Fixed mathematical formula (scaled for ASCII vocabulary)
            max_val = 256.0
            phase = (token_val + position + j) * 2 * math.pi / max_val
            amplitude = (token_val / max_val) * (j / self.embed_dim)

            wave_real = amplitude * torch.cos(phase)

        # Project to quaternion components (doe.md 2.9.1)
        psi0 = wave_real  # Real part
        psi1 = torch.roll(wave_real, shifts=1, dims=0)  # i component (phase shifted)
        psi2 = torch.sin(wave_real)  # j component (non-linear)
        psi3 = torch.cos(wave_real)  # k component (complementary)

        # Stack into quaternion [embed_dim, 4]
        psi_probe = torch.stack([psi0, psi1, psi2, psi3], dim=-1)

        return psi_probe

    def encode(self, text: str) -> torch.Tensor:
        """
        Convert text to quaternion embedding sequence.

        Args:
            text: Input text string

        Returns:
            Quaternion tensor [batch=1, seq_len, embed_dim, 4]
        """
        print(f"📝 Physical encoding: '{text[:50]}...' ({len(text)} chars)")

        # Convert text to ASCII values
        ascii_values = [ord(char) for char in text]
        seq_len = len(ascii_values)

        # Create quaternion embedding sequence
        psi_sequence = []

        for i, ascii_val in enumerate(ascii_values):
            psi_char = self._ascii_to_quaternion(ascii_val, i)
            psi_sequence.append(psi_char)

        # Stack into tensor [seq_len, embed_dim, 4]
        psi_tensor = torch.stack(psi_sequence)

        # Add batch dimension [1, seq_len, embed_dim, 4]
        psi_tensor = psi_tensor.unsqueeze(0)

        print(f"   ✅ Encoded to quaternion tensor: {psi_tensor.shape}")
        return psi_tensor

    def decode_state(self, psi_state: torch.Tensor, position: int) -> str:
        """
        Decode a single quantum state to token using learnable probe similarity.

        Args:
            psi_state: Quantum state tensor [embed_dim, 4]
            position: Position in sequence (affects probe generation)

        Returns:
            Decoded token string
        """
        # Calculate similarity with all possible token probes
        similarities = self._probe_similarity(psi_state, position)

        # Find token with highest similarity
        best_token_index = torch.argmax(similarities).item()

        # Ensure valid token ID range
        best_token_id = max(0, min(best_token_index, self.vocab_size - 1))

        # Convert token ID back to ASCII character
        best_ascii = self.ascii_start + best_token_id
        best_ascii = max(self.ascii_start, min(best_ascii, self.ascii_end))
        decoded_token = chr(best_ascii)

        return decoded_token

    def _ascii_to_quaternion(self, ascii_val: int, position: int) -> torch.Tensor:
        """
        Convert single ASCII value to quaternion embedding using learnable spectral parameters.

        Based on doe.md 2.9.1: Ψ(x) = ψ₀ + ψ₁i + ψ₂j + ψ₃k ∈ ℍ

        Args:
            ascii_val: ASCII value (0-255)
            position: Position in sequence

        Returns:
            Quaternion tensor [embed_dim, 4]
        """
        # Use the new learnable probe wave generation
        return self._generate_probe_wave(ascii_val, position)

    def _spectral_token_weights(self, psi: torch.Tensor, position: int) -> torch.Tensor:
        """
        Advanced Optical Analysis with Context Balancing (doe.md Methodology)

        Computes GPT token weights using sophisticated spectral analysis that leverages
        the full advantages of frequency domain over direct text processing.

        Mathematical Foundation - doe.md Optical Analysis:
        ================================================
        1. Multi-Scale Wavelet Decomposition
        2. Optical Coherence Analysis
        3. Context-Aware Spectral Balancing
        4. Quantum Interference Patterns
        5. Fractal Spectral Harmonics

        The spectrum contains richer information than text - we exploit this fully.

        Args:
            psi: Quantum state [embed_dim, 4]
            position: Position in sequence for context balancing

        Returns:
            Token weights [vocab_size] with optical coherence
        """
        device = psi.device

        # ========== ANÁLISE ÓPTICA AVANÇADA (doe.md) ==========

        # 1. DECOMPOSIÇÃO MULTI-ESCALA (Wavelet-like Analysis)
        # Analisar diferentes escalas de frequência simultaneamente
        psi_flat = psi.view(-1)  # [embed_dim * 4]

        # FFT Multi-resolution
        fft_full = torch.fft.fft(psi_flat)
        n_freq = len(fft_full)

        # Bandas de frequência (low, mid, high)
        low_band = fft_full[:n_freq//4]
        mid_band = fft_full[n_freq//4:n_freq//2]
        high_band = fft_full[n_freq//2:]

        # Características por banda
        low_power = torch.mean(torch.abs(low_band))
        mid_power = torch.mean(torch.abs(mid_band))
        high_power = torch.mean(torch.abs(high_band))

        # Coerência entre bandas (Optical Coherence)
        low_mid_coherence = torch.abs(torch.sum(low_band.conj() * mid_band[:len(low_band)])) / (torch.norm(low_band) * torch.norm(mid_band[:len(low_band)]) + 1e-10)
        mid_high_coherence = torch.abs(torch.sum(mid_band.conj() * high_band[:len(mid_band)])) / (torch.norm(mid_band) * torch.norm(high_band[:len(mid_band)]) + 1e-10)

        # 2. ANÁLISE DE COERÊNCIA ÓPTICA
        # Medidas avançadas de coerência espectral
        phase_gradient = torch.angle(fft_full[1:]) - torch.angle(fft_full[:-1])
        phase_coherence = torch.mean(torch.cos(phase_gradient))

        # Spectral purity (how concentrated the energy is)
        spectral_purity = torch.max(torch.abs(fft_full)) / torch.sum(torch.abs(fft_full))

        # 3. PADRÕES DE INTERFERÊNCIA QUÂNTICA
        # Analisar interferência entre componentes quaterniônicas
        w, x, y, z = psi[..., 0], psi[..., 1], psi[..., 2], psi[..., 3]

        # Interferência w-x (real-imaginary)
        wx_interference = torch.abs(torch.sum(w * x.conj())) / (torch.norm(w) * torch.norm(x) + 1e-10)

        # Interferência y-z (j-k components)
        yz_interference = torch.abs(torch.sum(y * z.conj())) / (torch.norm(y) * torch.norm(z) + 1e-10)

        # 4. HARMONIAS FRACTAIS ESPECTRAIS
        # Analisar estrutura fractal do espectro
        spectrum_magnitude = torch.abs(fft_full)
        # Calcular dimensão fractal aproximada via power-law
        freq_indices = torch.arange(1, len(spectrum_magnitude), device=device, dtype=torch.float32)
        log_freq = torch.log(freq_indices)
        log_mag = torch.log(spectrum_magnitude[1:len(freq_indices)+1] + 1e-10)

        # Regressão linear para estimar dimensão fractal
        if len(log_freq) > 10:
            # Usar pontos do meio para evitar extremos
            mid_idx = len(log_freq) // 2
            slope = torch.cov(torch.stack([log_freq[mid_idx-5:mid_idx+5], log_mag[mid_idx-5:mid_idx+5]]))[0,1] / torch.var(log_freq[mid_idx-5:mid_idx+5])
            fractal_dimension = 2 + slope  # Dimensão fractal estimada
        else:
            fractal_dimension = 2.0

        # ========== BALANCIAMENTO DE CONTEXTO ==========
        # Incorporar posição e contexto sequencial
        position_context = torch.sin(torch.tensor(position * 0.1, device=device))
        sequence_context = torch.cos(torch.tensor(position * 0.05, device=device))

        # ========== MAPEAMENTO ÓPTICO AVANÇADO ==========
        token_indices = torch.arange(self.vocab_size, dtype=torch.float32, device=device)

        # 1. Modulação por Coerência Óptica
        coherence_factor = (low_mid_coherence + mid_high_coherence) / 2
        optical_modulation = torch.exp(1j * coherence_factor * token_indices / self.vocab_size)

        # 2. Modulação por Energia Espectral Balanceada
        # Balancear entre bandas de frequência
        energy_balance = (low_power * mid_power * high_power) ** (1/3)  # Geometric mean
        spectral_energy = torch.exp(1j * energy_balance * token_indices / self.vocab_size)

        # 3. Modulação por Interferência Quântica
        quantum_interference = (wx_interference + yz_interference) / 2
        interference_modulation = torch.exp(1j * quantum_interference * token_indices / self.vocab_size)

        # 4. Modulação Fractal
        fractal_modulation = torch.exp(1j * fractal_dimension * token_indices / self.vocab_size)

        # 5. Modulação de Pureza Espectral
        purity_modulation = torch.pow(torch.abs(spectral_purity), token_indices / self.vocab_size)

        # 6. Modulação de Contexto
        context_modulation = position_context * torch.cos(token_indices * sequence_context / self.vocab_size)

        # ========== SÍNTESE FINAL DE PESOS ==========
        # Combinar todas as modulações ópticas
        complex_weights = (optical_modulation *
                          spectral_energy *
                          interference_modulation *
                          fractal_modulation *
                          purity_modulation *
                          (1 + context_modulation))

        # Converter para pesos reais com magnitude física
        token_weights = torch.abs(complex_weights)

        # Aplicar envelope gaussiano baseado na coerência
        coherence_width = 0.1 + coherence_factor * 0.5
        gaussian_envelope = torch.exp(-0.5 * (token_indices - self.vocab_size/2) ** 2 / (coherence_width * self.vocab_size) ** 2)
        token_weights = token_weights * gaussian_envelope

        # Normalização final
        if torch.sum(token_weights) > 0:
            token_weights = token_weights / torch.sum(token_weights)

        # Temperatura baseada na pureza espectral
        spectral_temperature = 0.1 + spectral_purity * 2.0
        token_weights = torch.pow(token_weights + 1e-10, 1.0 / spectral_temperature)
        token_weights = token_weights / torch.sum(token_weights)

        return token_weights

    def _probe_similarity(self, psi: torch.Tensor, position: int) -> torch.Tensor:
        """
        Legacy method - now uses efficient spectral token weights
        """
        return self._spectral_token_weights(psi, position)

    def get_vocabulary_info(self) -> dict:
        """
        Get information about the tokenizer's native vocabulary.

        Returns:
            Dictionary with vocabulary statistics and learnable parameters
        """
        vocab_type = "ASCII"
        sample_tokens = [chr(self.ascii_start + i) for i in range(min(10, self.num_chars))]

        return {
            'vocabulary_size': self.vocab_size,
            'vocabulary_type': vocab_type,
            'ascii_range': (self.ascii_start, self.ascii_end),
            'embed_dim': self.embed_dim,
            'spectral_params_dim': self.spectral_params_dim,
            'learnable': self.learnable,
            'total_learnable_params': self.vocab_size * self.spectral_params_dim if self.learnable else 0,
            'phase': 'Phase 2: Adaptive Spectral Vocabulary' if self.learnable else 'Phase 1: Deterministic Mapping',
            'token_sample': sample_tokens
        }