"""
Physical Tokenizer: Self-Contained Quantum Text Processing
==========================================================

Implements a physically-grounded tokenizer that eliminates dependency on external
tokenizers like GPT-2. Text identity is inseparable from its spectral representation.

Based on doe.md mathematical framework:
- Text → Quaternion Spectral Representation
- Probe Similarity Decoding (no external vocabularies)
- Complete self-contained quantum text processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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

        if self.learnable:
            # Phase 2: Learnable spectral embedding
            # Each character gets spectral_params_dim learnable parameters
            self.spectral_embedding = nn.Embedding(self.num_chars, spectral_params_dim)

            # Initialize with reasonable values to avoid training instability
            nn.init.xavier_uniform_(self.spectral_embedding.weight)

            print("🎵 Phase 2 ACTIVATED: Adaptive Spectral Vocabulary")
            print(f"   📊 Learnable parameters: {self.num_chars} chars × {spectral_params_dim} params = {self.num_chars * spectral_params_dim}")
        else:
            # Phase 1: Deterministic mathematical mapping (legacy)
            print("🔢 Phase 1 ACTIVE: Deterministic Mathematical Mapping")

        print(f"✅ PhysicalTokenizer initialized: embed_dim={embed_dim}, ascii_range=({self.ascii_start}, {self.ascii_end})")
        print(f"   📊 Character vocabulary: {self.num_chars} characters")
        print(f"   🎛️  Spectral control panel: {spectral_params_dim} parameters per character")

    def _generate_probe_wave(self, char_code: int, position: int) -> torch.Tensor:
        """
        Generate learnable probe wave for a character using spectral parameters.

        This method uses the learnable spectral embedding to create adaptive
        quantum wave signatures for each character.

        Args:
            char_code: ASCII code of the character (0-94 for printable ASCII)
            position: Position in sequence (affects phase modulation)

        Returns:
            Quaternion probe wave [embed_dim, 4]
        """
        # Convert ASCII code to embedding index (0-based)
        char_idx = char_code - self.ascii_start

        # Ensure valid range
        if char_idx < 0 or char_idx >= self.num_chars:
            # Fallback to space character
            char_idx = ord(' ') - self.ascii_start

        if self.learnable:
            # Phase 2: Use learnable spectral parameters
            params = self.spectral_embedding(torch.tensor(char_idx))

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
            ascii_val = char_code
            j = torch.arange(self.embed_dim, dtype=torch.float32)

            # Fixed mathematical formula
            phase = (ascii_val + position + j) * 2 * math.pi / 256.0
            amplitude = (ascii_val / 127.0) * (j / self.embed_dim)

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
        Decode a single quantum state to character using learnable probe similarity.

        Args:
            psi_state: Quantum state tensor [embed_dim, 4]
            position: Position in sequence (affects probe generation)

        Returns:
            Decoded character string (single character)
        """
        # Calculate similarity with all possible character probes
        similarities = self._probe_similarity(psi_state, position)

        # Find character with highest similarity
        best_char_index = torch.argmax(similarities).item()

        # Convert index back to ASCII code
        best_ascii = self.ascii_start + best_char_index

        # Ensure valid ASCII range
        best_ascii = max(self.ascii_start, min(best_ascii, self.ascii_end))

        # Convert ASCII back to character
        decoded_char = chr(best_ascii)

        return decoded_char

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

    def _probe_similarity(self, psi: torch.Tensor, position: int) -> torch.Tensor:
        """
        Calculate similarity between quantum state and learnable character probes.

        Uses optical probe measurement principle from doe.md with learnable spectral parameters.

        Args:
            psi: Quantum state [embed_dim, 4]
            position: Position in sequence

        Returns:
            Similarity scores for each ASCII character [num_chars]
        """
        similarities = []

        # Generate probe for each character using learnable parameters
        for char_code in range(self.ascii_start, self.ascii_end + 1):
            # Get learnable probe wave for this character
            probe_wave = self._generate_probe_wave(char_code, position)  # [embed_dim, 4]

            # Calculate cosine similarity between input state and probe
            similarity = F.cosine_similarity(psi.unsqueeze(0), probe_wave.unsqueeze(0), dim=-1)  # [embed_dim]

            # Sum similarities across embedding dimension
            total_similarity = torch.sum(similarity, dim=-1)  # scalar
            similarities.append(total_similarity)

        # Stack all character similarities
        return torch.stack(similarities)  # [num_chars]

    def get_vocabulary_info(self) -> dict:
        """
        Get information about the tokenizer's adaptive vocabulary.

        Returns:
            Dictionary with vocabulary statistics and learnable parameters
        """
        return {
            'vocabulary_size': self.num_chars,
            'ascii_range': (self.ascii_start, self.ascii_end),
            'embed_dim': self.embed_dim,
            'spectral_params_dim': self.spectral_params_dim,
            'learnable': self.learnable,
            'total_learnable_params': self.num_chars * self.spectral_params_dim if self.learnable else 0,
            'phase': 'Phase 2: Adaptive Spectral Vocabulary' if self.learnable else 'Phase 1: Deterministic Mapping',
            'character_sample': ''.join(chr(self.ascii_start + i) for i in range(min(10, self.num_chars)))
        }