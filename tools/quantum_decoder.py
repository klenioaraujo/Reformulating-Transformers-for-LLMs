#!/usr/bin/env python3
"""
ΨQRH Quantum Decoder
====================

Direct quantum-to-text conversion using probe similarity.
Based on the complete spectral pipeline reference implementation.

This decoder converts quantum states back to text by measuring similarity
between quantum wave functions and character-specific probe quaternions.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, List


class QuantumDecoder:
    """
    Direct Quantum-to-Text Decoder using probe similarity.

    This decoder reconstructs text by finding the character whose quantum
    probe quaternion has the highest similarity to the measured quantum state.
    Based on the reference implementation's approach.
    """

    def __init__(self, embed_dim: int = 64, device: str = 'cpu'):
        """
        Initialize the Quantum Decoder.

        Args:
            embed_dim: Embedding dimension used in quantum states
            device: Device for tensor operations
        """
        self.embed_dim = embed_dim
        self.device = device

        # ASCII printable characters (32-126)
        self.ascii_codes = torch.tensor(list(range(32, 127)), dtype=torch.float32, device=device)
        self.num_chars = len(self.ascii_codes)

        print(f"🔬 QuantumDecoder initialized: embed_dim={embed_dim}, vocab_size={self.num_chars}")

    def create_probe_quaternions(self, ascii_codes: torch.Tensor, position: int, embed_dim: int) -> torch.Tensor:
        """
        Create probe quaternions for given ASCII codes at a specific position.
        Simplified approach based on reference implementation.

        Args:
            ascii_codes: ASCII code values [num_chars]
            position: Position in sequence (for phase calculation)
            embed_dim: Embedding dimension

        Returns:
            Probe quaternions [num_chars, embed_dim, 4]
        """
        j = torch.arange(embed_dim, device=self.device).unsqueeze(0)  # [1, embed_dim]
        ascii_val = ascii_codes.unsqueeze(1)  # [num_chars, 1]

        # Create deterministic quaternion components (simplified)
        phase = (ascii_val + position + j) * 2 * math.pi / 256.0  # [num_chars, embed_dim]
        amplitude = (ascii_val / 127.0) * (j / embed_dim)  # [num_chars, embed_dim]

        # Simplified quaternion: focus on principal phase components
        psi_probe_0 = amplitude * torch.cos(phase)  # Real part
        psi_probe_1 = amplitude * torch.sin(phase)  # i component
        psi_probe_2 = torch.zeros_like(amplitude)   # j component (zeroed)
        psi_probe_3 = torch.zeros_like(amplitude)   # k component (zeroed)

        # Stack into quaternions
        probe_quaternions = torch.stack([psi_probe_0, psi_probe_1, psi_probe_2, psi_probe_3], dim=-1)

        return probe_quaternions  # [num_chars, embed_dim, 4]

    def decode_position(self, psi: torch.Tensor, position: int) -> Tuple[str, float]:
        """
        Decode a single quantum state to character using probe similarity.

        Args:
            psi: Quantum state [embed_dim, 4]
            position: Position in sequence

        Returns:
            Tuple of (decoded_character, confidence_score)
        """
        # Create probe quaternions for all possible characters
        probe_quaternions = self.create_probe_quaternions(self.ascii_codes, position, self.embed_dim)

        # Calculate cosine similarity between psi and all probes
        psi_expanded = psi.unsqueeze(0)  # [1, embed_dim, 4]

        # Compute similarity across all quaternion components
        similarity = F.cosine_similarity(psi_expanded, probe_quaternions, dim=-1)  # [num_chars, embed_dim]

        # Sum similarities across embedding dimensions
        total_similarity = torch.sum(similarity, dim=1)  # [num_chars]

        # Find best matching character
        best_idx = torch.argmax(total_similarity)
        best_score = total_similarity[best_idx].item()
        best_char_code = self.ascii_codes[best_idx].item()

        # Convert to character
        decoded_char = chr(int(best_char_code))

        # Normalize confidence to [0, 1]
        max_possible_score = self.embed_dim  # Maximum possible similarity sum
        confidence = min(best_score / max_possible_score, 1.0)

        return decoded_char, confidence

    def decode(self, psi_sequence: torch.Tensor) -> Tuple[str, List[float]]:
        """
        Decode a sequence of quantum states to text.

        Args:
            psi_sequence: Quantum state sequence [batch_size, seq_len, embed_dim, 4]

        Returns:
            Tuple of (decoded_text, confidence_scores)
        """
        if psi_sequence.dim() == 4:
            # Remove batch dimension if present
            psi_sequence = psi_sequence.squeeze(0)

        seq_len = psi_sequence.shape[0]
        decoded_chars = []
        confidences = []

        print(f"🔍 Decoding quantum sequence: {seq_len} positions")

        for i in range(seq_len):
            psi_position = psi_sequence[i]  # [embed_dim, 4]
            char, confidence = self.decode_position(psi_position, i)

            decoded_chars.append(char)
            confidences.append(confidence)

        decoded_text = ''.join(decoded_chars)

        print(f"   ✅ Quantum decoding complete: {len(decoded_text)} characters")
        print(f"   📊 Average confidence: {sum(confidences)/len(confidences):.3f}")

        return decoded_text, confidences

    def get_decoding_stats(self, psi_sequence: torch.Tensor) -> dict:
        """
        Get detailed statistics about the quantum decoding process.

        Args:
            psi_sequence: Quantum state sequence

        Returns:
            Dictionary with decoding statistics
        """
        decoded_text, confidences = self.decode(psi_sequence)

        stats = {
            'decoded_length': len(decoded_text),
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'min_confidence': min(confidences) if confidences else 0.0,
            'max_confidence': max(confidences) if confidences else 0.0,
            'high_confidence_chars': sum(1 for c in confidences if c > 0.8),
            'low_confidence_chars': sum(1 for c in confidences if c < 0.5),
        }

        return stats


def create_quantum_decoder(embed_dim: int = 64, device: str = 'cpu') -> QuantumDecoder:
    """
    Factory function to create a QuantumDecoder.

    Args:
        embed_dim: Embedding dimension
        device: Device for tensor operations

    Returns:
        Configured QuantumDecoder instance
    """
    return QuantumDecoder(embed_dim=embed_dim, device=device)


# Example usage and testing
if __name__ == "__main__":
    # Create decoder
    decoder = create_quantum_decoder(embed_dim=64, device='cpu')

    # Create test quantum states (simulated)
    seq_len = 10
    embed_dim = 64
    test_psi = torch.randn(seq_len, embed_dim, 4)

    # Decode
    decoded_text, confidences = decoder.decode(test_psi.unsqueeze(0))

    print(f"🎯 Decoded text: '{decoded_text}'")
    print(f"📊 Confidences: {[f'{c:.2f}' for c in confidences]}")

    # Get stats
    stats = decoder.get_decoding_stats(test_psi.unsqueeze(0))
    print("📈 Decoding Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")