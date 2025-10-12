#!/usr/bin/env python3
"""
ΨQRH Volumetric Decoder - DEPRECATED
====================================

⚠️  DEPRECATED: This module has been replaced by OpticalProbe which implements the true Padilha Wave Equation.
The VolumetricDecoder used simple cosine similarity and did not implement the required
physical wave-to-text conversion specified in doe.md section 2.9.5.

Use OpticalProbe instead: from src.core.optical_probe import OpticalProbe

The Padilha Wave Equation: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

Advanced decoder that uses the complete quaternion structure (w, x, y, z) for character
recognition, interpreting each component as orthogonal phases with different semantic roles.

Instead of collapsing quantum states to 1D similarity scores, this decoder analyzes the
full 4D volumetric signature of each character, enabling more precise and context-aware
text generation.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import numpy as np


class VolumetricDecoder:
    """
    Volumetric Decoder for ΨQRH pipeline - DEPRECATED.

    ⚠️  DEPRECATED: This class has been replaced by OpticalProbe which implements the true Padilha Wave Equation.
    The VolumetricDecoder used simple cosine similarity and did not implement the required
    physical wave-to-text conversion specified in doe.md section 2.9.5.

    Use OpticalProbe instead: from src.core.optical_probe import OpticalProbe

    The Padilha Wave Equation: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

    This decoder uses the complete quaternion structure to recognize characters by their
    4D volumetric signatures, where each component represents different semantic aspects:
    - w: amplitude/energy
    - x: vowel-like characteristics
    - y: consonant structure
    - z: punctuation/special characters
    """

    def __init__(self, vocab_size: int = 256, device: str = 'cpu', quantum_embedding=None):
        """
        Initialize the Volumetric Decoder.

        Args:
            vocab_size: Size of the character vocabulary
            device: Device for tensor operations
            quantum_embedding: Pre-trained quantum embedding layer for creating probes
        """
        self.device = device
        self.quantum_embedding = quantum_embedding

        # Define valid character set (printable ASCII: 32-126)
        self.valid_chars = [chr(i) for i in range(32, 127)]  # ' ' to '~'
        self.char_to_id = {char: i for i, char in enumerate(self.valid_chars)}
        self.id_to_char = {i: char for i, char in enumerate(self.valid_chars)}
        self.vocab_size = len(self.valid_chars)

        # Create volumetric probes for valid characters only
        self.volumetric_probes = self._create_volumetric_probes()
        print(f"🔬 Initialized volumetric probes for {self.vocab_size} valid characters (printable ASCII)")

    def _create_volumetric_probes(self) -> Dict[str, torch.Tensor]:
        """
        Create volumetric probes for each valid character.

        If quantum_embedding is available, use actual quantum embeddings to create probes.
        Otherwise, fall back to heuristic-based signatures.
        """
        probes = {}

        if self.quantum_embedding is not None:
            # Use actual quantum embeddings for more accurate probes
            print("🔬 Creating volumetric probes from quantum embeddings...")

            for char in self.valid_chars:
                char_id = ord(char)
                try:
                    # Generate quantum state for this character
                    char_ids = torch.tensor([[char_id]], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        quantum_state = self.quantum_embedding(char_ids).squeeze(0).squeeze(0)  # [embed_dim, 4]

                    # Extract volumetric features from the quantum state
                    probe_features = self._extract_volumetric_features(quantum_state)

                    probes[char] = probe_features.to(self.device)

                except Exception as e:
                    print(f"⚠️  Failed to create probe for char '{char}': {e}")
                    # Fallback to heuristic
                    probes[char] = self._create_heuristic_probe(char)

        else:
            # Fallback to heuristic-based probes
            print("🔬 Creating volumetric probes using heuristics...")
            for char in self.valid_chars:
                probes[char] = self._create_heuristic_probe(char)

        return probes

    def _create_heuristic_probe(self, char: str) -> torch.Tensor:
        """
        Create a heuristic-based volumetric probe for a character.

        Args:
            char: Character string

        Returns:
            8D probe vector with spectral and energy features
        """
        # Base 4D quaternion signature
        if char.islower():
            if char in 'aeiou':
                # Vowels: high energy in x-component
                base_sig = torch.tensor([0.3, 0.8, 0.2, 0.1])
            else:
                # Consonants: balanced with y-component emphasis
                base_sig = torch.tensor([0.4, 0.2, 0.7, 0.1])
        elif char.isupper():
            if char in 'AEIOU':
                # Uppercase vowels: higher amplitude
                base_sig = torch.tensor([0.5, 0.9, 0.3, 0.2])
            else:
                # Uppercase consonants: strong y-component
                base_sig = torch.tensor([0.5, 0.3, 0.8, 0.2])
        elif char.isdigit():
            # Digits: high z-component, structured
            base_sig = torch.tensor([0.6, 0.1, 0.1, 0.8])
        elif char in ' \t\n':
            # Whitespace: low energy, balanced
            base_sig = torch.tensor([0.1, 0.3, 0.3, 0.3])
        elif char in '.,!?;:':
            # Punctuation: high z-component
            base_sig = torch.tensor([0.2, 0.1, 0.2, 0.9])
        else:
            # Special characters: high z, varied
            base_sig = torch.tensor([0.3, 0.2, 0.2, 0.8])

        # Add spectral features (4 additional dimensions)
        # 5. Spectral centroid (0-1, higher for consonants)
        if char.islower() and char in 'aeiou':
            spectral_centroid = 0.3  # Vowels have lower frequency content
        elif char.isupper():
            spectral_centroid = 0.7  # Uppercase have higher frequency
        else:
            spectral_centroid = 0.5  # Default

        # 6. Phase coherence (0-1, higher for vowels)
        if char in 'aeiouAEIOU':
            phase_coherence = 0.8  # Vowels have more coherent phases
        else:
            phase_coherence = 0.4  # Consonants have less coherent phases

        # 7. Complexity (0-1, higher for complex characters)
        if char.isdigit() or char in '.,!?;:':
            complexity = 0.7  # Structured characters
        elif char.isalpha():
            complexity = 0.5  # Letters
        else:
            complexity = 0.3  # Simple characters

        # 8. Character-specific variation based on ASCII value
        char_id = ord(char)
        variation = (char_id % 10) / 10.0

        # Combine all features
        signature = torch.cat([
            base_sig,
            torch.tensor([spectral_centroid, phase_coherence, complexity, variation])
        ])

        # Add some noise for uniqueness
        signature = signature + torch.randn(8) * 0.1
        signature = torch.clamp(signature, 0.0, 1.0)

        # Normalize to unit length
        signature = signature / torch.norm(signature)

        return signature.to(self.device)

    def _extract_volumetric_features(self, psi_state: torch.Tensor) -> torch.Tensor:
        """
        Extract volumetric features from a quaternion state using spectral analysis.

        Args:
            psi_state: Quaternion tensor [embed_dim, 4] or [4]

        Returns:
            Volumetric feature vector [8] with spectral and energy features
        """
        if psi_state.dim() == 2:
            # [embed_dim, 4] - use the full state for spectral analysis
            psi_full = psi_state  # [embed_dim, 4]
        else:
            # Single quaternion - expand to sequence
            psi_full = psi_state.unsqueeze(0)  # [1, 4]

        # Extract features from each quaternion component
        features = []

        # 1. Total energy (norm of the entire state)
        total_energy = torch.norm(psi_full)
        features.append(total_energy)

        # 2-5. Energy distribution across quaternion components
        if psi_full.shape[0] > 1:
            # Average across embed_dim for component-wise energy
            component_energy = torch.norm(psi_full, dim=0)  # [4]
            total_comp_energy = component_energy.sum()
            if total_comp_energy > 0:
                energy_dist = component_energy / total_comp_energy
            else:
                energy_dist = torch.ones(4) / 4
        else:
            # Single quaternion
            component_energy = psi_full[0].abs()  # [4]
            total_comp_energy = component_energy.sum()
            if total_comp_energy > 0:
                energy_dist = component_energy / total_comp_energy
            else:
                energy_dist = torch.ones(4) / 4

        features.extend(energy_dist.tolist())

        # 6. Spectral centroid (frequency-weighted energy center)
        if psi_full.shape[0] > 1:
            # Compute FFT of each component
            freqs = torch.arange(psi_full.shape[0], dtype=torch.float32, device=psi_full.device)
            spectral_centroid = 0.0
            total_power = 0.0

            for comp in range(4):
                component_signal = psi_full[:, comp]
                fft_result = torch.fft.fft(component_signal)
                power_spectrum = torch.abs(fft_result) ** 2
                weighted_freq = (freqs * power_spectrum).sum()
                total_power += power_spectrum.sum()
                spectral_centroid += weighted_freq

            if total_power > 0:
                spectral_centroid = spectral_centroid / total_power / len(freqs)  # Normalize
            else:
                spectral_centroid = 0.5
        else:
            spectral_centroid = 0.5  # Default for single element

        features.append(spectral_centroid)

        # 7. Phase coherence (correlation between w and x components)
        if psi_full.shape[0] > 1:
            w_signal = psi_full[:, 0]
            x_signal = psi_full[:, 1]
            if torch.std(w_signal) > 0 and torch.std(x_signal) > 0:
                phase_coherence = torch.corrcoef(torch.stack([w_signal, x_signal]))[0, 1]
                phase_coherence = (phase_coherence + 1) / 2  # Normalize to [0, 1]
            else:
                phase_coherence = 0.5
        else:
            # For single quaternion, use angle between w and x
            w, x = psi_full[0, 0], psi_full[0, 1]
            if w != 0 or x != 0:
                angle = torch.atan2(x, w) / (2 * np.pi) + 0.5  # Normalize to [0, 1]
                phase_coherence = torch.clamp(angle, 0.0, 1.0)
            else:
                phase_coherence = 0.5

        features.append(phase_coherence)

        # 8. Complexity measure (entropy of energy distribution)
        if len(energy_dist) > 1:
            # Normalize energy distribution
            p = energy_dist / energy_dist.sum()
            # Compute entropy
            entropy = -torch.sum(p * torch.log(p + 1e-10))
            # Normalize by max entropy (log(4) for 4 components)
            max_entropy = torch.log(torch.tensor(4.0))
            complexity = entropy / max_entropy
        else:
            complexity = 0.5

        features.append(complexity)

        # Convert to tensor and normalize
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        features_tensor = features_tensor / torch.norm(features_tensor + 1e-10)

        return features_tensor

    def decode_position(self, psi_position: torch.Tensor, context: str = "") -> Tuple[str, float]:
        """
        Decode a single position using volumetric comparison with linguistic constraints.

        Args:
            psi_position: Quantum state for one position [embed_dim, 4]
            context: Previous characters for linguistic context

        Returns:
            Tuple of (decoded_character, confidence_score)
        """
        # Extract volumetric features
        features = self._extract_volumetric_features(psi_position)

        # Find best matching character among valid characters only
        candidates = []

        for char, probe in self.volumetric_probes.items():
            # Calculate Euclidean distance
            distance = torch.norm(features - probe).item()

            # Apply linguistic constraints
            linguistic_penalty = self._calculate_linguistic_penalty(char, context, distance)
            adjusted_distance = distance + linguistic_penalty

            candidates.append((char, adjusted_distance))

        # Sort by adjusted distance
        candidates.sort(key=lambda x: x[1])
        best_char, best_distance = candidates[0]

        # Calculate confidence based on distance (lower distance = higher confidence)
        # Normalize distance to confidence score
        if best_distance < 0.1:
            confidence = 1.0
        elif best_distance < 0.3:
            confidence = 0.8
        elif best_distance < 0.6:
            confidence = 0.5
        elif best_distance < 1.0:
            confidence = 0.3
        else:
            confidence = 0.1

        # Validate the decoded character
        decoded_char, final_confidence = self._validate_decoding_output(best_char, confidence)

        return decoded_char, final_confidence

    def _calculate_linguistic_penalty(self, char: str, context: str, base_distance: float) -> float:
        """
        Calculate linguistic penalty for a character given the context.

        Args:
            char: Character to evaluate
            context: Previous characters
            base_distance: Base distance from volumetric matching

        Returns:
            Linguistic penalty to add to distance
        """
        penalty = 0.0

        # Get last few characters for context
        recent_context = context[-3:] if len(context) >= 3 else context

        # Penalize excessive repetition
        if len(recent_context) >= 2 and recent_context.count(char) >= 2:
            penalty += 0.5  # Strong penalty for repetition

        # Penalize unlikely character sequences
        if len(recent_context) >= 1:
            last_char = recent_context[-1]

            # Penalize vowel after vowel (unless common diphthongs)
            if (char in 'aeiouAEIOU' and last_char in 'aeiouAEIOU' and
                char + last_char not in ['ea', 'ai', 'ou', 'oi', 'au']):
                penalty += 0.3

            # Penalize consonant clusters that are hard to pronounce
            if (char not in 'aeiouAEIOU' and last_char not in 'aeiouAEIOU' and
                len(recent_context) >= 2):
                cluster = recent_context[-1] + char
                hard_clusters = ['ck', 'dg', 'pf', 'gh', 'th', 'wh', 'ph', 'sh', 'ch']
                if cluster.lower() not in hard_clusters:
                    penalty += 0.2

        # Favor spaces after punctuation
        if len(recent_context) >= 1:
            last_char = recent_context[-1]
            if last_char in '.,!?;:)' and char != ' ':
                penalty += 0.4
            elif last_char not in ' .,!?;:)(' and char == ' ' and len(recent_context) < 10:
                penalty += 0.2  # Penalize early spaces

        # Reduce penalty for very close volumetric matches
        if base_distance < 0.2:
            penalty *= 0.5

        return penalty

    def _validate_decoding_output(self, char: str, confidence: float) -> Tuple[str, float]:
        """
        Validate decoding output - ZERO FALLBACK POLICY.

        Args:
            char: Decoded character
            confidence: Original confidence score

        Returns:
            Tuple of (validated_character, adjusted_confidence)

        Raises:
            ValueError: If character is invalid - no fallback allowed
        """
        # Check if character is in our valid set
        if char in self.valid_chars:
            return char, confidence

        # ZERO FALLBACK: Invalid characters cause immediate failure
        raise ValueError(f"Invalid character '{char}' decoded - ZERO FALLBACK POLICY: System must fail cleanly")


    def decode(self, psi_sequence: torch.Tensor) -> Tuple[str, List[float]]:
        """
        Decode a sequence of quantum states with validation and linguistic constraints.

        Args:
            psi_sequence: Sequence of quantum states [seq_len, embed_dim, 4]

        Returns:
            Tuple of (decoded_text, confidence_scores)
        """
        decoded_chars = []
        confidences = []

        for i in range(psi_sequence.shape[0]):
            psi_position = psi_sequence[i]  # [embed_dim, 4]
            # Build context from previously decoded characters
            context = ''.join(decoded_chars)
            char, confidence = self.decode_position(psi_position, context)
            decoded_chars.append(char)
            confidences.append(confidence)

        decoded_text = ''.join(decoded_chars)

        # Validate the entire sequence
        validated_text, validated_confidences = self._validate_generated_sequence(decoded_text, confidences)

        return validated_text, validated_confidences

    def _validate_generated_sequence(self, text: str, confidences: List[float]) -> Tuple[str, List[float]]:
        """
        Validate and clean up the generated sequence.

        Args:
            text: Generated text
            confidences: Confidence scores for each character

        Returns:
            Tuple of (validated_text, validated_confidences)
        """
        if not text:
            return text, confidences

        validated_chars = []
        validated_confidences = []

        for i, (char, conf) in enumerate(zip(text, confidences)):
            # Basic validation - ensure character is printable
            if ord(char) >= 32 and ord(char) <= 126:  # Printable ASCII
                validated_chars.append(char)
                validated_confidences.append(conf)
            else:
                # Replace invalid character with space and reduce confidence
                validated_chars.append(' ')
                validated_confidences.append(conf * 0.1)

        validated_text = ''.join(validated_chars)

        # Additional sequence-level validation
        # Remove excessive consecutive identical characters
        cleaned_text = []
        cleaned_confidences = []

        prev_char = None
        count = 0
        for char, conf in zip(validated_text, validated_confidences):
            if char == prev_char:
                count += 1
                if count <= 3:  # Allow up to 3 consecutive identical chars
                    cleaned_text.append(char)
                    cleaned_confidences.append(conf)
            else:
                count = 1
                prev_char = char
                cleaned_text.append(char)
                cleaned_confidences.append(conf)

        return ''.join(cleaned_text), cleaned_confidences

    def get_volumetric_quality_score(self, psi_sequence: torch.Tensor) -> Dict[str, float]:
        """
        Calculate volumetric quality metrics for a quantum state sequence.

        Args:
            psi_sequence: Sequence of quantum states [seq_len, embed_dim, 4]

        Returns:
            Dictionary with volumetric quality metrics
        """
        if psi_sequence.numel() == 0:
            return {
                'average_energy': 0.0,
                'energy_variance': 0.0,
                'phase_coherence': 0.0,
                'volumetric_stability': 0.0
            }

        # Extract features for all positions
        features_list = []
        for i in range(psi_sequence.shape[0]):
            features = self._extract_volumetric_features(psi_sequence[i])
            features_list.append(features)

        features_tensor = torch.stack(features_list)  # [seq_len, 4]

        # Calculate metrics
        energies = features_tensor[:, 0]  # Total energy per position
        average_energy = energies.mean().item()

        energy_variance = energies.var().item()

        # Phase coherence (how consistent phases are across sequence)
        phases = features_tensor[:, 3]  # Principal phase
        phase_coherence = 1.0 - phases.var().item()  # Lower variance = higher coherence

        # Volumetric stability (consistency of energy distribution)
        energy_dists = features_tensor[:, 1:3]  # w and x ratios
        stability = 1.0 - energy_dists.var(dim=0).mean().item()

        return {
            'average_energy': average_energy,
            'energy_variance': energy_variance,
            'phase_coherence': max(0.0, phase_coherence),
            'volumetric_stability': max(0.0, stability)
        }


def create_volumetric_decoder(vocab_size: int = 256, device: str = 'cpu', quantum_embedding=None) -> VolumetricDecoder:
    """
    Factory function to create a VolumetricDecoder.

    Args:
        vocab_size: Size of the character vocabulary (deprecated - now uses printable ASCII)
        device: Device for tensor operations
        quantum_embedding: Pre-trained quantum embedding layer for creating probes

    Returns:
        Configured VolumetricDecoder instance
    """
    return VolumetricDecoder(device=device, quantum_embedding=quantum_embedding)


# Example usage and testing
if __name__ == "__main__":
    # Create decoder
    decoder = create_volumetric_decoder(vocab_size=256, device='cpu')

    # Create test quantum states (simulated)
    seq_len = 10
    embed_dim = 64
    test_psi = torch.randn(seq_len, embed_dim, 4)

    # Decode
    decoded_text, confidences = decoder.decode(test_psi)

    print(f"🎯 Decoded text: '{decoded_text}'")
    print(f"📊 Confidences: {['.2f' for c in confidences]}")

    # Calculate volumetric quality
    quality = decoder.get_volumetric_quality_score(test_psi)
    print("🔬 Volumetric Quality Metrics:")
    for key, value in quality.items():
        print(".4f")