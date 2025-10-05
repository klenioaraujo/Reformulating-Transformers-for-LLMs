#!/usr/bin/env python3
"""
Enhanced Wave-to-Text Implementation
=====================================

Scientific fixes for the uniform probability distribution problem:
- Enhanced spectral mapping with strong character differentiation
- Much more aggressive exponential amplification
- Character-specific measurement parameters
- Linguistic frequency-based modulation
"""

import torch
import torch.fft as fft
import math
import numpy as np
from typing import Dict, Tuple, Optional
import sys
import os

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Import quaternion operations
try:
    from src.core.quaternion_operations import QuaternionOperations
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.core.quaternion_operations import QuaternionOperations

def create_enhanced_spectral_character_map(n_modes: int = 256) -> Dict[int, torch.Tensor]:
    """
    Creates spectral character mapping with STRONG differentiation.

    Principle: Each character should have a unique spectral pattern
    with significant differences (> 100x) between common and rare characters.
    """
    spectral_map = {}

    # Linguistic frequency mapping (Portuguese)
    linguistic_frequency = {
        # Very high frequency
        ' ': 18.0, 'a': 14.6, 'e': 12.6, 'o': 10.7,
        # High frequency
        's': 7.8, 'r': 6.5, 'i': 6.7, 'n': 5.0, 'd': 4.9,
        'm': 4.7, 't': 4.3, 'c': 3.9,
        # Medium frequency
        'l': 2.8, 'p': 2.5, 'u': 4.0, 'v': 1.7, 'g': 1.3,
        'h': 1.3, 'b': 1.0, 'f': 1.0, 'q': 1.0,
        # Low frequency
        'j': 0.4, 'z': 0.4, 'x': 0.2, 'k': 0.02,
        'w': 0.02, 'y': 0.01,
        # Punctuation
        '.': 6.5, ',': 6.0, '!': 0.5, '?': 0.5
    }

    for ascii_code in range(32, 127):
        char = chr(ascii_code)

        # Get linguistic frequency (default to very low for unknown chars)
        freq = linguistic_frequency.get(char.lower(), 0.001)

        # Scale fundamental frequency by linguistic frequency
        # Common chars: high frequency, rare chars: low frequency
        fundamental = (ascii_code / 127.0) * (freq / 2.0 + 1.0)

        # Create STRONGLY differentiated spectral pattern
        harmonics = []
        for k in range(1, n_modes + 1):
            # Strong modulation based on character properties
            amplitude = (
                np.sin(2 * np.pi * k * fundamental) *
                np.exp(-k * 0.05) *  # Gentle decay
                (1.0 + 0.8 * np.sin(k * ascii_code / 5.0)) *  # Strong character modulation
                (freq + 0.1)  # Scale by linguistic frequency
            )
            harmonics.append(amplitude)

        spectral_pattern = torch.tensor(harmonics, dtype=torch.float32)
        spectral_map[ascii_code] = spectral_pattern

    print(f"✅ Enhanced spectral map created with {len(spectral_map)} characters")
    return spectral_map

def enhanced_optical_probe(psi: torch.Tensor,
                          spectral_modes: torch.Tensor,
                          return_probabilities: bool = False) -> Tuple[int, Optional[torch.Tensor]]:
    """
    Enhanced optical probe with MUCH more aggressive amplification.

    Principle: For small differences, we need very strong exponential
    amplification (×100-×1000) to create meaningful differentiation.
    """
    n_chars, n_modes = spectral_modes.shape
    print(f"      🔬 [enhanced_optical_probe] Enhanced probe: {n_chars} chars")

    # Calculate inner products with enhanced method
    inner_products = []

    for i in range(n_chars):
        mode = spectral_modes[i]

        # Convert quaternion to complex representation
        psi_complex = torch.complex(psi[..., 0], psi[..., 1])

        # Inner product with enhanced calculation
        inner_product = torch.sum(mode * psi_complex)
        probability = torch.abs(inner_product) ** 2

        inner_products.append(probability.item())

    probabilities = torch.tensor(inner_products, dtype=torch.float32)

    print(f"      📊 [enhanced_optical_probe] Raw probabilities:")
    print(f"        - Mean: {probabilities.mean().item():.6f}")
    print(f"        - Std: {probabilities.std().item():.6f}")
    print(f"        - Range: [{probabilities.min().item():.6f}, {probabilities.max().item():.6f}]")

    # AGGRESSIVE AMPLIFICATION: ×100 (reduced from ×200 to avoid inf)
    probabilities = torch.exp(probabilities * 100.0)

    print(f"      📊 [enhanced_optical_probe] After exponential (×100):")
    print(f"        - Range: [{probabilities.min().item():.6f}, {probabilities.max().item():.6f}]")

    # Normalize
    probabilities = probabilities / (probabilities.sum() + 1e-8)

    print(f"      📊 [enhanced_optical_probe] After normalization:")
    print(f"        - Range: [{probabilities.min().item():.6f}, {probabilities.max().item():.6f}]")
    ratio_text = f"{probabilities.max().item()/probabilities.min().item():.1f}" if probabilities.min().item() > 0 else "inf"
    print(f"        - Ratio (max/min): {ratio_text}")

    # Check if we have sufficient differentiation
    max_prob = probabilities.max().item()
    min_prob = probabilities.min().item()

    if max_prob / min_prob < 100:  # If still insufficient
        print(f"      🔥 [enhanced_optical_probe] Applying ultra-strong softmax")
        probabilities = torch.softmax(probabilities * 100.0, dim=0)

    # Final selection
    char_index = torch.argmax(probabilities).item()
    max_prob = probabilities[char_index].item()

    print(f"      🎯 [enhanced_optical_probe] Selected: index {char_index}, prob {max_prob:.4f}")

    if return_probabilities:
        return char_index, probabilities
    else:
        return char_index, None

def enhanced_wave_to_character(psi: torch.Tensor,
                              spectral_map: dict,
                              temperature: float = 1.0) -> str:
    """
    Enhanced wave-to-character conversion with scientific fixes.
    """
    print(f"    🔬 [enhanced_wave_to_character] Enhanced conversion...")

    # Prepare spectral modes
    ascii_codes = sorted(spectral_map.keys())
    spectral_modes = torch.stack([spectral_map[code] for code in ascii_codes])

    # Enhanced optical probe
    char_index, probabilities = enhanced_optical_probe(psi, spectral_modes, return_probabilities=True)

    if probabilities is not None:
        # Show top characters with enhanced differentiation
        top_5_indices = torch.topk(probabilities, min(5, len(probabilities)))
        top_chars = [(ascii_codes[idx], chr(ascii_codes[idx]), probabilities[idx].item()) for idx in top_5_indices.indices]

        print(f"    🎯 [enhanced_wave_to_character] Top 5 characters:")
        for ascii_code, char, prob in top_chars:
            print(f"      - '{char}' (ASCII {ascii_code}): {prob:.4f}")

        # Temperature sampling
        if temperature != 1.0:
            probabilities = probabilities / temperature
            probabilities = probabilities / probabilities.sum()
            char_index = torch.multinomial(probabilities, 1).item()
            print(f"    🌡️  [enhanced_wave_to_character] Temperature sampling: new index {char_index}")

    # Convert to character
    ascii_code = ascii_codes[char_index]
    char = chr(ascii_code)

    print(f"    ✅ [enhanced_wave_to_character] Generated: '{char}' (ASCII {ascii_code})")
    return char

def enhanced_wave_to_text(psi_sequence: torch.Tensor,
                         spectral_map: dict,
                         temperature: float = 1.0,
                         min_seq_len: int = 5) -> str:
    """
    Enhanced wave-to-text conversion with scientific fixes.
    """
    print(f"🔍 [enhanced_wave_to_text] Enhanced decoding: seq_len={len(psi_sequence)}")

    characters = []

    # Ensure minimum sequence length
    target_seq_len = max(len(psi_sequence), min_seq_len)

    if len(psi_sequence) < target_seq_len:
        print(f"  🔄 [enhanced_wave_to_text] Extending sequence: {len(psi_sequence)} → {target_seq_len}")
        extended_sequence = []
        for i in range(target_seq_len):
            base_idx = i % len(psi_sequence)
            base_psi = psi_sequence[base_idx]
            # Add small variation
            noise = torch.randn_like(base_psi) * 0.01
            extended_sequence.append(base_psi + noise)
        psi_sequence = torch.stack(extended_sequence)

    # Generate each character
    for i in range(len(psi_sequence)):
        psi = psi_sequence[i]
        print(f"  📝 [enhanced_wave_to_text] Character {i+1}/{len(psi_sequence)}")

        char = enhanced_wave_to_character(psi, spectral_map, temperature=temperature)
        print(f"  ✅ [enhanced_wave_to_text] Character {i+1}: '{char}' (ASCII: {ord(char)})")
        characters.append(char)

    result = ''.join(characters)
    print(f"🎯 [enhanced_wave_to_text] Final text: '{result}'")
    return result

def test_enhanced_conversion():
    """Test the enhanced wave-to-text conversion"""
    print("\n" + "=" * 80)
    print("🧪 TESTING ENHANCED WAVE-TO-TEXT CONVERSION")
    print("=" * 80)

    # Create controlled quantum state
    def create_simple_quantum_state(text: str, embed_dim: int = 64) -> torch.Tensor:
        """Create deterministic quantum state from text"""
        ascii_values = [ord(char) for char in text]
        psi_sequence = []

        for i, ascii_val in enumerate(ascii_values):
            psi_char = torch.zeros(embed_dim, 4)
            for j in range(embed_dim):
                phase = (ascii_val + i + j) * 2 * math.pi / 256.0
                amplitude = (ascii_val / 127.0) * (j / embed_dim)

                psi_char[j, 0] = amplitude * math.cos(phase)
                psi_char[j, 1] = amplitude * math.sin(phase)
                psi_char[j, 2] = amplitude * math.cos(phase + math.pi/4)
                psi_char[j, 3] = amplitude * math.sin(phase + math.pi/4)

            psi_sequence.append(psi_char)

        return torch.stack(psi_sequence)

    # Test cases
    test_cases = ["hello", "aeiou", "test", "123"]

    for test_text in test_cases:
        print(f"\n📝 Testing: '{test_text}'")

        # Create quantum state
        psi_sequence = create_simple_quantum_state(test_text)

        # Create enhanced spectral map
        spectral_map = create_enhanced_spectral_character_map(n_modes=64)

        # Convert to text
        result = enhanced_wave_to_text(psi_sequence, spectral_map, temperature=1.0)

        print(f"🎯 Input: '{test_text}' → Output: '{result}'")

        # Calculate accuracy
        matches = sum(1 for a, b in zip(test_text, result) if a == b)
        accuracy = matches / len(test_text) if test_text else 0
        print(f"📊 Accuracy: {accuracy:.1%} ({matches}/{len(test_text)})")

    print("\n" + "=" * 80)
    print("✅ ENHANCED CONVERSION TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_enhanced_conversion()