#!/usr/bin/env python3
"""
Debug Script for Wave-to-Text Conversion Pipeline
=================================================

Scientific analysis of the wave-to-text conversion problem:
- Creates controlled test data to isolate the issue
- Analyzes probability distributions at each step
- Identifies root cause of uniform probabilities
- Tests with simple, predictable inputs
"""

import torch
import torch.fft as fft
import math
import numpy as np
from typing import Dict, Tuple, List
import sys
import os

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Import the actual wave-to-text components
from src.processing.wave_to_text import (
    optical_probe, wave_to_character,
    wave_to_character_with_sampling, wave_to_text,
    padilha_wave_measurement
)
from src.processing.text_to_wave import create_spectral_character_map
from src.processing.chaotic_wave_to_text import chaotic_wave_to_character

def create_controlled_quantum_state(text: str, embed_dim: int = 64) -> torch.Tensor:
    """
    Creates a controlled quantum state from text for debugging.

    Args:
        text: Input text
        embed_dim: Embedding dimension

    Returns:
        Quantum state tensor [seq_len, embed_dim, 4]
    """
    print(f"\n🔬 CREATING CONTROLLED QUANTUM STATE FOR: '{text}'")

    # Convert text to ASCII values
    ascii_values = [ord(char) for char in text]
    seq_len = len(ascii_values)

    # Create deterministic quantum state based on ASCII values
    psi_sequence = []

    for i, ascii_val in enumerate(ascii_values):
        # Create deterministic quaternion state based on character
        # This ensures each character produces a unique quantum state
        psi_char = torch.zeros(embed_dim, 4)

        # Use ASCII value to seed deterministic pattern
        # This creates a clear mapping from character to quantum state
        for j in range(embed_dim):
            # Deterministic pattern based on character and position
            phase = (ascii_val + i + j) * 2 * math.pi / 256.0
            amplitude = (ascii_val / 127.0) * (j / embed_dim)

            # Create quaternion state
            psi_char[j, 0] = amplitude * math.cos(phase)  # w component
            psi_char[j, 1] = amplitude * math.sin(phase)  # x component
            psi_char[j, 2] = amplitude * math.cos(phase + math.pi/4)  # y component
            psi_char[j, 3] = amplitude * math.sin(phase + math.pi/4)  # z component

        psi_sequence.append(psi_char)

    psi_tensor = torch.stack(psi_sequence)
    print(f"   ✅ Created quantum state: shape {psi_tensor.shape}")
    print(f"   📊 Character mapping: {[(i, chr(ascii_values[i])) for i in range(min(5, seq_len))]}")

    return psi_tensor

def analyze_spectral_mapping():
    """Analyze the spectral character mapping"""
    print("\n📊 ANALYZING SPECTRAL CHARACTER MAPPING")

    # Create spectral map
    spectral_map = create_spectral_character_map(n_modes=64)

    print(f"   📋 Spectral map contains {len(spectral_map)} characters")

    # Analyze a few characters
    test_chars = ['a', 'e', 'i', 'o', 'u', ' ', '.', '!']

    for char in test_chars:
        ascii_code = ord(char)
        spectral_pattern = spectral_map[ascii_code]

        print(f"   🔍 '{char}' (ASCII {ascii_code}):")
        print(f"      - Shape: {spectral_pattern.shape}")
        print(f"      - Mean: {spectral_pattern.mean().item():.4f}")
        print(f"      - Std: {spectral_pattern.std().item():.4f}")
        print(f"      - Range: [{spectral_pattern.min().item():.4f}, {spectral_pattern.max().item():.4f}]")

def test_optical_probe_with_controlled_data():
    """Test optical probe with controlled, predictable data"""
    print("\n🔬 TESTING OPTICAL PROBE WITH CONTROLLED DATA")

    # Create simple test: just the character 'a'
    text = "a"
    psi = create_controlled_quantum_state(text, embed_dim=64)[0]  # Take first character
    spectral_map = create_spectral_character_map(n_modes=64)

    # Prepare spectral modes
    ascii_codes = sorted(spectral_map.keys())
    spectral_modes = torch.stack([spectral_map[code] for code in ascii_codes])

    print(f"   📊 Testing with character 'a'")
    print(f"   📈 Quantum state shape: {psi.shape}")
    print(f"   📈 Spectral modes shape: {spectral_modes.shape}")

    # Test optical probe
    char_index, probabilities = optical_probe(psi, spectral_modes, return_probabilities=True)

    print(f"\n   🎯 OPTICAL PROBE RESULTS:")
    print(f"      - Selected index: {char_index}")
    print(f"      - Selected character: '{chr(ascii_codes[char_index])}'")

    if probabilities is not None:
        print(f"      - Probability distribution shape: {probabilities.shape}")
        print(f"      - Max probability: {probabilities.max().item():.4f}")
        print(f"      - Min probability: {probabilities.min().item():.4f}")
        print(f"      - Mean probability: {probabilities.mean().item():.4f}")

        # Check if 'a' has highest probability
        target_ascii = ord('a')
        target_index = ascii_codes.index(target_ascii)
        target_prob = probabilities[target_index].item()

        print(f"\n   🔍 ANALYSIS FOR TARGET CHARACTER 'a':")
        print(f"      - Target index: {target_index}")
        print(f"      - Target probability: {target_prob:.4f}")
        print(f"      - Is highest probability: {target_prob == probabilities.max().item()}")

        # Show top 10 characters
        top_10_indices = torch.topk(probabilities, min(10, len(probabilities)))
        print(f"\n   🏆 TOP 10 CHARACTERS BY PROBABILITY:")
        for i, idx in enumerate(top_10_indices.indices):
            ascii_code = ascii_codes[idx]
            prob = probabilities[idx].item()
            print(f"      {i+1:2d}. '{chr(ascii_code)}' (ASCII {ascii_code}): {prob:.4f}")

def test_padilha_wave_measurement():
    """Test the Padilha wave measurement function"""
    print("\n🌊 TESTING PADILHA WAVE MEASUREMENT")

    # Create a simple quantum state
    psi = torch.randn(64, 4) * 0.1

    # Test measurement at different positions
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    for pos in positions:
        intensity = padilha_wave_measurement(psi, pos, time=0.0)
        print(f"   📍 Position {pos:.2f}: intensity = {intensity:.6f}")

def test_chaotic_methods():
    """Test chaotic wave-to-text methods"""
    print("\n🌪️ TESTING CHAOTIC WAVE-TO-TEXT METHODS")

    # Create controlled quantum state
    text = "hello"
    psi_sequence = create_controlled_quantum_state(text, embed_dim=64)
    spectral_map = create_spectral_character_map(n_modes=64)

    # Test chaotic conversion
    try:
        result = chaotic_wave_to_character(
            psi_sequence[0], spectral_map,
            temperature=1.0,
            r_chaos=3.99,
            use_kuramoto=True
        )
        print(f"   ✅ Chaotic conversion result: '{result}'")
    except Exception as e:
        print(f"   ❌ Chaotic conversion failed: {e}")

def analyze_probability_distribution_issue():
    """
    Scientific analysis of the probability distribution issue
    """
    print("\n🔬 SCIENTIFIC ANALYSIS OF PROBABILITY DISTRIBUTION ISSUE")

    # Create test quantum state
    psi = torch.randn(64, 4) * 0.1
    spectral_map = create_spectral_character_map(n_modes=64)
    ascii_codes = sorted(spectral_map.keys())
    spectral_modes = torch.stack([spectral_map[code] for code in ascii_codes])

    print("\n📊 STEP 1: Analyze inner products before normalization")

    # Calculate raw inner products
    inner_products = []
    for i in range(len(spectral_modes)):
        mode = spectral_modes[i]

        # Convert quaternion to complex representation
        psi_complex = torch.complex(psi[..., 0], psi[..., 1])

        # Inner product
        inner_product = torch.sum(mode * psi_complex)
        probability = torch.abs(inner_product) ** 2

        inner_products.append(probability.item())

    inner_products = torch.tensor(inner_products)

    print(f"   📈 Raw inner products:")
    print(f"      - Shape: {inner_products.shape}")
    print(f"      - Mean: {inner_products.mean().item():.6f}")
    print(f"      - Std: {inner_products.std().item():.6f}")
    print(f"      - Range: [{inner_products.min().item():.6f}, {inner_products.max().item():.6f}]")

    print("\n📊 STEP 2: Analyze after exponential amplification")

    # Apply exponential amplification (as in current code)
    amplified = torch.exp(inner_products * 20.0)

    print(f"   📈 After exponential amplification:")
    print(f"      - Mean: {amplified.mean().item():.6f}")
    print(f"      - Std: {amplified.std().item():.6f}")
    print(f"      - Range: [{amplified.min().item():.6f}, {amplified.max().item():.6f}]")

    print("\n📊 STEP 3: Analyze after normalization")

    # Normalize
    normalized = amplified / (amplified.sum() + 1e-8)

    print(f"   📈 After normalization:")
    print(f"      - Mean: {normalized.mean().item():.6f}")
    print(f"      - Std: {normalized.std().item():.6f}")
    print(f"      - Range: [{normalized.min().item():.6f}, {normalized.max().item():.6f}]")
    print(f"      - Sum: {normalized.sum().item():.6f}")

    # Check if we have sufficient differentiation
    max_prob = normalized.max().item()
    min_prob = normalized.min().item()

    print(f"\n🔍 DIFFERENTIATION ANALYSIS:")
    print(f"   - Max probability: {max_prob:.6f}")
    print(f"   - Min probability: {min_prob:.6f}")
    print(f"   - Ratio (max/min): {max_prob/min_prob if min_prob > 0 else 'inf':.2f}")
    print(f"   - Sufficient differentiation: {'YES' if max_prob > 0.1 else 'NO'}")

def run_comprehensive_debug():
    """Run comprehensive debugging of the wave-to-text pipeline"""
    print("=" * 80)
    print("🔬 COMPREHENSIVE WAVE-TO-TEXT DEBUGGING")
    print("=" * 80)

    # 1. Analyze spectral mapping
    analyze_spectral_mapping()

    # 2. Test with controlled data
    test_optical_probe_with_controlled_data()

    # 3. Test Padilha wave measurement
    test_padilha_wave_measurement()

    # 4. Test chaotic methods
    test_chaotic_methods()

    # 5. Scientific analysis of probability distribution
    analyze_probability_distribution_issue()

    print("\n" + "=" * 80)
    print("🎯 DEBUGGING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_comprehensive_debug()