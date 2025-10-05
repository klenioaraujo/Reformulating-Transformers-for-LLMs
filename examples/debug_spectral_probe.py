#!/usr/bin/env python3
"""
Debug script for spectral optical probe
"""

import torch
import numpy as np
import sys
import os

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from examples.complete_spectral_pipeline_300_words import (
    text_to_quaternion_embedding,
    create_spectral_character_map,
    spectral_optical_probe
)


def debug_spectral_probe():
    """Debug the spectral optical probe step by step"""
    print("🔍 DEBUG SPECTRAL OPTICAL PROBE")
    print("=" * 60)

    # Test with a simple text
    test_text = "Hello"
    print(f"Test text: '{test_text}'")

    # Step 1: Convert to quaternion embedding
    print("\n📝 STEP 1: Text to Quaternion Embedding")
    psi_input = text_to_quaternion_embedding(test_text)
    print(f"   Shape: {psi_input.shape}")
    print(f"   First character embedding shape: {psi_input[0, 0].shape}")

    # Step 2: Create spectral character map
    print("\n📊 STEP 2: Spectral Character Map")
    spectral_map = create_spectral_character_map()
    print(f"   Characters in map: {len(spectral_map)}")

    # Get ASCII codes and spectral modes
    ascii_codes = sorted(spectral_map.keys())
    spectral_modes = torch.stack([spectral_map[code] for code in ascii_codes])
    print(f"   Spectral modes shape: {spectral_modes.shape}")

    # Step 3: Test probe for first character
    print("\n🔬 STEP 3: Testing Spectral Probe")
    first_psi = psi_input[0, 0]  # First character
    print(f"   First character quaternion shape: {first_psi.shape}")
    print(f"   First character quaternion range: [{first_psi.min():.4f}, {first_psi.max():.4f}]")

    # Debug the inner products
    print("\n📈 STEP 4: Inner Product Analysis")
    inner_products = []

    # Extract quaternion components
    psi_w = first_psi[..., 0]
    psi_x = first_psi[..., 1]
    psi_y = first_psi[..., 2]
    psi_z = first_psi[..., 3]

    # Create magnitude
    psi_magnitude = torch.sqrt(psi_w**2 + psi_x**2 + psi_y**2 + psi_z**2)
    print(f"   Quaternion magnitude range: [{psi_magnitude.min():.4f}, {psi_magnitude.max():.4f}]")

    # Normalize
    psi_magnitude = psi_magnitude / (torch.max(psi_magnitude) + 1e-8)
    print(f"   Normalized magnitude range: [{psi_magnitude.min():.4f}, {psi_magnitude.max():.4f}]")

    # Test with a few characters
    test_chars = ['H', 'e', 'l', 'o', ' ']
    for char in test_chars:
        ascii_code = ord(char)
        if ascii_code in spectral_map:
            mode = spectral_map[ascii_code]
            inner_product = torch.sum(mode * psi_magnitude)
            probability = torch.abs(inner_product) ** 2
            print(f"   '{char}' (ASCII {ascii_code}): inner_product={inner_product:.6f}, probability={probability:.6f}")

    # Step 5: Full probability distribution
    print("\n📊 STEP 5: Full Probability Distribution")
    probabilities = []
    for i in range(len(spectral_modes)):
        mode = spectral_modes[i]
        inner_product = torch.sum(mode * psi_magnitude)
        probability = torch.abs(inner_product) ** 2
        probabilities.append(probability.item())

    probabilities = torch.tensor(probabilities, dtype=torch.float32)
    print(f"   Raw probabilities range: [{probabilities.min():.6f}, {probabilities.max():.6f}]")

    # Normalize
    max_val = probabilities.max().item()
    if max_val > 1.0:
        probabilities = probabilities / max_val
    print(f"   Normalized probabilities range: [{probabilities.min():.6f}, {probabilities.max():.6f}]")

    # Apply softmax
    temperature = 0.1
    probabilities_exp = torch.exp(probabilities / temperature)
    probabilities_final = probabilities_exp / (probabilities_exp.sum() + 1e-8)
    print(f"   Final probabilities range: [{probabilities_final.min():.6f}, {probabilities_final.max():.6f}]")

    # Find top 5 characters
    top_k = 5
    top_probs, top_indices = torch.topk(probabilities_final, top_k)
    print(f"\n🏆 Top {top_k} characters:")
    for i in range(top_k):
        ascii_code = ascii_codes[top_indices[i].item()]
        char = chr(ascii_code)
        prob = top_probs[i].item()
        print(f"   {i+1}. '{char}' (ASCII {ascii_code}): {prob:.6f}")

    # Step 6: Test the actual probe function
    print("\n🔧 STEP 6: Testing Actual Probe Function")
    char_index = spectral_optical_probe(first_psi, spectral_modes)
    predicted_char = chr(ascii_codes[char_index])
    print(f"   Predicted character: '{predicted_char}' (ASCII {ascii_codes[char_index]})")
    print(f"   Expected character: '{test_text[0]}' (ASCII {ord(test_text[0])})")
    print(f"   Match: {'✅' if predicted_char == test_text[0] else '❌'}")

    print("\n" + "=" * 60)
    print("🎯 DEBUG COMPLETE")


if __name__ == "__main__":
    debug_spectral_probe()