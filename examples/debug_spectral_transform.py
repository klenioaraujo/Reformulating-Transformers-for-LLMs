#!/usr/bin/env python3
"""
Debug the SpectralQRHLayer transformation
"""

import torch
import torch.fft as fft
import math
import numpy as np

# Import the SpectralQRHLayer
from complete_spectral_pipeline_300_words import SpectralQRHLayer, invert_spectral_qrh

print("🔍 DEBUGGING SPECTRALQRHLAYER TRANSFORMATION")
print("=" * 60)

# Test with a single character
sample_char = 'T'
print(f"Testing character: '{sample_char}'")

# Create embedding
ascii_val = ord(sample_char)
embed_dim = 64

psi_char = torch.zeros(embed_dim, 4)
for j in range(embed_dim):
    phase = (ascii_val + 0 + j) * 2 * math.pi / 256.0
    amplitude = (ascii_val / 127.0) * (j / embed_dim)

    psi_char[j, 0] = amplitude * math.cos(phase)
    psi_char[j, 1] = amplitude * math.sin(phase)
    psi_char[j, 2] = amplitude * math.cos(phase + math.pi/4)
    psi_char[j, 3] = amplitude * math.sin(phase + math.pi/4)

# Add batch and sequence dimensions
psi_input = psi_char.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 4]
print(f"Input shape: {psi_input.shape}")

# Apply SpectralQRHLayer
print("\n🔍 APPLYING SPECTRALQRHLAYER")
qrh_layer = SpectralQRHLayer(embed_dim=64, alpha=1.0)
psi_qrh = qrh_layer.forward(psi_input)
print(f"Output shape: {psi_qrh.shape}")

# Apply inverse
print("\n🔍 APPLYING INVERSE TRANSFORM")
psi_inverted = invert_spectral_qrh(psi_qrh, qrh_layer)
print(f"Inverted shape: {psi_inverted.shape}")

# Check if they're similar
print("\n🔍 CHECKING RECONSTRUCTION ACCURACY")
diff = torch.abs(psi_input - psi_inverted)
max_diff = torch.max(diff).item()
mean_diff = torch.mean(diff).item()

print(f"Maximum difference: {max_diff:.6f}")
print(f"Mean difference: {mean_diff:.6f}")

# Check reconstruction
print("\n🔍 CHECKING TEXT RECONSTRUCTION")
from complete_spectral_pipeline_300_words import quantum_wave_to_text_vectorized

# Test with original
original_text = quantum_wave_to_text_vectorized(psi_input)
print(f"Original reconstruction: '{original_text}'")

# Test with inverted
inverted_text = quantum_wave_to_text_vectorized(psi_inverted)
print(f"Inverted reconstruction: '{inverted_text}'")

print(f"Match: {'✅' if original_text == inverted_text else '❌'}")

# Check if the problem is in the similarity calculation
print("\n🔍 CHECKING SIMILARITY CALCULATION")
from complete_spectral_pipeline_300_words import probe_similarity_vectorized

ascii_codes = torch.tensor(list(range(32, 127)), dtype=torch.float32)

# Test with original
psi_original = psi_input[0, 0]  # [64, 4]
similarities_original = probe_similarity_vectorized(psi_original, ascii_codes, 0, embed_dim)
best_original = torch.argmax(similarities_original)
char_original = chr(ascii_codes[best_original].int().item())

# Test with inverted
psi_inv = psi_inverted[0, 0]  # [64, 4]
similarities_inv = probe_similarity_vectorized(psi_inv, ascii_codes, 0, embed_dim)
best_inv = torch.argmax(similarities_inv)
char_inv = chr(ascii_codes[best_inv].int().item())

print(f"Original best match: '{char_original}' (score: {similarities_original[best_original]:.4f})")
print(f"Inverted best match: '{char_inv}' (score: {similarities_inv[best_inv]:.4f})")

# Check the actual scores for 'T'
correct_idx = torch.where(ascii_codes == ascii_val)[0][0]
score_original = similarities_original[correct_idx].item()
score_inv = similarities_inv[correct_idx].item()

print(f"\nScore for 'T' in original: {score_original:.4f}")
print(f"Score for 'T' in inverted: {score_inv:.4f}")

# Check if there's a consistent offset
print(f"\n🔍 CHECKING FOR OFFSET IN INVERTED:")
for offset in range(-10, 11):
    test_ascii = ascii_val + offset
    if test_ascii >= 32 and test_ascii < 127:
        test_idx = torch.where(ascii_codes == test_ascii)[0][0]
        test_score = similarities_inv[test_idx].item()
        print(f"  Offset {offset:3d}: '{chr(test_ascii)}' - score: {test_score:.4f}")