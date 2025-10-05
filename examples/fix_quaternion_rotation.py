#!/usr/bin/env python3
"""
Fix the quaternion rotation order in the inverse transformation
"""

import torch
import torch.fft as fft
import math

# Import the SpectralQRHLayer
from complete_spectral_pipeline_300_words import SpectralQRHLayer, QuaternionOperations

def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Compute conjugate of quaternion.
    """
    a, b, c, d = torch.unbind(q, dim=-1)
    return torch.stack([a, -b, -c, -d], dim=-1)

def invert_spectral_qrh_corrected(psi_qrh: torch.Tensor, qrh_layer: SpectralQRHLayer) -> torch.Tensor:
    """
    Corrected version of ΨQRH inverse transform with proper rotation order.
    """
    batch_size, seq_len, embed_dim, quat_dim = psi_qrh.shape

    # Step 1: Invert quaternion rotations (EXACT reverse order)
    q_left, q_right = qrh_layer.get_rotation_quaternions()
    q_left_inv = quaternion_conjugate(q_left)
    q_right_inv = quaternion_conjugate(q_right)

    psi_flat = psi_qrh.reshape(-1, 4)

    # IMPORTANT: The forward pass was: q_left * psi * q_right
    # So the inverse should be: q_right_inv * (q_left_inv * psi)
    # But we need to apply in reverse order: first invert left, then invert right

    # Invert left rotation first (reverse of forward pass)
    q_left_inv_expanded = q_left_inv.unsqueeze(0).expand(psi_flat.size(0), -1)
    psi_unrotated_left = QuaternionOperations.multiply(q_left_inv_expanded, psi_flat)

    # Invert right rotation
    q_right_inv_expanded = q_right_inv.unsqueeze(0).expand(psi_flat.size(0), -1)
    psi_unrotated = QuaternionOperations.multiply(psi_unrotated_left, q_right_inv_expanded)

    psi_unrotated = psi_unrotated.reshape(batch_size, seq_len, embed_dim, quat_dim)

    # Step 2: Apply FFT
    psi_unrotated_fft = fft.fft(psi_unrotated, dim=2)

    # Step 3: Invert spectral filter
    freqs = fft.fftfreq(embed_dim)
    k = 2 * math.pi * freqs.view(1, 1, embed_dim, 1)

    # Inverse filter - exact inverse of forward filter
    epsilon = 1e-10
    k_mag = torch.abs(k) + epsilon
    log_k = torch.log(k_mag)
    phase = torch.atan(log_k)

    # Forward filter was: exp(1j * alpha * phase)
    # Inverse should be: exp(-1j * alpha * phase)
    inverse_filter_response = torch.exp(-1j * qrh_layer.alpha * phase)

    # Step 4: Apply inverse filter
    psi_inverted_fft = psi_unrotated_fft * inverse_filter_response

    # Step 5: Inverse FFT
    psi_inverted = fft.ifft(psi_inverted_fft, dim=2).real

    return psi_inverted

def test_corrected_inverse():
    """Test the corrected inverse transformation"""
    print("🔍 TESTING CORRECTED INVERSE TRANSFORMATION")
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

    # Apply corrected inverse
    print("\n🔍 APPLYING CORRECTED INVERSE TRANSFORM")
    psi_inverted = invert_spectral_qrh_corrected(psi_qrh, qrh_layer)
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

if __name__ == "__main__":
    test_corrected_inverse()