#!/usr/bin/env python3
"""
Spectral Decoder - Decoder similar to complete_spectral_pipeline_300_words.py
=============================================================================

Converts generated text back to quaternion embedding, applies ΨQRH transform,
and reconstructs the text for validation.

Based on the perfect reconstruction method from 300_words pipeline.
"""

import torch
import math
from typing import Dict, Tuple, List
import sys
import os

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Import ΨQRH components
from src.core.quaternion_operations import QuaternionOperations
from src.processing.wave_to_text import wave_to_character
from src.processing.text_to_wave import create_spectral_character_map

def create_quaternion_embedding(text: str, embed_dim: int = 64) -> torch.Tensor:
    """
    Convert text to quaternion embedding as per doe.md Section 2.9.1

    Ψ(x) = ψ₀ + ψ₁i + ψ₂j + ψ₃k ∈ ℍ
    """
    print(f"📝 Converting text to quaternion embedding: {len(text)} characters")

    # Convert text to ASCII values
    ascii_values = [ord(char) for char in text]
    seq_len = len(ascii_values)

    # Create quaternion embedding [batch_size=1, seq_len, embed_dim, 4]
    psi = torch.zeros(1, seq_len, embed_dim, 4, dtype=torch.float32)

    for i, ascii_val in enumerate(ascii_values):
        # Store ascii_val directly for perfect reconstruction
        psi[0, i, 0, 0] = ascii_val / 127.0

        for j in range(embed_dim):
            # Create quaternion components based on character and position
            phase = (ascii_val + i + j) * 2 * math.pi / 256.0
            amplitude = (ascii_val / 127.0) * (j / embed_dim)

            # Quaternion components
            psi[0, i, j, 0] = amplitude * math.cos(phase)          # w (real)
            psi[0, i, j, 1] = amplitude * math.sin(phase)          # x (i)
            psi[0, i, j, 2] = amplitude * math.cos(phase + math.pi/4)  # y (j)
            psi[0, i, j, 3] = amplitude * math.sin(phase + math.pi/4)  # z (k)

    print(f"   ✅ Quaternion embedding created: shape {psi.shape}")
    return psi

def apply_psiqrh_transform(psi: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Apply complete ΨQRH transform as per doe.md Section 2.4:

    Ψ_QRH = R_left · F⁻¹{F(k) · F{Ψ}} · R_right

    Where:
    - F{} and F⁻¹{} are Fourier transforms
    - F(k) is the spectral filter: exp(i α · arctan(ln(|k| + ε)))
    - R_left, R_right are quaternion rotation operators
    """
    print("✅ Applying ΨQRH transform")
    batch_size, seq_len, embed_dim, _ = psi.shape

    # Step 1: Apply spectral filtering F(k) · F{Ψ}
    # FFT over embed_dim dimension
    psi_fft = torch.fft.fft(psi, dim=2)  # [batch, seq, embed_dim, 4]

    # Create spectral filter F(k) = exp(i α · arctan(ln(|k| + ε)))
    k = torch.arange(embed_dim, dtype=torch.float32, device=psi.device)
    k = k + 1e-10  # Avoid log(0)
    epsilon = 1e-10
    filter_kernel = torch.exp(1j * alpha * torch.arctan(torch.log(k + epsilon)))

    # Apply filter to each quaternion component - proper broadcasting
    # filter_kernel shape: [embed_dim]
    # psi_fft shape: [batch, seq, embed_dim, 4]
    for comp in range(4):
        psi_fft[:, :, :, comp] *= filter_kernel.unsqueeze(0).unsqueeze(0)

    # Step 2: Inverse FFT F⁻¹{...}
    psi_filtered = torch.fft.ifft(psi_fft, dim=2)

    # Step 3: Apply quaternion rotations R_left and R_right
    # Create unit quaternions for rotation
    theta_left, omega_left, phi_left = 0.1, 0.05, 0.02
    theta_right, omega_right, phi_right = 0.12, 0.06, 0.025

    # Left rotation quaternion
    q_left_w = math.cos(theta_left / 2)
    q_left_x = math.sin(theta_left / 2) * math.cos(omega_left)
    q_left_y = math.sin(theta_left / 2) * math.sin(omega_left) * math.cos(phi_left)
    q_left_z = math.sin(theta_left / 2) * math.sin(omega_left) * math.sin(phi_left)

    # Right rotation quaternion
    q_right_w = math.cos(theta_right / 2)
    q_right_x = math.sin(theta_right / 2) * math.cos(omega_right)
    q_right_y = math.sin(theta_right / 2) * math.sin(omega_right) * math.cos(phi_right)
    q_right_z = math.sin(theta_right / 2) * math.sin(omega_right) * math.sin(phi_right)

    # Apply rotations: R_left · ψ_filtered · R_right†
    # For each position in sequence
    psi_transformed = torch.zeros_like(psi_filtered)
    for b in range(batch_size):
        for s in range(seq_len):
            psi_pos = psi_filtered[b, s]  # [embed_dim, 4]

            # Apply left rotation: q_left * ψ
            psi_rot_left = QuaternionOperations.multiply(
                torch.tensor([q_left_w, q_left_x, q_left_y, q_left_z]).repeat(embed_dim, 1),
                psi_pos
            )

            # Apply right rotation: ψ_rot_left * q_right† (conjugate)
            q_right_conj = torch.tensor([q_right_w, -q_right_x, -q_right_y, -q_right_z]).repeat(embed_dim, 1)
            psi_rotated = QuaternionOperations.multiply(psi_rot_left, q_right_conj)

            psi_transformed[b, s] = psi_rotated

    print(f"   ✅ ΨQRH transform applied: input shape {psi.shape}, output shape {psi_transformed.shape}")
    return psi_transformed

def reconstruct_text(psi_sequence: torch.Tensor) -> str:
    """
    Reconstruct text using full 4D quaternion information (not just one component).
    Uses all quaternion components (w,x,y,z) for richer 3D representation.
    """
    print(f"🔍 Reconstructing text using full 4D quaternion information: {len(psi_sequence)} characters")

    # Create spectral character map
    spectral_map = create_spectral_character_map(n_modes=64)

    characters = []
    for i in range(len(psi_sequence)):
        psi_char = psi_sequence[i]  # [embed_dim, 4]

        # Use full quaternion state for richer measurement
        # Instead of just optical probe, combine multiple approaches
        char_candidates = []

        # Method 1: Standard optical probe
        char1 = wave_to_character(psi_char, spectral_map, temperature=0.1)
        char_candidates.append(char1)

        # Method 2: Use quaternion magnitude for different component
        # Extract from different quaternion components for 3D perspective
        for comp_idx in range(4):  # w, x, y, z components
            if comp_idx < psi_char.shape[1]:  # Ensure component exists
                # Create modified psi using different component as primary
                modified_psi = psi_char.clone()
                # Swap components to get different "views"
                temp = modified_psi[:, 0].clone()
                modified_psi[:, 0] = modified_psi[:, comp_idx]
                modified_psi[:, comp_idx] = temp

                try:
                    char_comp = wave_to_character(modified_psi, spectral_map, temperature=0.2)
                    char_candidates.append(char_comp)
                except:
                    pass

        # Method 3: Use quaternion norm across all components
        psi_norm = torch.norm(psi_char, dim=1)  # [embed_dim]
        # Reshape to [embed_dim, 1] and pad to [embed_dim, 4]
        psi_norm_reshaped = psi_norm.unsqueeze(1).repeat(1, 4)
        try:
            char_norm = wave_to_character(psi_norm_reshaped, spectral_map, temperature=0.2)
            char_candidates.append(char_norm)
        except:
            pass

        # Select character with highest "confidence" (most frequent)
        if char_candidates:
            from collections import Counter
            char_counts = Counter(char_candidates)
            selected_char = char_counts.most_common(1)[0][0]
        else:
            selected_char = char1  # Fallback

        characters.append(selected_char)

        if (i + 1) % 10 == 0:
            print(f"   ⏳ Processed {i+1}/{len(psi_sequence)} characters (using 4D quaternion)")

    reconstructed_text = ''.join(characters)
    print(f"   ✅ Text reconstruction complete: {len(reconstructed_text)} characters (4D quaternion)")
    return reconstructed_text

def decode_generated_text(generated_text: str, alpha: float = 1.0) -> str:
    """
    Decode generated text using ΨQRH spectral decoder.

    This mimics the perfect reconstruction pipeline from 300_words.
    """
    print("================================================================================\n🧪 SPECTRAL DECODER - ΨQRH FRAMEWORK\n================================================================================\n")
    print(f"Decoding generated text: '{generated_text}' (length: {len(generated_text)})\n")

    # Step 1: Convert text to quaternion embedding
    psi_embedding = create_quaternion_embedding(generated_text, embed_dim=64)

    # Step 2: Apply ΨQRH transform
    psi_transformed = apply_psiqrh_transform(psi_embedding, alpha=alpha)

    # Step 3: Reconstruct text (perfect reconstruction)
    reconstructed_text = reconstruct_text(psi_transformed[0])  # Remove batch dimension

    print("\n================================================================================\n🎯 DECODING COMPLETE\n================================================================================\n")
    print(f"Original generated text: '{generated_text}'")
    print(f"Decoded text: '{reconstructed_text}'")

    # Analysis
    matches = sum(1 for a, b in zip(generated_text, reconstructed_text) if a == b)
    accuracy = matches / len(generated_text) if len(generated_text) > 0 else 0
    print(f"\n📊 ANALYSIS:")
    print(f"   - Character matches: {matches}/{len(generated_text)}")
    print(f"   - Accuracy: {accuracy:.1%}")

    if accuracy == 1.0:
        print("   ✅ Perfect reconstruction achieved!")
    else:
        print("   ⚠️ Reconstruction not perfect - check implementation")

    return reconstructed_text

if __name__ == "__main__":
    # The generated output from the pipeline (latest run)
    generated_output = "                              f              r    "

    # Decode it
    decoded_text = decode_generated_text(generated_output)