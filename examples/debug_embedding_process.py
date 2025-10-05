#!/usr/bin/env python3
"""
Debug the embedding and reconstruction process
"""

import torch
import torch.fft as fft
import math
import numpy as np

# Test with a single character
sample_char = 'T'
print(f"🔍 DEBUGGING EMBEDDING FOR CHARACTER: '{sample_char}'")
print("=" * 60)

# Simulate the embedding process
ascii_val = ord(sample_char)
print(f"ASCII value: {ascii_val}")

embed_dim = 64
psi_char = torch.zeros(embed_dim, 4)

for j in range(embed_dim):
    phase = (ascii_val + 0 + j) * 2 * math.pi / 256.0
    amplitude = (ascii_val / 127.0) * (j / embed_dim)

    psi_char[j, 0] = amplitude * math.cos(phase)  # ψ₀ (real)
    psi_char[j, 1] = amplitude * math.sin(phase)  # ψ₁ (i)
    psi_char[j, 2] = amplitude * math.cos(phase + math.pi/4)  # ψ₂ (j)
    psi_char[j, 3] = amplitude * math.sin(phase + math.pi/4)  # ψ₃ (k)

print(f"Embedded quaternion shape: {psi_char.shape}")
print(f"First 5 quaternions:")
for j in range(5):
    print(f"  {j}: {psi_char[j].tolist()}")

# Now simulate the reconstruction
print("\n🔍 SIMULATING RECONSTRUCTION")
print("=" * 60)

ascii_codes = torch.tensor(list(range(32, 127)), dtype=torch.float32)
print(f"Testing {len(ascii_codes)} possible ASCII characters")

# Create probe quaternions for all ascii_codes
j = torch.arange(embed_dim).unsqueeze(0)  # [1, embed_dim]
ascii_val_test = ascii_codes.unsqueeze(1)  # [num_ascii, 1]

phase = (ascii_val_test + 0 + j) * 2 * math.pi / 256.0  # [num_ascii, embed_dim]
amplitude = (ascii_val_test / 127.0) * (j / embed_dim)  # [num_ascii, embed_dim]

psi_probe_0 = amplitude * torch.cos(phase)
psi_probe_1 = amplitude * torch.sin(phase)
psi_probe_2 = amplitude * torch.cos(phase + math.pi/4)
psi_probe_3 = amplitude * torch.sin(phase + math.pi/4)

probe_quaternions = torch.stack([psi_probe_0, psi_probe_1, psi_probe_2, psi_probe_3], dim=-1)  # [num_ascii, embed_dim, 4]

# Calculate similarity (cosine similarity)
psi_expanded = psi_char.unsqueeze(0)  # [1, embed_dim, 4]

similarity = torch.nn.functional.cosine_similarity(psi_expanded, probe_quaternions, dim=-1)  # [num_ascii, embed_dim]
total_similarity = torch.sum(similarity, dim=1)  # [num_ascii]

# Find best match
best_char_index = torch.argmax(total_similarity)
reconstructed_char = chr(ascii_codes[best_char_index].int().item())

print(f"\n🔍 RECONSTRUCTION RESULTS:")
print(f"Original character: '{sample_char}' (ASCII {ascii_val})")
print(f"Reconstructed character: '{reconstructed_char}' (ASCII {ord(reconstructed_char)})")
print(f"Match: {'✅' if sample_char == reconstructed_char else '❌'}")

# Show top 5 matches
print(f"\n🔍 TOP 5 MATCHES:")
top_indices = torch.topk(total_similarity, 5).indices
for i, idx in enumerate(top_indices):
    char = chr(ascii_codes[idx].int().item())
    score = total_similarity[idx].item()
    print(f"  {i+1}. '{char}' (ASCII {ascii_codes[idx].int().item()}) - score: {score:.4f}")

# Check the actual similarity scores for correct character
correct_idx = torch.where(ascii_codes == ascii_val)[0]
if len(correct_idx) > 0:
    correct_score = total_similarity[correct_idx[0]].item()
    print(f"\n🔍 CORRECT CHARACTER SCORE:")
    print(f"  '{sample_char}' score: {correct_score:.4f}")
    print(f"  Rank: {torch.sum(total_similarity > correct_score).item() + 1}")

# Check if there's a consistent offset
print(f"\n🔍 CHECKING FOR CONSISTENT OFFSET:")
for offset in range(-10, 11):
    test_ascii = ascii_val + offset
    if test_ascii >= 32 and test_ascii < 127:
        test_idx = torch.where(ascii_codes == test_ascii)[0]
        if len(test_idx) > 0:
            test_score = total_similarity[test_idx[0]].item()
            print(f"  Offset {offset:3d}: '{chr(test_ascii)}' (ASCII {test_ascii}) - score: {test_score:.4f}")