#!/usr/bin/env python3
"""
Debug unit projection by testing the _apply_normalization method directly
"""

import torch
from qrh_layer import QRHLayer, QRHConfig

def test_unit_projection_direct():
    """Test unit projection directly on the normalization method"""
    print("🔍 Testing Unit Projection Directly")
    print("-" * 40)

    config = QRHConfig(embed_dim=16, normalization_type='unit_projection')
    layer = QRHLayer(config)

    # Create quaternion tensor in the internal format [B, T, D, 4]
    batch_size, seq_len, embed_dim = 2, 8, 16
    Ψ_rotated = torch.randn(batch_size, seq_len, embed_dim, 4)

    print(f"   Input tensor shape: {Ψ_rotated.shape}")

    # Calculate original norms
    original_norms = torch.norm(Ψ_rotated, p=2, dim=-1)
    print(f"   Original norms - Mean: {original_norms.mean():.6f}, Std: {original_norms.std():.6f}")

    # Apply normalization directly
    Ψ_normalized = layer._apply_normalization(Ψ_rotated)

    # Calculate new norms
    new_norms = torch.norm(Ψ_normalized, p=2, dim=-1)
    print(f"   Normalized norms - Mean: {new_norms.mean():.6f}, Std: {new_norms.std():.6f}")

    # Check if they are unit quaternions
    unit_check = torch.allclose(new_norms, torch.ones_like(new_norms), atol=1e-5)
    print(f"   ✅ Unit quaternions: {unit_check}")

    if not unit_check:
        print(f"   Min norm: {new_norms.min():.6f}, Max norm: {new_norms.max():.6f}")

    return unit_check

def test_full_pipeline():
    """Test the complete pipeline to see where unit property is lost"""
    print("\n🔍 Testing Full Pipeline")
    print("-" * 40)

    config = QRHConfig(embed_dim=16, normalization_type='unit_projection')
    layer = QRHLayer(config)

    # Input
    x = torch.randn(2, 8, 64)  # 4 * 16 = 64

    # Step by step execution
    print("   Following pipeline step by step...")

    # 1. Validate input
    x_validated = layer._validate_input(x)
    print(f"   1. After validation: {x_validated.shape}")

    # 2. Preprocess
    Ψ = layer._preprocess_input(x_validated)
    print(f"   2. After preprocessing: {Ψ.shape}")

    # 3. Spectral filtering
    Ψ_filtered = layer._apply_spectral_filtering(Ψ)
    print(f"   3. After spectral filtering: {Ψ_filtered.shape}")

    # 4. Quaternion rotations
    Ψ_rotated = layer._apply_quaternion_rotations(Ψ_filtered)
    print(f"   4. After quaternion rotations: {Ψ_rotated.shape}")

    # Check norms before normalization
    norms_before = torch.norm(Ψ_rotated, p=2, dim=-1)
    print(f"   Norms before normalization - Mean: {norms_before.mean():.6f}")

    # 5. Apply normalization
    Ψ_normalized = layer._apply_normalization(Ψ_rotated)
    print(f"   5. After normalization: {Ψ_normalized.shape}")

    # Check norms after normalization
    norms_after = torch.norm(Ψ_normalized, p=2, dim=-1)
    print(f"   Norms after normalization - Mean: {norms_after.mean():.6f}")
    unit_check = torch.allclose(norms_after, torch.ones_like(norms_after), atol=1e-5)
    print(f"   ✅ Unit quaternions after normalization: {unit_check}")

    # 6. Postprocess
    final_output = layer._postprocess_output(Ψ_normalized, x_validated)
    print(f"   6. Final output: {final_output.shape}")

    # The final output goes through linear projections, so it won't preserve unit norms
    # But the normalized quaternions should be unit before postprocessing

    return unit_check

if __name__ == "__main__":
    print("🧪 UNIT PROJECTION DEBUG TESTS")
    print("=" * 50)

    test1 = test_unit_projection_direct()
    test2 = test_full_pipeline()

    if test1 and test2:
        print("\n✅ Unit projection working correctly!")
        print("   The normalization step produces unit quaternions.")
        print("   Final output differs due to linear projections in postprocessing.")
    else:
        print("\n❌ Unit projection has issues that need fixing.")