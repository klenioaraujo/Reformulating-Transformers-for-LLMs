#!/usr/bin/env python3
"""
Find Optimal Alpha Parameter for Spectral Filter
===============================================

Tests different alpha values to find the one that minimizes reconstruction error
in the ΨQRH spectral transformation pipeline.

Based on doe.md mathematical framework:
- F(k) = exp(iα · arctan(ln(|k| + ε)))
- Alpha controls the strength of spectral filtering
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Import required components
from examples.complete_spectral_pipeline_300_words import (
    SpectralQRHLayer,
    text_to_quaternion_embedding,
    invert_spectral_qrh
)

def test_alpha_reconstruction_error(alpha: float, test_text: str, embed_dim: int = 256) -> dict:
    """
    Test reconstruction error for a given alpha value.

    Args:
        alpha: Spectral filter parameter to test
        test_text: Text to use for testing
        embed_dim: Embedding dimension

    Returns:
        Dictionary with reconstruction metrics
    """
    try:
        # Step 1: Convert text to quaternion embedding
        psi_input = text_to_quaternion_embedding(test_text, embed_dim)

        # Step 2: Apply forward ΨQRH transform with given alpha
        qrh_layer = SpectralQRHLayer(embed_dim=embed_dim, alpha=alpha)
        psi_transformed = qrh_layer.forward(psi_input)

        # Step 3: Apply inverse transform
        psi_reconstructed = invert_spectral_qrh(psi_transformed, qrh_layer)

        # Step 4: Calculate reconstruction error metrics
        mse_error = F.mse_loss(psi_input, psi_reconstructed).item()
        mae_error = F.l1_loss(psi_input, psi_reconstructed).item()

        # Cosine similarity (flattened for global comparison)
        psi_input_flat = psi_input.flatten()
        psi_reconstructed_flat = psi_reconstructed.flatten()
        cos_similarity = F.cosine_similarity(
            psi_input_flat.unsqueeze(0),
            psi_reconstructed_flat.unsqueeze(0)
        ).item()

        # Energy preservation
        energy_preservation = torch.norm(psi_reconstructed) / torch.norm(psi_input)
        energy_preservation = energy_preservation.item()

        # Numerical stability (check for NaN/Inf)
        has_nan = torch.isnan(psi_reconstructed).any().item()
        has_inf = torch.isinf(psi_reconstructed).any().item()
        is_stable = not (has_nan or has_inf)

        return {
            'alpha': alpha,
            'mse_error': mse_error,
            'mae_error': mae_error,
            'cosine_similarity': cos_similarity,
            'energy_preservation': energy_preservation,
            'numerical_stability': is_stable,
            'input_norm': torch.norm(psi_input).item(),
            'reconstructed_norm': torch.norm(psi_reconstructed).item()
        }

    except Exception as e:
        return {
            'alpha': alpha,
            'error': str(e),
            'mse_error': float('inf'),
            'mae_error': float('inf'),
            'cosine_similarity': -1.0,
            'energy_preservation': 0.0,
            'numerical_stability': False
        }

def find_optimal_alpha():
    """
    Test different alpha values and find the optimal one that minimizes reconstruction error.
    """
    print("=" * 80)
    print("🔬 FIND OPTIMAL ALPHA PARAMETER FOR SPECTRAL FILTER")
    print("=" * 80)
    print("\nTesting different alpha values to minimize reconstruction error...")
    print("Using MSE between Ψ_input and Ψ_reconstructed as primary metric")
    print("\n" + "=" * 80)

    # Test alpha values
    alpha_values = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

    # Test text (shorter for faster testing)
    test_text = """
    The quick brown fox jumps over the lazy dog. This sentence contains every letter in the English alphabet.
    Natural language processing has evolved significantly with transformer architectures.
    """

    embed_dim = 256  # Use optimized embedding dimension
    results = []

    print(f"📊 Test Configuration:")
    print(f"   - Embedding dimension: {embed_dim}")
    print(f"   - Test text length: {len(test_text)} characters")
    print(f"   - Alpha values to test: {alpha_values}")
    print("\n🧪 Running experiments...")
    # Test each alpha value
    for i, alpha in enumerate(alpha_values):
        print(f"\n🧪 Testing alpha = {alpha} ({i+1}/{len(alpha_values)})")

        result = test_alpha_reconstruction_error(alpha, test_text, embed_dim)
        results.append(result)

        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print("   📊 Results:")
            print(f"   - MSE Error: {result['mse_error']:.6f}")
            print(f"   - Cosine Similarity: {result['cosine_similarity']:.6f}")
            print(f"   - Energy Preservation: {result['energy_preservation']:.6f}")
            print(f"   - MAE Error: {result['mae_error']:.6f}")
            print(f"   - Numerical stability: {'✅' if result['numerical_stability'] else '❌'}")

    # Filter out failed experiments
    valid_results = [r for r in results if 'error' not in r and r['numerical_stability']]

    if not valid_results:
        print("\n❌ ERROR: No valid experiments completed!")
        return None

    # Find optimal alpha (minimum MSE)
    best_result = min(valid_results, key=lambda x: x['mse_error'])
    optimal_alpha = best_result['alpha']

    print(f"\n" + "=" * 80)
    print("📈 EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    print("Alpha | MSE Error | Cosine Sim | Energy Pres | Stability")
    print("------|-----------|------------|-------------|-----------")

    for result in results:
        if 'error' in result:
            print(f"{result['alpha']:6.1f} | ERROR    | N/A        | N/A         | ❌")
        else:
            stability = "✅" if result['numerical_stability'] else "❌"
            print(f"{result['alpha']:6.1f} | {result['mse_error']:9.6f} | {result['cosine_similarity']:10.6f} | {result['energy_preservation']:11.6f} | {stability}")

    print(f"\nOPTIMAL ALPHA FOUND: {optimal_alpha}")
    print("   Best reconstruction metrics:")
    print(f"   - MSE Error: {best_result['mse_error']:.6f}")
    print(f"   - Cosine Similarity: {best_result['cosine_similarity']:.6f}")
    print(f"   - Energy Preservation: {best_result['energy_preservation']:.6f}")
    print(f"   - MAE Error: {best_result['mae_error']:.6f}")

    # Analysis and recommendations
    print(f"\n💡 ANALYSIS:")
    mse_values = [r['mse_error'] for r in valid_results]
    mse_range = max(mse_values) - min(mse_values)

    if mse_range < 0.001:
        print("   - Low variation in MSE across alpha values - alpha has minimal impact")
    elif mse_range < 0.01:
        print("   - Moderate variation - alpha tuning provides some benefit")
    else:
        print("   - High variation - alpha is a critical parameter for reconstruction quality")

    # Check if optimal alpha is at boundary
    if optimal_alpha == min(alpha_values):
        print("   ⚠️  Optimal alpha is at minimum boundary - consider testing lower values")
    elif optimal_alpha == max(alpha_values):
        print("   ⚠️  Optimal alpha is at maximum boundary - consider testing higher values")

    print(f"\n✅ RECOMMENDATION: Use alpha = {optimal_alpha} as default for SpectralQRHLayer")

    return optimal_alpha, results

if __name__ == "__main__":
    optimal_alpha, all_results = find_optimal_alpha()

    if optimal_alpha is not None:
        print(f"\n🎯 Optimal alpha parameter: {optimal_alpha}")
        print("   Use this value in SpectralQRHLayer initialization")
    else:
        print("\n❌ Failed to find optimal alpha parameter")