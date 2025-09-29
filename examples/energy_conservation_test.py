#!/usr/bin/env python3
"""
Test energy conservation improvements for ΨQRH Transformer
"""

import torch
import torch.nn as nn
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.architecture.psiqrh_transformer import PsiQRHTransformer
from src.validation.mathematical_validation import MathematicalValidator
from src.optimization.energy_normalizer import energy_preserve
from src.optimization.advanced_energy_controller import AdvancedEnergyController


def test_energy_normalizer():
    """Test the energy normalizer module"""
    print("Testing Energy Normalizer")
    print("=" * 50)

    # Create test data
    batch_size, seq_len, d_model = 2, 128, 512
    x = torch.randn(batch_size, seq_len, d_model)

    # Test basic energy normalizer
    normalized = energy_preserve(x, x * 2.0)  # Test with amplified signal

    # Calculate conservation
    input_energy = torch.norm(x, p=2).item()
    output_energy = torch.norm(normalized, p=2).item()
    conservation_ratio = output_energy / input_energy

    print(f"Basic Energy Normalizer:")
    print(f"  Input Energy: {input_energy:.6f}")
    print(f"  Output Energy: {output_energy:.6f}")
    print(f"  Conservation Ratio: {conservation_ratio:.6f}")
    print(f"  Target: 1.000000 ± 0.05")
    print(f"  Status: {'PASS' if abs(conservation_ratio - 1.0) <= 0.05 else 'FAIL'}")

    # Test advanced energy controller
    controller = AdvancedEnergyController(d_model, n_layers=1)
    controlled = controller(x, layer_idx=0)

    controlled_energy = torch.norm(controlled, p=2).item()
    controlled_ratio = controlled_energy / input_energy

    print(f"\nAdvanced Energy Controller:")
    print(f"  Controlled Energy: {controlled_energy:.6f}")
    print(f"  Controlled Ratio: {controlled_ratio:.6f}")
    print(f"  Status: {'PASS' if abs(controlled_ratio - 1.0) <= 0.05 else 'FAIL'}")

    return conservation_ratio, controlled_ratio


def test_enhanced_psiqrh():
    """Test ΨQRH with energy conservation"""
    print("\n" + "=" * 50)
    print("Testing Enhanced ΨQRH with Energy Conservation")
    print("=" * 50)

    # Create base ΨQRH transformer
    vocab_size = 1000
    d_model = 256
    base_model = PsiQRHTransformer(vocab_size=vocab_size, d_model=d_model)

    # Use base model with built-in energy preservation
    enhanced_model = base_model

    # Generate test input
    input_ids = torch.randint(0, vocab_size, (1, 64))

    # Test forward pass
    print("Running enhanced ΨQRH forward pass...")
    with torch.no_grad():
        output = enhanced_model(input_ids)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Test energy conservation
    validator = MathematicalValidator(tolerance=0.05)

    # Get input embeddings for proper energy calculation
    input_embeddings = base_model.token_embedding(input_ids)
    input_energy = torch.norm(input_embeddings, p=2).item()
    output_energy = torch.norm(output, p=2).item()
    conservation_ratio = output_energy / input_energy

    print(f"\nEnhanced ΨQRH Energy Conservation:")
    print(f"  Input Energy: {input_energy:.6f}")
    print(f"  Output Energy: {output_energy:.6f}")
    print(f"  Conservation Ratio: {conservation_ratio:.6f}")
    print(f"  Target: 1.000000 ± 0.05")
    print(f"  Status: {'PASS' if abs(conservation_ratio - 1.0) <= 0.05 else 'FAIL'}")

    return conservation_ratio


def compare_original_vs_enhanced():
    """Compare original ΨQRH vs enhanced with energy conservation"""
    print("\n" + "=" * 50)
    print("Comparing Original vs Enhanced ΨQRH")
    print("=" * 50)

    vocab_size = 1000
    d_model = 256

    # Create both models (same model, just testing the built-in energy preservation)
    original_model = PsiQRHTransformer(vocab_size=vocab_size, d_model=d_model)
    enhanced_model = original_model

    # Test input
    input_ids = torch.randint(0, vocab_size, (1, 64))

    # Test original model
    print("Testing Original ΨQRH...")
    with torch.no_grad():
        original_output = original_model(input_ids)

    input_embeddings = original_model.token_embedding(input_ids)
    original_input_energy = torch.norm(input_embeddings, p=2).item()
    original_output_energy = torch.norm(original_output, p=2).item()
    original_ratio = original_output_energy / original_input_energy

    # Test enhanced model
    print("Testing Enhanced ΨQRH...")
    with torch.no_grad():
        enhanced_output = enhanced_model(input_ids)

    enhanced_output_energy = torch.norm(enhanced_output, p=2).item()
    enhanced_ratio = enhanced_output_energy / original_input_energy

    print(f"\nComparison Results:")
    print(f"  Input Energy: {original_input_energy:.6f}")
    print(f"  Original Output Energy: {original_output_energy:.6f}")
    print(f"  Original Ratio: {original_ratio:.6f}")
    print(f"  Enhanced Output Energy: {enhanced_output_energy:.6f}")
    print(f"  Enhanced Ratio: {enhanced_ratio:.6f}")
    print(f"  Improvement: {abs(enhanced_ratio - 1.0) / abs(original_ratio - 1.0):.2f}x closer to 1.0")

    return original_ratio, enhanced_ratio


def main():
    """Main test function"""
    print("ΨQRH Energy Conservation Test Suite")
    print("=" * 50)

    # Test energy normalizer
    normalizer_ratio, controller_ratio = test_energy_normalizer()

    # Test enhanced ΨQRH
    enhanced_ratio = test_enhanced_psiqrh()

    # Compare original vs enhanced
    original_ratio, final_enhanced_ratio = compare_original_vs_enhanced()

    print("\n" + "=" * 50)
    print("Summary of Energy Conservation Improvements")
    print("=" * 50)

    print(f"Original ΨQRH Conservation Ratio: {original_ratio:.6f}")
    print(f"Enhanced ΨQRH Conservation Ratio: {final_enhanced_ratio:.6f}")
    print(f"Target Range: 0.95 - 1.05")

    original_deviation = abs(original_ratio - 1.0)
    enhanced_deviation = abs(final_enhanced_ratio - 1.0)

    print(f"\nOriginal Deviation from 1.0: {original_deviation:.6f}")
    print(f"Enhanced Deviation from 1.0: {enhanced_deviation:.6f}")

    if enhanced_deviation < original_deviation:
        improvement = (original_deviation - enhanced_deviation) / original_deviation * 100
        print(f"Improvement: {improvement:.1f}% closer to target")
    else:
        print("No improvement detected - further optimization needed")

    print("\n" + "=" * 50)
    print("Energy Conservation Test Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()