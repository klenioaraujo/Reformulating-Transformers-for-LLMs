#!/usr/bin/env python3
"""
Parseval Validation Test for ΨQRH

Verifies compliance with Parseval's Theorem in all spectral operations
"""

import torch
import torch.nn as nn
import sys
import os

# Add src and configs directories to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

from src.architecture.psiqrh_transformer import PsiQRHTransformer
from examples.config_loader import get_example_config
from src.core.utils import (
    validate_parseval,
    validate_parseval_local,
    parseval_checkpoint,
    energy_preserve,
    spectral_operation_with_parseval
)


def test_pure_fft_parseval():
    """Tests pure FFT compliance with Parseval (isolated from spectral operations)"""
    print("=== Pure FFT Parseval Compliance Test ===")
    print("=" * 50)

    # Test signal
    x = torch.randn(1, 128, 512)
    print(f"Signal shape: {x.shape}")

    # Pure FFT/IFFT cycle with orthonormal normalization
    x_fft = torch.fft.fft(x, dim=1, norm="ortho")
    x_ifft = torch.fft.ifft(x_fft, dim=1, norm="ortho")

    # Validate Parseval for pure FFT only
    fft_parseval_valid = validate_parseval_local(x, x_fft)
    ifft_parseval_valid = validate_parseval_local(x_fft, x_ifft)
    reconstruction_error = torch.norm(x - x_ifft.real, p=2).item()

    print(f"\nPure FFT Parseval valid: {'✅ YES' if fft_parseval_valid else '❌ NO'}")
    print(f"Pure IFFT Parseval valid: {'✅ YES' if ifft_parseval_valid else '❌ NO'}")
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    print(f"Perfect reconstruction: {'✅ YES' if reconstruction_error < 1e-4 else '❌ NO'}")

    return fft_parseval_valid and ifft_parseval_valid and reconstruction_error < 1e-4


def test_energy_preservation_function():
    """Tests energy preservation function"""
    print("\n=== Energy Preservation Function Test ===")
    print("=" * 50)

    # Test data
    batch_size, seq_len, d_model = 2, 128, 512
    x_input = torch.randn(batch_size, seq_len, d_model)
    x_output = torch.randn(batch_size, seq_len, d_model) * 3.0  # Different energy

    print(f"Input shape: {x_input.shape}")
    print(f"Output shape: {x_output.shape}")

    # Calculate energies before
    input_energy_before = torch.sum(x_input**2, dim=-1).mean().item()
    output_energy_before = torch.sum(x_output**2, dim=-1).mean().item()

    print(f"\nEnergies before:")
    print(f"   Input: {input_energy_before:.6f}")
    print(f"   Output: {output_energy_before:.6f}")
    print(f"   Ratio: {output_energy_before/input_energy_before:.6f}")

    # Apply preservation
    normalized = energy_preserve(x_input, x_output)

    # Calculate energies after
    normalized_energy = torch.sum(normalized**2, dim=-1).mean().item()
    ratio_after = normalized_energy / input_energy_before

    print(f"\nEnergies after:")
    print(f"   Normalized: {normalized_energy:.6f}")
    print(f"   Ratio: {ratio_after:.6f}")
    print(f"   Preservation: {'✅ PERFECT' if abs(ratio_after - 1.0) < 1e-6 else '✅ GOOD' if abs(ratio_after - 1.0) < 0.01 else '❌ POOR'}")

    return abs(ratio_after - 1.0) < 0.01


def test_psiqrh_parseval_compliance():
    """Tests ΨQRH with Parseval validation"""
    print("\n=== ΨQRH with Parseval Validation Test ===")
    print("=" * 50)

    # Load configuration for parseval validation test
    print("Loading configuration for parseval_validation_test.py...")
    config = get_example_config("parseval_validation_test.py")
    print(f"Config loaded: {config.get('vocab_size', 1000)} vocab, {config.get('d_model', 256)} d_model")

    print("Creating ΨQRH transformer...")
    model = PsiQRHTransformer(
        vocab_size=config.get('vocab_size', 1000),
        d_model=config.get('d_model', 256),
        n_layers=4,
        n_heads=8
    )

    # Test data
    input_ids = torch.randint(0, config.get('vocab_size', 1000), (1, 64))
    print(f"Input shape: {input_ids.shape}")

    # Validate Parseval during runtime
    print("\nValidating Parseval during forward pass...")

    with torch.no_grad():
        # Initial checkpoint
        input_embeddings = model.token_embedding(input_ids)
        initial_parseval = parseval_checkpoint(input_embeddings, "input_embeddings")

        # Forward pass
        output = model(input_ids)

        # Final checkpoint
        final_parseval = parseval_checkpoint(output, "final_output")

    # Calculate energy conservation
    input_energy = torch.sum(input_embeddings**2, dim=-1).mean().item()
    output_energy = torch.sum(output**2, dim=-1).mean().item()
    conservation_ratio = output_energy / input_energy

    print(f"\nResults:")
    print(f"   Input energy: {input_energy:.6f}")
    print(f"   Output energy: {output_energy:.6f}")
    print(f"   Conservation ratio: {conservation_ratio:.6f}")
    print(f"   Initial Parseval: {'✅ OK' if initial_parseval else '❌ FAILED'}")
    print(f"   Final Parseval: {'✅ OK' if final_parseval else '❌ FAILED'}")
    print(f"   Energy conservation: {'✅ OK' if abs(conservation_ratio - 1.0) < 0.05 else '❌ FAILED'}")

    return initial_parseval and final_parseval and abs(conservation_ratio - 1.0) < 0.05


def test_spectral_operation_energy_preservation():
    """Tests spectral operation wrapper for energy preservation (not Parseval)"""
    print("\n=== Spectral Operation Energy Preservation Test ===")
    print("=" * 50)

    # Test data
    x = torch.randn(1, 128, 512)
    print(f"Input shape: {x.shape}")

    # Define spectral operation that intentionally changes energy
    def spectral_operation(x_fft):
        # Apply filter that changes energy (non-unitary operation)
        return x_fft * 0.5

    # Execute with wrapper
    result = spectral_operation_with_parseval(
        x,
        spectral_operation,
        "test_spectral_op"
    )

    print(f"Result shape: {result.shape}")

    # Validate ENERGY preservation (not Parseval)
    input_energy = torch.sum(x**2).item()
    output_energy = torch.sum(result**2).item()
    energy_ratio = output_energy / input_energy

    print(f"Input energy: {input_energy:.6f}")
    print(f"Output energy: {output_energy:.6f}")
    print(f"Energy ratio: {energy_ratio:.6f}")
    print(f"Energy preserved: {'✅ YES' if abs(energy_ratio - 1.0) < 0.01 else '❌ NO'}")

    # Note: Parseval is NOT expected to be preserved for non-unitary operations
    return abs(energy_ratio - 1.0) < 0.01


def comprehensive_parseval_validation():
    """Comprehensive Parseval validation"""
    print("\n=== COMPREHENSIVE PARSEVAL VALIDATION ===")
    print("=" * 50)

    tests = [
        ("Pure FFT Parseval", test_pure_fft_parseval),
        ("Energy Preservation", test_energy_preservation_function),
        ("ΨQRH Energy Conservation", test_psiqrh_parseval_compliance),
        ("Spectral Operation Energy", test_spectral_operation_energy_preservation)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   Result: {status}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("PARSEVAL VALIDATION SUMMARY")
    print("=" * 50)

    passed = sum(1 for name, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎯 SYSTEM COMPLIANT WITH ENERGY CONSERVATION!")
        print("✅ Pure FFT operations preserve Parseval")
        print("✅ All spectral operations preserve energy")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("❌ Review energy conservation implementations")

    return passed == total


def main():
    """Main function"""
    print("ΨQRH - Energy Conservation & Parseval Validation")
    print("=" * 60)
    print("Objective: Verify energy conservation in ΨQRH operations")
    print("           Validate Parseval for pure FFT operations only")
    print("=" * 60)

    # Execute comprehensive validation
    all_passed = comprehensive_parseval_validation()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ENERGY CONSERVATION VALIDATION SUCCESSFUL")
        print("✅ Pure FFT operations preserve Parseval")
        print("✅ All ΨQRH operations preserve energy")
    else:
        print("❌ ENERGY CONSERVATION VALIDATION FAILED")
        print("❌ Review energy conservation implementations")
    print("=" * 60)


if __name__ == "__main__":
    main()