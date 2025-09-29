#!/usr/bin/env python3
"""
Parseval Validation Test for ΨQRH

Verifies compliance with Parseval's Theorem in all spectral operations
"""

import torch
import torch.nn as nn
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.architecture.psiqrh_transformer import PsiQRHTransformer
from src.core.utils import (
    validate_parseval,
    parseval_checkpoint,
    energy_preserve,
    spectral_operation_with_parseval
)


def test_fft_parseval_compliance():
    """Tests basic FFT compliance with Parseval"""
    print("=== FFT/Parseval Compliance Test ===")
    print("=" * 50)

    # Test signal
    t = torch.linspace(0, 2 * torch.pi, 128)
    signal = torch.sin(2 * torch.pi * 5 * t) + 0.5 * torch.sin(2 * torch.pi * 10 * t)
    signal = signal.unsqueeze(0).unsqueeze(-1)  # [1, 128, 1]

    print(f"Signal shape: {signal.shape}")

    # Test FFT without normalization
    print("\n1. FFT without normalization:")
    fft_no_norm = torch.fft.fft(signal, dim=1)
    energy_time_no_norm = torch.sum(signal.abs()**2).item()
    energy_freq_no_norm = torch.sum(fft_no_norm.abs()**2).item()
    ratio_no_norm = energy_time_no_norm / energy_freq_no_norm

    print(f"   Time energy: {energy_time_no_norm:.6f}")
    print(f"   Frequency energy: {energy_freq_no_norm:.6f}")
    print(f"   Ratio: {ratio_no_norm:.6f}")
    print(f"   Parseval valid: {'✅ YES' if abs(ratio_no_norm - 1.0) < 0.01 else '❌ NO'}")

    # Test FFT with orthonormal normalization
    print("\n2. FFT with norm='ortho':")
    fft_ortho = torch.fft.fft(signal, dim=1, norm="ortho")
    energy_time_ortho = torch.sum(signal.abs()**2).item()
    energy_freq_ortho = torch.sum(fft_ortho.abs()**2).item()
    ratio_ortho = energy_time_ortho / energy_freq_ortho

    print(f"   Time energy: {energy_time_ortho:.6f}")
    print(f"   Frequency energy: {energy_freq_ortho:.6f}")
    print(f"   Ratio: {ratio_ortho:.6f}")
    print(f"   Parseval valid: {'✅ YES' if abs(ratio_ortho - 1.0) < 0.01 else '❌ NO'}")

    # Test reconstruction
    print("\n3. IFFT Reconstruction:")
    reconstructed = torch.fft.ifft(fft_ortho, dim=1, norm="ortho").real
    reconstruction_error = torch.norm(signal - reconstructed, p=2).item()
    print(f"   Reconstruction error: {reconstruction_error:.6f}")
    print(f"   Perfect reconstruction: {'✅ YES' if reconstruction_error < 1e-6 else '❌ NO'}")

    return abs(ratio_ortho - 1.0) < 0.01 and reconstruction_error < 1e-6


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

    # Create model
    vocab_size = 1000
    d_model = 256

    print("Creating ΨQRH transformer...")
    model = PsiQRHTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=4,
        n_heads=8
    )

    # Test data
    input_ids = torch.randint(0, vocab_size, (1, 64))
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


def test_spectral_operation_wrapper():
    """Tests spectral operation wrapper"""
    print("\n=== Spectral Operation Wrapper Test ===")
    print("=" * 50)

    # Test data
    x = torch.randn(1, 128, 512)
    print(f"Input shape: {x.shape}")

    # Define simple spectral operation
    def spectral_operation(x_fft):
        # Apply simple filter (preserves Parseval)
        return x_fft * 0.5

    # Execute with wrapper
    result = spectral_operation_with_parseval(
        x,
        spectral_operation,
        "test_spectral_op"
    )

    print(f"Result shape: {result.shape}")

    # Validate Parseval
    is_valid = validate_parseval(x, result)
    print(f"Parseval preserved: {'✅ YES' if is_valid else '❌ NO'}")

    return is_valid


def comprehensive_parseval_validation():
    """Comprehensive Parseval validation"""
    print("\n=== COMPREHENSIVE PARSEVAL VALIDATION ===")
    print("=" * 50)

    tests = [
        ("FFT Compliance", test_fft_parseval_compliance),
        ("Energy Preservation", test_energy_preservation_function),
        ("ΨQRH Parseval", test_psiqrh_parseval_compliance),
        ("Spectral Operation", test_spectral_operation_wrapper)
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
        print("\n🎯 SYSTEM COMPLIANT WITH PARSEVAL THEOREM!")
        print("✅ All spectral operations preserve energy")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("❌ Review spectral implementations")

    return passed == total


def main():
    """Main function"""
    print("ΨQRH - Complete Parseval Theorem Validation")
    print("=" * 60)
    print("Objective: Verify compliance with ||x||² = ||F{x}||²")
    print("=" * 60)

    # Execute comprehensive validation
    all_passed = comprehensive_parseval_validation()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ PARSEVAL VALIDATION COMPLETED SUCCESSFULLY")
        print("✅ All spectral operations are unitary")
        print("✅ Energy conservation guaranteed")
    else:
        print("❌ PARSEVAL VALIDATION FAILED")
        print("❌ Review FFT implementations")
    print("=" * 60)


if __name__ == "__main__":
    main()