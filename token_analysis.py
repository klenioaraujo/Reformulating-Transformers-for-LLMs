#!/usr/bin/env python3
"""
Token Resonance Analysis - ΨQRH Optical Probe Diagnostics
=========================================================

Analyzes the token resonance patterns from the ΨQRH optical probe
to understand why certain tokens are selected and identify issues.
"""

import numpy as np
from typing import Dict, List, Tuple

def analyze_token_resonance_patterns():
    """
    Analyze the token resonance patterns from the provided pipeline outputs.
    """

    print("🔬 TOKEN RESONANCE ANALYSIS - ΨQRH OPTICAL PROBE")
    print("=" * 60)

    # Data from the user's provided outputs
    resonance_data = [
        {
            'iteration': 1,
            'resonance_spectrum': {
                1: 0.000217,
                3: 0.000193,
                21: 0.000191,
                23: 0.000182,
                19: 0.000174
            },
            'selected_token': 1,
            'max_resonance': 0.000217,
            'echo_quality': 1.0000,
            'vocab_size': 34
        },
        {
            'iteration': 2,
            'resonance_spectrum': {
                1: 0.000193,
                3: 0.000162,
                17: 0.000159,
                15: 0.000144,
                19: 0.000137
            },
            'selected_token': 1,
            'max_resonance': 0.000193,
            'echo_quality': 1.0000,
            'vocab_size': 34
        },
        {
            'iteration': 3,
            'resonance_spectrum': {
                8: 0.000152,
                1: 0.000146,
                17: 0.000124,
                15: 0.000123,
                10: 0.000113
            },
            'selected_token': 8,
            'max_resonance': 0.000152,
            'echo_quality': 1.0000,
            'vocab_size': 34
        }
    ]

    print("📊 ANALYZING RESONANCE PATTERNS ACROSS ITERATIONS")
    print("-" * 50)

    # Analyze resonance distribution
    all_resonances = []
    token_frequencies = {}
    max_resonances = []

    for data in resonance_data:
        spectrum = data['resonance_spectrum']
        all_resonances.extend(spectrum.values())
        max_resonances.append(data['max_resonance'])

        for token, resonance in spectrum.items():
            if token not in token_frequencies:
                token_frequencies[token] = []
            token_frequencies[token].append(resonance)

    # Statistical analysis
    resonance_array = np.array(all_resonances)
    print("\n🔢 STATISTICAL ANALYSIS:")
    print(f"   • Total resonance measurements: {len(resonance_array)}")
    print(f"   • Mean resonance: {resonance_array.mean():.6f}")
    print(f"   • Max resonance: {resonance_array.max():.6f}")
    print(f"   • Min resonance: {resonance_array.min():.6f}")
    print(f"   • Standard deviation: {resonance_array.std():.6f}")
    print(f"   • Range: {resonance_array.max() - resonance_array.min():.6f}")

    # Identify concerning patterns
    print("\n⚠️  CRITICAL ISSUES IDENTIFIED:")
    print(f"   • Very low resonance values (< 0.001) indicate weak optical coupling")
    print(f"   • Resonance range is only {resonance_array.max() - resonance_array.min():.6f}")
    print(f"   • This suggests the optical probe is not effectively measuring the quaternion state")

    # Token selection analysis
    print("\n🎯 TOKEN SELECTION ANALYSIS:")
    selected_tokens = [data['selected_token'] for data in resonance_data]
    print(f"   • Selected tokens: {selected_tokens}")
    print(f"   • Token diversity: {len(set(selected_tokens))} unique tokens out of {len(selected_tokens)} selections")

    # Character mapping (assuming ASCII mapping)
    print("\n🔤 CHARACTER MAPPING (ASCII):")
    for token in sorted(set(selected_tokens)):
        char = chr(32 + token) if 32 + token < 127 else f"INVALID({32 + token})"
        print(f"   • Token {token} → '{char}' (ASCII {32 + token})")

    # Resonance stability analysis
    print("\n📈 RESONANCE STABILITY:")
    stability_scores = []
    for i in range(len(resonance_data) - 1):
        current_max = resonance_data[i]['max_resonance']
        next_max = resonance_data[i + 1]['max_resonance']
        stability = 1.0 - abs(current_max - next_max) / max(current_max, next_max)
        stability_scores.append(stability)

    avg_stability = np.mean(stability_scores) if stability_scores else 0
    print(f"   • Average resonance stability: {avg_stability:.3f}")
    print(f"   • Echo quality consistently: {resonance_data[0]['echo_quality']:.4f}")

    # Root cause analysis
    print("\n🔍 ROOT CAUSE ANALYSIS:")
    print("   1. OPTICAL PROBE SIMPLIFICATION:")
    print("      • Using psi_last.mean().item() (scalar) for coupling")
    print("      • Ignores the full 4D quaternion structure")
    print("      • Should use proper quaternion inner product")

    print("   2. RESONANCE CALCULATION:")
    print("      • f(λ,t) = I₀·sin(ωt + αλ)·exp[i(ωt - kλ + βλ²)]")
    print("      • Coupling: |⟨f(λ,t), Ψ⟩|² where Ψ is scalar mean")
    print("      • Should be: |⟨f(λ,t), Ψ⟩|² where Ψ is full quaternion")

    print("   3. VOCABULARY SIZE:")
    print(f"      • Only {resonance_data[0]['vocab_size']} tokens analyzed")
    print("      • May be too small for effective resonance discrimination")

    print("   4. CALIBRATION ISSUES:")
    print("      • Recalibration triggered (ressonância < 0.001)")
    print("      • Indicates fundamental measurement problem")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("   1. Implement proper quaternion optical probe")
    print("   2. Use full 4D quaternion inner products")
    print("   3. Increase vocabulary size for better discrimination")
    print("   4. Implement multi-dimensional resonance measurement")
    print("   5. Add quaternion-specific coupling mechanisms")

    return {
        'resonance_stats': {
            'mean': float(resonance_array.mean()),
            'max': float(resonance_array.max()),
            'min': float(resonance_array.min()),
            'std': float(resonance_array.std())
        },
        'token_analysis': {
            'selected_tokens': selected_tokens,
            'unique_tokens': len(set(selected_tokens)),
            'token_frequencies': token_frequencies
        },
        'issues': [
            'Very low resonance values indicate weak coupling',
            'Optical probe uses scalar approximation instead of quaternion',
            'Limited vocabulary size affects discrimination',
            'Recalibration frequently triggered'
        ]
    }

def analyze_optical_probe_implementation():
    """
    Analyze the optical probe implementation in the code.
    """

    print("\n🔧 OPTICAL PROBE IMPLEMENTATION ANALYSIS")
    print("=" * 50)

    print("Current implementation issues:")
    print("1. psi_mean = psi_last.mean().item()  # Single scalar!")
    print("2. coupling = np.abs(f_lambda * psi_mean)**2  # 1D coupling")
    print("3. Missing quaternion structure utilization")

    print("\nCorrect implementation should be:")
    print("1. Use full quaternion state Ψ ∈ ℍ")
    print("2. Implement quaternion inner product ⟨f, Ψ⟩")
    print("3. f(λ,t) as quaternion wave function")
    print("4. |⟨f(λ,t), Ψ⟩|² with proper quaternion norm")

if __name__ == "__main__":
    results = analyze_token_resonance_patterns()
    analyze_optical_probe_implementation()

    print("\n📋 SUMMARY:")
    print(f"   • Resonance values are critically low (< 0.001)")
    print(f"   • Optical probe needs quaternion-aware implementation")
    print(f"   • Token selection shows limited diversity")
    print(f"   • Fundamental measurement issue identified")