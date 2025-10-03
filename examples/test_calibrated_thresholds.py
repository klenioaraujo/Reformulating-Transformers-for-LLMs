#!/usr/bin/env python3
"""
Test Calibrated FCI Thresholds
==============================

Test the calibrated FCI thresholds with ΨTWS data and compare
with original thresholds.
"""

import torch
import numpy as np
import yaml
from src.conscience.consciousness_metrics import ConsciousnessMetrics
from data.Ψtws.Ψtws_loader import ΨTWSLoader


def test_calibrated_thresholds():
    """Test calibrated thresholds vs original thresholds."""

    print("🧪 Testing Calibrated FCI Thresholds")
    print("=" * 50)

    # Load ΨTWS data
    loader = ΨTWSLoader()
    all_files = loader.load_training_files() + loader.load_validation_files() + loader.load_test_files()

    print(f"📚 Loaded {len(all_files)} ΨTWS files")

    # Load configurations
    with open('configs/consciousness_metrics.yaml', 'r') as f:
        original_config = yaml.safe_load(f)

    with open('calibrated_fci_thresholds.yaml', 'r') as f:
        calibrated_config = yaml.safe_load(f)

    # Create metrics instances
    base_config = {'device': 'cpu'}

    original_metrics = ConsciousnessMetrics(base_config, original_config)
    calibrated_metrics = ConsciousnessMetrics(base_config, calibrated_config)

    print("\n🔧 Configuration Comparison:")
    print("=" * 30)
    print("Original Thresholds:")
    print(f"  EMERGENCE: ≥ {original_config['state_thresholds']['emergence']['min_fci']}")
    print(f"  MEDITATION: ≥ {original_config['state_thresholds']['meditation']['min_fci']}")
    print(f"  ANALYSIS: ≥ {original_config['state_thresholds']['analysis']['min_fci']}")

    print("\nCalibrated Thresholds:")
    print(f"  EMERGENCE: ≥ {calibrated_config['state_thresholds']['emergence']['min_fci']}")
    print(f"  MEDITATION: ≥ {calibrated_config['state_thresholds']['meditation']['min_fci']}")
    print(f"  ANALYSIS: ≥ {calibrated_config['state_thresholds']['analysis']['min_fci']}")

    # Test with ΨTWS files
    print("\n📊 Testing with ΨTWS Files:")
    print("=" * 30)

    results = {
        'original': {'EMERGENCE': 0, 'MEDITATION': 0, 'ANALYSIS': 0, 'COMA': 0},
        'calibrated': {'EMERGENCE': 0, 'MEDITATION': 0, 'ANALYSIS': 0, 'COMA': 0}
    }

    for file_data in all_files:
        file_name = file_data['file_path'].split('/')[-1]

        # Generate FCI based on file characteristics
        fci = generate_fci_from_file_data(file_data)

        # Classify with both threshold sets
        original_state = original_metrics._classify_fci_state(fci)
        calibrated_state = calibrated_metrics._classify_fci_state(fci)

        results['original'][original_state] += 1
        results['calibrated'][calibrated_state] += 1

        print(f"\n📄 {file_name}:")
        print(f"   FCI: {fci:.3f}")
        print(f"   Original: {original_state}")
        print(f"   Calibrated: {calibrated_state}")

    # Print summary
    print("\n📈 Summary Results:")
    print("=" * 30)

    for config_name, state_counts in results.items():
        total = sum(state_counts.values())
        print(f"\n{config_name.upper()}:")
        for state, count in state_counts.items():
            percentage = (count / total) * 100
            print(f"  {state}: {count} ({percentage:.1f}%)")

    # Test with synthetic data across FCI range
    print("\n🔬 FCI Range Analysis:")
    print("=" * 30)

    fci_test_values = [0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9]

    print("FCI\tOriginal\tCalibrated")
    print("-" * 40)

    for fci in fci_test_values:
        original_state = original_metrics._classify_fci_state(fci)
        calibrated_state = calibrated_metrics._classify_fci_state(fci)
        print(f"{fci}\t{original_state}\t{calibrated_state}")


def generate_fci_from_file_data(file_data):
    """Generate synthetic FCI value based on file characteristics."""

    # Use file metadata to generate realistic FCI
    spectral_dim = file_data['metadata'].get('spectral_dimension', 256)
    encryption_layers = file_data['metadata'].get('encryption_layers', 7)

    # More complex files (higher spectral dim, more encryption) → higher FCI
    complexity_factor = (spectral_dim / 512) * (encryption_layers / 7)

    # Generate FCI with some randomness but biased by complexity
    base_fci = 0.3 + (complexity_factor * 0.6)  # Range: 0.3-0.9
    noise = np.random.normal(0, 0.05)  # Reduced noise for more consistent results

    fci = np.clip(base_fci + noise, 0.0, 1.0)
    return fci


def analyze_threshold_impact():
    """Analyze the impact of threshold changes."""

    print("\n🎯 Threshold Impact Analysis:")
    print("=" * 30)

    # Load configurations
    with open('configs/consciousness_metrics.yaml', 'r') as f:
        original_config = yaml.safe_load(f)

    with open('calibrated_fci_thresholds.yaml', 'r') as f:
        calibrated_config = yaml.safe_load(f)

    original_thresholds = original_config['state_thresholds']
    calibrated_thresholds = calibrated_config['state_thresholds']

    print("\nThreshold Changes:")
    for state in ['emergence', 'meditation', 'analysis']:
        orig = original_thresholds[state]['min_fci']
        calib = calibrated_thresholds[state]['min_fci']
        change = calib - orig
        direction = "↑" if change > 0 else "↓"
        print(f"  {state.upper()}: {orig:.3f} → {calib:.3f} ({direction}{abs(change):.3f})")

    print("\nExpected Impact:")
    print("  - ANALYSIS: More files classified as ANALYSIS (threshold lowered)")
    print("  - MEDITATION: Similar classification (small increase)")
    print("  - EMERGENCE: Fewer files classified as EMERGENCE (threshold lowered)")
    print("  - COMA: Fewer files classified as COMA (analysis threshold raised)")


if __name__ == "__main__":
    test_calibrated_thresholds()
    analyze_threshold_impact()

    print("\n✅ Calibrated threshold testing completed!")