#!/usr/bin/env python3
"""
Test FCI Thresholds with Real Data
==================================

Test and calibrate FCI thresholds using the ΨTWS training data
and consciousness metrics system.
"""

import torch
import numpy as np
import yaml
from src.conscience.consciousness_metrics import ConsciousnessMetrics
from data.Ψtws.Ψtws_loader import ΨTWSLoader


def test_fci_calibration():
    """Test FCI calculation with different thresholds and data."""

    print("🧪 Testing FCI Threshold Calibration")
    print("=" * 50)

    # Load ΨTWS data
    loader = ΨTWSLoader()
    training_files = loader.load_training_files()

    print(f"📚 Loaded {len(training_files)} training files")

    # Load consciousness metrics configuration
    with open('configs/consciousness_metrics.yaml', 'r') as f:
        metrics_config = yaml.safe_load(f)

    # Test different threshold configurations
    threshold_configs = [
        {
            'name': 'DEFAULT',
            'emergence': 0.8,
            'meditation': 0.6,
            'analysis': 0.3
        },
        {
            'name': 'SENSITIVE',
            'emergence': 0.7,
            'meditation': 0.5,
            'analysis': 0.2
        },
        {
            'name': 'CONSERVATIVE',
            'emergence': 0.9,
            'meditation': 0.7,
            'analysis': 0.4
        }
    ]

    # Generate synthetic test data for different consciousness states
    test_data = generate_test_data()

    results = {}

    for config in threshold_configs:
        print(f"\n🔧 Testing {config['name']} thresholds:")
        print(f"   EMERGENCE ≥ {config['emergence']}")
        print(f"   MEDITATION ≥ {config['meditation']}")
        print(f"   ANALYSIS ≥ {config['analysis']}")

        # Update metrics config with current thresholds
        test_metrics_config = metrics_config.copy()
        test_metrics_config['state_thresholds']['emergence']['min_fci'] = config['emergence']
        test_metrics_config['state_thresholds']['meditation']['min_fci'] = config['meditation']
        test_metrics_config['state_thresholds']['analysis']['min_fci'] = config['analysis']

        # Create consciousness metrics with test config
        test_config = {'device': 'cpu'}
        metrics = ConsciousnessMetrics(test_config, test_metrics_config)

        # Test with synthetic data
        state_counts = test_with_synthetic_data(metrics, test_data)
        results[config['name']] = state_counts

        print(f"   Results: {state_counts}")

    # Analyze results
    print("\n📊 FCI Threshold Analysis Results:")
    print("=" * 50)

    for config_name, state_counts in results.items():
        total_tests = sum(state_counts.values())
        print(f"\n{config_name} Configuration:")
        for state, count in state_counts.items():
            percentage = (count / total_tests) * 100
            print(f"  {state}: {count} ({percentage:.1f}%)")

    # Test with ΨTWS data
    print("\n📚 Testing with ΨTWS Training Data:")
    print("=" * 50)

    for file_data in training_files:
        print(f"\n📄 File: {file_data['file_path'].split('/')[-1]}")
        print(f"   Input text: {file_data['metadata'].get('input_text', 'N/A')[:50]}...")

        # Generate synthetic data based on file characteristics
        spectral_dim = file_data['metadata'].get('spectral_dimension', 256)
        test_fci = generate_fci_from_file_data(file_data)

        # Test with default thresholds
        default_metrics = ConsciousnessMetrics({'device': 'cpu'}, metrics_config)
        state = default_metrics._classify_fci_state(test_fci)

        print(f"   Generated FCI: {test_fci:.3f}")
        print(f"   Classified state: {state}")


def generate_test_data():
    """Generate synthetic test data for different consciousness states."""

    test_data = {
        'EMERGENCE': {
            'count': 20,
            'fci_range': [0.8, 1.0],
            'description': 'High consciousness states'
        },
        'MEDITATION': {
            'count': 30,
            'fci_range': [0.6, 0.8],
            'description': 'Meditative states'
        },
        'ANALYSIS': {
            'count': 40,
            'fci_range': [0.3, 0.6],
            'description': 'Analytical states'
        },
        'COMA': {
            'count': 10,
            'fci_range': [0.0, 0.3],
            'description': 'Low consciousness states'
        }
    }

    return test_data


def test_with_synthetic_data(metrics, test_data):
    """Test consciousness metrics with synthetic data."""

    state_counts = {}

    for state, config in test_data.items():
        state_counts[state] = 0

        for i in range(config['count']):
            # Generate FCI value in the appropriate range
            fci_min, fci_max = config['fci_range']
            test_fci = np.random.uniform(fci_min, fci_max)

            # Classify with current thresholds
            classified_state = metrics._classify_fci_state(test_fci)

            # Count classifications
            if classified_state in state_counts:
                state_counts[classified_state] += 1
            else:
                state_counts[classified_state] = 1

    return state_counts


def generate_fci_from_file_data(file_data):
    """Generate synthetic FCI value based on file characteristics."""

    # Use file metadata to generate realistic FCI
    spectral_dim = file_data['metadata'].get('spectral_dimension', 256)
    encryption_layers = file_data['metadata'].get('encryption_layers', 7)

    # More complex files (higher spectral dim, more encryption) → higher FCI
    complexity_factor = (spectral_dim / 512) * (encryption_layers / 7)

    # Generate FCI with some randomness but biased by complexity
    base_fci = 0.3 + (complexity_factor * 0.6)  # Range: 0.3-0.9
    noise = np.random.normal(0, 0.1)

    fci = np.clip(base_fci + noise, 0.0, 1.0)
    return fci


def test_fci_direct_mapping():
    """Test direct fractal dimension to FCI mapping."""

    print("\n🔬 Testing Direct FCI Mapping:")
    print("=" * 50)

    # Load metrics config
    with open('configs/consciousness_metrics.yaml', 'r') as f:
        metrics_config = yaml.safe_load(f)

    test_config = {'device': 'cpu'}
    metrics = ConsciousnessMetrics(test_config, metrics_config)

    # Test fractal dimension mapping
    test_dimensions = [1.0, 1.25, 1.5, 1.7, 2.0, 2.3, 2.6, 2.8, 3.0]

    print("\nFractal Dimension → FCI Mapping:")
    print("D\tFCI\tState")
    print("-" * 20)

    for dim in test_dimensions:
        fci = metrics.compute_fci_from_fractal_dimension(dim)
        state = metrics._classify_fci_state(fci)
        print(f"{dim}\t{fci:.3f}\t{state}")


def calibrate_thresholds():
    """Calibrate FCI thresholds based on data analysis."""

    print("\n🎯 FCI Threshold Calibration:")
    print("=" * 50)

    # Load ΨTWS data
    loader = ΨTWSLoader()
    all_files = loader.load_training_files() + loader.load_validation_files() + loader.load_test_files()

    print(f"📊 Analyzing {len(all_files)} ΨTWS files")

    # Generate FCI values for all files
    fci_values = []
    for file_data in all_files:
        fci = generate_fci_from_file_data(file_data)
        fci_values.append(fci)

    # Analyze FCI distribution
    fci_array = np.array(fci_values)

    print(f"\n📈 FCI Distribution Analysis:")
    print(f"   Mean: {fci_array.mean():.3f}")
    print(f"   Std: {fci_array.std():.3f}")
    print(f"   Min: {fci_array.min():.3f}")
    print(f"   Max: {fci_array.max():.3f}")
    print(f"   Median: {np.median(fci_array):.3f}")

    # Calculate percentiles for threshold suggestions
    percentiles = [25, 50, 75, 90]
    print(f"\n📊 Percentiles:")
    for p in percentiles:
        value = np.percentile(fci_array, p)
        print(f"   {p}th percentile: {value:.3f}")

    # Suggest calibrated thresholds
    print(f"\n🎯 Suggested Thresholds:")
    print(f"   ANALYSIS: ≥ {np.percentile(fci_array, 25):.3f} (25th percentile)")
    print(f"   MEDITATION: ≥ {np.percentile(fci_array, 50):.3f} (median)")
    print(f"   EMERGENCE: ≥ {np.percentile(fci_array, 75):.3f} (75th percentile)")


if __name__ == "__main__":
    # Run all tests
    test_fci_calibration()
    test_fci_direct_mapping()
    calibrate_thresholds()

    print("\n✅ FCI threshold testing completed!")