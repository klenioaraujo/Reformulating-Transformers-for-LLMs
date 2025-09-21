#!/usr/bin/env python3
"""
Test script to verify NaN resilience improvements
"""
import torch
import warnings
import sys
import os

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from qrh_layer import QRHLayer, QRHConfig

def test_nan_resilience():
    print("Testing NaN resilience in QRHLayer...")

    # Create layer
    config = QRHConfig(embed_dim=8, alpha=1.0, enable_warnings=False)  # Disable warnings for test
    layer = QRHLayer(config)

    # Test case 1: Input with some NaN values
    nan_input = torch.randn(2, 16, 32)
    nan_input[0, 0, 0] = float('nan')

    print(f"Input contains NaN: {torch.isnan(nan_input).any()}")
    print(f"Number of NaN values: {torch.isnan(nan_input).sum()}")

    try:
        with torch.no_grad():
            output = layer(nan_input)

        print(f"Output shape: {output.shape}")
        print(f"Output contains NaN: {torch.isnan(output).any()}")
        print(f"Output contains Inf: {torch.isinf(output).any()}")
        print(f"All output is NaN: {torch.isnan(output).all()}")

        # The test should pass if not all outputs are NaN
        test_passed = not torch.isnan(output).all()
        print(f"NaN resilience test: {'PASSED' if test_passed else 'FAILED'}")

        return test_passed

    except Exception as e:
        print(f"Error during forward pass: {e}")
        return False

def test_all_nan_input():
    print("\nTesting with all NaN input...")

    config = QRHConfig(embed_dim=8, alpha=1.0, enable_warnings=False)
    layer = QRHLayer(config)

    # All NaN input
    all_nan_input = torch.full((2, 16, 32), float('nan'))

    try:
        with torch.no_grad():
            output = layer(all_nan_input)

        print(f"Output shape: {output.shape}")
        print(f"Output contains NaN: {torch.isnan(output).any()}")
        print(f"Output contains Inf: {torch.isinf(output).any()}")
        print(f"Output is finite: {torch.isfinite(output).all()}")

        # For all NaN input, we expect finite output (due to bias terms)
        # The important thing is that it doesn't crash and doesn't produce NaN/Inf
        return torch.isfinite(output).all()

    except Exception as e:
        print(f"Error with all NaN input: {e}")
        return False

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        test1_passed = test_nan_resilience()
        test2_passed = test_all_nan_input()

        print(f"\nOverall NaN resilience: {'PASSED' if test1_passed and test2_passed else 'FAILED'}")