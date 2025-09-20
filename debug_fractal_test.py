#!/usr/bin/env python3
"""
Debug script to understand fractal analysis failures
"""

import numpy as np
import sys
import os

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from quartz_light_prototype import FractalAnalyzer

def test_fractal_analysis():
    analyzer = FractalAnalyzer()

    # Test uniform 2D data - should be close to dimension 2.0
    print("Testing fractal analysis on uniform 2D data...")

    for i in range(5):
        uniform_data = np.random.uniform(0, 1, (1000, 2))
        fractal_dim = analyzer.calculate_box_counting_dimension(uniform_data)
        error = abs(fractal_dim - 2.0)
        print(f"Test {i+1}: Calculated dim = {fractal_dim:.4f}, Error = {error:.4f}")

    # Test with different sample sizes
    print("\nTesting with different sample sizes:")
    for n_points in [500, 1000, 2000, 5000]:
        uniform_data = np.random.uniform(0, 1, (n_points, 2))
        fractal_dim = analyzer.calculate_box_counting_dimension(uniform_data)
        error = abs(fractal_dim - 2.0)
        print(f"N={n_points}: Calculated dim = {fractal_dim:.4f}, Error = {error:.4f}")

if __name__ == "__main__":
    test_fractal_analysis()