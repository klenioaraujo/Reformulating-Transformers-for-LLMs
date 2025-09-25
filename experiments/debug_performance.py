#!/usr/bin/env python3
"""
Debug script for performance benchmark issue
"""
import numpy as np
import time
import sys
import os

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.conceptual.quartz_light_prototype import FractalAnalyzer

def debug_fractal_analysis():
    print("Debugging fractal analysis performance issue...")

    analyzer = FractalAnalyzer()
    data_2d = np.random.rand(1000, 1000)

    print(f"Data shape: {data_2d.shape}")
    print(f"Data size: {data_2d.size}")

    try:
        start_time = time.time()
        fractal_dim = analyzer.calculate_box_counting_dimension(data_2d)
        fractal_time = time.time() - start_time

        print(f"Fractal dimension: {fractal_dim}")
        print(f"Analysis time: {fractal_time:.6f} seconds")
        print(f"Analysis time: {fractal_time * 1000:.2f} ms")

        return True

    except Exception as e:
        print(f"Error in fractal analysis: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_fractal_analysis()