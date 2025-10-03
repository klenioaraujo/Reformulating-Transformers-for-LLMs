#!/usr/bin/env python3
"""
Test script for GLS output generation
"""

import sys
import os

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.conscience.gls_output_generator import GLSOutputGenerator

def test_gls_generator():
    """Test the GLS output generator directly"""
    print("🧪 Testing GLS Output Generator...")

    # Create test consciousness results
    test_results = {
        'fci_evolution': [0.0033],  # FCI value
        'final_consciousness_state': type('State', (), {
            'name': 'COMA',
            'fractal_dimension': 1.008
        })(),
        'consciousness_distribution': None,
        'fractal_field': None,
        'processing_steps': 12,
        'convergence_achieved': True
    }

    # Create GLS generator
    gls_generator = GLSOutputGenerator()

    # Generate Processing code
    processing_code = gls_generator.generate_processing_code(test_results)
    print("\n📱 Processing Code:")
    print("=" * 50)
    print(processing_code[:500] + "..." if len(processing_code) > 500 else processing_code)

    # Generate p5.js code
    p5js_code = gls_generator.generate_p5js_code(test_results)
    print("\n🌐 p5.js Code:")
    print("=" * 50)
    print(p5js_code[:500] + "..." if len(p5js_code) > 500 else p5js_code)

    print("\n✅ GLS Output Generator test completed!")

if __name__ == "__main__":
    test_gls_generator()