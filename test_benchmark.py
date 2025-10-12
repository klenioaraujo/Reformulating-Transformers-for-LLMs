#!/usr/bin/env python3
"""
Test script for ΨQRH benchmark data generator
===========================================

Quick test to verify the benchmark script works correctly.
"""

import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        from src.architecture.psiqrh_transformer import PsiQRHTransformer
        print("✅ PsiQRHTransformer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PsiQRHTransformer: {e}")
        return False

    try:
        from generate_benchmark_data import generate_benchmark_data
        print("✅ Benchmark data generator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import benchmark functions: {e}")
        return False

    return True

def test_data_generation():
    """Test benchmark data generation"""
    print("Testing benchmark data generation...")

    try:
        from generate_benchmark_data import generate_benchmark_data

        wikitext_results, glue_results = generate_benchmark_data()

        print("✅ Benchmark data generated successfully")
        print(f"   WikiText results: {len(wikitext_results)} models")
        print(f"   GLUE results: {len(glue_results)} models")

        # Check key metrics
        psiqrh_wikitext = wikitext_results['psiqrh']
        psiqrh_glue = glue_results['psiqrh']

        print(f"   ΨQRH PPL: {psiqrh_wikitext['final_val_ppl']}")
        print(f"   ΨQRH Params: {psiqrh_wikitext['parameters']:,}")
        print(f"   ΨQRH GLUE Avg: {psiqrh_glue['average_score']:.2f}")

        return True
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ΨQRH Benchmark Test Suite")
    print("=" * 40)

    tests = [
        ("Imports", test_imports),
        ("Data Generation", test_data_generation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        if test_func():
            passed += 1
        print("-" * 30)

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Benchmark data generator is working.")
        print("\nTo generate benchmark data:")
        print("python generate_benchmark_data.py --generate-tables")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())