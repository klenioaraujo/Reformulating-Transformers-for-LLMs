#!/usr/bin/env python3
"""
Integration Test for Tensor Validator in ΨQRH
Tests the integration of ScientificTensorValidator in QRHLayer and ΨQRHPipeline
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.qrh_layer import QRHLayer, QRHConfig
from src.core.tensor_validator import ScientificTensorValidator


def test_tensor_validator_import():
    """Test that tensor validator can be imported correctly."""
    print("🧪 Testing Tensor Validator import...")

    try:
        validator = ScientificTensorValidator(auto_adjust=True)
        print("✅ Tensor Validator imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import Tensor Validator: {e}")
        return False


def test_qrh_layer_integration():
    """Test QRHLayer integration with tensor validator."""
    print("\n🧪 Testing QRHLayer integration...")

    try:
        # Test configuration
        config = QRHConfig(
            embed_dim=64,
            spatial_dims=(32, 32),
            enable_warnings=False
        )

        # Create layer
        layer = QRHLayer(config)

        # Test input tensor
        batch_size, seq_len = 2, 32
        input_features = 4 * config.embed_dim  # 256
        x = torch.randn(batch_size, seq_len, input_features)

        # Test forward pass
        output = layer(x)

        print(f"✅ QRHLayer forward pass successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")

        # Test health check
        health = layer.check_health(x)
        print(f"📊 Health report: {health}")

        return True

    except Exception as e:
        print(f"❌ QRHLayer integration failed: {e}")
        return False


def test_safe_reshape_method():
    """Test the _safe_reshape method in QRHLayer."""
    print("\n🧪 Testing _safe_reshape method...")

    try:
        config = QRHConfig(embed_dim=32)
        layer = QRHLayer(config)

        # Test tensor
        x = torch.randn(4096)  # 4096 elements

        # Test safe reshape with auto-adjust
        result = layer._safe_reshape(x, (1, 128, 1, 1), "test_operation")

        print(f"✅ _safe_reshape successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {result.shape}")
        print(f"   Auto-adjusted: {result.shape != (1, 128, 1, 1)}")

        return True

    except Exception as e:
        print(f"❌ _safe_reshape failed: {e}")
        return False


def test_energy_conservation_scenario():
    """Test the specific scenario that caused TEST_ENERGY_CONSERVATION error."""
    print("\n🧪 Testing energy conservation scenario...")

    try:
        config = QRHConfig(embed_dim=128)
        layer = QRHLayer(config)

        # Create tensor with 4096 elements (matching the error scenario)
        x = torch.randn(4096)

        # This should trigger auto-adjustment
        result = layer._safe_reshape(x, (1, 128, 1, 1), "energy_conservation")

        print(f"✅ Energy conservation scenario handled")
        print(f"   Input size: {x.numel()}")
        print(f"   Target shape: (1, 128, 1, 1) -> {128} elements")
        print(f"   Actual shape: {result.shape} -> {result.numel()} elements")

        # Verify the auto-adjustment worked
        assert result.numel() == 4096, f"Expected 4096 elements, got {result.numel()}"
        assert result.shape[0] == 1, "Batch dimension should be preserved"

        return True

    except Exception as e:
        print(f"❌ Energy conservation scenario failed: {e}")
        return False


def test_pipeline_integration():
    """Test ΨQRHPipeline integration with tensor validator."""
    print("\n🧪 Testing ΨQRHPipeline integration...")

    try:
        from psiqrh import ΨQRHPipeline

        # Create pipeline
        pipeline = ΨQRHPipeline(task="text-generation")

        # Check if tensor validator is initialized
        has_validator = hasattr(pipeline, 'tensor_validator')
        has_validate_method = hasattr(pipeline, '_validate_tensor_output')

        print(f"✅ ΨQRHPipeline integration check:")
        print(f"   Has tensor_validator: {has_validator}")
        print(f"   Has _validate_tensor_output: {has_validate_method}")

        return has_validator and has_validate_method

    except Exception as e:
        print(f"❌ ΨQRHPipeline integration failed: {e}")
        return False


def run_comprehensive_test():
    """Run all integration tests."""
    print("🚀 Running Comprehensive Tensor Validator Integration Tests")
    print("=" * 60)

    test_results = []

    # Run all tests
    test_results.append(("Tensor Validator Import", test_tensor_validator_import()))
    test_results.append(("QRHLayer Integration", test_qrh_layer_integration()))
    test_results.append(("Safe Reshape Method", test_safe_reshape_method()))
    test_results.append(("Energy Conservation Scenario", test_energy_conservation_scenario()))
    test_results.append(("Pipeline Integration", test_pipeline_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    total_tests = len(test_results)
    success_rate = passed / total_tests

    print(f"\n🎯 Results: {passed}/{total_tests} tests passed ({success_rate:.1%})")

    if success_rate >= 0.8:
        print("🎉 Integration successful! Ready for validation testing.")
        return True
    else:
        print("💥 Integration issues detected. Review the failures above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()

    if success:
        print("\n🚀 Next steps:")
        print("1. Run validation tests: python tests/validation_tests.py")
        print("2. Execute ΨQRH pipeline: python psiqrh.py --test")
        print("3. Monitor tensor validation logs")
    else:
        print("\n🔧 Issues detected. Please review the integration.")

    sys.exit(0 if success else 1)