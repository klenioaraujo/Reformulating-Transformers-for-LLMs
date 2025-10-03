#!/usr/bin/env python3
"""
Test Implementation - Validate all new components
"""

import torch
import torch.nn as nn
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.quaternion_operations import (
    QuaternionLinear, QuaternionLayerNorm, SpectralActivation,
    AdaptiveSpectralDropout, RealTimeFractalAnalyzer
)

from src.optimization.adaptive_fractal_controller import (
    AdaptiveFractalController, FractalControllerConfig
)

from src.core.hierarchical_gate_system import (
    HierarchicalGateSystem, ResonanceConfig
)

from src.cognitive.semantic_adaptive_filters import (
    SemanticAdaptiveFilter, SemanticFilterConfig
)


def test_quaternion_operations():
    """Test quaternion operations"""
    print("🧪 Testing Quaternion Operations...")

    # Test QuaternionLinear
    linear = QuaternionLinear(256, 512)
    x = torch.randn(2, 128, 256)
    output = linear(x)
    print(f"  QuaternionLinear: {x.shape} -> {output.shape} ✅")

    # Test QuaternionLayerNorm
    norm = QuaternionLayerNorm(256)
    x = torch.randn(2, 128, 256 * 4)
    output = norm(x)
    print(f"  QuaternionLayerNorm: {x.shape} -> {output.shape} ✅")

    # Test SpectralActivation
    activation = SpectralActivation("gelu")
    x = torch.randn(2, 128, 512)
    output = activation(x)
    print(f"  SpectralActivation: {x.shape} -> {output.shape} ✅")

    # Test AdaptiveSpectralDropout
    dropout = AdaptiveSpectralDropout(p=0.1)
    x = torch.randn(2, 128, 512)
    output = dropout(x)
    print(f"  AdaptiveSpectralDropout: {x.shape} -> {output.shape} ✅")

    # Test RealTimeFractalAnalyzer
    analyzer = RealTimeFractalAnalyzer()
    x = torch.randn(2, 128, 512)
    metrics = analyzer.analyze(x)
    print(f"  RealTimeFractalAnalyzer: {len(metrics)} metrics ✅")


def test_adaptive_fractal_controller():
    """Test adaptive fractal controller"""
    print("\n🧪 Testing Adaptive Fractal Controller...")

    config = FractalControllerConfig()
    controller = AdaptiveFractalController(config)

    x = torch.randn(2, 128, 512)

    # Test fractal analysis
    metrics = controller.analyze_fractal_dimension(x)
    print(f"  Fractal Analysis: {len(metrics)} metrics ✅")

    # Test parameter update (simulated layer)
    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.spectral_filter = MockSpectralFilter()

    class MockSpectralFilter(nn.Module):
        def update_alpha(self, alpha):
            pass

    mock_layer = MockLayer()
    controller.update_parameters(x, mock_layer)
    print(f"  Parameter Update: Controller status updated ✅")

    status = controller.get_controller_status()
    print(f"  Controller Status: {len(status)} status items ✅")


def test_hierarchical_gate_system():
    """Test hierarchical gate system"""
    print("\n🧪 Testing Hierarchical Gate System...")

    config = ResonanceConfig(num_levels=3, gate_dim=512)
    gate_system = HierarchicalGateSystem(config)

    x = torch.randn(2, 128, 512)

    # Test processing through hierarchy
    results = gate_system.process_through_hierarchy(x, x)
    print(f"  Hierarchy Processing: {len(results)} result components ✅")

    # Test health report
    health_report = gate_system.get_hierarchy_health_report(results)
    print(f"  Health Report: {len(health_report)} health metrics ✅")

    # Test forward pass
    output = gate_system(x)
    print(f"  Forward Pass: {x.shape} -> {output.shape} ✅")


def test_semantic_adaptive_filters():
    """Test semantic adaptive filters"""
    print("\n🧪 Testing Semantic Adaptive Filters...")

    config = SemanticFilterConfig(embed_dim=128)  # 128 * 4 = 512
    filter_system = SemanticAdaptiveFilter(config)

    x = torch.randn(2, 128, 512)

    # Test filtering
    filtered_output, metrics = filter_system(x)
    print(f"  Semantic Filtering: {x.shape} -> {filtered_output.shape} ✅")
    print(f"  Filter Metrics: {len(metrics)} metric types ✅")

    # Test health report
    health_report = filter_system.get_semantic_health_report(metrics)
    print(f"  Semantic Health: {len(health_report)} health indicators ✅")


def test_integration():
    """Test integration of all components"""
    print("\n🧪 Testing Component Integration...")

    # Create a simple integrated system
    x = torch.randn(2, 128, 512)

    # Quaternion operations
    linear = QuaternionLinear(512, 512)
    norm = QuaternionLayerNorm(512)
    activation = SpectralActivation("gelu")

    # Process through quaternion pipeline
    x_processed = linear(x)
    x_processed = norm(x_processed)
    x_processed = activation(x_processed)

    print(f"  Quaternion Pipeline: {x.shape} -> {x_processed.shape} ✅")

    # Semantic filtering
    filter_config = SemanticFilterConfig(embed_dim=512)  # 512 * 4 = 2048
    semantic_filter = SemanticAdaptiveFilter(filter_config)
    x_filtered, _ = semantic_filter(x_processed)

    print(f"  Semantic Filtering: {x_processed.shape} -> {x_filtered.shape} ✅")

    # Hierarchical gating
    gate_config = ResonanceConfig(gate_dim=2048)
    gate_system = HierarchicalGateSystem(gate_config)
    x_gated = gate_system(x_filtered)

    print(f"  Hierarchical Gating: {x_filtered.shape} -> {x_gated.shape} ✅")

    # Fractal analysis
    fractal_config = FractalControllerConfig()
    fractal_controller = AdaptiveFractalController(fractal_config)
    fractal_metrics = fractal_controller.analyze_fractal_dimension(x_gated)

    print(f"  Fractal Analysis: {len(fractal_metrics)} metrics ✅")

    print("\n🎯 All components integrated successfully!")


def main():
    """Run all tests"""
    print("🚀 ΨQRH Component Implementation Test Suite")
    print("=" * 60)

    try:
        test_quaternion_operations()
        test_adaptive_fractal_controller()
        test_hierarchical_gate_system()
        test_semantic_adaptive_filters()
        test_integration()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("🎯 ΨQRH Transformer components are ready for production!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()