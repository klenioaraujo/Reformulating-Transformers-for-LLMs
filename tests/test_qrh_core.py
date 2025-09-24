#!/usr/bin/env python3
"""
SIMPLE QRH CORE TEST

Direct test of the QRH system without complex production wrappers
to isolate the ScriptMethodStub issue
"""

import torch
import torch.nn as nn

def test_qrh_core():
    """Test core QRH components step by step"""
    print("🧪 Testing QRH Core Components")
    print("=" * 50)

    try:
        # Test 1: Import core QRH
        print("1️⃣ Testing QRH import...")
        from qrh_layer import QRHLayer, QRHConfig
        print("✅ QRH Layer imported successfully")

        # Test 2: Create QRH config
        print("2️⃣ Creating QRH configuration...")
        config = QRHConfig(embed_dim=32, alpha=1.2, use_learned_rotation=True)
        print(f"✅ QRH Config created: {config.embed_dim}D")

        # Test 3: Initialize QRH layer
        print("3️⃣ Initializing QRH layer...")
        qrh_layer = QRHLayer(config)
        print("✅ QRH Layer initialized")

        # Test 4: Create test input
        print("4️⃣ Creating test input...")
        batch_size = 2
        seq_len = 16
        embed_dim = config.embed_dim * 4  # QRH expects 4x embed_dim
        test_input = torch.randn(batch_size, seq_len, embed_dim)
        print(f"✅ Test input created: {test_input.shape}")

        # Test 5: Forward pass
        print("5️⃣ Testing forward pass...")
        with torch.no_grad():
            output = qrh_layer(test_input)
        print(f"✅ Forward pass successful: {output.shape}")

        # Test 6: Test semantic adaptive filters
        print("6️⃣ Testing semantic adaptive filters...")
        from semantic_adaptive_filters import ContradictionDetector, SemanticFilterConfig

        semantic_config = SemanticFilterConfig(embed_dim=config.embed_dim)
        contradiction_detector = ContradictionDetector(semantic_config)

        with torch.no_grad():
            contradictions = contradiction_detector.detect_contradictions(output)
        # Handle tuple return from detect_contradictions
        if isinstance(contradictions, tuple):
            contradictions = contradictions[0]
        print(f"✅ Semantic filters working: {contradictions.shape}")

        return True

    except Exception as e:
        print(f"❌ Test failed at step: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimized_components():
    """Test optimized components individually"""
    print("\n🚀 Testing Optimized Components")
    print("=" * 50)

    try:
        # Test optimized imports
        print("1️⃣ Testing optimized imports...")
        from optimized_components import (
            OptimizedSemanticConfig, OptimizedContinuumConfig,
            OptimizedResonanceConfig, FastContradictionDetector
        )
        print("✅ Optimized components imported")

        # Test configuration
        print("2️⃣ Creating optimized configurations...")
        semantic_config = OptimizedSemanticConfig(embed_dim=32)
        print("✅ Semantic config created")

        # Test fast contradiction detector with neurotransmitter alignment
        print("3️⃣ Testing fast contradiction detector with neurotransmitter alignment...")
        from synthetic_neurotransmitters import create_aligned_qrh_component, NeurotransmitterConfig

        detector = FastContradictionDetector(semantic_config)
        nt_config = NeurotransmitterConfig(embed_dim=semantic_config.embed_dim)
        aligned_detector = create_aligned_qrh_component(detector, nt_config)

        # Create test input with correct dimensions (4x embed_dim for QRH compatibility)
        test_input = torch.randn(2, 16, semantic_config.embed_dim * 4)

        with torch.no_grad():
            output = aligned_detector(test_input)
            # Handle different return types from neurotransmitter alignment
            if isinstance(output, tuple):
                scores = output[0] if len(output) > 0 else output
                analysis = output[1] if len(output) > 1 else None
            else:
                scores = output
                analysis = None

        print(f"✅ Fast detector with neurotransmitter alignment working: {scores.shape if isinstance(scores, torch.Tensor) else type(scores)}")

        return True

    except Exception as e:
        print(f"❌ Optimized components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_production_system():
    """Test production system creation"""
    print("\n🏭 Testing Production System")
    print("=" * 50)

    try:
        print("1️⃣ Testing production imports...")
        from production_system import ProductionSemanticQRH, ProductionConfig, ProductionMode
        print("✅ Production system imported")

        print("2️⃣ Creating production config...")
        config = ProductionConfig(
            mode=ProductionMode.BALANCED,
            embed_dim=24,
            enable_jit_compilation=False,  # Disable JIT for this test
            enable_health_checks=False     # Disable health checks for simplicity
        )
        print("✅ Production config created")

        print("3️⃣ Initializing production system...")
        system = ProductionSemanticQRH(config)
        print("✅ Production system initialized")

        return True

    except Exception as e:
        print(f"❌ Production system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 QRH System Core Testing Suite")
    print("Isolating components to find ScriptMethodStub issue\n")

    # Run tests step by step
    core_ok = test_qrh_core()
    optimized_ok = test_optimized_components()
    production_ok = test_production_system()

    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    print(f"QRH Core:           {'✅ PASS' if core_ok else '❌ FAIL'}")
    print(f"Optimized Components: {'✅ PASS' if optimized_ok else '❌ FAIL'}")
    print(f"Production System:   {'✅ PASS' if production_ok else '❌ FAIL'}")

    if all([core_ok, optimized_ok, production_ok]):
        print("\n🎉 All core components working! Issue likely in test framework.")
    else:
        print("\n⚠️ Issue identified in core components.")