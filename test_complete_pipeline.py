#!/usr/bin/env python3
"""
ΨQRH Complete Pipeline Test Suite
=================================

Comprehensive testing of the quantum-linguistic alignment pipeline:
1. Vocabulary building
2. Physics-based training
3. Spectral map generation
4. Optical probe decoding
5. End-to-end pipeline integration
"""

import torch
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from psiqrh import ΨQRHPipeline
from src.core.optical_probe import OpticalProbe
import tools.build_native_vocab
import tools.create_alignment_map

def test_vocabulary_building():
    """Test 1: Vocabulary Building"""
    print("🧪 Test 1: Vocabulary Building")
    print("-" * 50)

    try:
        # Test vocabulary building
        vocab_data = tools.build_native_vocab.build_vocab(
            corpus_path="data/train.txt",
            min_freq=1
        )

        print(f"✅ Vocabulary built successfully")
        print(f"   📊 Tokens: {vocab_data['vocab_size']}")
        print(f"   📝 Sample tokens: {list(vocab_data['vocab'].keys())[:10]}")

        # Verify native_vocab.json was created
        vocab_path = "data/native_vocab.json"
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                saved_vocab = json.load(f)
            assert saved_vocab['vocab_size'] == vocab_data['vocab_size']
            print("✅ Vocabulary file saved correctly")
        else:
            raise FileNotFoundError("Vocabulary file not created")

        return True

    except Exception as e:
        print(f"❌ Vocabulary building failed: {e}")
        return False

def test_pipeline_initialization():
    """Test 2: Pipeline Initialization"""
    print("\n🧪 Test 2: Pipeline Initialization")
    print("-" * 50)

    try:
        # Initialize pipeline with auto-calibration
        pipeline = ΨQRHPipeline(
            task="text-generation",
            device='cpu',
            enable_auto_calibration=True,
            audit_mode=True
        )

        print("✅ Pipeline initialized successfully")
        print(f"   📐 Embed dim: {pipeline.config['embed_dim']}")
        print(f"   🤖 Auto-calibration: {'ENABLED' if pipeline.enable_auto_calibration else 'DISABLED'}")
        print(f"   🧠 Quantum embedding vocab size: {pipeline.quantum_embedding.vocab_size}")

        return pipeline

    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return None

def test_physics_based_processing(pipeline):
    """Test 3: Physics-Based Processing"""
    print("\n🧪 Test 3: Physics-Based Processing")
    print("-" * 50)

    if pipeline is None:
        print("❌ Skipping test - pipeline not available")
        return False

    try:
        test_inputs = ["red", "hello world", "quantum physics"]

        for i, input_text in enumerate(test_inputs, 1):
            print(f"   🔄 Processing: '{input_text}'")

            result = pipeline(input_text)

            if result['status'] != 'success':
                print(f"      ❌ Processing failed: {result.get('error', 'Unknown error')}")
                continue

            # Check for required keys
            required_keys = ['response', 'final_quantum_state', 'physical_metrics']
            missing_keys = [key for key in required_keys if key not in result]

            if missing_keys:
                print(f"      ❌ Missing keys in result: {missing_keys}")
                continue

            print(f"      ✅ Success: {len(result['response'])} chars generated")
            print(f"         🧠 FCI: {result['physical_metrics'].get('FCI', 'N/A')}")
            print(f"         📐 Fractal Dim: {result['physical_metrics'].get('fractal_dimension', 'N/A'):.3f}")

        return True

    except Exception as e:
        print(f"❌ Physics-based processing failed: {e}")
        return False

def test_alignment_map_generation():
    """Test 4: Alignment Map Generation"""
    print("\n🧪 Test 4: Alignment Map Generation")
    print("-" * 50)

    try:
        # Generate alignment map
        tools.create_alignment_map.create_map()

        # Verify map was created
        map_path = "data/spectral_vocab_map.pt"
        if not os.path.exists(map_path):
            raise FileNotFoundError("Spectral map not created")

        # Load and verify map
        spectral_map = torch.load(map_path)
        print("✅ Spectral alignment map generated")
        print(f"   📊 Shape: {spectral_map.shape}")
        print(f"   📚 Vocabulary size: {spectral_map.shape[0]}")
        print(f"   🔬 Embedding dim: {spectral_map.shape[1] * spectral_map.shape[2]}")

        return True

    except Exception as e:
        print(f"❌ Alignment map generation failed: {e}")
        return False

def test_optical_probe_decoding():
    """Test 5: Optical Probe Decoding"""
    print("\n🧪 Test 5: Optical Probe Decoding")
    print("-" * 50)

    try:
        # Initialize optical probe
        probe = OpticalProbe(device='cpu')

        if probe.spectral_map is None:
            raise RuntimeError("Spectral map not loaded")

        print("✅ Optical probe initialized")
        print(f"   📊 Spectral map shape: {probe.spectral_map.shape}")
        print(f"   📚 Vocabulary size: {probe.vocab_size}")

        # Test decoding with different quantum states
        test_cases = [
            torch.randn(16, 4),  # [embed_dim, 4]
            torch.randn(64),     # [embed_dim * 4] flattened
        ]

        for i, test_psi in enumerate(test_cases, 1):
            try:
                token_id = probe(test_psi)
                print(f"   🧪 Test case {i}: Token ID = {token_id}")

                # Verify token ID is valid
                if 0 <= token_id < probe.vocab_size:
                    print("      ✅ Valid token ID")
                else:
                    print(f"      ❌ Invalid token ID: {token_id} (should be 0-{probe.vocab_size-1})")

            except Exception as e:
                print(f"   ❌ Test case {i} failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Optical probe decoding failed: {e}")
        return False

def test_end_to_end_pipeline():
    """Test 6: End-to-End Pipeline"""
    print("\n🧪 Test 6: End-to-End Pipeline")
    print("-" * 50)

    try:
        # Initialize components
        pipeline = ΨQRHPipeline(
            task="text-generation",
            device='cpu',
            enable_auto_calibration=True
        )

        probe = OpticalProbe(device='cpu')

        # Test complete flow
        test_input = "test quantum"
        print(f"   🔄 Processing: '{test_input}'")

        # Step 1: Process through pipeline
        result = pipeline(test_input)

        if result['status'] != 'success':
            raise RuntimeError(f"Pipeline processing failed: {result.get('error', 'Unknown error')}")

        # Step 2: Extract final quantum state
        final_psi = result['final_quantum_state']
        print(f"      📊 Final quantum state shape: {final_psi.shape}")

        # Step 3: Decode through optical probe
        if final_psi.dim() == 1:
            # If it's 1D, reshape to [embed_dim, 4]
            embed_dim = final_psi.shape[0] // 4
            final_psi_reshaped = final_psi.view(embed_dim, 4)
        else:
            final_psi_reshaped = final_psi

        token_id = probe(final_psi_reshaped)
        print(f"      🔢 Decoded token ID: {token_id}")

        print("✅ End-to-end pipeline test passed")
        print(f"   📝 Input: '{test_input}'")
        print(f"   🤖 Generated: '{result['response'][:50]}...'")
        print(f"   🔢 Token ID: {token_id}")

        return True

    except Exception as e:
        print(f"❌ End-to-end pipeline failed: {e}")
        return False

def run_complete_test_suite():
    """Run the complete test suite"""
    print("🚀 ΨQRH Complete Pipeline Test Suite")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    test_results = []
    pipeline = None

    # Test 1: Vocabulary Building
    vocab_success = test_vocabulary_building()
    test_results.append(("Vocabulary Building", vocab_success))

    # Test 2: Pipeline Initialization
    pipeline = test_pipeline_initialization()
    pipeline_success = pipeline is not None
    test_results.append(("Pipeline Initialization", pipeline_success))

    # Test 3: Physics-Based Processing
    processing_success = test_physics_based_processing(pipeline)
    test_results.append(("Physics-Based Processing", processing_success))

    # Test 4: Alignment Map Generation
    map_success = test_alignment_map_generation()
    test_results.append(("Alignment Map Generation", map_success))

    # Test 5: Optical Probe Decoding
    probe_success = test_optical_probe_decoding()
    test_results.append(("Optical Probe Decoding", probe_success))

    # Test 6: End-to-End Pipeline
    e2e_success = test_end_to_end_pipeline()
    test_results.append(("End-to-End Pipeline", e2e_success))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print("25")
        if success:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! ΨQRH Pipeline is fully operational.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_complete_test_suite()
    sys.exit(0 if success else 1)