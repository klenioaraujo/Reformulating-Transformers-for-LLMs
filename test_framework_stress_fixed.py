#!/usr/bin/env python3
"""
Fixed comprehensive stress test for the entire ΨQRH framework
Corrects tensor dimension coordination issues
"""

import torch
import time
import gc
import numpy as np
from qrh_layer import QRHLayer, QRHConfig
from negentropy_transformer_block import NegentropyTransformerBlock
from starfleet_glyph_system import StarfleetBridgeSimulator


def stress_test_memory_efficiency():
    """Test memory efficiency under large tensor loads - FIXED"""
    print("🧠 MEMORY EFFICIENCY STRESS TEST")
    print("-" * 50)

    # Test progressively larger tensors with CORRECT QRH dimensions
    test_configs = [
        # (batch, seq_len, qrh_embed_dim, description)
        (2, 32, 16, "Small"),
        (4, 64, 24, "Medium"),
        (8, 128, 32, "Large"),
        (16, 256, 48, "Extra Large")
    ]

    results = []

    for batch, seq_len, qrh_embed_dim, description in test_configs:
        # QRH layer expects input of size 4 * embed_dim
        input_dim = 4 * qrh_embed_dim
        print(f"\n   Testing {description}: {batch}x{seq_len}x{input_dim} (QRH embed_dim={qrh_embed_dim})")

        try:
            # Create QRH layer with all features enabled
            config = QRHConfig(
                embed_dim=qrh_embed_dim,
                alpha=1.0,
                normalization_type='layer_norm',
                spectral_dropout_rate=0.2
            )
            layer = QRHLayer(config)

            # Create input tensor with CORRECT dimensions
            x = torch.randn(batch, seq_len, input_dim)

            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated()
            else:
                start_memory = 0

            # Forward pass
            start_time = time.time()
            output = layer(x)
            duration = time.time() - start_time

            # Measure memory after
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
            else:
                memory_used = 0

            # Validate output
            shape_ok = x.shape == output.shape
            no_nan = not torch.isnan(output).any()
            no_inf = not torch.isinf(output).any()

            result = {
                'config': description,
                'size': f"{batch}x{seq_len}x{input_dim}",
                'duration_ms': duration * 1000,
                'memory_mb': memory_used,
                'shape_ok': shape_ok,
                'no_nan': no_nan,
                'no_inf': no_inf,
                'success': shape_ok and no_nan and no_inf
            }

            results.append(result)

            print(f"     Duration: {duration*1000:.2f}ms")
            print(f"     Memory: {memory_used:.2f}MB")
            print(f"     Success: {result['success']}")

            # Clean up
            del layer, x, output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"     ❌ Failed: {e}")
            results.append({
                'config': description,
                'size': f"{batch}x{seq_len}x{input_dim}",
                'success': False,
                'error': str(e)
            })

    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n   Memory stress test: {successful}/{len(results)} passed")

    return successful == len(results)


def stress_test_deep_stacking():
    """Test deep stacking of Negentropy Transformers"""
    print("\n🏗️ DEEP STACKING STRESS TEST")
    print("-" * 50)

    layer_counts = [1, 2, 4, 8, 16]
    results = []

    for num_layers in layer_counts:
        print(f"\n   Testing {num_layers} layers...")

        try:
            # Create stack of transformers with layer scaling
            layers = []
            for i in range(num_layers):
                transformer = NegentropyTransformerBlock(
                    d_model=64,
                    nhead=2,
                    dim_feedforward=128,
                    qrh_embed_dim=16,
                    init_layer_scale=1e-4,  # Critical for deep networks
                    qrh_normalization_type='layer_norm'
                )
                layers.append(transformer)

            # Test forward pass
            x = torch.randn(2, 8, 64)

            start_time = time.time()
            seals = []

            for i, layer in enumerate(layers):
                x, seal = layer(x)
                seals.append(seal)

                # Check for instability
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"     ❌ Instability at layer {i+1}")
                    break
            else:
                # All layers completed successfully
                duration = time.time() - start_time
                final_norm = torch.norm(x).item()

                print(f"     Duration: {duration*1000:.2f}ms")
                print(f"     Final norm: {final_norm:.6f}")
                print(f"     RG values: {[s.get('RG', 0) for s in seals[-3:]]}")
                print(f"     ✅ Stable: All {num_layers} layers")

                results.append({
                    'layers': num_layers,
                    'duration_ms': duration * 1000,
                    'final_norm': final_norm,
                    'success': True
                })

        except Exception as e:
            print(f"     ❌ Failed: {e}")
            results.append({
                'layers': num_layers,
                'success': False,
                'error': str(e)
            })

    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n   Deep stacking test: {successful}/{len(results)} passed")

    return successful == len(results)


def stress_test_configuration_combinations():
    """Test all possible configuration combinations"""
    print("\n🔧 CONFIGURATION COMBINATION STRESS TEST")
    print("-" * 50)

    # Test matrix of all feature combinations
    configurations = [
        # (normalization, spectral_dropout, layer_scale, description)
        (None, 0.0, 1.0, "Baseline"),
        ('layer_norm', 0.0, 1.0, "LayerNorm only"),
        ('unit_projection', 0.0, 1.0, "UnitProjection only"),
        (None, 0.3, 1.0, "SpectralDropout only"),
        (None, 0.0, 1e-4, "LayerScale only"),
        ('layer_norm', 0.2, 1e-4, "LayerNorm + SpectralDropout + LayerScale"),
        ('unit_projection', 0.3, 1e-5, "UnitProjection + SpectralDropout + LayerScale"),
        ('layer_norm', 0.5, 1e-3, "Extreme configuration")
    ]

    results = []

    for norm_type, dropout_rate, layer_scale, description in configurations:
        print(f"\n   Testing: {description}")

        try:
            # Create Negentropy Transformer with specific config
            transformer = NegentropyTransformerBlock(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                qrh_embed_dim=32,
                qrh_normalization_type=norm_type,
                init_layer_scale=layer_scale
            )

            # Update QRH layer for spectral dropout if needed
            if dropout_rate > 0:
                transformer.qrh_layer.config.spectral_dropout_rate = dropout_rate

            # Test forward pass
            x = torch.randn(2, 16, 128)
            output, seal = transformer(x)

            # Validate
            shape_ok = x.shape == output.shape
            no_nan = not torch.isnan(output).any()
            no_inf = not torch.isinf(output).any()
            seal_valid = seal.get('RG', 0) > 0

            success = shape_ok and no_nan and no_inf and seal_valid

            results.append({
                'config': description,
                'shape_ok': shape_ok,
                'no_nan': no_nan,
                'no_inf': no_inf,
                'seal_valid': seal_valid,
                'success': success
            })

            print(f"     Shape: {shape_ok}, NaN: {no_nan}, Inf: {no_inf}, Seal: {seal_valid}")
            print(f"     RG: {seal.get('RG', 0):.3f}")
            print(f"     Success: {success}")

        except Exception as e:
            print(f"     ❌ Failed: {e}")
            results.append({
                'config': description,
                'success': False,
                'error': str(e)
            })

    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n   Configuration test: {successful}/{len(results)} passed")

    return successful == len(results)


def stress_test_starfleet_integration():
    """Test Starfleet Glyph system under stress"""
    print("\n🖖 STARFLEET INTEGRATION STRESS TEST")
    print("-" * 50)

    bridge = StarfleetBridgeSimulator("USS Enterprise", "NCC-1701-D")

    # Test rapid mission switching
    missions = [
        "Emergency quantum containment protocol",
        "Diplomatic first contact verification",
        "Temporal paradox resolution analysis",
        "Subspace communication optimization",
        "Warp core stability assessment"
    ]

    formations = ["integrity_fusion", "protector_catalyst", "orient_pace"]

    print(f"   Testing {len(missions)} rapid mission switches...")

    start_time = time.time()
    successful_missions = 0

    for i, mission in enumerate(missions):
        try:
            formation = formations[i % len(formations)]

            bridge.set_mission(mission)
            bridge.activate_formation(formation)
            receipt = bridge.execute_cognitive_pass()

            # Validate receipt
            if (receipt.get('status') == 'OPERATIONAL' and
                receipt.get('temporal_seal') == 'Ω∞Ω' and
                receipt.get('rg_value', 0) > 0):
                successful_missions += 1

        except Exception as e:
            print(f"     ❌ Mission {i+1} failed: {e}")

    duration = time.time() - start_time
    missions_per_second = len(missions) / duration if duration > 0 else float('inf')

    print(f"   Completed: {successful_missions}/{len(missions)} missions")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Throughput: {missions_per_second:.1f} missions/second")

    return successful_missions == len(missions)


def stress_test_concurrent_operations():
    """Test concurrent operations and resource sharing - FIXED"""
    print("\n⚡ CONCURRENT OPERATIONS STRESS TEST")
    print("-" * 50)

    # Simulate multiple concurrent QRH operations with COORDINATED dimensions
    num_concurrent = 5
    operations = []

    print(f"   Testing {num_concurrent} concurrent QRH operations...")

    try:
        # Create multiple QRH layers with COORDINATED configs
        qrh_embed_dims = [16, 20, 24, 28, 32]  # Different QRH embed dims

        for i in range(num_concurrent):
            qrh_embed_dim = qrh_embed_dims[i]
            config = QRHConfig(
                embed_dim=qrh_embed_dim,
                alpha=1.0 + i*0.2,
                normalization_type=['layer_norm', 'unit_projection', None][i % 3],
                spectral_dropout_rate=i * 0.1
            )
            layer = QRHLayer(config)
            operations.append((layer, qrh_embed_dim))

        # Create test tensors with CORRECT dimensions (4 * embed_dim)
        tensors = []
        for i, (layer, qrh_embed_dim) in enumerate(operations):
            input_dim = 4 * qrh_embed_dim  # Correct input dimension
            tensor = torch.randn(2, 8, input_dim)
            tensors.append(tensor)

        # Process concurrently (simulated)
        start_time = time.time()
        outputs = []

        for i, ((layer, qrh_embed_dim), tensor) in enumerate(zip(operations, tensors)):
            print(f"     Processing operation {i+1}: QRH_dim={qrh_embed_dim}, input_shape={tensor.shape}")
            output = layer(tensor)
            outputs.append(output)

        duration = time.time() - start_time

        # Validate all outputs
        all_valid = True
        for i, (input_tensor, output) in enumerate(zip(tensors, outputs)):
            if (input_tensor.shape != output.shape or
                torch.isnan(output).any() or
                torch.isinf(output).any()):
                all_valid = False
                print(f"     ❌ Operation {i+1} invalid")
            else:
                print(f"     ✅ Operation {i+1} valid: {output.shape}")

        print(f"   Duration: {duration*1000:.2f}ms")
        print(f"   All operations valid: {all_valid}")

        return all_valid

    except Exception as e:
        print(f"     ❌ Concurrent operations failed: {e}")
        return False


def run_comprehensive_stress_tests():
    """Run all stress tests - FIXED VERSION"""
    print("💪 COMPREHENSIVE ΨQRH FRAMEWORK STRESS TESTS (FIXED)")
    print("=" * 70)

    tests = [
        ("Memory Efficiency", stress_test_memory_efficiency),
        ("Deep Stacking", stress_test_deep_stacking),
        ("Configuration Combinations", stress_test_configuration_combinations),
        ("Starfleet Integration", stress_test_starfleet_integration),
        ("Concurrent Operations", stress_test_concurrent_operations)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name.upper()} {'='*20}")
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name}: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Final summary
    print(f"\n{'='*20} STRESS TEST SUMMARY {'='*20}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print(f"\nStress tests passed: {passed}/{total}")
    print(f"Framework resilience: {(passed/total)*100:.1f}%")

    if passed == total:
        print("\n🎉 FRAMEWORK STRESS TESTING: 100% SUCCESSFUL!")
        print("   The ΨQRH framework demonstrates perfect resilience")
        print("   under extreme conditions and heavy loads.")
        print("\n🚀 READY FOR PRODUCTION DEPLOYMENT!")
    else:
        print("\n⚠️  FRAMEWORK NEEDS OPTIMIZATION")
        print("   Some stress tests failed. Review system stability.")

    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_stress_tests()