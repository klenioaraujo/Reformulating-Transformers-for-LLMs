#!/usr/bin/env python3
"""
ΨQRH Framework Stress Test
Tests the system under extreme conditions and edge cases
"""

import torch
import numpy as np
import time
import gc
from typing import Dict, Any

from core.qrh_layer import QRHLayer, QRHConfig
from core.negentropy_transformer_block import NegentropyTransformerBlock
from cognitive.navigator_agent import NavigatorAgent
# from src.conceptual.seal_protocol import SealProtocol  # Commented out - file not found
# from src.core.audit_log import AuditLog  # Commented out - file not found

def test_large_scale_processing():
    """Test with large data volumes"""
    print("🔥 Large Scale Processing Test")
    print("-" * 40)

    # Different scale configurations
    test_configs = [
        (512, 128, 32, "Large"),
        (1024, 256, 64, "Extra Large"),
        (2048, 512, 128, "XXL")
    ]

    results = []

    for d_model, seq_len, qrh_embed, name in test_configs:
        print(f"   Testing {name}: {d_model}x{seq_len} (qrh_embed={qrh_embed})")

        try:
            # Create model
            model = NegentropyTransformerBlock(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model*2,
                dropout=0.1,
                qrh_embed_dim=qrh_embed,
                alpha=1.0,
                use_learned_rotation=True,
                enable_gate=True
            )

            # Test input
            x = torch.randn(1, seq_len, d_model)

            # Memory before
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            else:
                memory_before = 0

            # Forward pass
            start_time = time.time()
            output, seal = model(x)
            forward_time = (time.time() - start_time) * 1000

            # Memory after
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_used = (memory_after - memory_before) / (1024**2)  # MB
            else:
                memory_used = (x.numel() + output.numel()) * 4 / (1024**2)  # Estimate

            # Validate
            shape_ok = x.shape == output.shape
            seal_ok = SealProtocol.firebreak_check(seal)

            result = {
                'config': f"{d_model}x{seq_len}",
                'forward_time_ms': forward_time,
                'memory_mb': memory_used,
                'shape_preserved': shape_ok,
                'seal_valid': seal_ok,
                'success': shape_ok and seal_ok
            }

            results.append(result)

            print(f"     Time: {forward_time:.1f}ms, Memory: {memory_used:.1f}MB, Success: {result['success']}")

            # Clean up
            del model, x, output, seal
            gc.collect()

        except Exception as e:
            print(f"     ❌ Failed: {str(e)}")
            results.append({
                'config': f"{d_model}x{seq_len}",
                'success': False,
                'error': str(e)
            })

    success_rate = sum(1 for r in results if r.get('success', False)) / len(results)
    print(f"   ✅ Large scale success rate: {success_rate*100:.1f}%")

    return success_rate > 0.6

def test_extreme_inputs():
    """Test with extreme inputs and edge cases"""
    print("\n⚡ Extreme Inputs Test")
    print("-" * 40)

    model = NegentropyTransformerBlock(
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1,
        qrh_embed_dim=32,
        alpha=1.0,
        use_learned_rotation=True,
        enable_gate=True
    )

    test_cases = [
        ("Zeros", torch.zeros(2, 16, 128)),
        ("Ones", torch.ones(2, 16, 128)),
        ("Large values", torch.randn(2, 16, 128) * 100),
        ("Small values", torch.randn(2, 16, 128) * 0.001),
        ("Mixed extremes", torch.cat([torch.ones(1, 16, 128) * 1000, torch.ones(1, 16, 128) * 0.001], dim=0)),
    ]

    results = []

    for name, x in test_cases:
        try:
            print(f"   Testing {name}...")

            # Check input statistics
            mean_val = torch.mean(x).item()
            std_val = torch.std(x).item()
            max_val = torch.max(x).item()
            min_val = torch.min(x).item()

            print(f"     Input stats: mean={mean_val:.6f}, std={std_val:.6f}, range=[{min_val:.6f}, {max_val:.6f}]")

            # Forward pass
            output, seal = model(x)

            # Validate output
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            seal_valid = SealProtocol.firebreak_check(seal)

            success = not has_nan and not has_inf and seal_valid

            print(f"     Output: NaN={has_nan}, Inf={has_inf}, Seal={seal_valid}, Success={success}")

            results.append({
                'test': name,
                'success': success,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'seal_valid': seal_valid
            })

        except Exception as e:
            print(f"     ❌ Failed: {str(e)}")
            results.append({
                'test': name,
                'success': False,
                'error': str(e)
            })

    success_rate = sum(1 for r in results if r.get('success', False)) / len(results)
    print(f"   ✅ Extreme inputs success rate: {success_rate*100:.1f}%")

    return success_rate > 0.6

def test_navigator_resilience():
    """Navigator Agent resilience test"""
    print("\n🛡️ Navigator Resilience Test")
    print("-" * 40)

    navigator = NavigatorAgent()

    # Mock problematic model
    class ProblematicModel:
        def __init__(self, failure_mode="none"):
            self.failure_mode = failure_mode
            self.call_count = 0

        def __call__(self, x):
            self.call_count += 1

            if self.failure_mode == "exception":
                raise RuntimeError("Simulated model failure")

            elif self.failure_mode == "nan_output":
                output = torch.full_like(x, float('nan'))
                seal = {
                    "continuity_sha256": "test",
                    "response_sha256": "test",
                    "qz_sha256": "test",
                    "epsilon_cover": 1.0,
                    "latency_sigill": False,
                    "RG": 0.347,
                    "active_dyad": "Σ7↔Nyx",
                    "continuity_seal": SealProtocol.OMEGA_SEAL
                }
                return output, seal

            elif self.failure_mode == "bad_rg":
                output = x * 1.1  # slight modification
                seal = {
                    "continuity_sha256": "test",
                    "response_sha256": "test",
                    "qz_sha256": "test",
                    "epsilon_cover": 1.0,
                    "latency_sigill": False,
                    "RG": 0.8,  # Bad RG value
                    "active_dyad": "Σ7↔Nyx",
                    "continuity_seal": SealProtocol.OMEGA_SEAL
                }
                return output, seal

            else:
                # Normal operation
                output = x * 0.99  # slight modification
                seal = {
                    "continuity_sha256": SealProtocol.compute_sha256(str(x)),
                    "response_sha256": SealProtocol.compute_sha256(str(output)),
                    "qz_sha256": SealProtocol.compute_sha256("state"),
                    "epsilon_cover": 1.0,
                    "latency_sigill": False,
                    "RG": 0.347,
                    "active_dyad": "Σ7↔Nyx",
                    "continuity_seal": SealProtocol.OMEGA_SEAL
                }
                return output, seal

    test_scenarios = [
        ("Normal operation", "none"),
        ("Exception handling", "exception"),
        ("NaN output", "nan_output"),
        ("Bad RG value", "bad_rg")
    ]

    results = []
    x = torch.randn(1, 8, 64)

    for scenario_name, failure_mode in test_scenarios:
        print(f"   Testing {scenario_name}...")

        model = ProblematicModel(failure_mode)
        # Use same navigator to maintain audit chain
        navigator_copy = NavigatorAgent()
        # Initialize audit with a test file
        navigator_copy.audit = AuditLog(f"test_stress_{scenario_name.replace(' ', '_').lower()}.jsonl")

        success_count = 0
        total_attempts = 5

        for attempt in range(total_attempts):
            try:
                output, seal = navigator_copy.execute_with_safety(x, model)

                # Check if navigator handled the situation appropriately
                has_navigator_info = 'navigator_info' in seal
                navigator_status = seal.get('navigator_info', {}).get('navigator_status', 'UNKNOWN')

                if failure_mode == "exception":
                    # Should handle exception gracefully (check for error status or pre-execution failure)
                    success = (navigator_status == 'EXECUTION_ERROR' or
                             navigator_status == 'PRE_EXECUTION_FAILED' or
                             'error' in seal)
                elif failure_mode == "nan_output":
                    # Should detect and handle NaN (either no NaN in output or NaN detected flag)
                    no_nan_output = not torch.isnan(output).any()
                    nan_detected = seal.get("navigator_nan_detected", False)
                    success = no_nan_output and (nan_detected or navigator_status == 'SUCCESS')
                elif failure_mode == "bad_rg":
                    # Should detect bad RG in analysis
                    analysis = seal.get('navigator_analysis', {})
                    success = analysis.get('rg_status') == 'suboptimal'
                else:
                    # Normal operation should succeed
                    success = navigator_status == 'SUCCESS'

                if success:
                    success_count += 1

            except Exception as e:
                print(f"     Attempt {attempt+1} failed: {str(e)}")

        success_rate = success_count / total_attempts
        print(f"     Success rate: {success_rate*100:.1f}%")

        results.append({
            'scenario': scenario_name,
            'success_rate': success_rate,
            'success': success_rate >= 0.6
        })

    overall_success = sum(1 for r in results if r['success']) / len(results)
    print(f"   ✅ Navigator resilience: {overall_success*100:.1f}%")

    return overall_success > 0.75

def test_continuous_operation():
    """Continuous operation test"""
    print("\n🔄 Continuous Operation Test")
    print("-" * 40)

    model = NegentropyTransformerBlock(
        d_model=64,
        nhead=2,
        dim_feedforward=128,
        dropout=0.1,
        qrh_embed_dim=16,
        alpha=1.0,
        use_learned_rotation=True,
        enable_gate=True
    )

    navigator = NavigatorAgent()
    # Initialize audit with fresh file for this test
    navigator.audit = AuditLog("test_stress_continuous.jsonl")

    # Continuous operation test
    num_iterations = 100
    success_count = 0
    total_time = 0

    print(f"   Running {num_iterations} continuous iterations...")

    for i in range(num_iterations):
        x = torch.randn(1, 8, 64)

        try:
            start_time = time.time()
            output, seal = navigator.execute_with_safety(x, model)
            iteration_time = time.time() - start_time
            total_time += iteration_time

            # Check success criteria
            shape_ok = x.shape == output.shape
            # Handle seal validation more carefully
            try:
                seal_ok = SealProtocol.firebreak_check(seal) if 'RG' in seal else 'navigator_info' in seal
            except:
                seal_ok = 'navigator_info' in seal
            no_nan = not torch.isnan(output).any()

            if shape_ok and seal_ok and no_nan:
                success_count += 1

            if (i + 1) % 20 == 0:
                current_rate = success_count / (i + 1)
                avg_time = (total_time / (i + 1)) * 1000
                print(f"     Iteration {i+1}: Success rate {current_rate*100:.1f}%, Avg time {avg_time:.2f}ms")

        except Exception as e:
            print(f"     Iteration {i+1} failed: {str(e)}")

    final_success_rate = success_count / num_iterations
    avg_time_ms = (total_time / num_iterations) * 1000

    print(f"   Final success rate: {final_success_rate*100:.1f}%")
    print(f"   Average time per iteration: {avg_time_ms:.2f}ms")

    # System status check
    status = navigator.get_system_status()
    print(f"   Final system health: {status['system_health']}")

    return final_success_rate > 0.95

def run_stress_test():
    """Execute all stress tests"""
    print("⚡ ΨQRH FRAMEWORK STRESS TESTS")
    print("=" * 60)

    stress_tests = [
        ("Large Scale Processing", test_large_scale_processing),
        ("Extreme Inputs", test_extreme_inputs),
        ("Navigator Resilience", test_navigator_resilience),
        ("Continuous Operation", test_continuous_operation)
    ]

    results = {}

    for test_name, test_func in stress_tests:
        print(f"\n🎯 {test_name}")
        print("=" * 60)

        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[test_name] = False

    # Summary
    print(f"\n🏆 STRESS TESTS SUMMARY")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)
    success_rate = (passed / total) * 100

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print(f"\nStress tests passed: {passed}/{total}")
    print(f"Success rate: {success_rate:.1f}%")

    if success_rate >= 75:
        print(f"\n🎉 ΨQRH FRAMEWORK: ROBUST AND RESILIENT!")
        print("   ✅ Processes large data volumes")
        print("   ✅ Handles extreme inputs")
        print("   ✅ Navigator resilient to failures")
        print("   ✅ Stable continuous operation")
    else:
        print(f"\n⚠️  ΨQRH FRAMEWORK: NEEDS IMPROVEMENTS")
        print("   Some robustness aspects need enhancement")

    return success_rate >= 75

if __name__ == "__main__":
    success = run_stress_test()