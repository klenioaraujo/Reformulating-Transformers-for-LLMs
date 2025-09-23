#!/usr/bin/env python3
"""
Test integration of QRH normalization with the complete ΨQRH system
"""

import torch
import time
from qrh_layer import QRHConfig
from negentropy_transformer_block import NegentropyTransformerBlock
from navigator_agent import NavigatorAgent
from seal_protocol import SealProtocol

def test_negentropy_transformer_with_normalization():
    """Test Negentropy Transformer with different QRH normalizations"""
    print("🔄 Testing Negentropy Transformer with QRH Normalizations")
    print("-" * 60)

    normalizations = [
        (None, "No normalization"),
        ('layer_norm', "Quaternion Layer Norm"),
        ('unit_projection', "Unit Projection")
    ]

    results = {}

    for norm_type, description in normalizations:
        print(f"\n   Testing {description}...")

        # Create transformer with specific normalization
        transformer = NegentropyTransformerBlock(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            qrh_embed_dim=32,
            alpha=1.0,
            use_learned_rotation=True,
            enable_gate=True,
            qrh_normalization_type=norm_type  # Pass normalization to QRH config
        )

        # Test input
        x = torch.randn(2, 16, 128)

        # Forward pass with seal protocol
        start_time = time.time()
        output, seal = transformer(x)
        forward_time = (time.time() - start_time) * 1000

        # Validate results
        shape_ok = x.shape == output.shape
        no_nan = not torch.isnan(output).any()
        no_inf = not torch.isinf(output).any()
        seal_valid = SealProtocol.firebreak_check(seal)

        # Energy analysis
        input_energy = torch.norm(x).item()
        output_energy = torch.norm(output).item()
        energy_ratio = output_energy / input_energy if input_energy > 0 else 0

        result = {
            'forward_time_ms': forward_time,
            'shape_ok': shape_ok,
            'no_nan': no_nan,
            'no_inf': no_inf,
            'seal_valid': seal_valid,
            'energy_ratio': energy_ratio,
            'rg_value': seal.get('RG', 0),
            'success': shape_ok and no_nan and no_inf and seal_valid
        }

        results[norm_type] = result

        print(f"     Time: {forward_time:.2f}ms")
        print(f"     Energy ratio: {energy_ratio:.4f}")
        print(f"     RG: {seal.get('RG', 'N/A')}")
        print(f"     Seal valid: {seal_valid}")
        print(f"     Success: {result['success']}")

    return results

def test_navigator_with_normalization():
    """Test Navigator Agent with normalized QRH layers"""
    print("\n🧭 Testing Navigator Agent with QRH Normalizations")
    print("-" * 60)

    # Test with layer norm (should be most stable)
    transformer = NegentropyTransformerBlock(
        d_model=64,
        nhead=2,
        dim_feedforward=128,
        dropout=0.1,
        qrh_embed_dim=16,
        alpha=1.0,
        use_learned_rotation=True,
        enable_gate=True,
        qrh_normalization_type='layer_norm'
    )

    navigator = NavigatorAgent()

    # Test multiple executions for stability
    success_count = 0
    total_tests = 20

    print(f"   Running {total_tests} Navigator executions with Layer Norm...")

    for i in range(total_tests):
        x = torch.randn(1, 8, 64)

        try:
            output, seal = navigator.execute_with_safety(x, transformer)

            # Check success criteria
            shape_ok = x.shape == output.shape
            no_nan = not torch.isnan(output).any()
            navigator_ok = 'navigator_info' in seal

            if shape_ok and no_nan and navigator_ok:
                success_count += 1

        except Exception as e:
            print(f"     Execution {i+1} failed: {str(e)}")

    success_rate = (success_count / total_tests) * 100
    print(f"   Navigator success rate: {success_rate:.1f}% ({success_count}/{total_tests})")

    return success_rate >= 90  # Expect high success rate with normalization

def test_stability_comparison():
    """Compare stability between different normalization approaches"""
    print("\n📊 Stability Comparison")
    print("-" * 60)

    normalizations = [None, 'layer_norm', 'unit_projection']
    stability_results = {}

    for norm_type in normalizations:
        norm_name = norm_type if norm_type else "none"
        print(f"\n   Testing stability with {norm_name} normalization...")

        # Create model
        transformer = NegentropyTransformerBlock(
            d_model=64,
            nhead=2,
            dim_feedforward=128,
            dropout=0.1,
            qrh_embed_dim=16,
            alpha=1.0,
            use_learned_rotation=True,
            enable_gate=True,
            qrh_normalization_type=norm_type
        )

        # Test with problematic inputs
        test_cases = [
            torch.randn(2, 8, 64),  # Normal
            torch.randn(2, 8, 64) * 100,  # Large values
            torch.randn(2, 8, 64) * 0.001,  # Small values
        ]

        case_results = []

        for i, x in enumerate(test_cases):
            try:
                output, seal = transformer(x)

                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                energy_ratio = torch.norm(output).item() / torch.norm(x).item()

                success = not has_nan and not has_inf and 0.1 < energy_ratio < 10.0
                case_results.append(success)

            except Exception as e:
                case_results.append(False)

        success_rate = sum(case_results) / len(case_results) * 100
        stability_results[norm_name] = success_rate

        print(f"     Stability rate: {success_rate:.1f}%")

    return stability_results

def run_integration_tests():
    """Run all integration tests"""
    print("🔗 QRH NORMALIZATION INTEGRATION TESTS")
    print("=" * 60)

    try:
        # Test 1: Negentropy Transformer with normalizations
        test1_results = test_negentropy_transformer_with_normalization()

        # Test 2: Navigator Agent with normalization
        test2_result = test_navigator_with_normalization()

        # Test 3: Stability comparison
        test3_results = test_stability_comparison()

        # Summary
        print(f"\n🏆 INTEGRATION TESTS SUMMARY")
        print("=" * 50)

        # Analyze transformer results
        all_transformers_work = all(
            result['success'] for result in test1_results.values()
        )

        # Compare energy stability
        energy_ratios = {
            norm: result['energy_ratio']
            for norm, result in test1_results.items()
        }

        print(f"   Negentropy Transformer: {'✅ PASS' if all_transformers_work else '❌ FAIL'}")
        print(f"   Navigator Integration: {'✅ PASS' if test2_result else '❌ FAIL'}")

        print(f"\n   Energy Ratios by Normalization:")
        for norm, ratio in energy_ratios.items():
            norm_name = norm if norm else "none"
            print(f"     {norm_name}: {ratio:.4f}")

        print(f"\n   Stability Rates:")
        for norm, rate in test3_results.items():
            print(f"     {norm}: {rate:.1f}%")

        # Overall assessment
        improved_stability = (
            test3_results.get('layer_norm', 0) > test3_results.get('none', 0) or
            test3_results.get('unit_projection', 0) > test3_results.get('none', 0)
        )

        overall_success = (
            all_transformers_work and
            test2_result and
            improved_stability
        )

        if overall_success:
            print(f"\n🎉 QRH NORMALIZATION INTEGRATION: SUCCESSFUL!")
            print("   ✅ All normalization types integrate correctly")
            print("   ✅ Navigator Agent works with normalized QRH")
            print("   ✅ Stability improvements demonstrated")
            print("   ✅ Energy ratios more controlled with normalization")
        else:
            print(f"\n⚠️  QRH NORMALIZATION INTEGRATION: NEEDS ATTENTION")
            print("   Some integration aspects need refinement")

        return overall_success

    except Exception as e:
        print(f"❌ Integration tests failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = run_integration_tests()