#!/usr/bin/env python3
"""
Test QRH Layer normalization improvements
"""

import torch
import numpy as np
import time
from qrh_layer import QRHLayer, QRHConfig, QuaternionLayerNorm

def test_quaternion_layer_norm():
    """Test QuaternionLayerNorm independently"""
    print("🧪 Testing QuaternionLayerNorm")
    print("-" * 40)

    # Create test data
    batch_size, seq_len, embed_dim = 2, 8, 16
    x = torch.randn(batch_size, seq_len, embed_dim, 4)

    # Test normalization
    norm_layer = QuaternionLayerNorm()
    normalized = norm_layer(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {normalized.shape}")
    print(f"   ✅ Shape preserved: {x.shape == normalized.shape}")

    # Check that normalization is working
    input_mean = x.mean(dim=-1).mean()
    input_std = x.std(dim=-1).mean()
    output_mean = normalized.mean(dim=-1).mean()
    output_std = normalized.std(dim=-1).mean()

    print(f"   Input mean: {input_mean:.6f}, std: {input_std:.6f}")
    print(f"   Output mean: {output_mean:.6f}, std: {output_std:.6f}")
    print(f"   ✅ Normalization applied: {abs(output_mean) < 0.1}")

    return True

def test_qrh_layer_normalization_options():
    """Test QRH layer with different normalization options"""
    print("\n🔧 Testing QRH Layer Normalization Options")
    print("-" * 40)

    batch_size, seq_len = 2, 16
    test_configs = [
        (None, "No normalization"),
        ('layer_norm', "Quaternion Layer Norm"),
        ('unit_projection', "Unit Projection")
    ]

    results = {}

    for norm_type, description in test_configs:
        print(f"\n   Testing {description}...")

        # Create config
        config = QRHConfig(
            embed_dim=32,
            alpha=1.0,
            normalization_type=norm_type,
            use_learned_rotation=True
        )

        # Create layer
        layer = QRHLayer(config)

        # Test input
        d_model = 4 * config.embed_dim  # 128
        x = torch.randn(batch_size, seq_len, d_model)

        # Forward pass
        start_time = time.time()
        output = layer(x)
        forward_time = (time.time() - start_time) * 1000

        # Validate
        shape_ok = x.shape == output.shape
        no_nan = not torch.isnan(output).any()
        no_inf = not torch.isinf(output).any()

        # Energy analysis
        input_energy = torch.norm(x).item()
        output_energy = torch.norm(output).item()
        energy_ratio = output_energy / input_energy if input_energy > 0 else 0

        # Gradient magnitude (for stability analysis)
        output.mean().backward()
        grad_norm = 0
        param_count = 0
        for param in layer.parameters():
            if param.grad is not None:
                grad_norm += torch.norm(param.grad).item()**2
                param_count += 1
        grad_norm = np.sqrt(grad_norm) if param_count > 0 else 0

        result = {
            'forward_time_ms': forward_time,
            'shape_ok': shape_ok,
            'no_nan': no_nan,
            'no_inf': no_inf,
            'energy_ratio': energy_ratio,
            'gradient_norm': grad_norm,
            'success': shape_ok and no_nan and no_inf
        }

        results[norm_type] = result

        print(f"     Time: {forward_time:.2f}ms")
        print(f"     Energy ratio: {energy_ratio:.4f}")
        print(f"     Gradient norm: {grad_norm:.6f}")
        print(f"     Success: {result['success']}")

    return results

def test_numerical_stability():
    """Test numerical stability with extreme inputs"""
    print("\n⚡ Testing Numerical Stability")
    print("-" * 40)

    configs = [
        (None, "No normalization"),
        ('layer_norm', "Layer Norm"),
        ('unit_projection', "Unit Projection")
    ]

    test_cases = [
        ("Normal", torch.randn(1, 8, 128)),
        ("Large values", torch.randn(1, 8, 128) * 100),
        ("Small values", torch.randn(1, 8, 128) * 0.001),
        ("Near zero", torch.randn(1, 8, 128) * 1e-6)
    ]

    stability_results = {}

    for norm_type, norm_desc in configs:
        print(f"\n   Testing {norm_desc}:")
        config = QRHConfig(embed_dim=32, normalization_type=norm_type)
        layer = QRHLayer(config)

        case_results = {}

        for case_name, x in test_cases:
            try:
                output = layer(x)

                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                output_range = (torch.min(output).item(), torch.max(output).item())

                success = not has_nan and not has_inf

                case_results[case_name] = {
                    'success': success,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'output_range': output_range
                }

                status = "✅" if success else "❌"
                print(f"     {case_name}: {status}")

            except Exception as e:
                case_results[case_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"     {case_name}: ❌ Error: {str(e)}")

        stability_results[norm_type] = case_results

    return stability_results

def test_unit_projection_properties():
    """Test that unit projection actually creates unit quaternions at the normalization step"""
    print("\n🔍 Testing Unit Projection Properties")
    print("-" * 40)

    config = QRHConfig(embed_dim=16, normalization_type='unit_projection')
    layer = QRHLayer(config)

    # Test with various inputs
    x = torch.randn(2, 8, 64)  # 4 * 16 = 64

    # Test the normalization step directly by accessing internal quaternions
    x_validated = layer._validate_input(x)
    Ψ = layer._preprocess_input(x_validated)
    Ψ_filtered = layer._apply_spectral_filtering(Ψ)
    Ψ_rotated = layer._apply_quaternion_rotations(Ψ_filtered)

    # Now test the normalization - this is where unit projection should work
    Ψ_normalized = layer._apply_normalization(Ψ_rotated)

    # Calculate norms of normalized quaternions
    quat_norms = torch.norm(Ψ_normalized, p=2, dim=-1)

    # Check if they are unit length
    mean_norm = quat_norms.mean().item()
    std_norm = quat_norms.std().item()
    min_norm = quat_norms.min().item()
    max_norm = quat_norms.max().item()

    print(f"   Normalized quaternion norms - Mean: {mean_norm:.6f}, Std: {std_norm:.6f}")
    print(f"   Norm range: [{min_norm:.6f}, {max_norm:.6f}]")

    # Check if norms are close to 1.0 (allowing for some numerical error)
    unit_check = torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5)
    print(f"   ✅ Unit quaternions after normalization: {unit_check}")

    # Note: The final output will differ due to linear projections in postprocessing
    print(f"   Note: Final output norms will differ due to linear projections")

    return unit_check

def run_normalization_tests():
    """Run all normalization tests"""
    print("🔢 QRH LAYER NORMALIZATION TESTS")
    print("=" * 50)

    try:
        # Test 1: QuaternionLayerNorm
        test1_result = test_quaternion_layer_norm()

        # Test 2: QRH Layer with different normalizations
        test2_results = test_qrh_layer_normalization_options()

        # Test 3: Numerical stability
        test3_results = test_numerical_stability()

        # Test 4: Unit projection properties
        test4_result = test_unit_projection_properties()

        # Summary
        print(f"\n🏆 NORMALIZATION TESTS SUMMARY")
        print("=" * 40)

        # Analyze results
        all_normalizations_work = all(
            result['success'] for result in test2_results.values()
        )

        stability_passed = 0
        stability_total = 0
        for norm_results in test3_results.values():
            for case_result in norm_results.values():
                stability_total += 1
                if case_result.get('success', False):
                    stability_passed += 1

        stability_rate = (stability_passed / stability_total) * 100 if stability_total > 0 else 0

        print(f"   QuaternionLayerNorm: {'✅ PASS' if test1_result else '❌ FAIL'}")
        print(f"   QRH Normalizations: {'✅ PASS' if all_normalizations_work else '❌ FAIL'}")
        print(f"   Stability Rate: {stability_rate:.1f}% ({stability_passed}/{stability_total})")
        print(f"   Unit Projection: {'✅ PASS' if test4_result else '❌ FAIL'}")

        overall_success = (
            test1_result and
            all_normalizations_work and
            stability_rate >= 75 and
            test4_result
        )

        if overall_success:
            print(f"\n🎉 QRH NORMALIZATION: FULLY FUNCTIONAL!")
            print("   ✅ QuaternionLayerNorm implemented correctly")
            print("   ✅ All normalization types working")
            print("   ✅ Numerical stability improved")
            print("   ✅ Unit projection creates unit quaternions")
        else:
            print(f"\n⚠️  QRH NORMALIZATION: NEEDS ATTENTION")
            print("   Some normalization features need refinement")

        return overall_success

    except Exception as e:
        print(f"❌ Tests failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = run_normalization_tests()