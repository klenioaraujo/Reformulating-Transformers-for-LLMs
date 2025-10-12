#!/usr/bin/env python3
"""
Architecture Validation of Inverse Cognitive Projector
======================================================

Validates that the Inverse Cognitive Projector architecture is viable by testing
forward/backward passes and gradient flow without full training.

Based on the task requirements:
- Test that the model can process inputs and produce outputs
- Verify gradient flow through the network
- Check numerical stability
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Import required components
from src.core.inverse_cognitive_projector import create_inverse_cognitive_projector
from tools.audit_analyzer import ΨQRHAuditAnalyzer

def validate_inverse_projector_architecture(embed_dim: int = 256, device: str = "cpu"):
    """
    Validate the Inverse Cognitive Projector architecture without full training.

    Args:
        embed_dim: Embedding dimension to test
        device: Device to run validation on

    Returns:
        Validation results dictionary
    """
    print("=" * 80)
    print("🔬 ARCHITECTURE VALIDATION: INVERSE COGNITIVE PROJECTOR")
    print("=" * 80)
    print("\nTesting architecture viability...")
    print(f"   - Embedding dimension: {embed_dim}")
    print(f"   - Device: {device}")
    print("\n" + "=" * 80)

    results = {
        'architecture_test': False,
        'forward_pass': False,
        'backward_pass': False,
        'gradient_flow': False,
        'numerical_stability': False,
        'parameter_count': 0,
        'output_shape_correct': False,
        'errors': []
    }

    try:
        # Create model
        print("🏗️  Creating Inverse Cognitive Projector model...")
        model = create_inverse_cognitive_projector(
            embed_dim=embed_dim,
            vocab_size=50257,
            hidden_dim=512,
            num_layers=3,
            dropout=0.1
        ).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results['parameter_count'] = total_params

        print(f"✅ Model created successfully")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")

        # Generate test input (ideal character probe)
        print("\n📝 Generating test input...")
        analyzer = ΨQRHAuditAnalyzer()
        test_probe = analyzer.generate_ascii_probes(embed_dim, device)[0]  # First character probe
        test_input = test_probe.unsqueeze(0)  # [1, embed_dim]

        print(f"✅ Test input generated: shape {test_input.shape}")

        # Test forward pass
        print("\n🔄 Testing forward pass...")
        model.eval()
        with torch.no_grad():
            try:
                output = model(test_input)
                print(f"✅ Forward pass successful: output shape {output.shape}")

                # Check output shape (can be [1, embed_dim] or [1, embed_dim, 4] depending on quantum_vocab)
                expected_shapes = [(1, embed_dim), (1, embed_dim, 4)]
                if output.shape in expected_shapes:
                    results['output_shape_correct'] = True
                    print("✅ Output shape correct")
                else:
                    results['errors'].append(f"Output shape mismatch: expected one of {expected_shapes}, got {output.shape}")
                    print(f"❌ Output shape incorrect: expected one of {expected_shapes}, got {output.shape}")

                results['forward_pass'] = True

            except Exception as e:
                results['errors'].append(f"Forward pass failed: {e}")
                print(f"❌ Forward pass failed: {e}")
                return results

        # Test backward pass and gradient flow
        print("\n🔄 Testing backward pass and gradient flow...")
        model.train()

        # Create a simple loss function (MSE to target)
        # Target should match output shape
        if output.dim() == 3:  # [1, embed_dim, 4]
            # Expand input to quaternion format for target
            target = test_input.unsqueeze(-1).expand(-1, -1, 4)  # [1, embed_dim, 4]
        else:  # [1, embed_dim]
            target = test_input.clone()  # Identity target
        criterion = nn.MSELoss()

        try:
            # Forward pass with gradients
            output = model(test_input)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Check gradient flow
            has_gradients = False
            zero_gradients = 0
            total_params = 0

            for name, param in model.named_parameters():
                if param.grad is not None:
                    total_params += 1
                    if torch.any(param.grad != 0):
                        has_gradients = True
                    else:
                        zero_gradients += 1

            if has_gradients:
                results['gradient_flow'] = True
                results['backward_pass'] = True
                print("✅ Backward pass successful")
                print(f"   - Parameters with gradients: {total_params}")
                print(f"   - Parameters with zero gradients: {zero_gradients}")
                print(".6f")
            else:
                results['errors'].append("No gradients flowing through the network")
                print("❌ No gradients flowing through the network")

        except Exception as e:
            results['errors'].append(f"Backward pass failed: {e}")
            print(f"❌ Backward pass failed: {e}")

        # Test numerical stability with multiple inputs
        print("\n🔢 Testing numerical stability...")
        model.eval()
        stable_tests = 0
        total_tests = 10

        with torch.no_grad():
            for i in range(total_tests):
                try:
                    # Generate random input in similar range to probes
                    random_input = torch.randn_like(test_input) * 0.1
                    output = model(random_input)

                    # Check for NaN/Inf
                    if torch.isfinite(output).all() and not torch.isnan(output).any():
                        stable_tests += 1
                    else:
                        results['errors'].append(f"Numerical instability in test {i+1}")
                except Exception as e:
                    results['errors'].append(f"Stability test {i+1} failed: {e}")

        stability_ratio = stable_tests / total_tests
        if stability_ratio >= 0.8:  # 80% success rate
            results['numerical_stability'] = True
            print(".1f")
        else:
            print(".1f")
            results['errors'].append(f"Poor numerical stability: {stable_tests}/{total_tests} tests passed")

        # Overall architecture test
        if (results['forward_pass'] and results['backward_pass'] and
            results['gradient_flow'] and results['numerical_stability'] and
            results['output_shape_correct']):
            results['architecture_test'] = True
            print("\n✅ ARCHITECTURE VALIDATION: PASSED")
            print("   The Inverse Cognitive Projector architecture is viable")
        else:
            print("\n❌ ARCHITECTURE VALIDATION: FAILED")
            print("   Issues found that need to be addressed")

        return results

    except Exception as e:
        results['errors'].append(f"Architecture validation failed: {e}")
        print(f"❌ Architecture validation failed: {e}")
        return results

def main():
    """Main function to run architecture validation"""
    embed_dim = 256  # Use optimized dimension
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run validation
    results = validate_inverse_projector_architecture(embed_dim=embed_dim, device=device)

    # Print detailed results
    print("\n" + "=" * 80)
    print("📋 VALIDATION RESULTS SUMMARY")
    print("=" * 80)

    print("Test Component              | Status")
    print("---------------------------|--------")
    print(f"Architecture Test          | {'✅ PASS' if results['architecture_test'] else '❌ FAIL'}")
    print(f"Forward Pass               | {'✅ PASS' if results['forward_pass'] else '❌ FAIL'}")
    print(f"Backward Pass              | {'✅ PASS' if results['backward_pass'] else '❌ FAIL'}")
    print(f"Gradient Flow              | {'✅ PASS' if results['gradient_flow'] else '❌ FAIL'}")
    print(f"Numerical Stability        | {'✅ PASS' if results['numerical_stability'] else '❌ FAIL'}")
    print(f"Output Shape Correct       | {'✅ PASS' if results['output_shape_correct'] else '❌ FAIL'}")

    print(f"\nModel Parameters: {results['parameter_count']:,}")

    if results['errors']:
        print(f"\n⚠️  Errors encountered ({len(results['errors'])}):")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"   - {error}")

    print("\n🔧 Next steps:")
    if results['architecture_test']:
        print("   ✅ Proceed with integration into full pipeline")
        print("   ✅ Architecture is viable for complex mappings")
        print("   ✅ Ready for full pre-training if needed")
    else:
        print("   ❌ Fix architectural issues before pipeline integration")
        print("   ❌ Investigate gradient flow and numerical stability")

if __name__ == "__main__":
    main()