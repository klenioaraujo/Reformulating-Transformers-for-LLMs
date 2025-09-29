#!/usr/bin/env python3
"""
Basic usage example for ΨQRH Transformer
"""

import torch
import torch.nn as nn
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.architecture.psiqrh_transformer import PsiQRHTransformer
from src.validation.mathematical_validation import MathematicalValidator


def demonstrate_basic_usage():
    """Demonstrate basic ΨQRH transformer usage"""
    print("ΨQRH Transformer - Basic Usage Example")
    print("=" * 50)

    # Model configuration
    vocab_size = 10000
    d_model = 512
    n_layers = 6
    n_heads = 8
    dim_feedforward = 2048

    # Create ΨQRH transformer
    print("Creating ΨQRH Transformer...")
    model = PsiQRHTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward
    )

    # Display model information
    model_info = model.get_model_info()
    print(f"\nModel Architecture:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # Generate sample input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)

    print(f"\nInput shape: {input_ids.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(input_ids)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    return model, input_ids


def demonstrate_mathematical_validation():
    """Demonstrate mathematical validation"""
    print("\n" + "=" * 50)
    print("Mathematical Validation")
    print("=" * 50)

    # Create model and sample input
    vocab_size = 1000
    d_model = 256
    model = PsiQRHTransformer(vocab_size=vocab_size, d_model=d_model)
    input_ids = torch.randint(0, vocab_size, (1, 64), dtype=torch.long)

    # Create validator
    validator = MathematicalValidator(tolerance=0.05)

    # Run comprehensive validation
    from src.core.quaternion_operations import QuaternionOperations
    quaternion_ops = QuaternionOperations()

    print("Running comprehensive mathematical validation...")
    validation_results = validator.comprehensive_validation(
        model, input_ids, quaternion_ops
    )

    # Generate and print report
    report = validator.generate_validation_report(validation_results)
    print("\n" + report)


def demonstrate_performance_comparison():
    """Demonstrate basic performance comparison"""
    print("\n" + "=" * 50)
    print("Basic Performance Comparison")
    print("=" * 50)

    import time

    # Model configurations
    vocab_size = 5000
    d_model = 512
    seq_length = 256
    batch_size = 4

    # Create ΨQRH transformer
    psiqrh_model = PsiQRHTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=6,
        n_heads=8
    )

    # Create standard transformer for comparison
    standard_model = nn.Transformer(
        d_model=d_model,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048
    )

    # Generate sample input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)

    # Measure ΨQRH inference time
    print("Measuring ΨQRH inference time...")
    start_time = time.time()
    with torch.no_grad():
        psiqrh_output = psiqrh_model(input_ids)
    psiqrh_time = time.time() - start_time

    # Measure memory usage
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print(f"ΨQRH Inference Time: {psiqrh_time:.4f} seconds")
    print(f"ΨQRH Output Shape: {psiqrh_output.shape}")

    # Note: Standard transformer would require proper setup
    print("\nNote: Full performance comparison requires proper standard transformer setup")
    print("This example demonstrates the basic ΨQRH usage pattern.")


def main():
    """Main demonstration function"""
    print("ΨQRH Transformer Demonstration")
    print("=" * 50)

    # Basic usage
    model, input_ids = demonstrate_basic_usage()

    # Mathematical validation
    demonstrate_mathematical_validation()

    # Performance comparison
    demonstrate_performance_comparison()

    print("\n" + "=" * 50)
    print("Demonstration Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run mathematical validation on your specific use case")
    print("2. Compare performance with standard transformers")
    print("3. Explore different model configurations")
    print("4. Check out the implementation roadmap for upcoming features")


if __name__ == "__main__":
    main()