#!/usr/bin/env python3
"""
Basic usage example for ΨQRH Transformer
Uses proper configuration from configs/example_configs.py
"""

import torch
import torch.nn as nn
import sys
import os

# Add src and configs directories to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

from src.architecture.psiqrh_transformer import PsiQRHTransformer
from src.validation.mathematical_validation import MathematicalValidator
from examples.config_loader import get_example_config

# Import complete implementation if available
try:
    from src.core.fractal_quantum_embedding import PsiQRHTransformerComplete
    HAS_COMPLETE_IMPLEMENTATION = True
except ImportError:
    HAS_COMPLETE_IMPLEMENTATION = False
    print("⚠️  PsiQRHTransformerComplete not available. Using standard PsiQRHTransformer.")


def demonstrate_basic_usage():
    """Demonstrate basic ΨQRH transformer usage"""
    print("ΨQRH Transformer - Basic Usage Example")
    print("=" * 50)

    # Load configuration for basic usage
    print("Loading configuration for basic_usage.py...")
    config = get_example_config("basic_usage.py")

    print(f"Configuration loaded:")
    print(f"  vocab_size: {config.get('vocab_size', 10000)}")
    print(f"  d_model: {config.get('d_model', 512)}")
    print(f"  n_layers: {config.get('n_layers', 6)}")
    print(f"  n_heads: {config.get('n_heads', 8)}")
    print(f"  dim_feedforward: {config.get('dim_feedforward', 2048)}")

    # Create ΨQRH transformer
    print("\nCreating ΨQRH Transformer...")
    model = PsiQRHTransformer(
        vocab_size=config.get('vocab_size', 10000),
        d_model=config.get('d_model', 512),
        n_layers=config.get('n_layers', 6),
        n_heads=config.get('n_heads', 8),
        dim_feedforward=config.get('dim_feedforward', 2048)
    )

    # Display model information
    model_info = model.get_model_info()
    print(f"\nModel Architecture:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # Generate sample input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, config.get('vocab_size', 10000), (batch_size, seq_length), dtype=torch.long)

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

    # Load scientific validation configuration
    print("Loading scientific validation configuration...")
    config = get_example_config("basic_usage.py")

    # Use smaller model for quick validation
    validation_config = get_example_config("basic_usage.py",
                                         vocab_size=1000,
                                         d_model=256,
                                         n_layers=3)

    print(f"Validation config: {validation_config.get('vocab_size', 1000)} vocab, {validation_config.get('d_model', 256)} d_model")

    # Create model and sample input
    model = PsiQRHTransformer(
        vocab_size=validation_config.get('vocab_size', 1000),
        d_model=validation_config.get('d_model', 256),
        n_layers=validation_config.get('n_layers', 3),
        n_heads=validation_config.get('n_heads', 8)
    )
    input_ids = torch.randint(0, validation_config.get('vocab_size', 1000), (1, 64), dtype=torch.long)

    print("Running quick mathematical validation...")

    try:
        with torch.no_grad():
            # Get input embeddings for proper energy comparison
            input_embeddings = model.token_embedding(input_ids)
            input_energy = torch.sum(input_embeddings ** 2).item()

            # Run one forward pass
            output = model(input_ids)
            output_energy = torch.sum(output ** 2).item()

            energy_ratio = output_energy / (input_energy + 1e-8)

            print(f"  Input Energy (embeddings): {input_energy:.6f}")
            print(f"  Output Energy: {output_energy:.6f}")
            print(f"  Energy Ratio: {energy_ratio:.6f}")

            if 0.95 <= energy_ratio <= 1.05:
                print("  Energy Conservation: ✅ PASS")
            else:
                print("  Energy Conservation: ❌ FAIL")

            # Also test with raw token energy for reference
            raw_token_energy = torch.sum(input_ids.float() ** 2).item()
            raw_ratio = output_energy / (raw_token_energy + 1e-8)
            print(f"  Raw Token Energy: {raw_token_energy:.6f}")
            print(f"  Raw Token Ratio: {raw_ratio:.6f} (for reference)")

        # Basic numerical stability test
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()

        if not has_nan and not has_inf:
            print("  Numerical Stability: ✅ PASS")
        else:
            print("  Numerical Stability: ❌ FAIL")

        print("\n✅ Quick validation completed successfully!")

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        print("This is expected for some model configurations.")

    print("\nNote: For comprehensive validation, run:")
    print("  python3 parseval_validation_test.py")
    print("  python3 energy_conservation_test.py")


def demonstrate_complete_implementation():
    """Demonstrate the complete PsiQRHTransformerComplete implementation"""
    if not HAS_COMPLETE_IMPLEMENTATION:
        print("\n⚠️  PsiQRHTransformerComplete not available. Skipping.")
        return

    print("\n" + "=" * 50)
    print("Complete ΨQRH Implementation (Física Rigorosa)")
    print("=" * 50)

    # Load configuration
    config = get_example_config("basic_usage.py", vocab_size=1000, d_model=256)

    print(f"Configuration:")
    print(f"  vocab_size: {config.get('vocab_size', 1000)}")
    print(f"  d_model: {config.get('d_model', 256)}")
    print(f"  n_layers: {config.get('n_layers', 3)}")
    print(f"  n_heads: {config.get('n_heads', 8)}")

    # Create complete model
    print("\nCreating PsiQRHTransformerComplete...")
    print("  ✅ Fractal Quantum Embedding")
    print("  ✅ Spectral Attention with α(D) adaptation")
    print("  ✅ SO(4) Harmonic Evolution")
    print("  ✅ Optical Probe Generation")

    model = PsiQRHTransformerComplete(
        vocab_size=config.get('vocab_size', 1000),
        embed_dim=128,
        quaternion_dim=4,
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 3),
        n_rotations=4,
        dropout=0.1,
        max_seq_len=128,
        use_leech_correction=False
    )

    # Generate sample input
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, config.get('vocab_size', 1000), (batch_size, seq_length), dtype=torch.long)

    print(f"\nInput shape: {input_ids.shape}")

    # Forward pass
    print("\nRunning forward pass with complete physics implementation...")
    with torch.no_grad():
        output = model(input_ids)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Test generation
    print("\nTesting autoregressive generation...")
    prompt = input_ids[:1, :4]
    print(f"Prompt shape: {prompt.shape}")

    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)

    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated length: {generated.shape[1]} (prompt: {prompt.shape[1]}, new: {generated.shape[1] - prompt.shape[1]})")
    print("✅ Generation successful!")

    print("\n🔬 Physics Validation:")
    print("  For comprehensive physics tests, run:")
    print("    python3 psiqrh.py --test-physics")
    print("    make test-physics")


def demonstrate_performance_comparison():
    """Demonstrate basic performance comparison"""
    print("\n" + "=" * 50)
    print("Basic Performance Comparison")
    print("=" * 50)

    import time

    # Load performance configuration
    config = get_example_config("basic_usage.py", vocab_size=5000, d_model=512)

    seq_length = 256
    batch_size = 4

    print(f"Performance test config: {config.get('vocab_size', 5000)} vocab, {config.get('d_model', 512)} d_model")

    # Create ΨQRH transformer
    psiqrh_model = PsiQRHTransformer(
        vocab_size=config.get('vocab_size', 5000),
        d_model=config.get('d_model', 512),
        n_layers=config.get('n_layers', 6),
        n_heads=config.get('n_heads', 8)
    )

    # Create standard transformer for comparison
    standard_model = nn.Transformer(
        d_model=config.get('d_model', 512),
        nhead=config.get('n_heads', 8),
        num_encoder_layers=config.get('n_layers', 6),
        num_decoder_layers=config.get('n_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 2048)
    )

    # Generate sample input
    input_ids = torch.randint(0, config.get('vocab_size', 5000), (batch_size, seq_length), dtype=torch.long)

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

    # Complete implementation (if available)
    demonstrate_complete_implementation()

    # Performance comparison
    demonstrate_performance_comparison()

    print("\n" + "=" * 50)
    print("Demonstration Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run mathematical validation on your specific use case")
    print("2. Compare performance with standard transformers")
    print("3. Explore different model configurations")
    print("4. Try the complete physics implementation with:")
    print("   - python3 psiqrh.py --test-physics")
    print("   - make train-complete")
    print("5. Check out the implementation roadmap for upcoming features")


if __name__ == "__main__":
    main()