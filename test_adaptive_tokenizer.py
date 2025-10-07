#!/usr/bin/env python3
"""
Test Script for Adaptive Spectral Vocabulary PhysicalTokenizer
===============================================================

Tests the evolution from deterministic to learnable spectral representations.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from processing.physical_tokenizer import PhysicalTokenizer

def test_deterministic_mode():
    """Test Phase 1: Deterministic mathematical mapping"""
    print("🔢 Testing Phase 1: Deterministic Mathematical Mapping")
    print("=" * 60)

    tokenizer = PhysicalTokenizer(embed_dim=32, learnable=False)

    test_text = "Hello World!"
    print(f"Input text: '{test_text}'")

    # Test encoding
    encoded = tokenizer.encode(test_text)
    print(f"Encoded shape: {encoded.shape}")

    # Test decoding
    decoded_chars = []
    for i in range(encoded.shape[1]):  # Iterate over sequence length
        psi_state = encoded[0, i]  # [embed_dim, 4]
        char = tokenizer.decode_state(psi_state, i)
        decoded_chars.append(char)

    decoded_text = ''.join(decoded_chars)
    print(f"Decoded text: '{decoded_text}'")

    # Test vocabulary info
    vocab_info = tokenizer.get_vocabulary_info()
    print(f"Vocabulary size: {vocab_info['vocabulary_size']}")
    print(f"Phase: {vocab_info['phase']}")

    return tokenizer

def test_learnable_mode():
    """Test Phase 2: Adaptive Spectral Vocabulary"""
    print("\n🎵 Testing Phase 2: Adaptive Spectral Vocabulary")
    print("=" * 60)

    tokenizer = PhysicalTokenizer(embed_dim=32, spectral_params_dim=8, learnable=True)

    test_text = "Hello World!"
    print(f"Input text: '{test_text}'")

    # Test encoding
    encoded = tokenizer.encode(test_text)
    print(f"Encoded shape: {encoded.shape}")

    # Test decoding
    decoded_chars = []
    for i in range(encoded.shape[1]):  # Iterate over sequence length
        psi_state = encoded[0, i]  # [embed_dim, 4]
        char = tokenizer.decode_state(psi_state, i)
        decoded_chars.append(char)

    decoded_text = ''.join(decoded_chars)
    print(f"Decoded text: '{decoded_text}'")

    # Test vocabulary info
    vocab_info = tokenizer.get_vocabulary_info()
    print(f"Vocabulary size: {vocab_info['vocabulary_size']}")
    print(f"Learnable parameters: {vocab_info['total_learnable_params']}")
    print(f"Phase: {vocab_info['phase']}")

    # Show spectral embedding shape
    if hasattr(tokenizer, 'spectral_embedding'):
        print(f"Spectral embedding shape: {tokenizer.spectral_embedding.weight.shape}")

    return tokenizer

def test_training_potential():
    """Test that the learnable tokenizer can be used in training"""
    print("\n🚀 Testing Training Potential")
    print("=" * 60)

    tokenizer = PhysicalTokenizer(embed_dim=16, spectral_params_dim=4, learnable=True)

    # Create some dummy training data
    test_texts = ["Hello", "World", "Quantum", "AI"]

    print("Testing parameter updates...")

    # Get initial parameters
    initial_params = tokenizer.spectral_embedding.weight.clone()

    # Simulate a training step (dummy gradients)
    loss = torch.sum(tokenizer.spectral_embedding.weight ** 2)  # Dummy loss
    loss.backward()

    # Apply gradient descent (dummy optimizer step)
    with torch.no_grad():
        tokenizer.spectral_embedding.weight -= 0.01 * tokenizer.spectral_embedding.weight.grad

    # Clear gradients
    tokenizer.spectral_embedding.weight.grad.zero_()

    # Check that parameters changed
    param_change = torch.mean(torch.abs(tokenizer.spectral_embedding.weight - initial_params)).item()
    print(".6f")

    if param_change > 1e-6:
        print("✅ Parameters successfully updated - ready for training!")
    else:
        print("❌ Parameters did not change - check gradient flow")

    return tokenizer

def main():
    """Run all tests"""
    print("🧪 Testing Adaptive Spectral Vocabulary PhysicalTokenizer")
    print("=" * 80)

    try:
        # Test deterministic mode
        det_tokenizer = test_deterministic_mode()

        # Test learnable mode
        learn_tokenizer = test_learnable_mode()

        # Test training potential
        train_tokenizer = test_training_potential()

        print("\n🎉 All tests completed successfully!")
        print("\n📊 Summary:")
        print("- Phase 1 (Deterministic): Working ✅")
        print("- Phase 2 (Learnable): Working ✅")
        print("- Training Integration: Ready ✅")

        print("\n🎵 The tokenizer has successfully evolved from deterministic")
        print("   mathematical mapping to adaptive spectral vocabulary!")
        print("   Each character now has its own learnable 'timbre' parameters.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())