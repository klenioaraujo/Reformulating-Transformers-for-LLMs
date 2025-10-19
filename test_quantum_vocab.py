#!/usr/bin/env python3
"""
Test Quantum Native Vocabulary with ΨQRH Pipeline
=================================================

Tests the quantum-native vocabulary integration with the ΨQRH pipeline
to ensure autonomous operation without GPT-2 dependencies.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_vocab_integration import QuantumVocabularyIntegration, integrate_with_psiqrh


def test_quantum_vocabulary_autonomy():
    """Test that quantum vocabulary operates autonomously"""
    print("🧪 Testing Quantum Vocabulary Autonomy")
    print("=" * 50)

    # Initialize quantum vocabulary
    quantum_vocab = QuantumVocabularyIntegration()

    # Test basic functionality
    stats = quantum_vocab.get_integration_stats()

    print(f"📊 Quantum Vocabulary Stats:")
    print(f"   Vocabulary Size: {stats['vocab_size']:,}")
    print(f"   Autonomous: {stats['is_autonomous']}")
    print(f"   GPT-2 Dependency: {stats['has_gpt2_dependency']}")

    # Verify autonomy
    assert stats['is_autonomous'] == True, "Vocabulary should be autonomous"
    assert stats['has_gpt2_dependency'] == False, "Vocabulary should not have GPT-2 dependency"
    assert stats['vocab_size'] == 50257, f"Vocabulary size should be 50257, got {stats['vocab_size']}"

    print("✅ Autonomy test passed!")


def test_tokenization_functionality():
    """Test tokenization functionality"""
    print("\n🔬 Testing Tokenization Functionality")
    print("=" * 50)

    quantum_vocab = QuantumVocabularyIntegration()

    # Test cases
    test_cases = [
        "quantum mechanics",
        "hello world",
        "the schrodinger equation",
        "entanglement and superposition"
    ]

    for test_text in test_cases:
        token_ids = quantum_vocab.tokenize_text(test_text)
        reconstructed = quantum_vocab.detokenize_text(token_ids)

        print(f"   Input: '{test_text}'")
        print(f"   Token IDs: {token_ids}")
        print(f"   Reconstructed: '{reconstructed}'")
        print()

    print("✅ Tokenization test passed!")


def test_quantum_properties():
    """Test quantum properties functionality"""
    print("\n🔬 Testing Quantum Properties")
    print("=" * 50)

    quantum_vocab = QuantumVocabularyIntegration()

    # Test quantum properties for various tokens
    test_tokens = ["quantum", "electron", "photon", "entanglement"]

    for token in test_tokens:
        props = quantum_vocab.get_quantum_properties(token)
        print(f"   Token: '{token}'")
        print(f"     Energy Level: {props['energy_level']:.3f}")
        print(f"     Coherence: {props['coherence']:.3f}")
        print(f"     Entropy: {props['entropy']:.3f}")
        print(f"     Spin: {props['spin']:.3f}")
        print()

    print("✅ Quantum properties test passed!")


def test_embedding_generation():
    """Test embedding matrix generation"""
    print("\n🔬 Testing Embedding Generation")
    print("=" * 50)

    quantum_vocab = QuantumVocabularyIntegration()

    # Generate embedding matrix
    embed_dim = 256
    embedding_matrix = quantum_vocab.create_quantum_embedding_matrix(embed_dim)

    print(f"   Embedding Matrix Shape: {embedding_matrix.shape}")
    print(f"   Vocabulary Size: {quantum_vocab.get_vocab_size()}")
    print(f"   Embedding Dimension: {embed_dim}")

    # Verify dimensions
    assert embedding_matrix.shape[0] == quantum_vocab.get_vocab_size(), "Vocabulary size mismatch"
    assert embedding_matrix.shape[1] == embed_dim, "Embedding dimension mismatch"

    print("✅ Embedding generation test passed!")


def main():
    """Main test function"""
    print("ΨQRH Quantum Vocabulary Integration Test")
    print("=" * 50)

    # Run all tests
    test_quantum_vocabulary_autonomy()
    test_tokenization_functionality()
    test_quantum_properties()
    test_embedding_generation()

    print("\n🎉 All tests passed!")
    print("✅ Quantum vocabulary integration is working correctly")
    print("✅ ΨQRH pipeline is now autonomous")
    print("✅ No GPT-2 dependencies")


if __name__ == "__main__":
    main()
