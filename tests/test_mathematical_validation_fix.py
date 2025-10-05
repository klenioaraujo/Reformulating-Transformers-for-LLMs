#!/usr/bin/env python3
"""
Unit tests for mathematical validation improvements

Tests the fixes for:
1. Proper input energy calculation
2. EmbeddingNotFoundError handling
3. Skip_on_no_embedding functionality
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.mathematical_validation import (
    MathematicalValidator,
    EmbeddingNotFoundError
)
from src.core.utils import compute_energy


class MockModelWithEmbedding(nn.Module):
    """Mock model with token_embedding"""
    def __init__(self, vocab_size=100, embed_dim=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        embedded = self.token_embedding(x)
        return self.linear(embedded)


class MockModelWithoutEmbedding(nn.Module):
    """Mock model without token_embedding"""
    def __init__(self, input_dim=64, output_dim=64):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Handle both float and int inputs
        if x.dtype != torch.float32:
            # Convert to float for linear layer
            x = x.float()
        return self.linear(x)


def test_energy_with_embedding():
    """Test energy conservation with model that has token_embedding"""
    print("Test 1: Energy conservation with token_embedding")

    validator = MathematicalValidator(tolerance=0.5)
    model = MockModelWithEmbedding(vocab_size=100, embed_dim=64)

    # Token IDs input
    x = torch.randint(0, 100, (2, 8))

    result = validator.validate_energy_conservation(model, x)

    assert result['validation_method'] == 'proper_embedding', \
        f"Expected 'proper_embedding', got {result['validation_method']}"
    assert result['input_energy'] is not None, "Input energy should not be None"
    assert result['output_energy'] is not None, "Output energy should not be None"
    assert result['conservation_ratio'] is not None, "Conservation ratio should not be None"

    print(f"  ✓ Input energy: {result['input_energy']:.6f}")
    print(f"  ✓ Output energy: {result['output_energy']:.6f}")
    print(f"  ✓ Conservation ratio: {result['conservation_ratio']:.6f}")
    print(f"  ✓ Is conserved: {result['is_conserved']}")


def test_energy_with_float_input():
    """Test energy conservation with model without embedding but float input"""
    print("\nTest 2: Energy conservation with float embeddings input")

    validator = MathematicalValidator(tolerance=0.5)
    model = MockModelWithoutEmbedding(input_dim=64, output_dim=64)

    # Float embeddings input (already embedded)
    x = torch.randn(2, 8, 64)

    result = validator.validate_energy_conservation(model, x)

    assert result['validation_method'] == 'proper_embedding', \
        f"Expected 'proper_embedding', got {result['validation_method']}"
    assert result['input_energy'] is not None, "Input energy should not be None"
    assert result['output_energy'] is not None, "Output energy should not be None"

    print(f"  ✓ Input energy: {result['input_energy']:.6f}")
    print(f"  ✓ Output energy: {result['output_energy']:.6f}")
    print(f"  ✓ Conservation ratio: {result['conservation_ratio']:.6f}")


def test_energy_without_embedding_raises_error():
    """Test that validation raises error when no embedding available"""
    print("\nTest 3: Raises EmbeddingNotFoundError without skip flag")

    validator = MathematicalValidator(tolerance=0.5)
    model = MockModelWithoutEmbedding(input_dim=8, output_dim=64)

    # Token IDs input (incompatible with model without embeddings)
    x = torch.randint(0, 100, (2, 8))

    try:
        result = validator.validate_energy_conservation(model, x, skip_on_no_embedding=False)
        assert False, "Should have raised EmbeddingNotFoundError"
    except EmbeddingNotFoundError as e:
        print(f"  ✓ Correctly raised EmbeddingNotFoundError: {str(e)[:80]}...")


def test_energy_with_skip_flag():
    """Test skip_on_no_embedding flag functionality"""
    print("\nTest 4: Skip validation when skip_on_no_embedding=True")

    validator = MathematicalValidator(tolerance=0.5)
    model = MockModelWithoutEmbedding(input_dim=8, output_dim=64)

    # Token IDs input (incompatible)
    x = torch.randint(0, 100, (2, 8))

    result = validator.validate_energy_conservation(model, x, skip_on_no_embedding=True)

    assert result['validation_method'] == 'skipped', \
        f"Expected 'skipped', got {result['validation_method']}"
    assert result['input_energy'] is None, "Input energy should be None when skipped"
    assert result['output_energy'] is not None, "Output energy should still be calculated"
    assert result['is_conserved'] is None, "is_conserved should be None when skipped"
    assert 'skip_reason' in result, "Should have skip_reason field"

    print(f"  ✓ Validation skipped correctly")
    print(f"  ✓ Skip reason: {result['skip_reason'][:80]}...")
    print(f"  ✓ Output energy still computed: {result['output_energy']:.6f}")


def test_no_fallback_to_output():
    """Test that we never use output as input (the old bug)"""
    print("\nTest 5: Verify no fallback to output as input")

    validator = MathematicalValidator(tolerance=0.5)
    model = MockModelWithoutEmbedding(input_dim=8, output_dim=64)

    # Token IDs input
    x = torch.randint(0, 100, (2, 8))

    # With skip flag, should not use output as input
    result = validator.validate_energy_conservation(model, x, skip_on_no_embedding=True)

    # The old buggy behavior would set input_energy = output_energy
    # New behavior: input_energy should be None when skipped
    assert result['input_energy'] is None, \
        "Input energy should be None (not output energy) when validation skipped"

    print(f"  ✓ Input energy is None (not fallback to output)")
    print(f"  ✓ Old bug is fixed")


if __name__ == "__main__":
    print("=" * 70)
    print("ΨQRH Mathematical Validation Fix - Unit Tests")
    print("=" * 70)

    tests = [
        test_energy_with_embedding,
        test_energy_with_float_input,
        test_energy_without_embedding_raises_error,
        test_energy_with_skip_flag,
        test_no_fallback_to_output
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
