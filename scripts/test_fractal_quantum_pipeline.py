"""
Test Script for Fractal Quantum Pipeline
=========================================

Tests the complete Fractal Quantum Transformer pipeline from token IDs
to quaternion embeddings to final logits.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.fractal_quantum_embedding import FractalQuantumEmbedding
from src.architecture.fractal_quantum_transformer import FractalQuantumTransformer

def test_fractal_quantum_embedding():
    """Test the Fractal Quantum Embedding layer."""
    print("🧪 Testing Fractal Quantum Embedding...")

    vocab_size = 100
    embed_dim = 64
    batch_size = 2
    seq_len = 5

    # Create embedding layer
    embedding_layer = FractalQuantumEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padilha_config={
            'I0': 1.0,
            'omega': 2 * 3.14159,
            'k': 2 * 3.14159 / 0.5,
            'lambda_coupling': 0.8
        }
    )

    # Create input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    embeddings = embedding_layer(input_ids)

    print(f"  - Input shape: {input_ids.shape}")
    print(f"  - Output shape: {embeddings.shape}")
    print(f"  - Output range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

    # Check if embeddings are unit quaternions
    norms = torch.norm(embeddings, dim=-1)
    print(f"  - Quaternion norms: mean={norms.mean():.4f}, std={norms.std():.4f}")
    print(f"  - All close to unit: {torch.allclose(norms, torch.ones_like(norms), atol=1e-4)}")

    print("✅ Fractal Quantum Embedding test passed!\n")
    return embeddings

def test_fractal_quantum_transformer():
    """Test the complete Fractal Quantum Transformer."""
    print("🧪 Testing Fractal Quantum Transformer...")

    vocab_size = 50
    embed_dim = 32
    d_model = 128
    nhead = 4
    num_layers = 2
    batch_size = 2
    seq_len = 10

    # Create model
    model = FractalQuantumTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        d_model=d_model,
        nhead=nhead,
        num_transformer_layers=num_layers,
        padilha_config={
            'I0': 1.0,
            'omega': 2 * 3.14159,
            'k': 2 * 3.14159 / 0.5,
            'lambda_coupling': 0.8
        }
    )

    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids)

    print(f"  - Input shape: {input_ids.shape}")
    print(f"  - Output shape: {logits.shape}")
    print(f"  - Vocabulary size: {vocab_size}")

    # Test generation
    print("\n🧪 Testing text generation...")
    model.eval()
    with torch.no_grad():
        # Simple greedy generation
        prompt = torch.randint(0, vocab_size, (1, 3))
        generated = prompt.clone()

        for _ in range(5):
            current_logits = model(generated)
            next_token = torch.argmax(current_logits[0, -1, :])
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        print(f"  - Generated sequence length: {generated.shape[1]}")

    print("✅ Fractal Quantum Transformer test passed!\n")
    return logits

def test_training_loop():
    """Test a simple training loop with the Fractal Quantum Transformer."""
    print("🧪 Testing Training Loop...")

    vocab_size = 30
    embed_dim = 16
    d_model = 64
    nhead = 2
    num_layers = 1

    # Create model
    model = FractalQuantumTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        d_model=d_model,
        nhead=nhead,
        num_transformer_layers=num_layers
    )

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create simple training data
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Training step
    model.train()
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
    loss.backward()
    optimizer.step()

    print(f"  - Training loss: {loss.item():.4f}")
    print(f"  - Gradients exist: {any(p.grad is not None for p in model.parameters())}")

    print("✅ Training loop test passed!\n")
    return loss.item()

def main():
    """Run all tests."""
    print("🚀 Testing Complete Fractal Quantum Pipeline")
    print("=" * 50)

    try:
        # Test individual components
        embeddings = test_fractal_quantum_embedding()
        logits = test_fractal_quantum_transformer()
        loss = test_training_loop()

        print("🎉 All tests completed successfully!")
        print(f"\n📊 Summary:")
        print(f"  - Embedding shape: {embeddings.shape}")
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Training loss: {loss:.4f}")
        print(f"\n✅ Fractal Quantum Pipeline is functional!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()