"""
Fractal Quantum Text Generator
==============================

Complete text generation system using the Fractal Quantum Transformer.
This demonstrates the full pipeline from token IDs to generated text,
using the physically-grounded embedding system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architecture.fractal_quantum_transformer import FractalQuantumTransformer

def create_simple_dataset():
    """Create a simple dataset for demonstration."""
    texts = [
        "hydrogen is the lightest element",
        "oxygen is essential for life",
        "carbon forms organic compounds",
        "nitrogen makes up most of the air",
        "iron is a magnetic metal",
        "carbon dioxide is a gas",
        "water is essential for life",
        "sodium chloride is salt",
        "gold is a precious metal",
        "silver conducts electricity"
    ]
    return texts

class FractalQuantumTextGenerator:
    """Text generator using the Fractal Quantum Transformer."""

    def __init__(self, vocab_size, embed_dim=32, d_model=128, nhead=4, num_layers=2):
        self.vocab_size = vocab_size
        self.model = FractalQuantumTransformer(
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

    def generate(self, prompt_ids, max_length=50, temperature=1.0,
                top_k=None, top_p=None, do_sample=True):
        """
        Generate text using the Fractal Quantum Transformer.

        Args:
            prompt_ids: Initial token IDs
            max_length: Maximum length of generated sequence
            temperature: Controls randomness (higher = more random)
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated token IDs
        """
        self.model.eval()
        generated_ids = prompt_ids.clone()

        with torch.no_grad():
            for _ in range(max_length - len(prompt_ids)):
                # Get logits for next token
                logits = self.model(generated_ids.unsqueeze(0))  # [1, seq, vocab]
                next_token_logits = logits[0, -1, :]  # [vocab]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')

                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('inf')

                # Get probabilities
                probs = F.softmax(next_token_logits, dim=-1)

                # Sample next token
                if do_sample:
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = torch.argmax(probs, dim=-1, keepdim=True)

                # Add to sequence
                generated_ids = torch.cat([generated_ids, next_token_id.squeeze().unsqueeze(0)])

                # Stop if end token (assuming 0 is special token)
                if next_token_id.item() == 0:
                    break

        return generated_ids

def train_fractal_quantum_generator(num_epochs=100, learning_rate=0.001):
    """Train the Fractal Quantum Text Generator."""
    print("🚀 Training Fractal Quantum Text Generator...")

    # Prepare dataset
    dataset = create_simple_dataset()
    all_chars = sorted(list(set("".join(dataset))))
    vocab_size = len(all_chars)
    char_to_id = {ch: i for i, ch in enumerate(all_chars)}
    id_to_char = {i: ch for i, ch in enumerate(all_chars)}

    print(f"  - Vocabulário: {vocab_size} caracteres")
    print(f"  - Dataset: {len(dataset)} frases")

    # Initialize generator
    generator = FractalQuantumTextGenerator(
        vocab_size=vocab_size,
        embed_dim=32,
        d_model=128,
        nhead=4,
        num_layers=2
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(generator.model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"  - Treinando por {num_epochs} épocas...")
    print("-" * 60)

    # Training loop
    for epoch in range(num_epochs):
        generator.model.train()
        total_loss = 0

        for sentence in dataset:
            # Convert to token IDs
            input_ids = torch.tensor([char_to_id[c] for c in sentence[:-1]], dtype=torch.long)
            target_ids = torch.tensor([char_to_id[c] for c in sentence[1:]], dtype=torch.long)

            # Forward pass
            optimizer.zero_grad()
            logits = generator.model(input_ids.unsqueeze(0))  # [1, seq-1, vocab]

            # Calculate loss
            loss = criterion(logits.view(-1, vocab_size), target_ids)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)

        if epoch % 10 == 0:
            print(f"  Época {epoch:03d}/{num_epochs} | Perda: {avg_loss:.4f}")

    print("-" * 60)
    print(f"\033[92m✅ Treinamento Concluído!\033[0m")

    return generator, char_to_id, id_to_char

def demonstrate_generation(generator, char_to_id, id_to_char):
    """Demonstrate text generation capabilities."""
    print("\n🤖 Demonstrando Geração de Texto com Fractal Quantum...")

    prompts = ["carbon ", "oxygen ", "water "]

    for prompt in prompts:
        print(f"\n  - Prompt: '{prompt}'")

        # Convert prompt to IDs
        prompt_ids = torch.tensor([char_to_id[c] for c in prompt], dtype=torch.long)

        # Generate text
        generated_ids = generator.generate(
            prompt_ids,
            max_length=20,
            temperature=0.8,
            top_k=3,
            do_sample=True
        )

        # Convert back to text
        generated_text = "".join([id_to_char[id.item()] for id in generated_ids])

        print(f"  - Geração: '{generated_text}'")

    print("=" * 60)

if __name__ == "__main__":
    # Train and demonstrate
    generator, char_to_id, id_to_char = train_fractal_quantum_generator(num_epochs=50)
    demonstrate_generation(generator, char_to_id, id_to_char)