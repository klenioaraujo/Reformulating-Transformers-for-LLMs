#!/usr/bin/env python3
"""
Quantum Vocabulary Integration for ΨQRH Pipeline
================================================

Integrates the quantum-native vocabulary with the ΨQRH pipeline,
eliminating GPT-2 dependencies and enabling autonomous operation.
"""

import torch
import json
from typing import Dict, List


class QuantumVocabularyIntegration:
    """
    Integrates quantum-native vocabulary with ΨQRH pipeline
    """

    def __init__(self, vocab_path: str = "quantum_native_vocab.json"):
        self.vocab_path = vocab_path
        self.quantum_vocab = None
        self.token_to_id = None
        self.id_to_token = None
        self.vocab_size = 0

        self._load_quantum_vocabulary()

    def _load_quantum_vocabulary(self):
        """Load quantum-native vocabulary"""
        print(f"📚 Loading quantum-native vocabulary from {self.vocab_path}...")

        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        self.quantum_vocab = vocab_data.get('quantum_vocabulary', {})
        self.token_to_id = vocab_data.get('token_to_id', {})
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.quantum_vocab)

        print(f"✅ Loaded quantum vocabulary: {self.vocab_size} tokens")
        print(f"🎯 Autonomous: True")
        print(f"🔗 GPT-2 Dependency: False")

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.vocab_size

    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize text using quantum vocabulary

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        # Simple tokenization - split by whitespace and map to IDs
        tokens = text.split()
        token_ids = []

        for token in tokens:
            # Try exact match first
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Try character-level tokenization for unknown tokens
                for char in token:
                    if char in self.token_to_id:
                        token_ids.append(self.token_to_id[char])
                    else:
                        # Fallback to unknown token
                        token_ids.append(0)  # Assuming 0 is unknown

        return token_ids

    def detokenize_text(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text

        Args:
            token_ids: List of token IDs

        Returns:
            Detokenized text
        """
        tokens = []

        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append("<UNK>")

        return ' '.join(tokens)

    def get_quantum_properties(self, token: str) -> Dict:
        """
        Get quantum properties for a token

        Args:
            token: Token string

        Returns:
            Dictionary of quantum properties
        """
        if token in self.quantum_vocab:
            return self.quantum_vocab[token]['quantum_properties']
        else:
            return {
                'energy_level': 0.0,
                'coherence': 0.5,
                'entropy': 0.0,
                'spin': 0.0,
                'mass': 1.0,
                'charge': 0.0,
                'frequency': 1.0,
                'wavelength': 1.0
            }

    def create_quantum_embedding_matrix(self, embed_dim: int) -> torch.Tensor:
        """
        Create quantum embedding matrix

        Args:
            embed_dim: Embedding dimension

        Returns:
            Embedding matrix tensor
        """
        # Create embedding matrix with quantum properties
        embedding_matrix = torch.randn(self.vocab_size, embed_dim)

        # Scale embeddings by quantum properties
        for token, info in self.quantum_vocab.items():
            token_id = info['token_id']
            props = info['quantum_properties']

            # Scale embedding by energy level
            energy_scale = props['energy_level'] + 0.5  # Range: 0.5 to 1.5
            embedding_matrix[token_id] *= energy_scale

        return embedding_matrix

    def get_integration_stats(self) -> Dict:
        """Get integration statistics"""
        return {
            'vocab_size': self.vocab_size,
            'token_mappings': len(self.token_to_id),
            'quantum_properties': len(self.quantum_vocab),
            'is_autonomous': True,
            'has_gpt2_dependency': False,
            'integration_status': 'active'
        }


def integrate_with_psiqrh(pipeline_instance) -> QuantumVocabularyIntegration:
    """
    Main integration function for ΨQRH pipeline

    Args:
        pipeline_instance: ΨQRH pipeline instance

    Returns:
        QuantumVocabularyIntegration instance
    """
    print("\n🔗 Integrating Quantum Vocabulary with ΨQRH Pipeline...")

    # Initialize integration
    integrator = QuantumVocabularyIntegration()

    # Update pipeline configuration
    if hasattr(pipeline_instance, 'config'):
        pipeline_instance.config['vocab_size'] = integrator.get_vocab_size()
        pipeline_instance.config['quantum_vocab'] = True
        pipeline_instance.config['gpt2_dependency'] = False

    print("✅ Quantum vocabulary integration complete!")

    # Show integration stats
    stats = integrator.get_integration_stats()
    print(f"📊 Integration Statistics:")
    print(f"   Vocabulary size: {stats['vocab_size']:,}")
    print(f"   Token mappings: {stats['token_mappings']:,}")
    print(f"   Quantum properties: {stats['quantum_properties']:,}")
    print(f"   Autonomous: {stats['is_autonomous']}")
    print(f"   GPT-2 Dependency: {stats['has_gpt2_dependency']}")

    return integrator


if __name__ == "__main__":
    # Test the integration
    print("🧪 Testing Quantum Vocabulary Integration")
    print("=" * 50)

    integrator = QuantumVocabularyIntegration()

    # Test tokenization
    test_text = "quantum mechanics is fascinating"
    token_ids = integrator.tokenize_text(test_text)
    print(f"\n🔬 Tokenization Test:")
    print(f"   Input: '{test_text}'")
    print(f"   Token IDs: {token_ids}")

    # Test detokenization
    reconstructed = integrator.detokenize_text(token_ids)
    print(f"\n🔬 Detokenization Test:")
    print(f"   Token IDs: {token_ids}")
    print(f"   Output: '{reconstructed}'")

    # Test quantum properties
    test_token = "quantum"
    props = integrator.get_quantum_properties(test_token)
    print(f"\n🔬 Quantum Properties Test:")
    print(f"   Token: '{test_token}'")
    print(f"   Energy Level: {props['energy_level']:.3f}")
    print(f"   Coherence: {props['coherence']:.3f}")
    print(f"   Entropy: {props['entropy']:.3f}")

    print("\n✅ Quantum vocabulary integration test completed!")
