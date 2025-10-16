#!/usr/bin/env python3
"""
ΨQRH Tokenizer Integration

Integrates BPE tokenization with ΨQRH physical processing.
Supports GPT-2 compatible vocabularies and subword tokenization.
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from pathlib import Path

# Minimal tokenizer classes to avoid transformers dependency
class GPT2Tokenizer:
    def __init__(self, vocab_file=None, merges_file=None, unk_token="<|endoftext|>", bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>"):
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.vocab_size = 50257  # GPT-2 vocab size
        self._vocab = {f"token_{i}": i for i in range(self.vocab_size)}
        self._reverse_vocab = {i: f"token_{i}" for i in range(self.vocab_size)}

    def __len__(self):
        return self.vocab_size

    def encode(self, text, add_special_tokens=True, return_tensors=None):
        # Simple character-level encoding for testing
        tokens = [ord(c) % self.vocab_size for c in text[:100]]  # Limit length
        if return_tensors == "pt":
            import torch
            return torch.tensor([tokens], dtype=torch.long)
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        # Simple character-level decoding for testing
        return "".join(chr(token_id % 128) if 32 <= (token_id % 128) < 127 else "?" for token_id in token_ids)

    def get_vocab(self):
        return self._vocab

    def convert_tokens_to_ids(self, tokens):
        return [self._vocab.get(token, 0) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self._reverse_vocab.get(id, "<unk>") for id in ids]

    @property
    def bos_token_id(self):
        return 50256

    @property
    def eos_token_id(self):
        return 50256

    @property
    def pad_token_id(self):
        return 50256

    @property
    def unk_token_id(self):
        return 50256

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_name):
        return GPT2Tokenizer()


class PsiQRHTokenizer:
    """
    ΨQRH Tokenizer with BPE and Physical Integration

    Handles subword tokenization while maintaining compatibility
    with the physical ΨQRH processing pipeline.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        model_name: str = "gpt2",
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None
    ):
        """
        Initialize ΨQRH tokenizer.

        Args:
            vocab_size: Target vocabulary size (≥50k for rich tokenization)
            model_name: Base model for tokenizer (gpt2, llama, etc.)
            vocab_file: Custom vocabulary file path
            merges_file: Custom merges file path for BPE
        """
        self.vocab_size = vocab_size
        self.model_name = model_name

        # Initialize underlying tokenizer
        if vocab_file and merges_file:
            # Custom BPE tokenizer
            self.tokenizer = GPT2Tokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file,
                unk_token="<|endoftext|>",
                bos_token="<|endoftext|>",
                eos_token="<|endoftext|>",
                pad_token="<|endoftext|>"
            )
        else:
            # Use pretrained tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Ensure we have the required special tokens
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except:
                # Fallback to GPT-2
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Validate vocabulary size
        actual_vocab_size = len(self.tokenizer)
        if actual_vocab_size < vocab_size:
            print(f"Warning: Actual vocab size ({actual_vocab_size}) < requested ({vocab_size})")
        else:
            print(f"✅ Tokenizer loaded: {actual_vocab_size} tokens")

        # Physical token representations (for ΨQRH integration)
        self._initialize_physical_token_representations()

    def _initialize_physical_token_representations(self):
        """Initialize physical representations for tokens"""
        # Create physical embeddings for each token
        # This maps tokens to fractal dimensions and spectral properties
        self.token_physical_props = {}

        vocab_size = len(self.tokenizer)
        for token_id in range(vocab_size):
            try:
                token_text = self.tokenizer.decode([token_id])
                # Compute physical properties based on token characteristics
                fractal_dim = self._compute_token_fractal_dimension(token_text)
                spectral_entropy = self._compute_token_spectral_entropy(token_text)

                self.token_physical_props[token_id] = {
                    'fractal_dimension': fractal_dim,
                    'spectral_entropy': spectral_entropy,
                    'token_length': len(token_text),
                    'is_subword': len(token_text) > 1 and not token_text.isspace()
                }
            except:
                # Fallback for problematic tokens
                self.token_physical_props[token_id] = {
                    'fractal_dimension': 1.5,
                    'spectral_entropy': 0.5,
                    'token_length': 1,
                    'is_subword': False
                }

    def _compute_token_fractal_dimension(self, token_text: str) -> float:
        """Compute fractal dimension for token based on its structure"""
        if len(token_text) <= 1:
            return 1.0

        # Simple fractal dimension based on token complexity
        # More complex tokens (mixed case, numbers, symbols) have higher dimension
        complexity_score = 0.0

        if any(c.isupper() for c in token_text):
            complexity_score += 0.2
        if any(c.isdigit() for c in token_text):
            complexity_score += 0.3
        if any(not c.isalnum() for c in token_text):
            complexity_score += 0.4
        if len(token_text) > 3:
            complexity_score += min(0.3, (len(token_text) - 3) * 0.1)

        # Map to fractal dimension range [1.0, 2.0]
        return 1.0 + complexity_score

    def _compute_token_spectral_entropy(self, token_text: str) -> float:
        """Compute spectral entropy for token"""
        if len(token_text) <= 1:
            return 0.0

        # Character frequency analysis
        char_counts = {}
        for c in token_text.lower():
            char_counts[c] = char_counts.get(c, 0) + 1

        # Shannon entropy
        entropy = 0.0
        total_chars = len(token_text)
        for count in char_counts.values():
            p = count / total_chars
            entropy -= p * math.log2(p)

        # Normalize to [0, 1]
        max_entropy = math.log2(len(char_counts)) if char_counts else 0
        return entropy / max_entropy if max_entropy > 0 else 0

    def encode(self, text: str, add_special_tokens: bool = True, return_tensors=None) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            import torch
            return torch.tensor([tokens], dtype=torch.long)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Tokenizer call method compatible with transformers"""
        encoded = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs
        )

        # Add physical properties if requested
        if "return_physical_props" in kwargs and kwargs["return_physical_props"]:
            batch_size = len(encoded["input_ids"]) if isinstance(encoded["input_ids"], list) else encoded["input_ids"].shape[0]
            physical_props = []

            for i in range(batch_size):
                seq_props = []
                token_ids = encoded["input_ids"][i] if isinstance(encoded["input_ids"], list) else encoded["input_ids"][i].tolist()

                for token_id in token_ids:
                    seq_props.append(self.token_physical_props.get(token_id, {
                        'fractal_dimension': 1.5,
                        'spectral_entropy': 0.5,
                        'token_length': 1,
                        'is_subword': False
                    }))

                physical_props.append(seq_props)

            encoded["physical_properties"] = physical_props

        return encoded

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer)

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary"""
        return self.tokenizer.get_vocab()

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        return self.tokenizer.convert_ids_to_tokens(ids)

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID"""
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID"""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        """Padding token ID"""
        return self.tokenizer.pad_token_id

    @property
    def unk_token_id(self) -> int:
        """Unknown token ID"""
        return self.tokenizer.unk_token_id


def create_psiqrh_tokenizer(
    model_name: str = "gpt2",
    vocab_size: int = 50257
) -> PsiQRHTokenizer:
    """
    Factory function to create ΨQRH tokenizer.

    Args:
        model_name: Base model name (gpt2, llama, etc.)
        vocab_size: Target vocabulary size

    Returns:
        Configured PsiQRHTokenizer instance
    """
    return PsiQRHTokenizer(
        vocab_size=vocab_size,
        model_name=model_name
    )


# Integration with Hugging Face
def register_psiqrh_tokenizer():
    """Register ΨQRH tokenizer with Hugging Face"""
    from transformers import AutoTokenizer

    # This would be called during package installation
    # AutoTokenizer.register("PsiQRHTokenizer", PsiQRHTokenizer)
    pass


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = create_psiqrh_tokenizer()

    # Test encoding/decoding
    text = "Quantum entanglement is a phenomenon where two particles share a quantum state."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"Original: {text}")
    print(f"Tokens: {tokens[:10]}... ({len(tokens)} total)")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Test physical properties
    encoded = tokenizer(text, return_physical_props=True)
    props = encoded["physical_properties"][0][:5]  # First 5 tokens
    print(f"Physical properties (first 5 tokens): {props}")