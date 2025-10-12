#!/usr/bin/env python3
"""
Native Vocabulary Generator for ΨQRH Pipeline
=============================================

Creates a native vocabulary from scratch by analyzing a text corpus,
completely decoupling ΨQRH from external vocabularies like GPT-2.

This implements true vocabulary autonomy - the system builds its own
linguistic foundation based on the training data.
"""

import json
import os
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional
import re


class NativeVocabularyGenerator:
    """
    Generates a native vocabulary from text corpus analysis.

    This class analyzes raw text data to build:
    - Token-to-ID mapping
    - Frequency statistics
    - Character-level and word-level vocabularies
    - Special tokens for ΨQRH operations
    """

    def __init__(self, min_freq: int = 1, max_vocab_size: Optional[int] = None):
        """
        Initialize the vocabulary generator.

        Args:
            min_freq: Minimum frequency for a token to be included
            max_vocab_size: Maximum vocabulary size (None for unlimited)
        """
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        # Special tokens for ΨQRH operations
        self.special_tokens = {
            '<pad>': 0,      # Padding token
            '<unk>': 1,      # Unknown token
            '<bos>': 2,      # Beginning of sequence
            '<eos>': 3,      # End of sequence
            '<mask>': 4,     # Masking token for training
        }

        print("🧠 Native Vocabulary Generator initialized")
        print(f"   📊 Min frequency: {min_freq}")
        print(f"   📊 Max vocab size: {max_vocab_size or 'unlimited'}")
        print(f"   🎯 Special tokens: {len(self.special_tokens)}")

    def analyze_corpus(self, corpus_path: str) -> Dict[str, any]:
        """
        Analyze a text corpus to extract vocabulary statistics.

        Args:
            corpus_path: Path to the text corpus file

        Returns:
            Dictionary with corpus analysis results
        """
        print(f"📖 Analyzing corpus: {corpus_path}")

        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        # Read the corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Basic statistics
        total_chars = len(text)
        total_lines = text.count('\n') + 1

        # Character-level analysis
        char_counter = Counter(text)
        unique_chars = len(char_counter)

        # Word-level analysis (split by whitespace and punctuation)
        words = re.findall(r'\b\w+\b', text.lower())
        word_counter = Counter(words)
        unique_words = len(word_counter)

        # Most common characters and words
        top_chars = char_counter.most_common(20)
        top_words = word_counter.most_common(20)

        analysis = {
            'total_characters': total_chars,
            'total_lines': total_lines,
            'unique_characters': unique_chars,
            'unique_words': unique_words,
            'top_characters': top_chars,
            'top_words': top_words,
            'character_frequencies': dict(char_counter),
            'word_frequencies': dict(word_counter.most_common(1000))  # Top 1000 words
        }

        print(f"   📊 Corpus stats: {total_chars} chars, {total_lines} lines")
        print(f"   🔤 Unique chars: {unique_chars}, words: {unique_words}")

        return analysis

    def build_vocabulary(self, corpus_path: str, output_path: str,
                        tokenization_method: str = 'character') -> Dict[str, any]:
        """
        Build a complete vocabulary from the corpus.

        Args:
            corpus_path: Path to the text corpus
            output_path: Where to save the vocabulary
            tokenization_method: 'character' or 'word' level tokenization

        Returns:
            Vocabulary metadata
        """
        print(f"🏗️ Building {tokenization_method}-level vocabulary...")

        # Analyze corpus first
        analysis = self.analyze_corpus(corpus_path)

        # Build token-to-ID mapping
        vocab = dict(self.special_tokens)  # Start with special tokens
        next_id = len(self.special_tokens)

        if tokenization_method == 'character':
            # Character-level vocabulary
            sorted_chars = sorted(analysis['character_frequencies'].keys())
            for char in sorted_chars:
                if analysis['character_frequencies'][char] >= self.min_freq:
                    vocab[char] = next_id
                    next_id += 1

        elif tokenization_method == 'word':
            # Word-level vocabulary
            for word, freq in analysis['word_frequencies'].items():
                if freq >= self.min_freq:
                    vocab[word] = next_id
                    next_id += 1

        else:
            raise ValueError(f"Unknown tokenization method: {tokenization_method}")

        # Apply vocabulary size limit if specified
        if self.max_vocab_size and len(vocab) > self.max_vocab_size:
            print(f"   ✂️ Trimming vocabulary to {self.max_vocab_size} tokens")

            # Keep special tokens, trim the rest
            special_vocab = {k: v for k, v in vocab.items() if k in self.special_tokens}
            regular_tokens = [(k, v) for k, v in vocab.items() if k not in self.special_tokens]

            # Sort by frequency (we'd need frequency info here)
            # For now, just take the first N regular tokens
            regular_vocab = dict(regular_tokens[:self.max_vocab_size - len(special_vocab)])

            vocab = {**special_vocab, **regular_vocab}

        # Create inverse mapping (ID to token)
        id_to_token = {v: k for k, v in vocab.items()}

        # Save vocabulary
        vocab_data = {
            'metadata': {
                'tokenization_method': tokenization_method,
                'min_frequency': self.min_freq,
                'max_vocab_size': self.max_vocab_size,
                'corpus_analysis': analysis,
                'special_tokens': list(self.special_tokens.keys()),
                'created_by': 'NativeVocabularyGenerator'
            },
            'vocab_size': len(vocab),
            'token_to_id': vocab,
            'id_to_token': id_to_token
        }

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Vocabulary saved to: {output_path}")
        print(f"   📊 Final vocab size: {len(vocab)} tokens")
        print(f"   🎯 Special tokens: {len(self.special_tokens)}")
        print(f"   🔤 Regular tokens: {len(vocab) - len(self.special_tokens)}")

        return vocab_data

    def load_vocabulary(self, vocab_path: str) -> Dict[str, any]:
        """
        Load a previously generated vocabulary.

        Args:
            vocab_path: Path to the vocabulary file

        Returns:
            Vocabulary data dictionary
        """
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        print(f"📚 Loaded vocabulary: {vocab_data['vocab_size']} tokens")
        return vocab_data

    def get_tokenizer_from_vocab(self, vocab_data: Dict[str, any]):
        """
        Create a tokenizer function from vocabulary data.

        Args:
            vocab_data: Vocabulary data from load_vocabulary()

        Returns:
            Tokenizer function
        """
        token_to_id = vocab_data['token_to_id']
        unk_token = vocab_data['metadata']['special_tokens'][1]  # <unk>

        def tokenize(text: str) -> List[int]:
            """Convert text to token IDs"""
            tokenization_method = vocab_data['metadata']['tokenization_method']

            if tokenization_method == 'character':
                tokens = list(text)
            elif tokenization_method == 'word':
                tokens = re.findall(r'\b\w+\b', text.lower())
            else:
                raise ValueError(f"Unknown tokenization method: {tokenization_method}")

            # Convert to IDs
            token_ids = []
            for token in tokens:
                token_id = token_to_id.get(token, token_to_id.get(unk_token, 1))
                token_ids.append(token_id)

            return token_ids

        return tokenize


def generate_native_vocabulary(corpus_path: str = "data/train.txt",
                             output_path: str = "data/native_vocab.json",
                             tokenization_method: str = "character",
                             min_freq: int = 1,
                             max_vocab_size: Optional[int] = None):
    """
    Convenience function to generate native vocabulary.

    Args:
        corpus_path: Path to corpus file
        output_path: Where to save vocabulary
        tokenization_method: 'character' or 'word'
        min_freq: Minimum token frequency
        max_vocab_size: Maximum vocabulary size
    """
    generator = NativeVocabularyGenerator(min_freq=min_freq, max_vocab_size=max_vocab_size)
    vocab_data = generator.build_vocabulary(corpus_path, output_path, tokenization_method)

    print("\n🎉 Native vocabulary generation complete!")
    print(f"   📁 Corpus: {corpus_path}")
    print(f"   💾 Output: {output_path}")
    print(f"   🔤 Method: {tokenization_method}")
    print(f"   📊 Size: {vocab_data['vocab_size']} tokens")

    return vocab_data


if __name__ == "__main__":
    # Generate character-level vocabulary from train.txt
    vocab_data = generate_native_vocabulary(
        corpus_path="data/train.txt",
        output_path="data/native_vocab.json",
        tokenization_method="character",
        min_freq=1,
        max_vocab_size=256  # ASCII range
    )

    # Test the tokenizer
    generator = NativeVocabularyGenerator()
    loaded_vocab = generator.load_vocabulary("data/native_vocab.json")
    tokenizer = generator.get_tokenizer_from_vocab(loaded_vocab)

    test_text = "ΨQRH framework"
    token_ids = tokenizer(test_text)
    print(f"\n🧪 Test tokenization:")
    print(f"   Text: '{test_text}'")
    print(f"   Tokens: {token_ids}")