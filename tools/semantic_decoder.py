#!/usr/bin/env python3
"""
ΨQRH Semantic Beam Search Decoder
==================================

Advanced decoder that uses beam search with semantic coherence to generate
contextually relevant and grammatically correct text from quantum states.

This decoder explores multiple decoding paths and prefers sequences that form
valid words, significantly improving semantic quality over greedy decoding.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Set, Optional
from pathlib import Path
import json
import heapq
from dataclasses import dataclass


@dataclass
class BeamCandidate:
    """Represents a candidate sequence in the beam search."""
    sequence: List[str]  # The sequence of characters/tokens
    score: float         # Combined score (model + semantic bonus)
    model_score: float   # Score from the quantum model
    semantic_score: float # Bonus for forming valid word prefixes
    position: int        # Current position in the sequence


class SemanticBeamSearchDecoder:
    """
    Semantic Beam Search Decoder for ΨQRH pipeline.

    This decoder uses beam search to explore multiple decoding paths and
    incorporates semantic coherence by giving bonuses to sequences that
    form prefixes of valid words from a dictionary.
    """

    def __init__(self, beam_width: int = 5, dictionary_path: str = "data/english_dictionary.txt",
                 semantic_bonus_weight: float = 0.3, max_sequence_length: int = 100):
        """
        Initialize the Semantic Beam Search Decoder.

        Args:
            beam_width: Number of candidates to maintain in the beam
            dictionary_path: Path to the dictionary file (one word per line)
            semantic_bonus_weight: Weight for semantic coherence bonus (0.0 to 1.0)
            max_sequence_length: Maximum length of generated sequences
        """
        self.beam_width = beam_width
        self.dictionary_path = Path(dictionary_path)
        self.semantic_bonus_weight = semantic_bonus_weight
        self.max_sequence_length = max_sequence_length

        # Load the dictionary
        self.dictionary = self._load_dictionary()
        print(f"📚 Loaded dictionary with {len(self.dictionary)} words")

        # Create prefix sets for efficient prefix matching
        self.prefixes = self._build_prefix_sets()
        print(f"🔍 Built prefix sets for semantic coherence checking")

    def _load_dictionary(self) -> Set[str]:
        """Load the English dictionary from file."""
        dictionary = set()

        try:
            if self.dictionary_path.exists():
                with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word and len(word) >= 2:  # Only words with 2+ characters
                            dictionary.add(word)
            else:
                print(f"⚠️  Dictionary file not found: {self.dictionary_path}")
                print("   Using built-in basic dictionary...")

                # Fallback basic dictionary
                basic_words = {
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                    'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                    'can', 'shall', 'must', 'let', 'make', 'go', 'take', 'come', 'see', 'know', 'get',
                    'give', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'think', 'say', 'mean', 'want',
                    'use', 'need', 'help', 'turn', 'run', 'move', 'live', 'try', 'call', 'keep', 'begin',
                    'quantum', 'mechanics', 'physics', 'science', 'theory', 'model', 'system', 'process',
                    'language', 'text', 'word', 'sentence', 'meaning', 'context', 'semantic', 'coherence'
                }
                dictionary.update(basic_words)

        except Exception as e:
            print(f"❌ Error loading dictionary: {e}")
            # Minimal fallback
            dictionary = {'the', 'a', 'quantum', 'mechanics', 'physics'}

        return dictionary

    def _build_prefix_sets(self) -> Dict[int, Set[str]]:
        """Build sets of valid prefixes for different lengths."""
        prefixes = {}

        for word in self.dictionary:
            for i in range(1, len(word) + 1):
                prefix = word[:i]
                if i not in prefixes:
                    prefixes[i] = set()
                prefixes[i].add(prefix)

        return prefixes

    def _calculate_semantic_bonus(self, sequence: List[str]) -> float:
        """
        Calculate semantic coherence bonus for a sequence.

        Args:
            sequence: List of characters forming the current sequence

        Returns:
            Semantic bonus score (0.0 to 1.0)
        """
        if not sequence:
            return 0.0

        # Join the sequence into a string
        text = ''.join(sequence).lower()

        # Check if the entire sequence forms a valid word
        if text in self.dictionary:
            return 1.0

        # Check if it's a valid prefix of any word
        seq_len = len(text)
        if seq_len in self.prefixes and text in self.prefixes[seq_len]:
            # Bonus decreases with prefix length (prefer longer prefixes)
            return 0.5 * (seq_len / max(len(w) for w in self.dictionary if w.startswith(text)))

        # Check for word boundaries (spaces indicate completed words)
        words = text.split()
        if len(words) > 1:
            # Calculate bonus based on completed words
            completed_words = sum(1 for word in words[:-1] if word in self.dictionary)
            last_word_bonus = 0.5 if words[-1] in self.dictionary else 0.0
            return (completed_words + last_word_bonus) / len(words)

        return 0.0

    def decode(self, top_k_predictions: List[List[Tuple[str, float]]],
               max_length: Optional[int] = None) -> str:
        """
        Decode using semantic beam search.

        Args:
            top_k_predictions: List of lists, where each inner list contains
                              (character, score) tuples for each position
            max_length: Maximum sequence length (overrides instance setting)

        Returns:
            The best decoded sequence as a string
        """
        if not top_k_predictions:
            return ""

        max_len = max_length or self.max_sequence_length
        sequence_length = len(top_k_predictions)

        # Initialize beam with empty sequence
        initial_candidate = BeamCandidate(
            sequence=[],
            score=0.0,
            model_score=0.0,
            semantic_score=0.0,
            position=0
        )

        beam = [initial_candidate]

        # Beam search through each position
        for pos in range(min(sequence_length, max_len)):
            if not beam:
                break

            new_candidates = []

            for candidate in beam:
                # Get predictions for this position
                if pos < len(top_k_predictions):
                    position_predictions = top_k_predictions[pos]
                else:
                    # If we run out of predictions, use a default continuation
                    position_predictions = [('.', 0.1), (' ', 0.05)]

                # Expand each candidate with each possible next character
                for char, model_prob in position_predictions:
                    # Create new sequence
                    new_sequence = candidate.sequence + [char]
                    new_model_score = candidate.model_score + model_prob

                    # Calculate semantic bonus
                    semantic_bonus = self._calculate_semantic_bonus(new_sequence)
                    new_semantic_score = candidate.semantic_score + semantic_bonus

                    # Combined score
                    combined_score = (new_model_score +
                                    self.semantic_bonus_weight * new_semantic_score)

                    # Create new candidate
                    new_candidate = BeamCandidate(
                        sequence=new_sequence,
                        score=combined_score,
                        model_score=new_model_score,
                        semantic_score=new_semantic_score,
                        position=pos + 1
                    )

                    new_candidates.append(new_candidate)

            # Select top beam_width candidates
            beam = heapq.nlargest(self.beam_width, new_candidates,
                                key=lambda x: x.score)

            # Early stopping if we have a very good candidate
            best_score = beam[0].score if beam else 0
            if best_score > 10.0:  # Threshold for very confident sequences
                break

        # Return the best sequence
        if beam:
            best_candidate = max(beam, key=lambda x: x.score)
            result = ''.join(best_candidate.sequence)

            # Clean up the result (remove trailing punctuation if it makes sense)
            result = self._post_process_sequence(result)

            return result

        return ""

    def _post_process_sequence(self, sequence: str) -> str:
        """Post-process the decoded sequence for better readability."""
        if not sequence:
            return sequence

        # Remove excessive repeated characters
        cleaned = []
        prev_char = None
        repeat_count = 0

        for char in sequence:
            if char == prev_char:
                repeat_count += 1
                if repeat_count <= 2:  # Allow up to 2 repeats
                    cleaned.append(char)
            else:
                repeat_count = 1
                cleaned.append(char)
                prev_char = char

        return ''.join(cleaned)

    def get_semantic_quality_score(self, text: str) -> Dict[str, float]:
        """
        Calculate semantic quality metrics for a given text.

        Args:
            text: The text to evaluate

        Returns:
            Dictionary with semantic quality metrics
        """
        words = text.lower().split()
        total_words = len(words)

        if total_words == 0:
            return {
                'word_validity_ratio': 0.0,
                'average_word_length': 0.0,
                'semantic_coherence_score': 0.0
            }

        # Calculate word validity ratio
        valid_words = sum(1 for word in words if word in self.dictionary)
        word_validity_ratio = valid_words / total_words

        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / total_words

        # Calculate semantic coherence (combination of validity and length)
        semantic_coherence = word_validity_ratio * (avg_word_length / 10.0)  # Normalize length

        return {
            'word_validity_ratio': word_validity_ratio,
            'average_word_length': avg_word_length,
            'semantic_coherence_score': semantic_coherence,
            'valid_words': valid_words,
            'total_words': total_words
        }


def create_semantic_decoder(beam_width: int = 5,
                          dictionary_path: str = "data/english_dictionary.txt") -> SemanticBeamSearchDecoder:
    """
    Factory function to create a SemanticBeamSearchDecoder.

    Args:
        beam_width: Number of candidates to maintain in beam search
        dictionary_path: Path to the dictionary file

    Returns:
        Configured SemanticBeamSearchDecoder instance
    """
    return SemanticBeamSearchDecoder(
        beam_width=beam_width,
        dictionary_path=dictionary_path
    )


# Example usage and testing
if __name__ == "__main__":
    # Create a test decoder
    decoder = create_semantic_decoder(beam_width=3)

    # Example top-k predictions (simulated)
    # Each position has a list of (character, score) tuples
    test_predictions = [
        [('Q', 0.8), ('q', 0.6), (' ', 0.2)],  # Position 0
        [('u', 0.9), ('U', 0.1), ('a', 0.3)],  # Position 1
        [('a', 0.7), ('o', 0.4), ('i', 0.2)],  # Position 2
        [('n', 0.8), ('m', 0.3), ('t', 0.1)],  # Position 3
        [('t', 0.6), (' ', 0.5), ('s', 0.2)],  # Position 4
        [('u', 0.7), (' ', 0.4), ('m', 0.1)],  # Position 5
    ]

    # Decode using beam search
    result = decoder.decode(test_predictions, max_length=10)

    print(f"🎯 Decoded sequence: '{result}'")

    # Calculate semantic quality
    quality = decoder.get_semantic_quality_score(result)
    print("📊 Semantic Quality Metrics:")
    for key, value in quality.items():
        if isinstance(value, float):
            print(".3f")
        else:
            print(f"   {key}: {value}")