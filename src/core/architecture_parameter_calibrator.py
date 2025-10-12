#!/usr/bin/env python3
"""
Architecture Parameter Calibrator for ΨQRH Pipeline
==================================================

Auto-calibrates architecture parameters based on input characteristics:

Parameters calibrated:
- embed_dim: Based on text complexity
- num_heads: Based on semantic diversity
- hidden_dim: Based on vocabulary richness
- num_layers: Based on required depth
"""

import torch
import math
from typing import Dict, Any, List
import re


class ArchitectureParameterCalibrator:
    """
    Calibrates architecture parameters based on input analysis
    """

    def __init__(self):
        """Initialize the architecture parameter calibrator"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze text complexity metrics

        Args:
            text: Input text

        Returns:
            Dict with complexity metrics
        """
        # Basic text statistics
        text_length = len(text)
        word_count = len(text.split())
        char_count = len([c for c in text if c.isalnum()])

        # Lexical diversity (unique words / total words)
        words = text.lower().split()
        unique_words = len(set(words))
        lexical_diversity = unique_words / max(word_count, 1)

        # Sentence complexity (average words per sentence)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_words_per_sentence = word_count / max(len(sentences), 1)

        # Character diversity
        unique_chars = len(set(text))
        char_diversity = unique_chars / 95.0  # Normalized by printable ASCII

        # Entropy-based complexity (Shannon entropy)
        if text:
            char_freq = {}
            for c in text:
                char_freq[c] = char_freq.get(c, 0) + 1

            entropy = 0
            for freq in char_freq.values():
                p = freq / len(text)
                entropy -= p * math.log2(p) if p > 0 else 0

            normalized_entropy = entropy / math.log2(len(text)) if len(text) > 1 else 0
        else:
            normalized_entropy = 0

        return {
            'text_length': text_length,
            'word_count': word_count,
            'lexical_diversity': lexical_diversity,
            'avg_words_per_sentence': avg_words_per_sentence,
            'char_diversity': char_diversity,
            'normalized_entropy': normalized_entropy
        }

    def analyze_semantic_diversity(self, text: str) -> float:
        """
        Analyze semantic diversity based on word categories

        Args:
            text: Input text

        Returns:
            Semantic diversity score (0-1)
        """
        words = text.lower().split()

        # Simple semantic categorization
        categories = {
            'nouns': ['the', 'a', 'an', 'this', 'that', 'these', 'those'],
            'verbs': ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'],
            'adjectives': ['good', 'bad', 'big', 'small', 'long', 'short', 'high', 'low', 'fast', 'slow'],
            'quantifiers': ['many', 'few', 'some', 'all', 'none', 'most', 'least'],
            'connectors': ['and', 'or', 'but', 'so', 'because', 'although', 'however']
        }

        category_counts = {}
        for category, keywords in categories.items():
            count = sum(1 for word in words if word in keywords)
            category_counts[category] = count

        # Diversity score based on category distribution
        total_keywords = sum(category_counts.values())
        if total_keywords == 0:
            return 0.5  # Default diversity

        # Shannon entropy of category distribution
        entropy = 0
        for count in category_counts.values():
            if count > 0:
                p = count / total_keywords
                entropy -= p * math.log2(p)

        # Normalize to 0-1 range
        max_entropy = math.log2(len(categories))
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0.5

        return diversity_score

    def analyze_vocabulary_richness(self, text: str) -> float:
        """
        Analyze vocabulary richness

        Args:
            text: Input text

        Returns:
            Vocabulary richness score (0-1)
        """
        words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        if not words:
            return 0.0

        unique_words = len(set(words))
        total_words = len(words)

        # Type-token ratio (lexical richness)
        ttr = unique_words / total_words

        # Hapax legomena ratio (words appearing only once)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
        hapax_ratio = hapax_count / total_words

        # Combined richness score
        richness_score = (ttr + hapax_ratio) / 2.0

        return richness_score

    def calibrate_embed_dim(self, complexity_metrics: Dict[str, float]) -> int:
        """
        Calibrate embedding dimension based on text complexity

        embed_dim = max(32, min(256, int(64 * complexity_factor)))
        """
        # Complexity factor based on multiple metrics
        entropy_factor = complexity_metrics['normalized_entropy']
        lexical_factor = complexity_metrics['lexical_diversity']
        length_factor = min(complexity_metrics['text_length'] / 1000.0, 1.0)  # Normalize length

        complexity_factor = (entropy_factor + lexical_factor + length_factor) / 3.0

        # Base dimension scaled by complexity
        embed_dim = int(64 * (0.5 + complexity_factor))  # Range: 32-160

        # Clamp to reasonable range and ensure even number
        embed_dim = max(32, min(256, embed_dim))
        embed_dim = embed_dim - (embed_dim % 2)  # Ensure even

        return embed_dim

    def calibrate_num_heads(self, semantic_diversity: float) -> int:
        """
        Calibrate number of attention heads based on semantic diversity

        num_heads = max(4, min(16, int(8 * diversity_factor)))
        """
        # Scale based on semantic diversity
        num_heads = int(8 * (0.5 + semantic_diversity * 1.5))  # Range: 4-20

        # Clamp to reasonable range and ensure divisibility
        num_heads = max(4, min(16, num_heads))

        # Ensure embed_dim will be divisible by num_heads (checked later)
        return num_heads

    def calibrate_hidden_dim(self, vocab_richness: float) -> int:
        """
        Calibrate hidden dimension based on vocabulary richness

        hidden_dim = max(256, min(1024, int(512 * richness_factor)))
        """
        # Scale based on vocabulary richness
        hidden_dim = int(512 * (0.5 + vocab_richness))  # Range: 256-768

        # Clamp to reasonable range
        hidden_dim = max(256, min(1024, hidden_dim))

        return hidden_dim

    def calibrate_num_layers(self, complexity_metrics: Dict[str, float]) -> int:
        """
        Calibrate number of layers based on required depth

        num_layers = max(2, min(6, int(3 * depth_factor)))
        """
        # Depth factor based on complexity and length
        complexity_factor = complexity_metrics['normalized_entropy']
        length_factor = min(complexity_metrics['text_length'] / 500.0, 1.0)

        depth_factor = (complexity_factor + length_factor) / 2.0

        num_layers = int(3 * (0.5 + depth_factor * 0.8))  # Range: 2-5

        # Clamp to reasonable range
        num_layers = max(2, min(6, num_layers))

        return num_layers

    def calibrate_all(self, text: str) -> Dict[str, Any]:
        """
        Calibrate all architecture parameters

        Args:
            text: Input text

        Returns:
            Dict with calibrated architecture parameters
        """
        # Analyze input characteristics
        complexity_metrics = self.analyze_text_complexity(text)
        semantic_diversity = self.analyze_semantic_diversity(text)
        vocab_richness = self.analyze_vocabulary_richness(text)

        # Calibrate parameters
        embed_dim = self.calibrate_embed_dim(complexity_metrics)
        num_heads = self.calibrate_num_heads(semantic_diversity)
        hidden_dim = self.calibrate_hidden_dim(vocab_richness)
        num_layers = self.calibrate_num_layers(complexity_metrics)

        # Ensure compatibility: embed_dim must be divisible by num_heads
        if embed_dim % num_heads != 0:
            # Adjust embed_dim to be divisible by num_heads
            embed_dim = (embed_dim // num_heads) * num_heads
            embed_dim = max(32, embed_dim)  # Ensure minimum size

        return {
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            # Include analysis metrics for debugging
            'complexity_metrics': complexity_metrics,
            'semantic_diversity': semantic_diversity,
            'vocab_richness': vocab_richness
        }

    def validate_architecture_consistency(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate architecture parameter consistency

        Args:
            params: Calibrated parameters

        Returns:
            Validation results
        """
        embed_dim = params['embed_dim']
        num_heads = params['num_heads']
        hidden_dim = params['hidden_dim']
        num_layers = params['num_layers']

        # Check parameter ranges
        range_checks = {
            'embed_dim_range': 32 <= embed_dim <= 256,
            'num_heads_range': 4 <= num_heads <= 16,
            'hidden_dim_range': 256 <= hidden_dim <= 1024,
            'num_layers_range': 2 <= num_layers <= 6
        }

        # Architecture consistency checks
        architecture_checks = {
            'embed_divisible_by_heads': embed_dim % num_heads == 0,
            'hidden_greater_than_embed': hidden_dim >= embed_dim,
            'reasonable_depth': num_layers <= 6  # Prevent overfitting
        }

        all_checks_pass = all(range_checks.values()) and all(architecture_checks.values())

        return {
            'range_checks': range_checks,
            'architecture_checks': architecture_checks,
            'all_checks_pass': all_checks_pass
        }