#!/usr/bin/env python3
"""
Enhanced Token Analysis with Dynamic Quantum Vocabulary Integration

Integrates the dynamic quantum vocabulary system with token analysis
for enhanced semantic connectivity and quantum word references.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import time
from collections import deque

# Import dynamic quantum vocabulary system
from dynamic_quantum_vocabulary import DynamicQuantumVocabulary

class EnhancedDCFTokenAnalysis:
    """
    Enhanced DCF Token Analysis with Dynamic Quantum Vocabulary Integration

    This system enhances the original DCF token analysis by:
    1. Integrating dynamic quantum vocabulary with word weights
    2. Using quantum word references for semantic connectivity
    3. Applying quantum influence factors to token selection
    4. Providing enhanced semantic analysis with quantum terminology
    """

    def __init__(self, config_path: Optional[str] = None, device: str = "cpu",
                 enable_cognitive_priming: bool = True,
                 quantum_vocab_representations: Optional[torch.Tensor] = None,
                 char_to_idx: Optional[Dict[str, int]] = None,
                 enable_dynamic_vocabulary: bool = True):
        """
        Initialize enhanced DCF token analysis with dynamic quantum vocabulary.

        Args:
            config_path: Path to configuration file
            device: Computation device
            enable_cognitive_priming: Enable cognitive priming
            quantum_vocab_representations: Quantum vocabulary representations
            char_to_idx: Character to index mapping
            enable_dynamic_vocabulary: Enable dynamic quantum vocabulary
        """
        self.device = device
        self.enable_cognitive_priming = enable_cognitive_priming
        self.enable_dynamic_vocabulary = enable_dynamic_vocabulary

        # Quantum vocabulary for semantic connectivity
        self.quantum_vocab_representations = quantum_vocab_representations
        self.char_to_idx = char_to_idx if char_to_idx is not None else {}

        # Initialize dynamic quantum vocabulary
        if enable_dynamic_vocabulary:
            self.dynamic_vocab = DynamicQuantumVocabulary()
            print("      ✅ Dynamic quantum vocabulary initialized")
        else:
            self.dynamic_vocab = None

        # Import and initialize original DCF components
        try:
            from src.processing.token_analysis import DCFTokenAnalysis
            self.dcf_analyzer = DCFTokenAnalysis(
                config_path=config_path,
                device=device,
                enable_cognitive_priming=enable_cognitive_priming,
                quantum_vocab_representations=quantum_vocab_representations,
                char_to_idx=char_to_idx
            )
            print("      ✅ Original DCF analyzer initialized")
        except Exception as e:
            print(f"      ⚠️  Failed to initialize DCF analyzer: {e}")
            self.dcf_analyzer = None

        print("🎯 Enhanced DCF Token Analysis initialized")
        print(f"   🔄 Dynamic Vocabulary: {self.dynamic_vocab is not None}")
        print(f"   🧠 Original DCF: {self.dcf_analyzer is not None}")

    def analyze_tokens_with_quantum_vocab(self, logits: torch.Tensor,
                                         candidate_indices: Optional[torch.Tensor] = None,
                                         embeddings: Optional[torch.Tensor] = None,
                                         input_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced token analysis with dynamic quantum vocabulary integration.

        Args:
            logits: Model logits [vocab_size] or [batch_size, vocab_size]
            candidate_indices: Candidate token indices
            embeddings: Token embeddings
            input_text: Original input text for quantum vocabulary analysis

        Returns:
            Enhanced analysis results with quantum vocabulary integration
        """
        start_time = time.time()

        # Run original DCF analysis
        if self.dcf_analyzer:
            dcf_result = self.dcf_analyzer.analyze_tokens(
                logits, candidate_indices, embeddings
            )
        else:
            # Fallback basic analysis
            dcf_result = self._fallback_analysis(logits, candidate_indices)

        # Enhance with dynamic quantum vocabulary if available
        if self.dynamic_vocab and input_text:
            enhanced_result = self._enhance_with_quantum_vocab(
                dcf_result, input_text, logits, candidate_indices
            )
        else:
            enhanced_result = dcf_result

        processing_time = time.time() - start_time
        enhanced_result['enhanced_processing_time'] = processing_time

        return enhanced_result

    def _enhance_with_quantum_vocab(self, dcf_result: Dict[str, Any],
                                   input_text: str,
                                   logits: torch.Tensor,
                                   candidate_indices: Optional[torch.Tensor]) -> Dict[str, Any]:
        """
        Enhance DCF results with dynamic quantum vocabulary analysis.

        Args:
            dcf_result: Original DCF analysis results
            input_text: Input text for quantum vocabulary analysis
            logits: Model logits
            candidate_indices: Candidate token indices

        Returns:
            Enhanced analysis results
        """
        print("   🔬 Enhancing analysis with dynamic quantum vocabulary...")

        # Get quantum vocabulary analysis for input text
        quantum_analysis = self.dynamic_vocab.get_weighted_words_for_text(input_text)

        # Create quantum prompt
        quantum_prompt = self.dynamic_vocab.create_quantum_prompt(input_text, verbose=False)

        # Analyze selected token with quantum vocabulary
        selected_token = dcf_result.get('selected_token', 0)
        token_quantum_analysis = self._analyze_token_with_quantum_vocab(
            selected_token, input_text, logits, candidate_indices
        )

        # Enhance cluster analysis with quantum semantics
        enhanced_clusters = self._enhance_cluster_analysis(
            dcf_result.get('cluster_analysis', {}),
            input_text,
            logits,
            candidate_indices
        )

        # Build enhanced result
        enhanced_result = dcf_result.copy()
        enhanced_result.update({
            'quantum_vocabulary_analysis': {
                'input_text': input_text,
                'quantum_prompt': quantum_prompt,
                'weighted_words': quantum_analysis,
                'vocabulary_stats': self.dynamic_vocab.get_vocabulary_stats(),
                'selected_token_analysis': token_quantum_analysis
            },
            'enhanced_cluster_analysis': enhanced_clusters,
            'semantic_connectivity_enhanced': True,
            'quantum_word_references': True,
            'analysis_method': 'Enhanced DCF with Dynamic Quantum Vocabulary'
        })

        print("   ✅ Quantum vocabulary enhancement complete")
        return enhanced_result

    def _analyze_token_with_quantum_vocab(self, token_id: int, input_text: str,
                                         logits: torch.Tensor,
                                         candidate_indices: Optional[torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze selected token using dynamic quantum vocabulary.

        Args:
            token_id: Selected token ID
            input_text: Input text
            logits: Model logits
            candidate_indices: Candidate token indices

        Returns:
            Quantum vocabulary analysis for the token
        """
        # Convert token ID to character if possible
        char_representation = self._token_to_char(token_id)

        if char_representation:
            # Get quantum word analysis for this character
            quantum_words = self.dynamic_vocab.character_mappings.get(char_representation, [])

            if quantum_words:
                # Select best quantum word based on weights
                best_word = max(quantum_words, key=lambda x: x['weight'])

                return {
                    'token_id': token_id,
                    'character': char_representation,
                    'quantum_word': best_word['word'],
                    'quantum_weight': best_word['weight'],
                    'quantum_references': best_word['quantum_references'],
                    'energy_level': best_word['energy_level'],
                    'semantic_analysis': self._analyze_semantic_context(
                        best_word['word'], input_text
                    )
                }

        # Fallback analysis
        return {
            'token_id': token_id,
            'character': 'unknown',
            'quantum_word': 'QUANTUM_UNKNOWN',
            'quantum_weight': 0.0,
            'quantum_references': ['unknown quantum state'],
            'energy_level': 'unknown_state',
            'semantic_analysis': {'context': 'unknown', 'relevance': 0.0}
        }

    def _token_to_char(self, token_id: int) -> Optional[str]:
        """
        Convert token ID to character representation.

        Args:
            token_id: Token ID

        Returns:
            Character representation or None
        """
        # Simple mapping for demonstration
        # In practice, this would use the actual tokenizer
        if 0 <= token_id < 128:  # ASCII range
            return chr(token_id)

        # Extended mapping for common tokens
        common_tokens = {
            1000: 'the', 1001: 'and', 1002: 'of', 1003: 'to', 1004: 'a',
            1005: 'in', 1006: 'for', 1007: 'is', 1008: 'on', 1009: 'that',
            1010: 'by', 1011: 'this', 1012: 'with', 1013: 'i', 1014: 'you',
            1015: 'it', 1016: 'not', 1017: 'or', 1018: 'be', 1019: 'are',
            1020: 'from', 1021: 'at', 1022: 'as', 1023: 'your', 1024: 'all'
        }

        if token_id in common_tokens:
            return common_tokens[token_id][0] if common_tokens[token_id] else None

        return None

    def _analyze_semantic_context(self, quantum_word: str, input_text: str) -> Dict[str, Any]:
        """
        Analyze semantic context between quantum word and input text.

        Args:
            quantum_word: Quantum word
            input_text: Input text

        Returns:
            Semantic context analysis
        """
        # Simple semantic analysis
        quantum_lower = quantum_word.lower()
        text_lower = input_text.lower()

        # Check for semantic relevance
        relevance_score = 0.0

        # Scientific terms get higher relevance for technical text
        scientific_terms = ['quantum', 'physics', 'electron', 'proton', 'neutron', 'photon']
        if any(term in quantum_lower for term in scientific_terms):
            relevance_score += 0.3

        # Common words get relevance for general text
        common_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all']
        if any(word in quantum_lower for word in common_words):
            relevance_score += 0.2

        # Length-based relevance
        if len(quantum_word) <= 3:
            relevance_score += 0.1  # Short words often more relevant
        elif len(quantum_word) >= 8:
            relevance_score += 0.2  # Long words often more specific

        return {
            'context': f"Quantum word '{quantum_word}' in input context",
            'relevance': min(relevance_score, 1.0),
            'semantic_category': self._classify_semantic_category(quantum_word),
            'quantum_affinity': self._calculate_quantum_affinity(quantum_word)
        }

    def _classify_semantic_category(self, word: str) -> str:
        """Classify word into semantic category."""
        word_lower = word.lower()

        if any(term in word_lower for term in ['quantum', 'physics', 'electron']):
            return 'scientific'
        elif any(term in word_lower for term in ['the', 'and', 'for', 'are']):
            return 'common'
        elif len(word) <= 3:
            return 'short'
        elif len(word) >= 8:
            return 'complex'
        else:
            return 'general'

    def _calculate_quantum_affinity(self, word: str) -> float:
        """Calculate quantum affinity score for word."""
        word_lower = word.lower()

        affinity = 0.0

        # Scientific terms have high quantum affinity
        scientific_terms = ['quantum', 'physics', 'electron', 'proton', 'neutron', 'photon']
        if any(term in word_lower for term in scientific_terms):
            affinity += 0.8

        # Common words have medium affinity
        common_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all']
        if any(word in word_lower for word in common_words):
            affinity += 0.4

        # Complex words have some affinity
        if len(word) >= 8:
            affinity += 0.2

        return min(affinity, 1.0)

    def _enhance_cluster_analysis(self, cluster_analysis: Dict[str, Any],
                                 input_text: str,
                                 logits: torch.Tensor,
                                 candidate_indices: Optional[torch.Tensor]) -> Dict[str, Any]:
        """
        Enhance cluster analysis with quantum vocabulary semantics.

        Args:
            cluster_analysis: Original cluster analysis
            input_text: Input text
            logits: Model logits
            candidate_indices: Candidate token indices

        Returns:
            Enhanced cluster analysis
        """
        enhanced_clusters = cluster_analysis.copy()

        # Enhance dominant cluster with quantum semantics
        dominant_cluster = cluster_analysis.get('dominant_cluster', {})
        if dominant_cluster:
            enhanced_dominant = self._enhance_cluster_with_quantum(
                dominant_cluster, input_text, logits, candidate_indices
            )
            enhanced_clusters['enhanced_dominant_cluster'] = enhanced_dominant

        # Add quantum vocabulary statistics
        if self.dynamic_vocab:
            enhanced_clusters['quantum_vocab_stats'] = self.dynamic_vocab.get_vocabulary_stats()

        return enhanced_clusters

    def _enhance_cluster_with_quantum(self, cluster: Dict[str, Any],
                                     input_text: str,
                                     logits: torch.Tensor,
                                     candidate_indices: Optional[torch.Tensor]) -> Dict[str, Any]:
        """
        Enhance individual cluster with quantum vocabulary analysis.

        Args:
            cluster: Cluster information
            input_text: Input text
            logits: Model logits
            candidate_indices: Candidate token indices

        Returns:
            Enhanced cluster with quantum analysis
        """
        enhanced_cluster = cluster.copy()

        # Analyze leader token with quantum vocabulary
        leader_token = cluster.get('leader_token', 0)
        leader_analysis = self._analyze_token_with_quantum_vocab(
            leader_token, input_text, logits, candidate_indices
        )

        enhanced_cluster['quantum_leader_analysis'] = leader_analysis

        # Calculate quantum coherence for cluster
        quantum_coherence = self._calculate_cluster_quantum_coherence(cluster)
        enhanced_cluster['quantum_coherence'] = quantum_coherence

        # Add semantic interpretation
        enhanced_cluster['quantum_interpretation'] = self._generate_quantum_interpretation(
            cluster, leader_analysis
        )

        return enhanced_cluster

    def _calculate_cluster_quantum_coherence(self, cluster: Dict[str, Any]) -> float:
        """Calculate quantum coherence score for cluster."""
        order_parameter = cluster.get('order_parameter', 0.5)
        size = cluster.get('size', 1)

        # Coherence based on order parameter and size
        coherence = order_parameter * (1.0 + 0.1 * min(size, 10))
        return min(coherence, 1.0)

    def _generate_quantum_interpretation(self, cluster: Dict[str, Any],
                                        leader_analysis: Dict[str, Any]) -> str:
        """Generate quantum interpretation for cluster."""
        order_parameter = cluster.get('order_parameter', 0.5)
        size = cluster.get('size', 1)
        quantum_word = leader_analysis.get('quantum_word', 'UNKNOWN')

        if order_parameter > 0.8:
            return f"High quantum coherence: Cluster of {size} tokens synchronized around '{quantum_word}' with strong semantic binding"
        elif order_parameter > 0.5:
            return f"Moderate quantum coherence: {size} tokens forming conceptual group around '{quantum_word}'"
        else:
            return f"Low quantum coherence: Weak semantic binding in cluster around '{quantum_word}'"

    def _fallback_analysis(self, logits: torch.Tensor,
                          candidate_indices: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Fallback analysis when DCF analyzer is not available."""
        # Simple top-1 selection
        if candidate_indices is not None:
            selected_token = candidate_indices[0].item() if hasattr(candidate_indices[0], 'item') else candidate_indices[0]
        else:
            selected_token = torch.argmax(logits).item()

        return {
            'selected_token': selected_token,
            'final_probability': 1.0,
            'fci_value': 0.5,
            'consciousness_state': 'UNKNOWN',
            'synchronization_order': 0.5,
            'processing_time': 0.001,
            'analysis_report': 'Fallback analysis - DCF not available',
            'cluster_analysis': {},
            'semantic_analysis': {'semantic_reasoning': False}
        }

    def get_enhanced_analysis_stats(self) -> Dict[str, Any]:
        """Get enhanced analysis statistics."""
        stats = {
            'dynamic_vocabulary_enabled': self.dynamic_vocab is not None,
            'dcf_analyzer_enabled': self.dcf_analyzer is not None,
            'quantum_word_references': True,
            'semantic_connectivity_enhanced': True
        }

        if self.dynamic_vocab:
            stats.update(self.dynamic_vocab.get_vocabulary_stats())

        return stats

# Main interface function
def analyze_tokens_enhanced(logits: torch.Tensor,
                          config_path: Optional[str] = None,
                          device: str = "cpu",
                          embeddings: Optional[torch.Tensor] = None,
                          input_text: Optional[str] = None,
                          enable_dynamic_vocabulary: bool = True) -> Dict[str, Any]:
    """
    Enhanced token analysis interface with dynamic quantum vocabulary.

    Args:
        logits: Model logits
        config_path: Configuration path
        device: Computation device
        embeddings: Token embeddings
        input_text: Input text for quantum vocabulary analysis
        enable_dynamic_vocabulary: Enable dynamic quantum vocabulary

    Returns:
        Enhanced analysis results
    """
    analyzer = EnhancedDCFTokenAnalysis(
        config_path=config_path,
        device=device,
        enable_dynamic_vocabulary=enable_dynamic_vocabulary
    )

    return analyzer.analyze_tokens_with_quantum_vocab(
        logits=logits,
        embeddings=embeddings,
        input_text=input_text
    )

if __name__ == "__main__":
    # Test the enhanced token analysis
    print("🧪 Testing Enhanced Token Analysis with Dynamic Quantum Vocabulary...")
    print("=" * 70)

    # Simulate logits
    vocab_size = 1000
    logits = torch.randn(vocab_size)
    input_text = "what color is the sky?"

    # Run enhanced analysis
    result = analyze_tokens_enhanced(
        logits=logits,
        device="cpu",
        input_text=input_text,
        enable_dynamic_vocabulary=True
    )

    print("\n📊 Enhanced Analysis Results:")
    print("=" * 70)
    print(f"Selected Token: {result['selected_token']}")
    print(f"FCI Value: {result['fci_value']:.4f}")
    print(f"Consciousness State: {result['consciousness_state']}")
    print(f"Processing Time: {result.get('enhanced_processing_time', 0):.3f}s")

    # Show quantum vocabulary analysis
    if 'quantum_vocabulary_analysis' in result:
        quantum_analysis = result['quantum_vocabulary_analysis']
        print(f"\n🔬 Quantum Vocabulary Analysis:")
        print(f"   Input Text: {quantum_analysis['input_text']}")
        print(f"   Quantum Prompt: {quantum_analysis['quantum_prompt']}")

        if 'selected_token_analysis' in quantum_analysis:
            token_analysis = quantum_analysis['selected_token_analysis']
            print(f"   Selected Token Analysis:")
            print(f"      Character: {token_analysis['character']}")
            print(f"      Quantum Word: {token_analysis['quantum_word']}")
            print(f"      Quantum Weight: {token_analysis['quantum_weight']:.2f}")
            print(f"      Energy Level: {token_analysis['energy_level']}")

    print("\n✅ Enhanced token analysis test completed!")