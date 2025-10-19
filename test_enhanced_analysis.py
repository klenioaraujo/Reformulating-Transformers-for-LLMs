#!/usr/bin/env python3
"""
Test Enhanced Token Analysis with Dynamic Quantum Vocabulary

This test script demonstrates the enhanced token analysis system
without requiring the full ΨQRH pipeline dependencies.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_token_analysis import EnhancedDCFTokenAnalysis

def test_enhanced_analysis():
    """Test the enhanced token analysis system."""
    print("🧪 Testing Enhanced Token Analysis System")
    print("=" * 60)

    # Create enhanced analyzer
    analyzer = EnhancedDCFTokenAnalysis(
        device="cpu",
        enable_dynamic_vocabulary=True
    )

    print("✅ Enhanced analyzer created successfully")

    # Test with sample data
    vocab_size = 100
    logits = torch.randn(vocab_size)
    input_text = "what color is the sky?"

    print(f"\n🔬 Testing with input: '{input_text}'")
    print(f"   Logits shape: {logits.shape}")

    # Get enhanced analysis
    result = analyzer.analyze_tokens_with_quantum_vocab(
        logits=logits,
        input_text=input_text
    )

    print("\n📊 Analysis Results:")
    print("=" * 60)
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
            print(f"\n   Selected Token Analysis:")
            print(f"      Character: {token_analysis['character']}")
            print(f"      Quantum Word: {token_analysis['quantum_word']}")
            print(f"      Quantum Weight: {token_analysis['quantum_weight']:.2f}")
            print(f"      Energy Level: {token_analysis['energy_level']}")
            print(f"      Quantum References: {token_analysis['quantum_references'][:2]}")

    # Show enhanced cluster analysis
    if 'enhanced_cluster_analysis' in result:
        clusters = result['enhanced_cluster_analysis']
        print(f"\n📈 Enhanced Cluster Analysis:")
        print(f"   Total Clusters: {clusters.get('total_clusters', 0)}")

        if 'enhanced_dominant_cluster' in clusters:
            dominant = clusters['enhanced_dominant_cluster']
            print(f"   Dominant Cluster:")
            print(f"      Size: {dominant.get('size', 0)}")
            print(f"      Order Parameter: {dominant.get('order_parameter', 0):.3f}")
            print(f"      Quantum Coherence: {dominant.get('quantum_coherence', 0):.3f}")

    # Show statistics
    stats = analyzer.get_enhanced_analysis_stats()
    print(f"\n📊 System Statistics:")
    print(f"   Dynamic Vocabulary: {stats['dynamic_vocabulary_enabled']}")
    print(f"   DCF Analyzer: {stats['dcf_analyzer_enabled']}")
    print(f"   Quantum Word References: {stats['quantum_word_references']}")

    if 'total_english_words' in stats:
        print(f"   English Words: {stats['total_english_words']:,}")
        print(f"   Character Mappings: {stats['total_character_mappings']}")

    print("\n✅ Enhanced token analysis test completed successfully!")

def test_quantum_vocabulary_standalone():
    """Test the dynamic quantum vocabulary system standalone."""
    print("\n🧪 Testing Dynamic Quantum Vocabulary Standalone")
    print("=" * 60)

    from dynamic_quantum_vocabulary import DynamicQuantumVocabulary

    # Create vocabulary
    vocab = DynamicQuantumVocabulary()

    # Test with sample texts
    test_texts = [
        "what color is the sky?",
        "Hello Quantum World",
        "AI and machine learning"
    ]

    for text in test_texts:
        print(f"\n🔬 Testing: '{text}'")
        print("-" * 40)

        # Create quantum prompt
        quantum_prompt = vocab.create_quantum_prompt(text, verbose=True)
        print(f"📝 Quantum Prompt: {quantum_prompt}")

    # Show vocabulary statistics
    stats = vocab.get_vocabulary_stats()
    print(f"\n📊 Vocabulary Statistics:")
    print(f"   English words: {stats['total_english_words']:,}")
    print(f"   Character mappings: {stats['total_character_mappings']}")
    print(f"   Quantum references: {stats['total_quantum_references']:,}")
    print(f"   Avg words per character: {stats['average_words_per_character']:.1f}")

    print("\n✅ Dynamic quantum vocabulary test completed!")

if __name__ == "__main__":
    # Run standalone quantum vocabulary test
    test_quantum_vocabulary_standalone()

    print("\n" + "=" * 70)
    print("🚀 MAIN ENHANCED ANALYSIS TEST")
    print("=" * 70)

    # Run enhanced analysis test
    test_enhanced_analysis()