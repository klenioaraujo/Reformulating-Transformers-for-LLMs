#!/usr/bin/env python3
"""
ΨQRH Contextual Probing Evaluation Script
==========================================

Compares simple average vs weighted average contextual probing methods.
Evaluates character-level and word-level accuracy for semantic coherence.

This script evaluates the improvement in character-level accuracy when using
weighted averaging in contextual probing instead of simple averaging, and
measures semantic coherence through word-level accuracy.

Usage:
    python run_evaluation.py
"""

import torch
import json
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Set

# Import ΨQRH components
from psiqrh import ΨQRHPipeline
from src.core.optical_probe import OpticalProbe


def load_english_dictionary() -> Set[str]:
    """Load a dictionary of English words for word-level accuracy evaluation."""
    # Common English words dictionary (basic set for evaluation)
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'can', 'shall', 'must', 'let', 'make', 'go', 'take', 'come', 'see', 'know', 'get',
        'give', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'think', 'say', 'mean', 'want',
        'use', 'need', 'help', 'turn', 'run', 'move', 'live', 'try', 'call', 'keep', 'begin',
        'seem', 'help', 'talk', 'turn', 'start', 'might', 'show', 'hear', 'play', 'run', 'move',
        'like', 'well', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'own', 'same', 'different', 'new', 'old', 'young',
        'high', 'low', 'right', 'left', 'first', 'last', 'good', 'bad', 'big', 'small', 'long',
        'short', 'hot', 'cold', 'full', 'empty', 'open', 'close', 'light', 'dark', 'hard',
        'soft', 'heavy', 'light', 'strong', 'weak', 'easy', 'difficult', 'fast', 'slow',
        'early', 'late', 'here', 'there', 'now', 'then', 'before', 'after', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'above', 'below', 'left', 'right',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which', 'who', 'when', 'where',
        'why', 'how', 'time', 'day', 'year', 'way', 'man', 'woman', 'child', 'life', 'world',
        'hand', 'eye', 'head', 'face', 'place', 'thing', 'point', 'case', 'fact', 'idea',
        'part', 'kind', 'form', 'system', 'program', 'problem', 'question', 'answer', 'reason',
        'result', 'change', 'end', 'number', 'group', 'level', 'order', 'book', 'word', 'letter',
        'sentence', 'language', 'model', 'framework', 'quantum', 'processing', 'natural',
        'machine', 'learning', 'context', 'information', 'understanding', 'generation'
    }

    print(f"📚 Loaded English dictionary with {len(common_words)} words")
    return common_words


def load_test_text() -> str:
    """Load test text for evaluation (different from training text)."""
    # Use a different text from the training data for proper evaluation
    test_text = """
Quantum mechanics provides the foundation for understanding physical phenomena at atomic scales.
The uncertainty principle states that it is impossible to simultaneously know both position and momentum
with arbitrary precision. Wave functions describe the quantum state of systems, evolving according to
the Schrödinger equation. Superposition allows particles to exist in multiple states simultaneously
until measured. Entanglement creates correlations between particles that persist regardless of distance.
These principles form the basis of quantum computing and quantum information theory.
"""

    # Clean and return test text
    return test_text.strip()


def create_modified_pipeline(simple_average: bool = False) -> ΨQRHPipeline:
    """
    Create a ΨQRH pipeline with modified contextual probing.

    Args:
        simple_average: If True, use simple averaging (equal weights).
                       If False, use weighted averaging (0.2, 0.6, 0.2).
    """
    # Create pipeline with audit mode enabled
    pipeline = ΨQRHPipeline(
        task="text-generation",
        device="cpu",  # Use CPU for evaluation
        enable_auto_calibration=False,  # Disable for consistent evaluation
        audit_mode=True
    )

    # Modify the contextual probing method based on the parameter
    if simple_average:
        # Temporarily modify the contextual probing to use simple averaging
        original_method = pipeline.find_closest_char_projection_contextual

        def simple_average_contextual_probing(psi_sequence, position, context_window=1, candidate_tokens=None, top_k=5):
            """Modified contextual probing with simple averaging (equal weights)."""
            # Temporarily override the weights to be equal
            start_idx = max(0, position - context_window)
            end_idx = min(psi_sequence.shape[1] - 1, position + context_window)

            # Collect quantum states in the context window
            context_states = []
            context_weights = []

            for j in range(start_idx, end_idx + 1):
                # Use equal weights for all positions (simple average)
                weight = 1.0  # Equal weight for all positions
                context_states.append(psi_sequence[0, j])  # [embed_dim, 4]
                context_weights.append(weight)

            # Handle case where no context states are found
            if not context_states:
                print(f"   ⚠️  No context states found for position {position}, using center position only")
                # Fallback: use the center position if available, otherwise use zeros
                if position < psi_sequence.shape[1]:
                    psi_contextual = psi_sequence[0, position]  # [embed_dim, 4]
                else:
                    psi_contextual = torch.zeros(pipeline.config['embed_dim'], 4, device=psi_sequence.device)
            else:
                # Convert to tensors
                context_states = torch.stack(context_states)  # [window_size, embed_dim, 4]
                context_weights = torch.tensor(context_weights, dtype=torch.float32, device=psi_sequence.device)  # [window_size]

                # Compute weighted average of quantum states in context
                weights_normalized = context_weights / context_weights.sum()
                psi_contextual = torch.sum(context_states * weights_normalized.view(-1, 1, 1), dim=0)  # [embed_dim, 4]

            # Find closest characters using the contextual quantum state
            return pipeline.find_closest_char_projection(psi_contextual, position, candidate_tokens, top_k)

        # Monkey patch the method
        pipeline.find_closest_char_projection_contextual = simple_average_contextual_probing

    return pipeline


def calculate_word_accuracy(generated_text: str, english_dict: Set[str]) -> Dict:
    """
    Calculate word-level accuracy by checking how many generated words are valid English words.

    Args:
        generated_text: The generated text to evaluate
        english_dict: Set of valid English words

    Returns:
        Dictionary with word accuracy metrics
    """
    # Split text into words (basic tokenization)
    words = generated_text.lower().split()

    # Remove punctuation and filter out very short words
    clean_words = []
    for word in words:
        # Remove common punctuation
        clean_word = ''.join(c for c in word if c.isalnum())
        if len(clean_word) >= 2:  # Only consider words with 2+ characters
            clean_words.append(clean_word)

    # Calculate word accuracy
    valid_words = sum(1 for word in clean_words if word in english_dict)
    total_words = len(clean_words)

    word_accuracy = valid_words / total_words if total_words > 0 else 0.0

    return {
        'word_accuracy': word_accuracy,
        'valid_words': valid_words,
        'total_words': total_words,
        'clean_words': clean_words[:10]  # Show first 10 for debugging
    }


def evaluate_accuracy(pipeline: ΨQRHPipeline, test_text: str, method_name: str, english_dict: Set[str]) -> Dict:
    """
    Evaluate character-level and word-level accuracy for a given pipeline and test text using OpticalProbe.

    Args:
        pipeline: ΨQRH pipeline to evaluate
        test_text: Ground truth text
        method_name: Name of the method being evaluated
        english_dict: Dictionary of valid English words

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n🔬 Evaluating {method_name}...")

    start_time = time.time()

    try:
        # Create optical probe using Padilha Wave Equation
        optical_probe = OpticalProbe(vocab_size=95, device=pipeline.device)

        # Generate quantum states from the test text
        # First, convert text to character IDs
        char_ids = torch.tensor([ord(c) for c in test_text], dtype=torch.long, device=pipeline.device).unsqueeze(0)  # [1, seq_len]

        # Generate quantum states
        with torch.no_grad():
            psi_sequence = pipeline.quantum_embedding(char_ids)  # [1, seq_len, embed_dim//4, 4]

        # Reshape for decoder: [seq_len, embed_dim, 4]
        psi_sequence = psi_sequence.squeeze(0)  # [seq_len, embed_dim//4, 4]
        # Expand embed_dim if needed
        embed_dim = pipeline.config['embed_dim']
        if psi_sequence.shape[1] < embed_dim:
            # Pad or repeat to reach embed_dim
            padding = embed_dim - psi_sequence.shape[1]
            psi_sequence = torch.cat([psi_sequence, psi_sequence[:, :padding]], dim=1)  # [seq_len, embed_dim, 4]

        # Decode using optical probe with Padilha Wave Equation
        generated_text = optical_probe(psi_sequence)
        confidences = [1.0] * len(generated_text)  # Optical probe doesn't provide confidences

        # Calculate character-level accuracy
        min_length = min(len(test_text), len(generated_text))
        char_matches = sum(1 for i in range(min_length) if test_text[i] == generated_text[i])
        char_accuracy = char_matches / len(test_text) if len(test_text) > 0 else 0.0

        # Calculate word-level accuracy
        word_metrics = calculate_word_accuracy(generated_text, english_dict)

        processing_time = time.time() - start_time

        print(f"   📏 Generated length: {len(generated_text)} characters")
        print(f"   🎯 Character accuracy: {char_accuracy:.3f}")
        print(f"   📝 Word accuracy: {word_metrics['word_accuracy']:.3f}")
        print(f"   📝 Sample words: {word_metrics['clean_words']}")
        print(f"   🔬 Average confidence: {np.mean(confidences):.3f}")

        return {
            'method': method_name,
            'char_accuracy': char_accuracy,
            'word_accuracy': word_metrics['word_accuracy'],
            'char_matches': char_matches,
            'word_matches': word_metrics['valid_words'],
            'total_chars': len(test_text),
            'total_words': word_metrics['total_words'],
            'generated_length': len(generated_text),
            'processing_time': processing_time,
            'generated_text': generated_text,
            'input_text': test_text,
            'word_details': word_metrics,
            'average_confidence': float(np.mean(confidences)),
            'confidences': confidences
        }

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"   ❌ Evaluation failed: {e}")
        return {
            'method': method_name,
            'error': str(e),
            'processing_time': processing_time
        }


def run_volumetric_evaluation():
    """Run the complete optical evaluation using OpticalProbe with Padilha Wave Equation."""
    print("🔬 ΨQRH Optical Decoding Evaluation - Padilha Wave Equation & Word Validity")
    print("=" * 70)

    # Load English dictionary and test text
    english_dict = load_english_dictionary()
    test_text = load_test_text()
    print(f"📝 Test text loaded: {len(test_text)} characters")
    print(f"   Preview: {test_text[:100]}...")

    # Create pipeline
    print("\n🔧 Creating evaluation pipeline...")
    pipeline = ΨQRHPipeline(
        task="text-generation",
        device="cpu",
        enable_auto_calibration=False,
        audit_mode=False
    )

    # Run evaluation
    results = []

    result = evaluate_accuracy(pipeline, test_text, "Volumetric Decoding", english_dict)
    results.append(result)

    # Generate comprehensive final report
    print("\n" + "=" * 70)
    print("📊 FINAL OPTICAL EVALUATION REPORT")
    print("=" * 70)

    if 'char_accuracy' in result:
        char_acc = result['char_accuracy']
        word_acc = result['word_accuracy']
        avg_confidence = result.get('average_confidence', 0.0)

        print("OPTICAL DECODING PERFORMANCE (Padilha Wave Equation):")
        print(f"   Character Accuracy: {char_acc:.3f}")
        print(f"   Word Validity:      {word_acc:.3f}")
        print(f"   Average Confidence: {avg_confidence:.3f}")

        # Check optical decoding success criteria
        word_target = 0.99
        if word_acc > word_target:
            print(f"\n✅ SUCCESS: Word validity ({word_acc:.3f}) exceeds target of {word_target:.3f}")
            print("   🎉 Optical decoding with Padilha Wave Equation fully achieved!")
        else:
            print(f"\n📈 Word validity ({word_acc:.3f}) below target of {word_target:.3f}")
            print("   Continue spectral coherence training for better results")

        print("\n📝 INPUT TEXT:")
        print("-" * 50)
        print(test_text)

        print("\n📝 DECODED TEXT (Optical - Padilha Wave Equation):")
        print("-" * 50)
        generated = result.get('generated_text', 'N/A')
        print(generated)

    else:
        print("❌ Evaluation failed due to errors")

    # Save detailed results
    output_file = Path("results/optical_evaluation.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'evaluation_timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'test_text_length': len(test_text),
            'english_dict_size': len(english_dict),
            'results': results,
            'comparison': {
                'baseline_target': 0.056,
                'char_improvement': char_improvement if 'char_improvement' in locals() else None,
                'word_improvement': word_improvement if 'word_improvement' in locals() else None
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed results saved to: {output_file}")

    return results


if __name__ == "__main__":
    # Run the evaluation
    results = run_volumetric_evaluation()

    print("\n✅ Optical evaluation completed!")