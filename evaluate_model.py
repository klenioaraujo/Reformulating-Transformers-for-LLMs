#!/usr/bin/env python3
"""
ΨQRH Model Semantic Evaluation Framework
========================================

Comprehensive evaluation framework for assessing semantic quality and coherence
of trained ΨQRH models. Includes BLEU scores, word validity metrics, and
comparative reporting.

This script evaluates trained models against reference texts and generates
detailed reports on semantic performance improvements.
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import argparse
import numpy as np

# Import ΨQRH components
from psiqrh import ΨQRHPipeline
from tools.semantic_decoder import create_semantic_decoder, SemanticBeamSearchDecoder

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("⚠️  NLTK not available - BLEU scores will be disabled")


class ΨQRHEvaluator:
    """Comprehensive evaluator for ΨQRH models."""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu',
                 semantic_decoder: Optional[SemanticBeamSearchDecoder] = None):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to trained model checkpoint (optional)
            device: Device to use for evaluation
            semantic_decoder: Pre-configured semantic decoder (optional)
        """
        self.device = device
        self.model_path = Path(model_path) if model_path else None

        # Initialize pipeline
        self.pipeline = self._load_pipeline()

        # Initialize semantic decoder
        self.semantic_decoder = semantic_decoder or create_semantic_decoder(beam_width=5)

        # Evaluation results
        self.results = {
            'model_info': {},
            'evaluation_metrics': {},
            'test_cases': [],
            'summary': {}
        }

    def _load_pipeline(self) -> ΨQRHPipeline:
        """Load the ΨQRH pipeline, optionally from checkpoint."""
        pipeline = ΨQRHPipeline(
            task="text-generation",
            device=self.device,
            enable_auto_calibration=False,
            audit_mode=False
        )

        if self.model_path and self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Load trained components
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']

                    if 'context_funnel' in model_state and hasattr(pipeline, 'context_funnel'):
                        pipeline.context_funnel.load_state_dict(model_state['context_funnel'])
                        print("✅ Loaded trained Context Funnel")

                    if 'inverse_projector' in model_state and hasattr(pipeline, 'inverse_projector'):
                        pipeline.inverse_projector.load_state_dict(model_state['inverse_projector'])
                        print("✅ Loaded trained Inverse Projector")

                # Load training statistics
                if 'training_stats' in checkpoint:
                    self.results['model_info']['training_stats'] = checkpoint['training_stats']

                print(f"✅ Loaded trained model from: {self.model_path}")

            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}")
                print("   Using untrained pipeline for baseline evaluation")

        return pipeline

    def load_test_data(self, test_data_path: str) -> List[Dict[str, str]]:
        """
        Load test data for evaluation.

        Args:
            test_data_path: Path to test data JSON file

        Returns:
            List of test cases with input and reference output
        """
        test_data_file = Path(test_data_path)

        if not test_data_file.exists():
            print(f"⚠️  Test data file not found: {test_data_file}")
            print("   Creating synthetic test data...")

            # Create synthetic test data
            test_data = self._create_synthetic_test_data()
            self._save_test_data(test_data, test_data_file)
            return test_data

        try:
            with open(test_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'test_cases' in data:
                return data['test_cases']
            else:
                print("⚠️  Unexpected test data format")
                return self._create_synthetic_test_data()

        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return self._create_synthetic_test_data()

    def _create_synthetic_test_data(self) -> List[Dict[str, str]]:
        """Create synthetic test data for evaluation."""
        test_cases = [
            {
                'input': 'quantum mechanics',
                'reference': 'Quantum mechanics provides the foundation for understanding physical phenomena at atomic scales.'
            },
            {
                'input': 'uncertainty principle',
                'reference': 'The uncertainty principle states that it is impossible to simultaneously know both position and momentum with arbitrary precision.'
            },
            {
                'input': 'wave function',
                'reference': 'Wave functions describe the quantum state of systems, evolving according to the Schrödinger equation.'
            },
            {
                'input': 'superposition',
                'reference': 'Superposition allows particles to exist in multiple states simultaneously until measured.'
            },
            {
                'input': 'entanglement',
                'reference': 'Entanglement creates correlations between particles that persist regardless of distance.'
            }
        ]

        return test_cases

    def _save_test_data(self, test_data: List[Dict[str, str]], file_path: Path):
        """Save test data to file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved test data to: {file_path}")
        except Exception as e:
            print(f"⚠️  Could not save test data: {e}")

    def compute_bleu_score(self, candidate: str, reference: str) -> float:
        """
        Compute BLEU score between candidate and reference texts.

        Args:
            candidate: Generated text
            reference: Reference text

        Returns:
            BLEU score (0.0 to 1.0)
        """
        if not HAS_NLTK:
            return 0.0

        try:
            # Tokenize texts
            candidate_tokens = candidate.lower().split()
            reference_tokens = reference.lower().split()

            # Use smoothing for short texts
            smoothing = SmoothingFunction().method1

            # Compute BLEU score
            bleu = sentence_bleu([reference_tokens], candidate_tokens,
                               smoothing_function=smoothing)

            return float(bleu)

        except Exception as e:
            print(f"⚠️  BLEU computation error: {e}")
            return 0.0

    def compute_word_validity_metrics(self, text: str) -> Dict[str, float]:
        """
        Compute word validity metrics for generated text.

        Args:
            text: Text to evaluate

        Returns:
            Dictionary with word validity metrics
        """
        return self.semantic_decoder.get_semantic_quality_score(text)

    def evaluate_single_case(self, input_text: str, reference_text: str) -> Dict[str, Any]:
        """
        Evaluate a single test case.

        Args:
            input_text: Input text for the model
            reference_text: Reference output text

        Returns:
            Dictionary with evaluation results for this case
        """
        print(f"🔬 Evaluating: '{input_text}'")

        try:
            # Generate output using the pipeline
            result = self.pipeline(input_text)

            # Extract generated text
            if isinstance(result, dict) and 'response' in result:
                generated_text = result.get('response', '')
            else:
                generated_text = str(result) if result else ''

            # Clean up generated text
            generated_text = generated_text.strip()

            if not generated_text:
                print("⚠️  No text generated")
                generated_text = "[NO OUTPUT]"

            # Compute BLEU score
            bleu_score = self.compute_bleu_score(generated_text, reference_text)

            # Compute word validity metrics
            validity_metrics = self.compute_word_validity_metrics(generated_text)

            # Compute character-level accuracy (for comparison)
            char_accuracy = self._compute_character_accuracy(generated_text, reference_text)

            case_result = {
                'input': input_text,
                'reference': reference_text,
                'generated': generated_text,
                'bleu_score': bleu_score,
                'char_accuracy': char_accuracy,
                'word_validity_ratio': validity_metrics['word_validity_ratio'],
                'average_word_length': validity_metrics['average_word_length'],
                'semantic_coherence_score': validity_metrics['semantic_coherence_score'],
                'valid_words': validity_metrics['valid_words'],
                'total_words': validity_metrics['total_words']
            }

            print(".3f")
            print(".3f")
            print(".1f")

            return case_result

        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {
                'input': input_text,
                'reference': reference_text,
                'generated': '[ERROR]',
                'error': str(e),
                'bleu_score': 0.0,
                'char_accuracy': 0.0,
                'word_validity_ratio': 0.0,
                'semantic_coherence_score': 0.0
            }

    def _compute_character_accuracy(self, generated: str, reference: str) -> float:
        """Compute character-level accuracy."""
        if not reference:
            return 0.0

        min_len = min(len(generated), len(reference))
        matches = sum(1 for i in range(min_len) if generated[i] == reference[i])

        return matches / len(reference)

    def evaluate_all_cases(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate all test cases and compute aggregate metrics.

        Args:
            test_cases: List of test cases

        Returns:
            Dictionary with complete evaluation results
        """
        print("🧪 Starting comprehensive model evaluation...")
        print("=" * 60)

        case_results = []

        for i, case in enumerate(test_cases, 1):
            print(f"\n📊 Test Case {i}/{len(test_cases)}")
            print("-" * 30)

            case_result = self.evaluate_single_case(case['input'], case['reference'])
            case_results.append(case_result)

        # Compute aggregate metrics
        valid_results = [r for r in case_results if 'error' not in r]

        if valid_results:
            avg_bleu = np.mean([r['bleu_score'] for r in valid_results])
            avg_char_accuracy = np.mean([r['char_accuracy'] for r in valid_results])
            avg_word_validity = np.mean([r['word_validity_ratio'] for r in valid_results])
            avg_semantic_coherence = np.mean([r['semantic_coherence_score'] for r in valid_results])

            # Compute improvement over random baseline
            random_baseline_bleu = 0.01  # Very low baseline
            random_baseline_validity = 0.05  # 5% random word validity

            bleu_improvement = (avg_bleu - random_baseline_bleu) / random_baseline_bleu * 100
            validity_improvement = (avg_word_validity - random_baseline_validity) / random_baseline_validity * 100

            summary = {
                'num_test_cases': len(test_cases),
                'num_valid_results': len(valid_results),
                'average_bleu_score': avg_bleu,
                'average_char_accuracy': avg_char_accuracy,
                'average_word_validity_ratio': avg_word_validity,
                'average_semantic_coherence': avg_semantic_coherence,
                'bleu_improvement_over_random': bleu_improvement,
                'validity_improvement_over_random': validity_improvement,
                'evaluation_timestamp': datetime.now().isoformat()
            }
        else:
            summary = {
                'num_test_cases': len(test_cases),
                'num_valid_results': 0,
                'error': 'No valid evaluation results'
            }

        results = {
            'model_info': {
                'model_path': str(self.model_path) if self.model_path else 'baseline_untrained',
                'device': self.device,
                'semantic_decoder_beam_width': self.semantic_decoder.beam_width
            },
            'summary': summary,
            'test_cases': case_results
        }

        self.results = results
        return results

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            output_path: Path to save the report (optional)

        Returns:
            Report as a formatted string
        """
        report_lines = []
        report_lines.append("# ΨQRH Model Semantic Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Model information
        report_lines.append("## Model Information")
        report_lines.append(f"- **Model**: {self.results['model_info']['model_path']}")
        report_lines.append(f"- **Device**: {self.results['model_info']['device']}")
        report_lines.append(f"- **Semantic Decoder Beam Width**: {self.results['model_info']['semantic_decoder_beam_width']}")

        if 'training_stats' in self.results['model_info']:
            stats = self.results['model_info']['training_stats']
            report_lines.append(f"- **Training Epochs**: {stats.get('epoch', 'N/A')}")
            report_lines.append(f"- **Final Loss**: {stats.get('total_loss', 'N/A'):.6f}")

        report_lines.append("")

        # Summary metrics
        summary = self.results.get('summary', {})
        report_lines.append("## Summary Metrics")
        report_lines.append(f"- **Test Cases Evaluated**: {summary.get('num_valid_results', 0)}/{summary.get('num_test_cases', 0)}")

        if 'average_bleu_score' in summary:
            report_lines.append(".3f")
            report_lines.append(".3f")
            report_lines.append(".3f")
            report_lines.append(".3f")
            report_lines.append(".1f")
            report_lines.append(".1f")

        report_lines.append("")

        # Detailed test case results
        report_lines.append("## Detailed Test Case Results")
        report_lines.append("")

        for i, case in enumerate(self.results.get('test_cases', []), 1):
            report_lines.append(f"### Test Case {i}")
            report_lines.append(f"**Input**: {case['input']}")
            report_lines.append(f"**Reference**: {case['reference']}")
            report_lines.append(f"**Generated**: {case['generated']}")

            if 'bleu_score' in case:
                report_lines.append(".3f")
                report_lines.append(".3f")
                report_lines.append(".1f")

            if 'error' in case:
                report_lines.append(f"**Error**: {case['error']}")

            report_lines.append("")

        # Save report if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))

            print(f"💾 Report saved to: {output_file}")

        return '\n'.join(report_lines)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="ΨQRH Model Semantic Evaluation")
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test-data', type=str, default='data/test_cases.json',
                       help='Path to test data JSON file')
    parser.add_argument('--output-dir', type=str, default='reports/evaluation',
                       help='Directory to save evaluation reports')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for evaluation')
    parser.add_argument('--beam-width', type=int, default=5,
                       help='Beam width for semantic decoder')

    args = parser.parse_args()

    print("🧪 ΨQRH Model Semantic Evaluation")
    print("=" * 50)

    # Create semantic decoder
    semantic_decoder = create_semantic_decoder(beam_width=args.beam_width)

    # Create evaluator
    evaluator = ΨQRHEvaluator(
        model_path=args.model_path,
        device=args.device,
        semantic_decoder=semantic_decoder
    )

    # Load test data
    test_cases = evaluator.load_test_data(args.test_data)

    # Run evaluation
    results = evaluator.evaluate_all_cases(test_cases)

    # Generate and display report
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"evaluation_report_{timestamp}.md"

    report = evaluator.generate_report(str(report_path))

    print("\n" + "=" * 50)
    print("📊 EVALUATION REPORT SUMMARY")
    print("=" * 50)

    summary = results.get('summary', {})
    if 'average_bleu_score' in summary:
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".1f")
        print(".1f")

    print(f"\n💾 Detailed report saved to: {report_path}")

    # Check success criteria
    if 'average_word_validity_ratio' in summary:
        validity_ratio = summary['average_word_validity_ratio']
        if validity_ratio > 0.1:  # 10% improvement target
            print("✅ SUCCESS: Word validity ratio exceeds 10% target!")
        else:
            print(".1f")
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()