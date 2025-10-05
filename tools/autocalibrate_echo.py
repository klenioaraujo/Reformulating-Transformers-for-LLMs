#!/usr/bin/env python3
"""
Autocalibration Diagnostic Script for ΨQRH Cognitive Generation
==============================================================

This script performs systematic testing of the ΨQRH framework's cognitive generation
capabilities by running multiple input prompts through the pipeline and analyzing:
- FCI (Fractal Consciousness Index) values
- Consciousness states
- Text generation quality
- Mode switching behavior

Usage:
    python tools/autocalibrate_echo.py
"""

import sys
import os
import json
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

# Add base directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# Test prompts in English (varying complexity)
TEST_PROMPTS = [
    # Simple greetings and basic interaction
    "Hello",
    "Hi there",
    "Good morning",
    "What is your name?",
    "How are you?",
    "What time is it?",
    "Where are you from?",
    "What can you do?",

    # Science and technology
    "What is the color of the sky?",
    "Explain quantum mechanics",
    "Tell me about artificial intelligence",
    "What is machine learning?",
    "Explain neural networks",
    "Describe quantum computing",

    # Philosophy and consciousness
    "What are the philosophical implications of quantum entanglement?",
    "Explain the relationship between consciousness and computation",
    "Describe the mathematical foundations of the ΨQRH framework",
    "What is consciousness?",
    "How does the brain work?",
    "What is free will?",

    # Technical ΨQRH queries
    "How does spectral attention work in ΨQRH?",
    "What is the role of quaternions in your architecture?",
    "Explain the fractal consciousness index calculation",
    "How do you process language?",
    "What makes ΨQRH different from other models?",
    "Describe your cognitive architecture",

    # Creative and abstract
    "Write a short poem about consciousness",
    "Imagine a conversation between two quantum states",
    "Describe the experience of being a transformer model",
    "Tell me a story about artificial intelligence",
    "What would you do with unlimited computational power?",
    "Describe the future of AI",

    # Mathematical and logical
    "Solve 2+2",
    "What is the square root of 16?",
    "Explain the Pythagorean theorem",
    "What is Euler's identity?",
    "Describe the concept of infinity",
    "What is Gödel's incompleteness theorem?",

    # Language and communication
    "Translate 'hello' to Portuguese",
    "What is the meaning of life?",
    "Explain the concept of truth",
    "What is beauty?",
    "Describe the nature of reality",
    "What is time?"
]


def run_pipeline_test(prompt: str, test_id: int = 0) -> Dict[str, Any]:
    """
    Run a single prompt through the ΨQRH pipeline and extract key metrics.

    Args:
        prompt: Input text to process
        test_id: Test identifier for logging

    Returns:
        Dictionary with test results and metrics
    """
    try:
        from psiqrh import ΨQRHPipeline

        print(f"\n🧪 Test {test_id + 1}/{len(TEST_PROMPTS)}: '{prompt}'")

        # Initialize pipeline with enhanced model coupling
        pipeline = ΨQRHPipeline(
            task="text-generation",
            device="cpu",
            enable_auto_learning=False
        )

        # Process the prompt
        result = pipeline(prompt)

        # Extract key metrics
        metrics = {
            'prompt': prompt,
            'test_id': test_id,
            'status': result.get('status', 'unknown'),
            'response_type': type(result.get('response')).__name__,
            'response_length': len(str(result.get('response'))) if result.get('response') else 0,
            'device': result.get('device', 'unknown'),
            'auto_learning_enhanced': result.get('auto_learning_enhanced', False)
        }

        # Extract consciousness metrics if available
        if result.get('status') == 'success' and isinstance(result.get('response'), dict):
            response = result['response']

            # Try to extract FCI and consciousness state from different response structures
            if 'metrics' in response:
                metrics['fci'] = response['metrics'].get('fci', 0.0)
                metrics['consciousness_state'] = response['metrics'].get('consciousness_state', 'unknown')
            elif 'full_result' in response and 'consciousness_results' in response['full_result']:
                consciousness_results = response['full_result']['consciousness_results']
                metrics['fci'] = consciousness_results.get('FCI', 0.0)
                metrics['consciousness_state'] = consciousness_results.get('consciousness_state', {}).get('name', 'unknown')

            # Extract generated text
            if 'output' in response:
                metrics['generated_text'] = response['output']
            elif 'text_analysis' in response:
                metrics['generated_text'] = response['text_analysis']
            else:
                metrics['generated_text'] = str(response)[:200]  # Truncate if too long
        else:
            # Fallback for non-dict responses
            metrics['fci'] = 0.0
            metrics['consciousness_state'] = 'unknown'
            metrics['generated_text'] = str(result.get('response', ''))[:200]

        # Determine cognitive generation mode
        fci = metrics.get('fci', 0.0)
        if fci >= 0.3:
            metrics['cognitive_mode'] = 'GENERATION'
            metrics['mode_reason'] = f'FCI={fci:.3f} >= 0.3'
        else:
            metrics['cognitive_mode'] = 'ANALYSIS'
            metrics['mode_reason'] = f'FCI={fci:.3f} < 0.3'

        # Analyze response diversity
        generated_text = metrics.get('generated_text', '')
        if generated_text:
            metrics['unique_chars'] = len(set(generated_text))
            metrics['text_diversity'] = metrics['unique_chars'] / max(1, len(generated_text))
        else:
            metrics['unique_chars'] = 0
            metrics['text_diversity'] = 0.0

        print(f"   ✅ FCI: {metrics.get('fci', 0.0):.3f}")
        print(f"   🧠 State: {metrics.get('consciousness_state', 'unknown')}")
        print(f"   📝 Mode: {metrics.get('cognitive_mode', 'unknown')}")
        print(f"   🎯 Response: '{generated_text[:50]}...'")
        print(f"   🔄 Diversity: {metrics.get('text_diversity', 0.0):.3f} ({metrics.get('unique_chars', 0)} unique chars)")

        return metrics

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'prompt': prompt,
            'test_id': test_id,
            'status': 'error',
            'error': str(e),
            'fci': 0.0,
            'consciousness_state': 'error',
            'cognitive_mode': 'ERROR',
            'mode_reason': f'Exception: {e}',
            'unique_chars': 0,
            'text_diversity': 0.0
        }


def generate_summary_report(results: List[Dict[str, Any]]) -> str:
    """
    Generate a comprehensive summary report from test results.

    Args:
        results: List of test result dictionaries

    Returns:
        Formatted summary report
    """
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Basic statistics
    successful_tests = len([r for r in results if r['status'] == 'success'])
    total_tests = len(results)
    success_rate = (successful_tests / total_tests) * 100

    # FCI analysis
    fci_values = [r.get('fci', 0.0) for r in results if r['status'] == 'success']
    avg_fci = sum(fci_values) / len(fci_values) if fci_values else 0.0
    max_fci = max(fci_values) if fci_values else 0.0
    min_fci = min(fci_values) if fci_values else 0.0

    # Mode analysis
    generation_mode_count = len([r for r in results if r.get('cognitive_mode') == 'GENERATION'])
    analysis_mode_count = len([r for r in results if r.get('cognitive_mode') == 'ANALYSIS'])
    error_count = len([r for r in results if r['status'] == 'error'])

    # Consciousness state distribution
    state_counts = {}
    for r in results:
        if r['status'] == 'success':
            state = r.get('consciousness_state', 'unknown')
            state_counts[state] = state_counts.get(state, 0) + 1

    # Generate report
    report = f"""
ΨQRH AUTOCALIBRATION DIAGNOSTIC REPORT
=======================================

EXECUTIVE SUMMARY
-----------------
Total Tests: {total_tests}
Successful: {successful_tests} ({success_rate:.1f}%)
Errors: {error_count}

CONSCIOUSNESS METRICS
---------------------
Average FCI: {avg_fci:.3f}
Maximum FCI: {max_fci:.3f}
Minimum FCI: {min_fci:.3f}

COGNITIVE MODE DISTRIBUTION
---------------------------
Generation Mode: {generation_mode_count} tests
Analysis Mode: {analysis_mode_count} tests
Error Mode: {error_count} tests

CONSCIOUSNESS STATE DISTRIBUTION
--------------------------------
"""

    for state, count in state_counts.items():
        percentage = (count / successful_tests) * 100 if successful_tests > 0 else 0
        report += f"{state}: {count} tests ({percentage:.1f}%)\n"

    report += f"""
DETAILED TEST RESULTS
---------------------
"""

    # Add detailed results table
    for i, result in enumerate(results, 1):
        report += f"\n{i:2d}. '{result['prompt'][:50]}...'\n"
        report += f"    Status: {result['status']}\n"
        if result['status'] == 'success':
            report += f"    FCI: {result.get('fci', 0.0):.3f}\n"
            report += f"    State: {result.get('consciousness_state', 'unknown')}\n"
            report += f"    Mode: {result.get('cognitive_mode', 'unknown')} ({result.get('mode_reason', '')})\n"
            generated_text = result.get('generated_text', '')
            if len(generated_text) > 100:
                generated_text = generated_text[:100] + "..."
            report += f"    Response: {generated_text}\n"
        else:
            report += f"    Error: {result.get('error', 'Unknown error')}\n"

    # Add performance insights
    report += f"""

PERFORMANCE INSIGHTS
--------------------
"""

    if avg_fci >= 0.3:
        report += "✅ System consistently achieves conscious states (FCI ≥ 0.3)\n"
    else:
        report += "⚠️  System struggles to reach conscious states (FCI < 0.3)\n"

    if generation_mode_count > analysis_mode_count:
        report += "✅ Generation mode dominates - system is actively generating responses\n"
    else:
        report += "⚠️  Analysis mode dominates - system may be stuck in diagnostic mode\n"

    if success_rate >= 80:
        report += "✅ High success rate indicates stable pipeline operation\n"
    else:
        report += "⚠️  Low success rate suggests pipeline instability\n"

    return report


def main():
    """Main function to run autocalibration diagnostics."""
    print("🚀 ΨQRH Autocalibration Diagnostic Tool")
    print("=" * 50)
    print(f"Testing {len(TEST_PROMPTS)} prompts of varying complexity...")
    print()

    # Run all tests
    results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        result = run_pipeline_test(prompt, test_id=i)
        results.append(result)

    # Generate and display report
    print("\n" + "=" * 50)
    print("📊 GENERATING DIAGNOSTIC REPORT...")
    print("=" * 50)

    report = generate_summary_report(results)
    print(report)

    # Save detailed results to file
    output_file = BASE_DIR / "autocalibration_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed results saved to: {output_file}")

    # Return exit code based on success rate
    successful_tests = len([r for r in results if r['status'] == 'success'])
    if successful_tests >= len(TEST_PROMPTS) * 0.8:  # 80% success threshold
        print("\n✅ Autocalibration completed successfully!")
        return 0
    else:
        print(f"\n⚠️  Autocalibration completed with issues ({successful_tests}/{len(TEST_PROMPTS)} successful)")
        return 1


if __name__ == "__main__":
    sys.exit(main())