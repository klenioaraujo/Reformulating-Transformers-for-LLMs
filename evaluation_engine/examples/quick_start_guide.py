#!/usr/bin/env python3
"""
ΨQRH Evaluation Engine - Quick Start Guide
==========================================

This script provides practical examples of how to use the ΨQRH Evaluation Engine
for integrating with HELM and LM-Eval frameworks.

Run this script to see the engine in action and understand how to integrate
ΨQRH models with standard evaluation frameworks.

Usage:
    python quick_start_guide.py
"""

import sys
import os
from pathlib import Path

# Add evaluation engine to path
EVAL_ENGINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EVAL_ENGINE_DIR))

def main():
    print("🚀 ΨQRH Evaluation Engine - Quick Start Guide")
    print("=" * 60)
    print()

    # Example 1: Basic Engine Usage
    print("📋 Example 1: Basic Engine Usage")
    print("-" * 40)

    try:
        from psiqrh_evaluation_engine import ΨQRHEvaluationEngine

        # Initialize engine with custom configuration
        config = {
            'embed_dim': 32,  # Smaller for demo
            'alpha': 1.5,
            'beta': 0.01,
            'use_spectral_filtering': True,
            'use_fractal_analysis': True
        }

        engine = ΨQRHEvaluationEngine(model_config=config, debug=True)

        # Test various types of inputs
        test_inputs = [
            "What is the mathematical relationship between quaternions and 3D rotations?",
            "Explain the concept of fractal dimension in simple terms.",
            "How does the Fourier transform work in signal processing?",
            "What is consciousness and how might it relate to information processing?"
        ]

        print(f"Testing {len(test_inputs)} different input types:")
        print()

        for i, text in enumerate(test_inputs, 1):
            print(f"Test {i}: {text[:50]}...")
            result = engine.process_text(text, max_tokens=100)

            if result['status'] == 'success':
                print(f"  ✅ Success!")
                print(f"  📊 Fractal Dimension: {result['fractal_dimension']:.3f}")
                print(f"  ⚡ Energy Conservation: {result['energy_conservation']:.3f}")
                print(f"  🎯 Response: {result['text'][:80]}...")
            else:
                print(f"  ❌ Error: {result.get('error', 'Unknown')}")

            print()

    except Exception as e:
        print(f"❌ Error in basic engine test: {e}")
        print("This might be due to missing ΨQRH core components.")
        print("The engine will use fallback implementations for demonstration.")

    print()

    # Example 2: HELM Integration
    print("📋 Example 2: HELM Integration")
    print("-" * 40)

    try:
        from helm_client import create_psiqrh_helm_client

        # Create HELM-compatible client
        helm_client = create_psiqrh_helm_client(
            psiqrh_config={
                'embed_dim': 32,
                'alpha': 1.5,
                'use_spectral_filtering': True
            }
        )

        print("HELM client created successfully!")

        # Simulate HELM requests
        class MockHELMRequest:
            def __init__(self, prompt, max_tokens=100, temperature=1.0):
                self.prompt = prompt
                self.max_tokens = max_tokens
                self.temperature = temperature
                self.top_p = 1.0
                self.frequency_penalty = 0.0
                self.presence_penalty = 0.0
                self.stop_sequences = None

        helm_test_prompts = [
            "The theory of relativity states that",
            "In quantum mechanics, the uncertainty principle",
            "Quaternion multiplication is defined as"
        ]

        for i, prompt in enumerate(helm_test_prompts, 1):
            print(f"\nHELM Test {i}: {prompt}")
            helm_request = MockHELMRequest(prompt, max_tokens=80)
            result = helm_client.make_request(helm_request)

            if result.success:
                completion = result.completions[0]
                print(f"  ✅ Success! (Log Prob: {completion.logprob:.3f})")
                print(f"  📝 Response: {completion.text}")

                # Show ΨQRH metadata if available
                if hasattr(result, 'raw_response') and 'psiqrh_metadata' in result.raw_response:
                    metadata = result.raw_response['psiqrh_metadata']
                    if metadata.get('fractal_dimension'):
                        print(f"  🔬 Fractal Dim: {metadata['fractal_dimension']:.3f}")
                    if metadata.get('energy_conservation'):
                        print(f"  ⚡ Energy: {metadata['energy_conservation']:.3f}")
            else:
                print(f"  ❌ Failed: {result.raw_response.get('error', 'Unknown error')}")

        # Show performance statistics
        print(f"\n📊 HELM Client Performance:")
        stats = helm_client.get_performance_stats()
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Average Time: {stats['average_processing_time']:.3f}s")
        print(f"  Requests/sec: {stats['requests_per_second']:.2f}")

    except Exception as e:
        print(f"❌ Error in HELM integration test: {e}")

    print()

    # Example 3: LM-Eval Integration
    print("📋 Example 3: LM-Eval Integration")
    print("-" * 40)

    try:
        from lm_eval_model import create_psiqrh_lm_eval_model

        # Create LM-Eval-compatible model
        lm_eval_model = create_psiqrh_lm_eval_model(
            batch_size=1,
            psiqrh_config={
                'embed_dim': 32,
                'alpha': 1.5,
                'use_spectral_filtering': True
            }
        )

        print("LM-Eval model created successfully!")
        print(f"Model: {lm_eval_model.model_name}")
        print(f"Device: {lm_eval_model.device}")
        print(f"Max Length: {lm_eval_model.max_length}")

        # Test text generation
        print(f"\n🔧 Testing text generation:")
        generation_requests = [
            ("The capital of France is", [".", "\n"]),
            ("In mathematics, a quaternion", []),
            ("The Padilha Wave Equation describes", [".", "equation"])
        ]

        generation_results = lm_eval_model.generate_until(generation_requests)

        for i, (request, result) in enumerate(zip(generation_requests, generation_results)):
            context = request[0]
            print(f"  Request {i+1}: {context}")
            print(f"  Generated: {result}")
            print()

        # Test log-likelihood calculation
        print(f"🔧 Testing log-likelihood calculation:")
        loglik_requests = [
            ("The theory of relativity was developed by", "Einstein"),
            ("Two plus two equals", "four"),
            ("Quaternion multiplication is", "non-commutative")
        ]

        loglik_results = lm_eval_model.loglikelihood(loglik_requests)

        for i, ((context, completion), (logprob, is_greedy)) in enumerate(zip(loglik_requests, loglik_results)):
            print(f"  Request {i+1}: '{context}' + '{completion}'")
            print(f"  Log Prob: {logprob:.3f}, Greedy: {is_greedy}")
            print()

        # Show model information
        print(f"📊 Model Information:")
        model_info = lm_eval_model.get_model_info()
        print(f"  Framework: {model_info['framework']}")
        print(f"  Total Requests: {model_info['performance']['total_requests']}")
        print(f"  Total Tokens: {model_info['performance']['total_tokens_generated']}")

    except Exception as e:
        print(f"❌ Error in LM-Eval integration test: {e}")

    print()

    # Example 4: Configuration Examples
    print("📋 Example 4: Configuration Examples")
    print("-" * 40)

    print("Example ΨQRH configurations for different use cases:")
    print()

    configs = {
        "High Performance": {
            'embed_dim': 128,
            'alpha': 2.0,
            'beta': 0.005,
            'use_spectral_filtering': True,
            'use_fractal_analysis': True,
            'quaternion_precision': 'float32'
        },

        "Memory Efficient": {
            'embed_dim': 32,
            'alpha': 1.0,
            'beta': 0.02,
            'use_spectral_filtering': False,
            'use_fractal_analysis': True,
            'quaternion_precision': 'float16'
        },

        "Mathematical Focus": {
            'embed_dim': 64,
            'alpha': 1.5,
            'beta': 0.01,
            'use_spectral_filtering': True,
            'use_fractal_analysis': True,
            'quaternion_precision': 'double'
        }
    }

    for name, config in configs.items():
        print(f"🔧 {name} Configuration:")
        for key, value in config.items():
            print(f"    {key}: {value}")
        print()

    # Example 5: Evaluation Workflow
    print("📋 Example 5: Complete Evaluation Workflow")
    print("-" * 40)

    print("Step-by-step evaluation workflow:")
    print()

    workflow_steps = [
        "1. Configure ΨQRH parameters based on task requirements",
        "2. Initialize evaluation engine with desired framework",
        "3. Load evaluation datasets (HELM scenarios or LM-Eval tasks)",
        "4. Run evaluation with progress monitoring",
        "5. Collect standard metrics (accuracy, F1, etc.)",
        "6. Collect ΨQRH-specific metrics (energy conservation, etc.)",
        "7. Compare with baseline models",
        "8. Generate comprehensive report",
        "9. Analyze results and identify optimization opportunities"
    ]

    for step in workflow_steps:
        print(f"  {step}")

    print()
    print("📂 Configuration files are available in the config/ directory:")
    print("  - helm_config.yaml: HELM evaluation configuration")
    print("  - lm_eval_config.yaml: LM-Eval evaluation configuration")
    print("  - psiqrh_evaluation_config.json: Universal configuration")

    print()

    # Example 6: Best Practices
    print("📋 Example 6: Best Practices and Tips")
    print("-" * 40)

    best_practices = [
        "🎯 Choose embed_dim based on task complexity (32-128 for most tasks)",
        "⚡ Enable spectral filtering for improved accuracy",
        "🔬 Use fractal analysis for automatic parameter tuning",
        "💾 Enable caching for repeated evaluations",
        "📊 Monitor energy conservation ratio (should be close to 1.0)",
        "🔧 Start with default parameters and adjust based on results",
        "📈 Use batch processing for large-scale evaluations",
        "🏃 Profile performance to identify bottlenecks",
        "📋 Document configuration choices for reproducibility",
        "🔍 Validate mathematical properties before production use"
    ]

    print("Key best practices for ΨQRH evaluation:")
    print()
    for practice in best_practices:
        print(f"  {practice}")

    print()

    # Summary
    print("🎉 Quick Start Guide Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Explore the configuration files in config/")
    print("2. Run the integration tests with your own data")
    print("3. Compare ΨQRH with baseline models on your tasks")
    print("4. Optimize parameters based on evaluation results")
    print()
    print("For more information, see:")
    print("  - README.md: Complete documentation")
    print("  - examples/: Additional usage examples")
    print("  - config/: Configuration templates")
    print()
    print("Happy evaluating! 🚀")


if __name__ == "__main__":
    main()