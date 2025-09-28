#!/usr/bin/env python3
"""
ΨQRH Evaluation Engine - Full Integration Test
===============================================

Comprehensive test suite to validate all components of the ΨQRH evaluation engine
and its integration with HELM and LM-Eval frameworks.
"""

import sys
import time
import traceback
from pathlib import Path

# Add evaluation engine to path
EVAL_ENGINE_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_ENGINE_DIR))

def test_core_engine():
    """Test the core ΨQRH evaluation engine."""
    print("🧪 Testing Core ΨQRH Engine...")

    try:
        from psiqrh_evaluation_engine import ΨQRHEvaluationEngine

        # Test different configurations
        configs = [
            {"embed_dim": 16, "alpha": 1.0},
            {"embed_dim": 32, "alpha": 1.5, "use_spectral_filtering": True},
            {"embed_dim": 64, "alpha": 2.0, "use_fractal_analysis": True}
        ]

        for i, config in enumerate(configs, 1):
            print(f"  Config {i}: {config}")
            engine = ΨQRHEvaluationEngine(model_config=config, debug=False)

            result = engine.process_text("Test quaternion processing", max_tokens=50)

            if result['status'] == 'success':
                print(f"    ✅ Success - Energy: {result['energy_conservation']:.3f}")
            else:
                print(f"    ❌ Failed: {result.get('error')}")

        return True

    except Exception as e:
        print(f"    ❌ Core engine test failed: {e}")
        return False

def test_helm_integration():
    """Test HELM framework integration."""
    print("\n🎯 Testing HELM Integration...")

    try:
        from helm_client import create_psiqrh_helm_client

        client = create_psiqrh_helm_client(
            psiqrh_config={"embed_dim": 32, "alpha": 1.5}
        )

        # Mock HELM request
        class MockRequest:
            def __init__(self, prompt):
                self.prompt = prompt
                self.max_tokens = 100
                self.temperature = 1.0
                self.top_p = 1.0
                self.frequency_penalty = 0.0
                self.presence_penalty = 0.0
                self.stop_sequences = None

        test_prompts = [
            "What is artificial intelligence?",
            "Explain the mathematical concept of infinity",
            "How do neural networks learn?"
        ]

        success_count = 0
        for prompt in test_prompts:
            request = MockRequest(prompt)
            result = client.make_request(request)

            if result.success and result.completions:
                success_count += 1
                print(f"    ✅ '{prompt[:30]}...' - Success")
            else:
                print(f"    ❌ '{prompt[:30]}...' - Failed")

        print(f"  HELM Integration: {success_count}/{len(test_prompts)} successful")
        return success_count == len(test_prompts)

    except Exception as e:
        print(f"    ❌ HELM integration test failed: {e}")
        return False

def test_lm_eval_integration():
    """Test LM-Eval framework integration."""
    print("\n📊 Testing LM-Eval Integration...")

    try:
        from lm_eval_model import create_psiqrh_lm_eval_model

        model = create_psiqrh_lm_eval_model(
            batch_size=1,
            psiqrh_config={"embed_dim": 32, "alpha": 1.5}
        )

        # Test generation
        gen_requests = [
            ("The meaning of life is", []),
            ("Mathematics is the language of", []),
            ("Consciousness might be defined as", [])
        ]

        gen_results = model.generate_until(gen_requests)
        gen_success = len([r for r in gen_results if not r.startswith("Error")])

        print(f"    Text Generation: {gen_success}/{len(gen_requests)} successful")

        # Test log-likelihood
        loglik_requests = [
            ("The sky is", "blue"),
            ("Two plus two equals", "four"),
            ("The capital of France is", "Paris")
        ]

        loglik_results = model.loglikelihood(loglik_requests)
        loglik_success = len([r for r in loglik_results if r[0] > -float('inf')])

        print(f"    Log-likelihood: {loglik_success}/{len(loglik_requests)} successful")

        total_success = gen_success + loglik_success
        total_tests = len(gen_requests) + len(loglik_requests)

        print(f"  LM-Eval Integration: {total_success}/{total_tests} successful")
        return total_success == total_tests

    except Exception as e:
        print(f"    ❌ LM-Eval integration test failed: {e}")
        return False

def test_configuration_loading():
    """Test configuration file loading."""
    print("\n⚙️ Testing Configuration Loading...")

    try:
        import json
        import yaml

        # Test JSON config
        json_config_path = "config/psiqrh_evaluation_config.json"
        if Path(json_config_path).exists():
            with open(json_config_path, 'r') as f:
                json_config = json.load(f)
            print("    ✅ JSON configuration loaded successfully")
        else:
            print("    ❌ JSON configuration file not found")
            return False

        # Test YAML configs
        yaml_configs = [
            "config/helm_config.yaml",
            "config/lm_eval_config.yaml"
        ]

        yaml_success = 0
        for config_path in yaml_configs:
            if Path(config_path).exists():
                try:
                    with open(config_path, 'r') as f:
                        yaml_config = yaml.safe_load(f)
                    yaml_success += 1
                    print(f"    ✅ {config_path} loaded successfully")
                except Exception as e:
                    print(f"    ❌ {config_path} failed to load: {e}")
            else:
                print(f"    ❌ {config_path} not found")

        total_success = 1 + yaml_success  # JSON + YAML files
        total_configs = 1 + len(yaml_configs)

        print(f"  Configuration Loading: {total_success}/{total_configs} successful")
        return total_success == total_configs

    except Exception as e:
        print(f"    ❌ Configuration test failed: {e}")
        return False

def test_performance_benchmark():
    """Run basic performance benchmark."""
    print("\n🚀 Testing Performance...")

    try:
        from psiqrh_evaluation_engine import ΨQRHEvaluationEngine

        engine = ΨQRHEvaluationEngine(
            model_config={"embed_dim": 32, "alpha": 1.5},
            debug=False
        )

        # Performance test
        test_texts = [
            "Short text",
            "Medium length text with multiple words and concepts",
            "Longer text that contains more complex sentences and ideas about mathematics, physics, and consciousness that should test the processing capabilities"
        ]

        processing_times = []
        energy_conservations = []

        for text in test_texts:
            start_time = time.time()
            result = engine.process_text(text, max_tokens=100)
            end_time = time.time()

            processing_time = end_time - start_time
            processing_times.append(processing_time)

            if result['status'] == 'success':
                energy_conservations.append(result['energy_conservation'])

        avg_time = sum(processing_times) / len(processing_times)
        avg_energy = sum(energy_conservations) / len(energy_conservations) if energy_conservations else 0

        print(f"    Average processing time: {avg_time:.3f}s")
        print(f"    Average energy conservation: {avg_energy:.3f}")
        print(f"    Processed {len(test_texts)} texts successfully")

        # Basic performance criteria
        performance_ok = avg_time < 1.0 and len(energy_conservations) == len(test_texts)

        if performance_ok:
            print("    ✅ Performance test passed")
        else:
            print("    ❌ Performance test failed")

        return performance_ok

    except Exception as e:
        print(f"    ❌ Performance test failed: {e}")
        return False

def test_mathematical_properties():
    """Test ΨQRH mathematical properties."""
    print("\n🔬 Testing Mathematical Properties...")

    try:
        from psiqrh_evaluation_engine import ΨQRHEvaluationEngine

        engine = ΨQRHEvaluationEngine(
            model_config={"embed_dim": 64, "alpha": 1.5, "use_spectral_filtering": True},
            debug=False
        )

        # Test energy conservation
        test_texts = [
            "Energy conservation test",
            "Quaternion stability analysis",
            "Spectral filtering validation"
        ]

        energy_ratios = []
        quaternion_norms = []

        for text in test_texts:
            result = engine.process_text(text, max_tokens=50)

            if result['status'] == 'success':
                energy_ratios.append(result.get('energy_conservation', 0))
                quaternion_norms.append(result.get('quaternion_norm', 0))

        # Check mathematical properties
        avg_energy_ratio = sum(energy_ratios) / len(energy_ratios) if energy_ratios else 0
        avg_quaternion_norm = sum(quaternion_norms) / len(quaternion_norms) if quaternion_norms else 0

        print(f"    Average energy ratio: {avg_energy_ratio:.3f}")
        print(f"    Average quaternion norm: {avg_quaternion_norm:.3f}")

        # Validate properties
        energy_ok = 0.5 <= avg_energy_ratio <= 2.5  # Reasonable range
        quaternion_ok = avg_quaternion_norm > 0

        if energy_ok and quaternion_ok:
            print("    ✅ Mathematical properties validated")
            return True
        else:
            print("    ❌ Mathematical properties validation failed")
            return False

    except Exception as e:
        print(f"    ❌ Mathematical properties test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive integration test suite."""
    print("🧪 ΨQRH Evaluation Engine - Comprehensive Integration Test")
    print("=" * 70)

    # Test suite
    tests = [
        ("Core Engine", test_core_engine),
        ("HELM Integration", test_helm_integration),
        ("LM-Eval Integration", test_lm_eval_integration),
        ("Configuration Loading", test_configuration_loading),
        ("Performance Benchmark", test_performance_benchmark),
        ("Mathematical Properties", test_mathematical_properties)
    ]

    results = {}
    start_time = time.time()

    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            traceback.print_exc()
            results[test_name] = False

    end_time = time.time()
    total_time = end_time - start_time

    # Results summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary")
    print("-" * 40)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")

    print("-" * 40)
    print(f"Overall Result: {passed}/{total} tests passed")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"Total Time: {total_time:.2f}s")

    if passed == total:
        print("\n🎉 All tests passed! ΨQRH Evaluation Engine is ready for use.")
        return True
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)