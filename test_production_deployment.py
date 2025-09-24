#!/usr/bin/env python3
"""
PRODUCTION DEPLOYMENT VALIDATION TESTS

Comprehensive test suite for production readiness validation:
1. Performance benchmarks against targets
2. Real-world scenario validation
3. Health monitoring verification
4. Deployment optimization testing
5. Error handling and recovery
"""

import torch
import torch.nn as nn
import time
import warnings
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict

from production_system import (
    ProductionSemanticQRH, ProductionConfig, ProductionMode, ProductionHealthMonitor
)
from real_world_scenarios import ScenarioType


class ProductionValidationSuite:
    """Complete production validation test suite"""

    def __init__(self):
        self.results = {}
        self.test_configs = self._create_test_configurations()

    def _create_test_configurations(self) -> Dict[str, ProductionConfig]:
        """Create test configurations for different deployment modes"""
        return {
            'high_performance': ProductionConfig(
                mode=ProductionMode.HIGH_PERFORMANCE,
                embed_dim=24,
                target_latency_ms=30.0,
                target_throughput_tokens_per_sec=3000.0
            ),
            'high_accuracy': ProductionConfig(
                mode=ProductionMode.HIGH_ACCURACY,
                embed_dim=48,
                target_latency_ms=100.0,
                target_throughput_tokens_per_sec=1000.0
            ),
            'balanced': ProductionConfig(
                mode=ProductionMode.BALANCED,
                embed_dim=32,
                target_latency_ms=50.0,
                target_throughput_tokens_per_sec=2000.0
            ),
            'memory_efficient': ProductionConfig(
                mode=ProductionMode.MEMORY_EFFICIENT,
                embed_dim=16,
                max_sequence_length=256,
                batch_size=4,
                max_memory_mb=50.0
            )
        }

    def run_complete_validation(self) -> Dict:
        """Execute complete production validation suite"""
        print("🚀 Starting Complete Production Validation Suite")
        print("=" * 60)

        overall_results = {
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode_results': {},
            'overall_summary': {}
        }

        # Test each production mode
        for mode_name, config in self.test_configs.items():
            print(f"\n📊 Testing {mode_name.upper()} mode...")
            mode_results = self._test_production_mode(mode_name, config)
            overall_results['mode_results'][mode_name] = mode_results

            # Print mode summary
            self._print_mode_summary(mode_name, mode_results)

        # Generate overall summary
        overall_results['overall_summary'] = self._generate_overall_summary(overall_results['mode_results'])

        print("\n" + "=" * 60)
        print("📋 OVERALL VALIDATION SUMMARY")
        print("=" * 60)
        self._print_overall_summary(overall_results['overall_summary'])

        return overall_results

    def _test_production_mode(self, mode_name: str, config: ProductionConfig) -> Dict:
        """Test a specific production mode configuration"""
        try:
            # Initialize production system
            system = ProductionSemanticQRH(config)
            system.optimize_for_deployment()

            results = {
                'config': asdict(config),
                'initialization': {'status': 'success'},
                'performance_tests': {},
                'scenario_tests': {},
                'health_monitoring': {},
                'error_handling': {},
                'deployment_readiness': {}
            }

            # 1. Performance benchmarks
            print(f"  ⚡ Running performance benchmarks...")
            results['performance_tests'] = self._test_performance(system, config)

            # 2. Real-world scenario validation
            print(f"  🌍 Testing real-world scenarios...")
            results['scenario_tests'] = self._test_scenarios(system, config)

            # 3. Health monitoring validation
            print(f"  🏥 Validating health monitoring...")
            results['health_monitoring'] = self._test_health_monitoring(system, config)

            # 4. Error handling and recovery
            print(f"  🛡️ Testing error handling...")
            results['error_handling'] = self._test_error_handling(system, config)

            # 5. Deployment readiness check
            print(f"  📦 Checking deployment readiness...")
            results['deployment_readiness'] = self._test_deployment_readiness(system, config)

            return results

        except Exception as e:
            return {
                'config': asdict(config),
                'initialization': {'status': 'failed', 'error': str(e)},
                'fatal_error': True
            }

    def _test_performance(self, system: ProductionSemanticQRH, config: ProductionConfig) -> Dict:
        """Test performance against targets"""
        batch_sizes = [1, 2, 4, config.batch_size]
        sequence_lengths = [16, 32, 64, min(128, config.max_sequence_length)]

        performance_results = {
            'latency_tests': [],
            'throughput_tests': [],
            'memory_tests': [],
            'targets_met': {}
        }

        # Latency tests
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                if batch_size <= config.batch_size and seq_len <= config.max_sequence_length:
                    latencies = []

                    for _ in range(10):  # Multiple runs for accurate measurement
                        input_tensor = torch.randn(batch_size, seq_len, config.embed_dim * 4)

                        start_time = time.time()
                        with torch.no_grad():
                            _ = system(input_tensor)
                        end_time = time.time()

                        latencies.append((end_time - start_time) * 1000)  # Convert to ms

                    avg_latency = sum(latencies) / len(latencies)
                    min_latency = min(latencies)
                    max_latency = max(latencies)

                    performance_results['latency_tests'].append({
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                        'avg_latency_ms': avg_latency,
                        'min_latency_ms': min_latency,
                        'max_latency_ms': max_latency,
                        'meets_target': avg_latency <= config.target_latency_ms
                    })

        # Throughput tests
        batch_size = config.batch_size
        seq_len = min(64, config.max_sequence_length)
        num_batches = 50

        input_tensor = torch.randn(batch_size, seq_len, config.embed_dim * 4)
        total_tokens = batch_size * seq_len * num_batches

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_batches):
                _ = system(input_tensor)
        end_time = time.time()

        throughput = total_tokens / (end_time - start_time)

        performance_results['throughput_tests'].append({
            'tokens_per_second': throughput,
            'total_tokens': total_tokens,
            'total_time': end_time - start_time,
            'meets_target': throughput >= config.target_throughput_tokens_per_sec
        })

        # Memory usage (if CUDA)
        if config.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            input_tensor = torch.randn(config.batch_size, config.max_sequence_length, config.embed_dim * 4, device='cuda')
            _ = system(input_tensor)

            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            performance_results['memory_tests'].append({
                'peak_memory_mb': peak_memory_mb,
                'meets_target': peak_memory_mb <= config.max_memory_mb
            })

        # Evaluate targets
        avg_latencies = [test['avg_latency_ms'] for test in performance_results['latency_tests']]
        latency_target_met = all(lat <= config.target_latency_ms for lat in avg_latencies) if avg_latencies else False

        throughput_target_met = all(test['meets_target'] for test in performance_results['throughput_tests'])

        memory_target_met = all(test['meets_target'] for test in performance_results['memory_tests']) if performance_results['memory_tests'] else True

        performance_results['targets_met'] = {
            'latency': latency_target_met,
            'throughput': throughput_target_met,
            'memory': memory_target_met,
            'overall': latency_target_met and throughput_target_met and memory_target_met
        }

        return performance_results

    def _test_scenarios(self, system: ProductionSemanticQRH, config: ProductionConfig) -> Dict:
        """Test real-world scenarios"""
        scenario_results = {
            'conversation_tests': [],
            'document_tests': [],
            'mixed_content_tests': [],
            'overall_success_rate': 0.0
        }

        test_scenarios = [
            (ScenarioType.CONVERSATION, "conversation", 10),
            (ScenarioType.DOCUMENT, "document", 10),
            (ScenarioType.MIXED_CONTENT, "mixed_content", 10)
        ]

        total_tests = 0
        successful_tests = 0

        for scenario_type, scenario_name, num_tests in test_scenarios:
            scenario_successes = 0

            for i in range(num_tests):
                try:
                    # Generate test input
                    seq_len = min(32, config.max_sequence_length)
                    input_tensor = torch.randn(2, seq_len, config.embed_dim * 4)

                    # Run scenario test
                    output, metrics = system(input_tensor, scenario_type=scenario_type, return_detailed_metrics=True)

                    # Evaluate success based on quality metrics
                    quality_score = metrics.get('signal_clarity_score', 0.0)
                    success = quality_score >= config.min_signal_clarity

                    if success:
                        scenario_successes += 1
                        successful_tests += 1

                    total_tests += 1

                except Exception as e:
                    total_tests += 1
                    # Failure recorded implicitly (scenario_successes not incremented)

            scenario_success_rate = scenario_successes / num_tests if num_tests > 0 else 0.0
            scenario_results[f'{scenario_name}_tests'].append({
                'scenario_type': scenario_type.value,
                'success_rate': scenario_success_rate,
                'successful_runs': scenario_successes,
                'total_runs': num_tests,
                'meets_target': scenario_success_rate >= 0.8  # 80% target
            })

        scenario_results['overall_success_rate'] = successful_tests / total_tests if total_tests > 0 else 0.0

        return scenario_results

    def _test_health_monitoring(self, system: ProductionSemanticQRH, config: ProductionConfig) -> Dict:
        """Test health monitoring functionality"""
        if not config.enable_health_checks:
            return {'status': 'disabled', 'monitoring_functional': False}

        # Generate some forward passes to populate health data
        input_tensor = torch.randn(2, 16, config.embed_dim * 4)

        for i in range(config.health_check_interval + 5):
            try:
                _ = system(input_tensor)
            except Exception:
                pass  # Some failures expected for error rate testing

        # Get health report
        health_status = system.get_health_status()
        system_info = system.get_system_info()

        return {
            'status': 'enabled',
            'monitoring_functional': health_status.get('status') in ['HEALTHY', 'DEGRADED'],
            'health_report': health_status,
            'system_info': system_info,
            'metrics_tracked': len(health_status.get('metrics', {})) > 0,
            'recommendations_provided': len(health_status.get('recommendations', [])) > 0
        }

    def _test_error_handling(self, system: ProductionSemanticQRH, config: ProductionConfig) -> Dict:
        """Test error handling and recovery"""
        error_tests = []

        # Test 1: Invalid input dimensions
        try:
            invalid_input = torch.randn(2, 16, 10)  # Wrong embed_dim
            _ = system(invalid_input)
            error_tests.append({'test': 'invalid_dimensions', 'handled': False})
        except ValueError:
            error_tests.append({'test': 'invalid_dimensions', 'handled': True})
        except Exception:
            error_tests.append({'test': 'invalid_dimensions', 'handled': False})

        # Test 2: NaN input
        try:
            nan_input = torch.full((2, 16, config.embed_dim * 4), float('nan'))
            _ = system(nan_input)
            error_tests.append({'test': 'nan_input', 'handled': False})
        except ValueError:
            error_tests.append({'test': 'nan_input', 'handled': True})
        except Exception:
            error_tests.append({'test': 'nan_input', 'handled': False})

        # Test 3: Oversized sequence
        try:
            if config.max_sequence_length < 1000:  # Only test if limit is reasonable
                oversized_input = torch.randn(1, config.max_sequence_length + 100, config.embed_dim * 4)
                _ = system(oversized_input)
                error_tests.append({'test': 'oversized_sequence', 'handled': False})
        except ValueError:
            error_tests.append({'test': 'oversized_sequence', 'handled': True})
        except Exception:
            error_tests.append({'test': 'oversized_sequence', 'handled': False})

        total_tests = len(error_tests)
        handled_correctly = sum(1 for test in error_tests if test['handled'])

        return {
            'error_tests': error_tests,
            'total_error_tests': total_tests,
            'correctly_handled': handled_correctly,
            'error_handling_rate': handled_correctly / total_tests if total_tests > 0 else 0.0
        }

    def _test_deployment_readiness(self, system: ProductionSemanticQRH, config: ProductionConfig) -> Dict:
        """Test deployment readiness"""
        readiness_checks = {}

        # Check model is in eval mode
        readiness_checks['eval_mode'] = not system.training

        # Check JIT compilation (if enabled)
        if config.enable_jit_compilation:
            readiness_checks['jit_compiled'] = hasattr(system, 'optimized_qrh') and isinstance(system.optimized_qrh, torch.jit.ScriptModule)
        else:
            readiness_checks['jit_compiled'] = True  # Not required

        # Check warmup completed (forward passes > 0)
        readiness_checks['warmed_up'] = system.forward_count > 0

        # Check configuration validity
        readiness_checks['config_valid'] = all([
            config.embed_dim > 0,
            config.max_sequence_length > 0,
            config.batch_size > 0,
            config.target_latency_ms > 0,
            config.target_throughput_tokens_per_sec > 0
        ])

        # Check health monitoring (if enabled)
        if config.enable_health_checks:
            health_status = system.get_health_status()
            readiness_checks['health_monitoring_active'] = health_status.get('status') != 'NO_DATA'
        else:
            readiness_checks['health_monitoring_active'] = True  # Not required

        # Overall readiness
        all_ready = all(readiness_checks.values())

        return {
            'deployment_ready': all_ready,
            'readiness_checks': readiness_checks,
            'deployment_score': sum(readiness_checks.values()) / len(readiness_checks)
        }

    def _print_mode_summary(self, mode_name: str, results: Dict):
        """Print summary for a specific mode"""
        print(f"    📈 {mode_name.upper()} Results:")

        if results.get('fatal_error'):
            print(f"      ❌ FATAL ERROR: {results['initialization']['error']}")
            return

        # Performance summary
        perf = results.get('performance_tests', {})
        if perf.get('targets_met', {}).get('overall'):
            print(f"      ✅ Performance: All targets met")
        else:
            print(f"      ⚠️ Performance: Some targets missed")

        # Scenario summary
        scenario = results.get('scenario_tests', {})
        success_rate = scenario.get('overall_success_rate', 0.0)
        if success_rate >= 0.8:
            print(f"      ✅ Scenarios: {success_rate:.1%} success rate")
        else:
            print(f"      ⚠️ Scenarios: {success_rate:.1%} success rate (target: 80%)")

        # Deployment readiness
        deployment = results.get('deployment_readiness', {})
        if deployment.get('deployment_ready'):
            print(f"      ✅ Deployment: Ready")
        else:
            print(f"      ⚠️ Deployment: Issues detected")

    def _generate_overall_summary(self, mode_results: Dict) -> Dict:
        """Generate overall validation summary"""
        total_modes = len(mode_results)
        successful_modes = 0
        performance_issues = 0
        scenario_issues = 0
        deployment_issues = 0

        for mode_name, results in mode_results.items():
            if results.get('fatal_error'):
                continue

            # Count successful modes (all targets met)
            perf_ok = results.get('performance_tests', {}).get('targets_met', {}).get('overall', False)
            scenario_ok = results.get('scenario_tests', {}).get('overall_success_rate', 0.0) >= 0.8
            deploy_ok = results.get('deployment_readiness', {}).get('deployment_ready', False)

            if perf_ok and scenario_ok and deploy_ok:
                successful_modes += 1

            if not perf_ok:
                performance_issues += 1
            if not scenario_ok:
                scenario_issues += 1
            if not deploy_ok:
                deployment_issues += 1

        return {
            'total_modes_tested': total_modes,
            'successful_modes': successful_modes,
            'overall_success_rate': successful_modes / total_modes if total_modes > 0 else 0.0,
            'performance_issues': performance_issues,
            'scenario_issues': scenario_issues,
            'deployment_issues': deployment_issues,
            'production_ready': successful_modes == total_modes and total_modes > 0
        }

    def _print_overall_summary(self, summary: Dict):
        """Print overall validation summary"""
        if summary['production_ready']:
            print("🎉 PRODUCTION READY: All modes passed validation!")
        else:
            print("⚠️ PRODUCTION ISSUES DETECTED")

        print(f"📊 Success Rate: {summary['overall_success_rate']:.1%} ({summary['successful_modes']}/{summary['total_modes_tested']} modes)")

        if summary['performance_issues'] > 0:
            print(f"⚡ Performance Issues: {summary['performance_issues']} modes")

        if summary['scenario_issues'] > 0:
            print(f"🌍 Scenario Issues: {summary['scenario_issues']} modes")

        if summary['deployment_issues'] > 0:
            print(f"📦 Deployment Issues: {summary['deployment_issues']} modes")


def main():
    """Run complete production validation suite"""
    print("🔬 Enhanced Semantic QRH System - Production Validation")
    print("Testing production readiness with performance, quality, and deployment validation")
    print()

    # Initialize validation suite
    validator = ProductionValidationSuite()

    # Run complete validation
    results = validator.run_complete_validation()

    # Save detailed results
    results_file = 'production_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {results_file}")

    return results


if __name__ == "__main__":
    results = main()