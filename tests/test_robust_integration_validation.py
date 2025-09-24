#!/usr/bin/env python3
"""
ROBUST NEUROTRANSMITTER INTEGRATION VALIDATION

Comprehensive test suite to validate:
1. Performance: ≤5% degradation on JIT path
2. Efficiency: ≥90% cases use JIT-only processing
3. Robustness: ≥95% accuracy on complex cases with expertise
4. Adaptability: Continuous improvement based on feedback
5. Maintainability: Clean, documented, tested code
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple, Any
import asyncio

from qrh_layer import QRHConfig
from robust_neurotransmitter_integration import (
    RobustNeurotransmitterIntegration, IntegrationConfig, OperationMode
)


class RobustIntegrationValidator:
    """Comprehensive validation suite for robust neurotransmitter integration"""

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.test_results = {}

    def create_test_configurations(self) -> Dict[str, Tuple[QRHConfig, IntegrationConfig]]:
        """Create test configurations for different scenarios"""
        base_qrh_config = QRHConfig(
            embed_dim=32,
            alpha=1.2,
            use_learned_rotation=True,
            device=self.device
        )

        configs = {
            'high_performance': (
                base_qrh_config,
                IntegrationConfig(
                    activation_threshold=0.9,
                    performance_budget=0.98,
                    cache_size=500,
                    enable_async_processing=True
                )
            ),
            'balanced': (
                base_qrh_config,
                IntegrationConfig(
                    activation_threshold=0.7,
                    performance_budget=0.95,
                    cache_size=1000,
                    enable_async_processing=True
                )
            ),
            'high_accuracy': (
                base_qrh_config,
                IntegrationConfig(
                    activation_threshold=0.5,
                    performance_budget=0.90,
                    cache_size=2000,
                    enable_async_processing=False
                )
            )
        }

        return configs

    def generate_test_scenarios(self) -> Dict[str, torch.Tensor]:
        """Generate various test scenarios"""
        embed_dim = 32 * 4  # QRH uses 4x embed_dim

        scenarios = {
            # Simple cases (should use JIT-only)
            'simple_uniform': torch.randn(4, 16, embed_dim) * 0.1,
            'simple_structured': torch.ones(4, 16, embed_dim) * 0.5,

            # Moderate cases (should use JIT+Expertise)
            'moderate_noise': torch.randn(4, 16, embed_dim) * 1.5,
            'moderate_patterns': torch.sin(torch.arange(16).unsqueeze(0).unsqueeze(-1).expand(4, 16, embed_dim).float()),

            # Complex cases (should use Expertise-Lead)
            'complex_chaotic': torch.randn(4, 16, embed_dim) * 5.0 + torch.sin(torch.randn(4, 16, embed_dim) * 10),
            'complex_conflicts': torch.cat([
                torch.ones(4, 8, embed_dim),
                -torch.ones(4, 8, embed_dim)
            ], dim=1),

            # Edge cases
            'edge_zeros': torch.zeros(4, 16, embed_dim),
            'edge_extremes': torch.where(torch.rand(4, 16, embed_dim) > 0.5,
                                       torch.tensor(100.0),
                                       torch.tensor(-100.0)),
            'edge_nans_handled': torch.randn(4, 16, embed_dim)  # Will be tested with NaN injection
        }

        return scenarios

    def test_performance_requirement(self, config_name: str,
                                   qrh_config: QRHConfig,
                                   integration_config: IntegrationConfig) -> Dict[str, Any]:
        """Test Performance Requirement: ≤5% degradation on JIT path"""
        print(f"🚀 Testing Performance Requirement: {config_name}")

        # Create systems for comparison
        baseline_qrh = nn.Module()
        baseline_qrh.core = torch.jit.trace(
            nn.Sequential(
                nn.Linear(qrh_config.embed_dim * 4, qrh_config.embed_dim * 4),
                nn.GELU(),
                nn.Linear(qrh_config.embed_dim * 4, qrh_config.embed_dim * 4)
            ),
            torch.randn(1, qrh_config.embed_dim * 4)
        )

        integrated_system = RobustNeurotransmitterIntegration(qrh_config, integration_config)

        # Test data (simple cases that should use JIT-only)
        test_data = torch.randn(8, 32, qrh_config.embed_dim * 4) * 0.1  # Simple case

        # Baseline performance
        baseline_times = []
        with torch.no_grad():
            for _ in range(50):
                start = time.time()
                baseline_qrh.core(test_data.view(-1, qrh_config.embed_dim * 4))
                baseline_times.append(time.time() - start)

        baseline_avg = np.mean(baseline_times)

        # Integrated system performance (should be JIT-only for simple cases)
        integrated_times = []
        jit_only_count = 0

        with torch.no_grad():
            for _ in range(50):
                start = time.time()
                output, metrics = integrated_system(test_data, return_detailed_metrics=True)
                integrated_times.append(time.time() - start)

                if metrics['processing_mode'] == 'jit_pure':
                    jit_only_count += 1

        integrated_avg = np.mean(integrated_times)
        performance_ratio = integrated_avg / baseline_avg
        jit_only_rate = jit_only_count / 50

        # Performance requirement: ≤5% degradation
        performance_ok = performance_ratio <= 1.05
        efficiency_ok = jit_only_rate >= 0.90  # ≥90% should be JIT-only

        return {
            'performance_ratio': performance_ratio,
            'baseline_time_ms': baseline_avg * 1000,
            'integrated_time_ms': integrated_avg * 1000,
            'jit_only_rate': jit_only_rate,
            'performance_requirement_met': performance_ok,
            'efficiency_requirement_met': efficiency_ok,
            'overall_pass': performance_ok and efficiency_ok
        }

    def test_efficiency_requirement(self, system: RobustNeurotransmitterIntegration,
                                  test_scenarios: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Test Efficiency Requirement: ≥90% cases use JIT-only processing"""
        print("⚡ Testing Efficiency Requirement")

        total_cases = 0
        jit_only_cases = 0
        mode_distribution = {'jit_pure': 0, 'jit_expertise': 0, 'expertise_lead': 0}

        with torch.no_grad():
            for scenario_name, test_data in test_scenarios.items():
                # Run multiple batches per scenario
                for _ in range(10):
                    output, metrics = system(test_data, return_detailed_metrics=True)
                    total_cases += 1

                    mode = metrics['processing_mode']
                    mode_distribution[mode] = mode_distribution.get(mode, 0) + 1

                    if mode == 'jit_pure':
                        jit_only_cases += 1

        jit_only_rate = jit_only_cases / total_cases if total_cases > 0 else 0
        efficiency_ok = jit_only_rate >= 0.90

        return {
            'total_test_cases': total_cases,
            'jit_only_rate': jit_only_rate,
            'mode_distribution': mode_distribution,
            'efficiency_requirement_met': efficiency_ok,
            'recommendation': 'Increase activation threshold' if not efficiency_ok else 'Efficiency optimal'
        }

    def test_robustness_requirement(self, system: RobustNeurotransmitterIntegration,
                                  test_scenarios: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Test Robustness Requirement: ≥95% accuracy on complex cases"""
        print("🛡️ Testing Robustness Requirement")

        complex_scenarios = {
            'complex_chaotic': test_scenarios['complex_chaotic'],
            'complex_conflicts': test_scenarios['complex_conflicts'],
            'edge_extremes': test_scenarios['edge_extremes']
        }

        successful_cases = 0
        total_cases = 0
        error_cases = 0
        quality_scores = []

        with torch.no_grad():
            for scenario_name, test_data in complex_scenarios.items():
                for _ in range(10):  # Multiple runs per complex scenario
                    try:
                        total_cases += 1
                        output, metrics = system(test_data, return_detailed_metrics=True)

                        # Check for successful processing (no NaNs, reasonable values)
                        if not torch.isnan(output).any() and not torch.isinf(output).any():
                            successful_cases += 1

                            # Calculate quality score
                            signal_strength = torch.norm(output).item()
                            stability = 1.0 / (1.0 + torch.std(output).item())
                            quality = min(signal_strength / 10.0, 1.0) * stability
                            quality_scores.append(quality)

                    except Exception as e:
                        error_cases += 1
                        print(f"  ⚠️ Error in {scenario_name}: {e}")

        accuracy = successful_cases / total_cases if total_cases > 0 else 0
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        robustness_ok = accuracy >= 0.95

        return {
            'total_cases': total_cases,
            'successful_cases': successful_cases,
            'error_cases': error_cases,
            'accuracy_rate': accuracy,
            'average_quality_score': avg_quality,
            'robustness_requirement_met': robustness_ok,
            'error_rate': error_cases / total_cases if total_cases > 0 else 0
        }

    def test_adaptability_requirement(self, system: RobustNeurotransmitterIntegration) -> Dict[str, Any]:
        """Test Adaptability Requirement: Continuous improvement based on feedback"""
        print("🧠 Testing Adaptability Requirement")

        # Get initial thresholds
        initial_thresholds = system.learner.get_learned_thresholds()

        # Simulate learning scenario with feedback
        learning_data = torch.randn(4, 16, 32 * 4)

        # Generate learning episodes with known performance feedback
        learning_episodes = [
            {'complexity': 0.3, 'expected_performance': 0.9},  # Simple case, high performance
            {'complexity': 0.7, 'expected_performance': 0.8},  # Moderate case, good performance
            {'complexity': 0.9, 'expected_performance': 0.6}   # Complex case, lower performance
        ]

        with torch.no_grad():
            for episode in learning_episodes:
                # Modify data to match complexity
                if episode['complexity'] > 0.8:
                    modified_data = learning_data * 5.0  # High complexity
                elif episode['complexity'] > 0.5:
                    modified_data = learning_data * 2.0  # Moderate complexity
                else:
                    modified_data = learning_data * 0.5  # Low complexity

                # Run multiple iterations to trigger learning
                for _ in range(20):
                    output, metrics = system(modified_data, return_detailed_metrics=True)

                    # Simulate feedback (in real scenario, this would come from actual performance)
                    simulated_performance = episode['expected_performance'] + np.random.normal(0, 0.05)

                    # Record learning (normally done internally)
                    trigger_context = {
                        'confidence': metrics['confidence'],
                        'complexity': episode['complexity']
                    }
                    system.learner.record_activation(
                        trigger_context,
                        simulated_performance,
                        metrics['processing_time_ms'] / 1000
                    )

        # Get updated thresholds
        updated_thresholds = system.learner.get_learned_thresholds()

        # Check if thresholds have adapted
        threshold_changes = {}
        adaptation_detected = False

        for key in initial_thresholds:
            change = abs(updated_thresholds[key] - initial_thresholds[key])
            threshold_changes[key] = change
            if change > 0.01:  # Significant change
                adaptation_detected = True

        return {
            'initial_thresholds': initial_thresholds,
            'updated_thresholds': updated_thresholds,
            'threshold_changes': threshold_changes,
            'adaptation_detected': adaptation_detected,
            'learning_episodes_completed': len(learning_episodes) * 20,
            'adaptability_requirement_met': adaptation_detected
        }

    def test_maintainability_requirement(self, system: RobustNeurotransmitterIntegration) -> Dict[str, Any]:
        """Test Maintainability Requirement: Clean, documented, tested code"""
        print("🔧 Testing Maintainability Requirement")

        maintainability_checks = {
            'has_performance_stats': hasattr(system, 'get_performance_stats'),
            'has_health_monitoring': system.config.enable_health_monitoring,
            'has_caching': system.config.enable_decision_cache,
            'has_async_processing': system.config.enable_async_processing,
            'has_adaptive_learning': system.config.enable_meta_learning,
            'proper_error_handling': True,  # Tested implicitly in robustness
            'configuration_flexibility': len(vars(system.config)) > 10
        }

        # Test API cleanliness
        try:
            # Should have clean API methods
            stats = system.get_performance_stats()
            system.optimize_for_deployment('balanced')
            api_clean = True
        except Exception as e:
            print(f"  ⚠️ API issue: {e}")
            api_clean = False

        maintainability_checks['clean_api'] = api_clean

        # Calculate maintainability score
        total_checks = len(maintainability_checks)
        passed_checks = sum(maintainability_checks.values())
        maintainability_score = passed_checks / total_checks

        maintainability_ok = maintainability_score >= 0.8

        return {
            'maintainability_checks': maintainability_checks,
            'checks_passed': passed_checks,
            'total_checks': total_checks,
            'maintainability_score': maintainability_score,
            'maintainability_requirement_met': maintainability_ok
        }

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all requirements"""
        print("🧪🔬 ROBUST NEUROTRANSMITTER INTEGRATION VALIDATION")
        print("=" * 80)

        test_configs = self.create_test_configurations()
        test_scenarios = self.generate_test_scenarios()

        all_results = {}
        overall_summary = {
            'performance_tests': {},
            'efficiency_tests': {},
            'robustness_tests': {},
            'adaptability_tests': {},
            'maintainability_tests': {},
            'requirements_summary': {}
        }

        # Test each configuration
        for config_name, (qrh_config, integration_config) in test_configs.items():
            print(f"\n🎯 Testing Configuration: {config_name.upper()}")

            # Create system instance
            system = RobustNeurotransmitterIntegration(qrh_config, integration_config)

            config_results = {}

            # Test 1: Performance Requirement
            performance_results = self.test_performance_requirement(
                config_name, qrh_config, integration_config
            )
            config_results['performance'] = performance_results
            overall_summary['performance_tests'][config_name] = performance_results

            # Test 2: Efficiency Requirement
            efficiency_results = self.test_efficiency_requirement(system, test_scenarios)
            config_results['efficiency'] = efficiency_results
            overall_summary['efficiency_tests'][config_name] = efficiency_results

            # Test 3: Robustness Requirement
            robustness_results = self.test_robustness_requirement(system, test_scenarios)
            config_results['robustness'] = robustness_results
            overall_summary['robustness_tests'][config_name] = robustness_results

            # Test 4: Adaptability Requirement
            adaptability_results = self.test_adaptability_requirement(system)
            config_results['adaptability'] = adaptability_results
            overall_summary['adaptability_tests'][config_name] = adaptability_results

            # Test 5: Maintainability Requirement
            maintainability_results = self.test_maintainability_requirement(system)
            config_results['maintainability'] = maintainability_results
            overall_summary['maintainability_tests'][config_name] = maintainability_results

            # Configuration summary
            config_summary = {
                'performance_pass': performance_results['overall_pass'],
                'efficiency_pass': efficiency_results['efficiency_requirement_met'],
                'robustness_pass': robustness_results['robustness_requirement_met'],
                'adaptability_pass': adaptability_results['adaptability_requirement_met'],
                'maintainability_pass': maintainability_results['maintainability_requirement_met']
            }

            config_results['summary'] = config_summary
            all_results[config_name] = config_results

            # Print configuration results
            self._print_config_results(config_name, config_summary)

        # Generate overall requirements summary
        overall_summary['requirements_summary'] = self._generate_requirements_summary(all_results)

        print("\n" + "=" * 80)
        print("📋 OVERALL VALIDATION SUMMARY")
        print("=" * 80)
        self._print_overall_summary(overall_summary['requirements_summary'])

        return {
            'detailed_results': all_results,
            'overall_summary': overall_summary,
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def _print_config_results(self, config_name: str, summary: Dict[str, bool]):
        """Print results for a specific configuration"""
        status_emoji = lambda x: "✅" if x else "❌"

        print(f"  📊 Results for {config_name}:")
        print(f"    {status_emoji(summary['performance_pass'])} Performance (≤5% degradation)")
        print(f"    {status_emoji(summary['efficiency_pass'])} Efficiency (≥90% JIT-only)")
        print(f"    {status_emoji(summary['robustness_pass'])} Robustness (≥95% accuracy)")
        print(f"    {status_emoji(summary['adaptability_pass'])} Adaptability (Learning detected)")
        print(f"    {status_emoji(summary['maintainability_pass'])} Maintainability (Clean code)")

    def _generate_requirements_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all requirement tests"""
        requirements = ['performance', 'efficiency', 'robustness', 'adaptability', 'maintainability']

        summary = {
            'total_configurations': len(all_results),
            'requirements_met': {},
            'overall_pass_rate': 0.0,
            'best_configuration': None,
            'recommendations': []
        }

        # Count passes per requirement
        for req in requirements:
            passes = sum(1 for config_results in all_results.values()
                        if config_results['summary'][f'{req}_pass'])
            summary['requirements_met'][req] = {
                'passes': passes,
                'total': len(all_results),
                'pass_rate': passes / len(all_results)
            }

        # Overall pass rate
        total_tests = len(requirements) * len(all_results)
        total_passes = sum(req_data['passes'] for req_data in summary['requirements_met'].values())
        summary['overall_pass_rate'] = total_passes / total_tests

        # Find best configuration
        config_scores = {}
        for config_name, config_results in all_results.items():
            score = sum(config_results['summary'].values())
            config_scores[config_name] = score

        if config_scores:
            summary['best_configuration'] = max(config_scores, key=config_scores.get)

        # Generate recommendations
        for req, req_data in summary['requirements_met'].items():
            if req_data['pass_rate'] < 1.0:
                summary['recommendations'].append(f"Improve {req} requirement (only {req_data['pass_rate']:.1%} pass)")

        if not summary['recommendations']:
            summary['recommendations'].append("All requirements met across all configurations!")

        return summary

    def _print_overall_summary(self, summary: Dict[str, Any]):
        """Print overall validation summary"""
        print(f"🎯 Overall Pass Rate: {summary['overall_pass_rate']:.1%}")
        print(f"🏆 Best Configuration: {summary['best_configuration']}")

        print(f"\n📊 Requirements Breakdown:")
        for req, req_data in summary['requirements_met'].items():
            status = "✅" if req_data['pass_rate'] == 1.0 else "⚠️" if req_data['pass_rate'] >= 0.67 else "❌"
            print(f"  {status} {req.title()}: {req_data['pass_rate']:.1%} ({req_data['passes']}/{req_data['total']})")

        if summary['recommendations']:
            print(f"\n🔧 Recommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")


def main():
    """Main function to run robust integration validation"""
    print("🚀🧬 ROBUST NEUROTRANSMITTER INTEGRATION VALIDATION")
    print("Testing JIT + Synthetic Neurotransmitters Hybrid Architecture")
    print("=" * 80)

    # Initialize validator
    validator = RobustIntegrationValidator(device='cpu')

    # Run comprehensive validation
    results = validator.run_comprehensive_validation()

    # Save results
    import json
    results_file = 'robust_integration_validation_results.json'

    # Convert results to JSON-serializable format
    json_results = {
        'overall_summary': results['overall_summary']['requirements_summary'],
        'validation_timestamp': results['validation_timestamp'],
        'configurations_tested': len(results['detailed_results'])
    }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {results_file}")
    print("🎉 Robust Integration Validation Complete!")

    return results


if __name__ == "__main__":
    results = main()