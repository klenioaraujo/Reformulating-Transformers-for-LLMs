#!/usr/bin/env python3
"""
ΨQRH Complete Test Suite Runner - Full System Validation
========================================================

Master test runner that executes all ΨQRH test suites:
- Component functionality tests
- Security validation tests
- Integration tests
- Performance benchmarks
- System validation
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class ΨQRHCompleteTestRunner:
    """Master test runner for complete ΨQRH system validation."""

    def __init__(self):
        self.all_results = {}
        self.system_status = {}
        self.start_time = time.time()

    def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run all test suites for complete system validation."""

        print("🚀 Starting ΨQRH Complete Test Suite")
        print("=" * 70)
        print("This will run all test suites to validate the complete ΨQRH system")
        print("=" * 70)

        # 1. Run Component Functionality Tests
        print("\n1. Running Component Functionality Tests...")
        component_results = self._run_component_tests()
        self.all_results['component_tests'] = component_results

        # 2. Run Security Validation Tests
        print("\n2. Running Security Validation Tests...")
        security_results = self._run_security_tests()
        self.all_results['security_tests'] = security_results

        # 3. Run Integration Tests
        print("\n3. Running Integration Tests...")
        integration_results = self._run_integration_tests()
        self.all_results['integration_tests'] = integration_results

        # 4. Run Performance Benchmarks
        print("\n4. Running Performance Benchmarks...")
        performance_results = self._run_performance_benchmarks()
        self.all_results['performance_tests'] = performance_results

        # 5. Generate System Status
        print("\n5. Generating System Status Report...")
        self._generate_system_status()

        # Calculate total execution time
        total_time = time.time() - self.start_time

        self.all_results['execution_summary'] = {
            'total_execution_time': total_time,
            'test_suites_run': 4,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        return self.all_results

    def _run_component_tests(self) -> Dict[str, Any]:
        """Run component functionality tests."""
        try:
            from .ΨQRH_test_prompt_engine import ΨQRHTestPromptEngine

            test_engine = ΨQRHTestPromptEngine()
            component_results = test_engine.run_comprehensive_test_suite()

            # Extract summary
            total_tests = 0
            passed_tests = 0

            for category, result in component_results.items():
                if 'tests' in result:
                    category_tests = len(result['tests'])
                    category_passed = sum(1 for test in result['tests'] if test['status'] == 'passed')

                    total_tests += category_tests
                    passed_tests += category_passed

            return {
                'results': component_results,
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
                    'status': 'passed' if all(result.get('status') == 'passed'
                                           for result in component_results.values()) else 'failed'
                }
            }

        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }

    def _run_security_tests(self) -> Dict[str, Any]:
        """Run security validation tests."""
        try:
            from .security_validation_test import ΨCWSSecurityValidator

            security_validator = ΨCWSSecurityValidator()
            security_results = security_validator.run_security_validation_suite()

            # Extract summary
            total_tests = 0
            passed_tests = 0

            for category, result in security_results.items():
                if 'tests' in result:
                    category_tests = len(result['tests'])
                    category_passed = sum(1 for test in result['tests'] if test['status'] == 'passed')

                    total_tests += category_tests
                    passed_tests += category_passed

            return {
                'results': security_results,
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
                    'security_level': 'VERY HIGH' if (passed_tests / total_tests) >= 0.9 else
                                    'HIGH' if (passed_tests / total_tests) >= 0.7 else
                                    'MEDIUM' if (passed_tests / total_tests) >= 0.5 else 'LOW',
                    'status': 'passed' if all(result.get('status') == 'passed'
                                           for result in security_results.values()) else 'failed'
                }
            }

        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }

    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        try:
            from .integration_test_runner import ΨQRHIntegrationTestRunner

            integration_runner = ΨQRHIntegrationTestRunner()
            integration_results = integration_runner.run_integration_test_suite()

            # Extract summary
            total_tests = 0
            passed_tests = 0

            for category, result in integration_results.items():
                if 'tests' in result:
                    category_tests = len(result['tests'])
                    category_passed = sum(1 for test in result['tests'] if test['status'] == 'passed')

                    total_tests += category_tests
                    passed_tests += category_passed

            integration_score = passed_tests / total_tests if total_tests > 0 else 0

            return {
                'results': integration_results,
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'integration_score': integration_score,
                    'integration_level': 'EXCELLENT' if integration_score >= 0.9 else
                                       'GOOD' if integration_score >= 0.7 else
                                       'FAIR' if integration_score >= 0.5 else 'POOR',
                    'status': 'passed' if all(result.get('status') == 'passed'
                                           for result in integration_results.values()) else 'failed'
                }
            }

        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }

    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        try:
            # Simple performance benchmarks
            benchmarks = {}

            # Benchmark imports
            import_start = time.time()
            import torch
            import numpy as np
            benchmarks['import_time'] = time.time() - import_start

            # Benchmark tensor operations
            tensor_start = time.time()
            for _ in range(1000):
                x = torch.randn(100, 100)
                y = torch.randn(100, 100)
                z = torch.matmul(x, y)
            benchmarks['tensor_operations'] = time.time() - tensor_start

            # Benchmark file operations
            file_start = time.time()
            test_file = Path("benchmark_test.txt")
            with open(test_file, 'w') as f:
                f.write("Performance benchmark test data" * 100)
            if test_file.exists():
                test_file.unlink()
            benchmarks['file_operations'] = time.time() - file_start

            return {
                'benchmarks': benchmarks,
                'status': 'passed' if all(time < 10 for time in benchmarks.values()) else 'warning'
            }

        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }

    def _generate_system_status(self):
        """Generate overall system status."""
        # Calculate overall status
        component_status = self.all_results.get('component_tests', {}).get('summary', {}).get('status', 'unknown')
        security_status = self.all_results.get('security_tests', {}).get('summary', {}).get('status', 'unknown')
        integration_status = self.all_results.get('integration_tests', {}).get('summary', {}).get('status', 'unknown')
        performance_status = self.all_results.get('performance_tests', {}).get('status', 'unknown')

        all_passed = all(status == 'passed' for status in
                        [component_status, security_status, integration_status, performance_status])

        # Calculate overall scores
        component_score = self.all_results.get('component_tests', {}).get('summary', {}).get('pass_rate', 0)
        security_score = self.all_results.get('security_tests', {}).get('summary', {}).get('pass_rate', 0)
        integration_score = self.all_results.get('integration_tests', {}).get('summary', {}).get('integration_score', 0)

        overall_score = (component_score + security_score + integration_score) / 3

        self.system_status = {
            'overall_status': 'PASSED' if all_passed else 'FAILED',
            'overall_score': overall_score,
            'component_status': component_status,
            'security_status': security_status,
            'integration_status': integration_status,
            'performance_status': performance_status,
            'system_health': 'EXCELLENT' if overall_score >= 0.9 else
                           'GOOD' if overall_score >= 0.7 else
                           'FAIR' if overall_score >= 0.5 else 'POOR'
        }

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive test report."""
        if not self.all_results:
            return "No test results available. Run tests first."

        report = []
        report.append("ΨQRH COMPREHENSIVE TEST REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Execution Time: {self.all_results['execution_summary']['total_execution_time']:.2f}s")
        report.append("")

        # System Status Summary
        report.append("SYSTEM STATUS SUMMARY")
        report.append("-" * 40)
        report.append(f"Overall Status: {self.system_status['overall_status']}")
        report.append(f"System Health: {self.system_status['system_health']}")
        report.append(f"Overall Score: {self.system_status['overall_score']*100:.1f}%")
        report.append("")

        # Component Tests Summary
        component_summary = self.all_results.get('component_tests', {}).get('summary', {})
        report.append("COMPONENT FUNCTIONALITY TESTS")
        report.append("-" * 40)
        report.append(f"Status: {component_summary.get('status', 'unknown').upper()}")
        report.append(f"Pass Rate: {component_summary.get('pass_rate', 0)*100:.1f}%")
        report.append(f"Tests: {component_summary.get('passed_tests', 0)}/{component_summary.get('total_tests', 0)} passed")
        report.append("")

        # Security Tests Summary
        security_summary = self.all_results.get('security_tests', {}).get('summary', {})
        report.append("SECURITY VALIDATION TESTS")
        report.append("-" * 40)
        report.append(f"Status: {security_summary.get('status', 'unknown').upper()}")
        report.append(f"Security Level: {security_summary.get('security_level', 'UNKNOWN')}")
        report.append(f"Pass Rate: {security_summary.get('pass_rate', 0)*100:.1f}%")
        report.append(f"Tests: {security_summary.get('passed_tests', 0)}/{security_summary.get('total_tests', 0)} passed")
        report.append("")

        # Integration Tests Summary
        integration_summary = self.all_results.get('integration_tests', {}).get('summary', {})
        report.append("INTEGRATION TESTS")
        report.append("-" * 40)
        report.append(f"Status: {integration_summary.get('status', 'unknown').upper()}")
        report.append(f"Integration Level: {integration_summary.get('integration_level', 'UNKNOWN')}")
        report.append(f"Integration Score: {integration_summary.get('integration_score', 0)*100:.1f}%")
        report.append(f"Tests: {integration_summary.get('passed_tests', 0)}/{integration_summary.get('total_tests', 0)} passed")
        report.append("")

        # Performance Benchmarks
        performance_results = self.all_results.get('performance_tests', {})
        report.append("PERFORMANCE BENCHMARKS")
        report.append("-" * 40)
        report.append(f"Status: {performance_results.get('status', 'unknown').upper()}")
        if 'benchmarks' in performance_results:
            for benchmark, time_taken in performance_results['benchmarks'].items():
                report.append(f"{benchmark}: {time_taken:.4f}s")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)

        if self.system_status['system_health'] == 'EXCELLENT':
            report.append("✅ System is in excellent condition")
            report.append("✅ Continue with current development practices")
            report.append("✅ Consider adding more advanced test scenarios")
        elif self.system_status['system_health'] == 'GOOD':
            report.append("⚠️ System is in good condition with minor issues")
            report.append("⚠️ Review failed tests and optimize components")
            report.append("⚠️ Consider performance optimizations")
        else:
            report.append("❌ System requires immediate attention")
            report.append("❌ Focus on fixing critical component failures")
            report.append("❌ Review security and integration issues")

        return "\n".join(report)

    def save_detailed_results(self, output_dir: str = "test_results"):
        """Save detailed test results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save comprehensive results
        results_file = output_path / "ΨQRH_comprehensive_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'system_status': self.system_status,
                'all_results': self.all_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)

        # Save individual test suite results
        individual_files = {
            'component_tests.json': self.all_results.get('component_tests', {}),
            'security_tests.json': self.all_results.get('security_tests', {}),
            'integration_tests.json': self.all_results.get('integration_tests', {}),
            'performance_tests.json': self.all_results.get('performance_tests', {})
        }

        for filename, data in individual_files.items():
            file_path = output_path / filename
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"📊 Detailed results saved to: {output_path}")


def main():
    """Main function to run complete test suite."""

    print("ΨQRH Complete System Validation")
    print("=" * 50)
    print("This will validate the entire ΨQRH architecture")
    print("including components, security, integration, and performance.")
    print("=" * 50)

    # Initialize complete test runner
    complete_runner = ΨQRHCompleteTestRunner()

    # Run complete test suite
    print("\nStarting comprehensive test suite...")
    all_results = complete_runner.run_complete_test_suite()

    # Generate and display comprehensive report
    comprehensive_report = complete_runner.generate_comprehensive_report()
    print("\n" + comprehensive_report)

    # Save detailed results
    complete_runner.save_detailed_results()

    # Final status
    if complete_runner.system_status['overall_status'] == 'PASSED':
        print("\n🎉 ΨQRH SYSTEM VALIDATION PASSED!")
        print("The system is ready for production use.")
    else:
        print("\n⚠️ ΨQRH SYSTEM VALIDATION ISSUES DETECTED!")
        print("Please review the recommendations above.")

    print(f"\n📊 Detailed results saved to: test_results/")


if __name__ == "__main__":
    main()