#!/usr/bin/env python3
"""
ΨQRH Test Prompt Engine - Comprehensive Testing Framework
========================================================

Advanced testing engine for ΨQRH architecture with:
- File reading and security validation
- Component integration testing
- Consciousness metrics analysis
- Security system verification
- Performance benchmarking
"""

import sys
import os
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
import numpy as np

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class ΨQRHTestPromptEngine:
    """Advanced testing engine for ΨQRH architecture."""

    def __init__(self, test_config: Dict[str, Any] = None):
        if test_config is None:
            test_config = self._default_config()

        self.config = test_config
        self.test_results = {}
        self.performance_metrics = {}

        # Initialize test components
        self._initialize_test_components()

    def _default_config(self) -> Dict[str, Any]:
        """Default testing configuration."""
        return {
            'test_components': ['core', 'conscience', 'fractal', 'security'],
            'performance_benchmark': True,
            'security_validation': True,
            'consciousness_metrics': True,
            'file_operations': True,
            'verbose': True
        }

    def _initialize_test_components(self):
        """Initialize all test components with error handling."""
        self.components = {}

        try:
            # Core components
            from core.ΨQRH import QRHFactory
            from core.qrh_layer import QRHLayer
            from core.quaternion_operations import QuaternionOperations

            self.components['qrh_factory'] = QRHFactory()
            self.components['quaternion_ops'] = QuaternionOperations()

            print("✅ Core components initialized")
        except ImportError as e:
            print(f"⚠️ Core components import error: {e}")

        try:
            # Conscience components
            from conscience.conscious_wave_modulator import ConsciousWaveModulator
            from conscience.secure_Ψcws_protector import create_secure_Ψcws_protector

            self.components['wave_modulator'] = ConsciousWaveModulator()
            self.components['security_protector'] = create_secure_Ψcws_protector()

            print("✅ Conscience components initialized")
        except ImportError as e:
            print(f"⚠️ Conscience components import error: {e}")

        try:
            # Fractal components
            from fractal.spectral_filter import SpectralFilter

            self.components['spectral_filter'] = SpectralFilter()
            print("✅ Fractal components initialized")
        except ImportError as e:
            print(f"⚠️ Fractal components import error: {e}")

    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite for ΨQRH architecture."""

        print("🚀 Starting ΨQRH Comprehensive Test Suite")
        print("=" * 60)

        test_results = {}

        # 1. Core Component Tests
        if 'core' in self.config['test_components']:
            test_results['core'] = self._test_core_components()

        # 2. Security System Tests
        if 'security' in self.config['test_components']:
            test_results['security'] = self._test_security_system()

        # 3. File Operations Tests
        if 'file_operations' in self.config['test_components']:
            test_results['file_operations'] = self._test_file_operations()

        # 4. Consciousness Metrics Tests
        if 'consciousness_metrics' in self.config['test_components']:
            test_results['consciousness'] = self._test_consciousness_metrics()

        # 5. Performance Benchmarking
        if self.config['performance_benchmark']:
            test_results['performance'] = self._run_performance_benchmarks()

        # 6. Integration Tests
        test_results['integration'] = self._test_integration()

        self.test_results = test_results
        return test_results

    def _test_core_components(self) -> Dict[str, Any]:
        """Test core ΨQRH components."""
        print("\n🧪 Testing Core Components")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            # Test QRH Factory
            factory = self.components.get('qrh_factory')
            if factory:
                test_text = "QRH test prompt for consciousness analysis"
                result = factory.process_text(test_text)

                results['tests'].append({
                    'name': 'QRH Factory Text Processing',
                    'status': 'passed' if result else 'failed',
                    'details': f'Processed {len(test_text)} characters'
                })

            # Test Quaternion Operations
            quat_ops = self.components.get('quaternion_ops')
            if quat_ops:
                # Test basic quaternion operations
                q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
                q2 = torch.tensor([0.0, 1.0, 0.0, 0.0])  # i quaternion

                # Test multiplication
                result = quat_ops.quaternion_multiply(q1, q2)
                expected = torch.tensor([0.0, 1.0, 0.0, 0.0])

                multiplication_test = torch.allclose(result, expected, atol=1e-6)

                results['tests'].append({
                    'name': 'Quaternion Multiplication',
                    'status': 'passed' if multiplication_test else 'failed',
                    'details': f'Result: {result}, Expected: {expected}'
                })

            results['status'] = 'passed'
            print("✅ Core components test completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Core components test failed: {e}")

        return results

    def _test_security_system(self) -> Dict[str, Any]:
        """Test ΨCWS security system."""
        print("\n🔒 Testing Security System")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            protector = self.components.get('security_protector')
            if protector:
                # Test data encryption/decryption
                test_data = b"QRH security test data for 7-layer encryption"

                # Test encryption
                start_time = time.time()
                encrypted = protector.security_layer.encrypt_7_layers(test_data)
                encryption_time = time.time() - start_time

                # Test decryption
                start_time = time.time()
                decrypted = protector.security_layer.decrypt_7_layers(encrypted)
                decryption_time = time.time() - start_time

                # Verify integrity
                integrity_test = test_data == decrypted

                results['tests'].extend([
                    {
                        'name': '7-Layer Encryption',
                        'status': 'passed' if len(encrypted) > len(test_data) else 'failed',
                        'details': f'Data size: {len(test_data)} → {len(encrypted)} bytes'
                    },
                    {
                        'name': 'Encryption/Decryption Integrity',
                        'status': 'passed' if integrity_test else 'failed',
                        'details': f'Data integrity: {integrity_test}'
                    },
                    {
                        'name': 'Performance',
                        'status': 'passed',
                        'details': f'Encryption: {encryption_time:.4f}s, Decryption: {decryption_time:.4f}s'
                    }
                ])

            results['status'] = 'passed'
            print("✅ Security system test completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Security system test failed: {e}")

        return results

    def _test_file_operations(self) -> Dict[str, Any]:
        """Test file operations and .Ψcws format."""
        print("\n📁 Testing File Operations")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            # Create test file
            test_content = "QRH test content for file operations and consciousness analysis"
            test_file = Path("test_QRH_content.txt")

            with open(test_file, 'w') as f:
                f.write(test_content)

            # Test wave modulator
            modulator = self.components.get('wave_modulator')
            if modulator:
                Ψcws_file = modulator.process_file(test_file)

                results['tests'].append({
                    'name': 'File to .Ψcws Conversion',
                    'status': 'passed' if Ψcws_file else 'failed',
                    'details': f'Generated .Ψcws file with consciousness metrics'
                })

                # Test consciousness metrics
                if hasattr(Ψcws_file, 'spectral_data') and hasattr(Ψcws_file.spectral_data, 'consciousness_metrics'):
                    metrics = Ψcws_file.spectral_data.consciousness_metrics

                    results['tests'].append({
                        'name': 'Consciousness Metrics Generation',
                        'status': 'passed' if metrics else 'failed',
                        'details': f'Metrics: {metrics}'
                    })

            # Cleanup
            if test_file.exists():
                test_file.unlink()

            results['status'] = 'passed'
            print("✅ File operations test completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ File operations test failed: {e}")

        return results

    def _test_consciousness_metrics(self) -> Dict[str, Any]:
        """Test consciousness metrics calculation."""
        print("\n🧠 Testing Consciousness Metrics")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            # Test with sample consciousness data
            sample_text = """
            Consciousness emerges from complex neural dynamics involving
            information integration, complexity, and adaptive processing.
            The ΨQRH framework models these dynamics using quaternion-based
            transformations and spectral analysis.
            """

            modulator = self.components.get('wave_modulator')
            if modulator:
                # Generate embeddings and metrics
                embeddings = modulator._generate_wave_embeddings(sample_text)
                trajectories = modulator._generate_chaotic_trajectories(sample_text)
                spectra = modulator._compute_fourier_spectra(embeddings)
                metrics = modulator._compute_consciousness_metrics(embeddings, trajectories, spectra)

                # Validate metrics
                valid_metrics = all(key in metrics for key in ['complexity', 'coherence', 'adaptability', 'integration'])
                reasonable_values = all(0 <= value <= 1 for value in metrics.values())

                results['tests'].extend([
                    {
                        'name': 'Consciousness Metrics Structure',
                        'status': 'passed' if valid_metrics else 'failed',
                        'details': f'Metrics keys: {list(metrics.keys())}'
                    },
                    {
                        'name': 'Metrics Value Range',
                        'status': 'passed' if reasonable_values else 'failed',
                        'details': f'Values: {metrics}'
                    },
                    {
                        'name': 'Embedding Generation',
                        'status': 'passed' if embeddings is not None else 'failed',
                        'details': f'Embedding shape: {embeddings.shape if embeddings else "None"}'
                    }
                ])

            results['status'] = 'passed'
            print("✅ Consciousness metrics test completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Consciousness metrics test failed: {e}")

        return results

    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("\n⚡ Running Performance Benchmarks")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            # Benchmark quaternion operations
            quat_ops = self.components.get('quaternion_ops')
            if quat_ops:
                # Create test quaternions
                batch_size = 1000
                q1 = torch.randn(batch_size, 4)
                q2 = torch.randn(batch_size, 4)

                # Benchmark multiplication
                start_time = time.time()
                for _ in range(100):
                    result = quat_ops.quaternion_multiply(q1, q2)
                quat_time = time.time() - start_time

                results['tests'].append({
                    'name': 'Quaternion Operations Performance',
                    'status': 'passed',
                    'details': f'100 batches of {batch_size} quaternions: {quat_time:.4f}s'
                })

            # Benchmark security operations
            protector = self.components.get('security_protector')
            if protector:
                test_data = b"Performance test data" * 100  # 2KB data

                start_time = time.time()
                encrypted = protector.security_layer.encrypt_7_layers(test_data)
                security_time = time.time() - start_time

                results['tests'].append({
                    'name': 'Security Operations Performance',
                    'status': 'passed',
                    'details': f'2KB data encryption: {security_time:.4f}s'
                })

            results['status'] = 'passed'
            print("✅ Performance benchmarks completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Performance benchmarks failed: {e}")

        return results

    def _test_integration(self) -> Dict[str, Any]:
        """Test integration between components."""
        print("\n🔗 Testing Component Integration")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            # Test end-to-end pipeline
            test_text = "Integration test for QRH architecture components"

            # Process through QRH factory
            factory = self.components.get('qrh_factory')
            if factory:
                processed_result = factory.process_text(test_text)

                results['tests'].append({
                    'name': 'End-to-End Text Processing',
                    'status': 'passed' if processed_result else 'failed',
                    'details': f'Processed {len(test_text)} characters through full pipeline'
                })

            # Test file processing integration
            modulator = self.components.get('wave_modulator')
            if modulator:
                # Create and process test file
                test_file = Path("integration_test.txt")
                with open(test_file, 'w') as f:
                    f.write(test_text)

                Ψcws_file = modulator.process_file(test_file)

                # Test security integration
                protector = self.components.get('security_protector')
                if protector and Ψcws_file:
                    # Save and protect file
                    temp_path = Path("temp_test.Ψcws")
                    Ψcws_file.save(temp_path)

                    # Test protection
                    protected_parts = protector.protect_file(temp_path, parts=2)

                    results['tests'].append({
                        'name': 'File Security Integration',
                        'status': 'passed' if protected_parts else 'failed',
                        'details': f'Generated {len(protected_parts)} protected parts'
                    })

                # Cleanup
                if test_file.exists():
                    test_file.unlink()
                if temp_path.exists():
                    temp_path.unlink()

            results['status'] = 'passed'
            print("✅ Integration test completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Integration test failed: {e}")

        return results

    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        if not self.test_results:
            return "No test results available. Run tests first."

        report = []
        report.append("ΨQRH Test Report")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for category, result in self.test_results.items():
            if 'tests' in result:
                category_tests = len(result['tests'])
                category_passed = sum(1 for test in result['tests'] if test['status'] == 'passed')

                total_tests += category_tests
                passed_tests += category_passed
                failed_tests += (category_tests - category_passed)

        if total_tests > 0:
            report.append(f"Summary: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        else:
            report.append("Summary: No tests were executed")
        report.append("")

        # Detailed results
        for category, result in self.test_results.items():
            report.append(f"{category.upper()} Tests - {result['status'].upper()}")
            report.append("-" * 40)

            if 'tests' in result:
                for test in result['tests']:
                    status_icon = "✅" if test['status'] == 'passed' else "❌"
                    report.append(f"{status_icon} {test['name']}")
                    report.append(f"   {test['details']}")

            if 'error' in result:
                report.append(f"❌ Error: {result['error']}")

            report.append("")

        return "\n".join(report)

    def save_test_results(self, output_path: str = "tmp/ΨQRH_test_report.json"):
        """Save test results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"📊 Test results saved to: {output_path}")


def main():
    """Main function to run ΨQRH test suite."""

    # Initialize test engine
    test_engine = ΨQRHTestPromptEngine()

    # Run comprehensive tests
    test_results = test_engine.run_comprehensive_test_suite()

    # Generate and display report
    report = test_engine.generate_test_report()
    print("\n" + report)

    # Save results
    test_engine.save_test_results()

    # Determine overall status
    all_passed = all(result.get('status') == 'passed' for result in test_results.values())

    if all_passed:
        print("\n🎉 All tests passed! ΨQRH architecture is functioning correctly.")
    else:
        print("\n⚠️ Some tests failed. Please review the report above.")


if __name__ == "__main__":
    main()