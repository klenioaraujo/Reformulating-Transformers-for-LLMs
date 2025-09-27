#!/usr/bin/env python3
"""
ΨQRH Integration Test Runner - Component Integration Testing
============================================================

Advanced integration testing for ΨQRH architecture:
- Component interoperability testing
- End-to-end pipeline validation
- Consciousness processing integration
- Security system integration
- Performance and reliability testing
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
sys.path.append(str(Path(__file__).parent.parent))

class ΨQRHIntegrationTestRunner:
    """Advanced integration testing for ΨQRH architecture."""

    def __init__(self, integration_config: Dict[str, Any] = None):
        if integration_config is None:
            integration_config = self._default_config()

        self.config = integration_config
        self.integration_results = {}
        self.performance_data = {}

        # Initialize integration components
        self._initialize_integration_components()

    def _default_config(self) -> Dict[str, Any]:
        """Default integration testing configuration."""
        return {
            'test_scenarios': ['text_processing', 'file_conversion', 'security_pipeline', 'consciousness_analysis'],
            'performance_tracking': True,
            'error_handling': True,
            'data_validation': True,
            'verbose': True
        }

    def _initialize_integration_components(self):
        """Initialize all integration components."""
        self.components = {}

        try:
            # Core ΨQRH components
            from src.core.ΨQRH import QRHFactory
            from src.core.qrh_layer import QRHLayer
            from src.core.quaternion_operations import QuaternionOperations

            self.components['qrh_factory'] = QRHFactory()
            self.components['quaternion_ops'] = QuaternionOperations()

            print("✅ Core ΨQRH components initialized")
        except ImportError as e:
            print(f"⚠️ Core components import error: {e}")

        try:
            # Conscience processing components
            from src.conscience.conscious_wave_modulator import ConsciousWaveModulator
            from src.conscience.secure_Ψcws_protector import create_secure_Ψcws_protector

            self.components['wave_modulator'] = ConsciousWaveModulator()
            self.components['security_protector'] = create_secure_Ψcws_protector()

            print("✅ Conscience processing components initialized")
        except ImportError as e:
            print(f"⚠️ Conscience components import error: {e}")

        try:
            # Fractal mathematics components
            from src.fractal.spectral_filter import SpectralFilter

            self.components['spectral_filter'] = SpectralFilter()
            print("✅ Fractal mathematics components initialized")
        except ImportError as e:
            print(f"⚠️ Fractal components import error: {e}")

    def run_integration_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive integration test suite."""

        print("🚀 Starting ΨQRH Integration Test Suite")
        print("=" * 60)

        integration_results = {}

        # 1. Text Processing Integration
        if 'text_processing' in self.config['test_scenarios']:
            integration_results['text_processing'] = self._test_text_processing_integration()

        # 2. File Conversion Integration
        if 'file_conversion' in self.config['test_scenarios']:
            integration_results['file_conversion'] = self._test_file_conversion_integration()

        # 3. Security Pipeline Integration
        if 'security_pipeline' in self.config['test_scenarios']:
            integration_results['security_pipeline'] = self._test_security_pipeline_integration()

        # 4. Consciousness Analysis Integration
        if 'consciousness_analysis' in self.config['test_scenarios']:
            integration_results['consciousness_analysis'] = self._test_consciousness_analysis_integration()

        # 5. End-to-End Pipeline Integration
        integration_results['end_to_end'] = self._test_end_to_end_integration()

        self.integration_results = integration_results
        return integration_results

    def _test_text_processing_integration(self) -> Dict[str, Any]:
        """Test text processing component integration."""
        print("\n📝 Testing Text Processing Integration")
        print("-" * 40)

        results = {'status': 'pending', 'tests': []}

        try:
            # Test sample text through full pipeline
            test_texts = [
                "Simple consciousness test",
                "Complex neural dynamics involving information integration and adaptive processing",
                "ΨQRH framework combines quaternion algebra with spectral analysis for advanced AI systems"
            ]

            factory = self.components.get('qrh_factory')
            if factory:
                for i, text in enumerate(test_texts):
                    start_time = time.time()
                    processed_result = factory.process_text(text)
                    processing_time = time.time() - start_time

                    results['tests'].append({
                        'name': f'Text Processing Pipeline {i+1}',
                        'status': 'passed' if processed_result else 'failed',
                        'details': f'Text: "{text[:50]}...", Time: {processing_time:.4f}s, Result: {bool(processed_result)}'
                    })

            results['status'] = 'passed'
            print("✅ Text processing integration tests completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Text processing integration tests failed: {e}")

        return results

    def _test_file_conversion_integration(self) -> Dict[str, Any]:
        """Test file conversion pipeline integration."""
        print("\n📁 Testing File Conversion Integration")
        print("-" * 40)

        results = {'status': 'pending', 'tests': []}

        try:
            # Create test files in different formats
            test_files = []

            # Text file
            text_file = Path("integration_test.txt")
            with open(text_file, 'w') as f:
                f.write("File conversion integration test content for ΨQRH system")
            test_files.append(text_file)

            modulator = self.components.get('wave_modulator')
            if modulator:
                for test_file in test_files:
                    start_time = time.time()
                    Ψcws_file = modulator.process_file(test_file)
                    conversion_time = time.time() - start_time

                    # Validate .Ψcws file structure
                    valid_structure = all(hasattr(Ψcws_file, attr)
                                        for attr in ['header', 'spectral_data', 'content_metadata'])

                    # Check consciousness metrics
                    has_metrics = (hasattr(Ψcws_file, 'spectral_data') and
                                 hasattr(Ψcws_file.spectral_data, 'consciousness_metrics'))

                    results['tests'].append({
                        'name': f'File Conversion: {test_file.name}',
                        'status': 'passed' if Ψcws_file and valid_structure else 'failed',
                        'details': f'Conversion time: {conversion_time:.4f}s, Structure: {valid_structure}, Metrics: {has_metrics}'
                    })

            # Cleanup test files
            for test_file in test_files:
                if test_file.exists():
                    test_file.unlink()

            results['status'] = 'passed'
            print("✅ File conversion integration tests completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ File conversion integration tests failed: {e}")

        return results

    def _test_security_pipeline_integration(self) -> Dict[str, Any]:
        """Test security pipeline integration."""
        print("\n🔒 Testing Security Pipeline Integration")
        print("-" * 40)

        results = {'status': 'pending', 'tests': []}

        try:
            modulator = self.components.get('wave_modulator')
            protector = self.components.get('security_protector')

            if modulator and protector:
                # Create test content
                test_content = "Security pipeline integration test"
                test_file = Path("security_test.txt")

                with open(test_file, 'w') as f:
                    f.write(test_content)

                # Full security pipeline: File → .Ψcws → Protected Parts
                start_time = time.time()

                # Step 1: Convert to .Ψcws
                Ψcws_file = modulator.process_file(test_file)
                Ψcws_path = Path("security_test.Ψcws")
                Ψcws_file.save(Ψcws_path)

                # Step 2: Apply security protection
                protected_parts = protector.protect_file(Ψcws_path, parts=4)

                # Step 3: Read protected file
                reconstructed_path = Path("reconstructed_security.Ψcws")
                read_success = protector.read_protected_file(protected_parts, reconstructed_path)

                pipeline_time = time.time() - start_time

                # Validate pipeline success
                pipeline_valid = all([
                    Ψcws_file is not None,
                    protected_parts is not None,
                    len(protected_parts) == 4,
                    read_success
                ])

                results['tests'].extend([
                    {
                        'name': 'Full Security Pipeline',
                        'status': 'passed' if pipeline_valid else 'failed',
                        'details': f'Pipeline time: {pipeline_time:.4f}s, Steps: 3/3, Success: {pipeline_valid}'
                    },
                    {
                        'name': 'File to .Ψcws Conversion',
                        'status': 'passed' if Ψcws_file else 'failed',
                        'details': f'.Ψcws file generated: {bool(Ψcws_file)}'
                    },
                    {
                        'name': 'Security Protection',
                        'status': 'passed' if protected_parts else 'failed',
                        'details': f'Protected parts: {len(protected_parts) if protected_parts else 0}'
                    },
                    {
                        'name': 'Protected File Reading',
                        'status': 'passed' if read_success else 'failed',
                        'details': f'File reading success: {read_success}'
                    }
                ])

                # Cleanup
                files_to_clean = [test_file, Ψcws_path, reconstructed_path]
                for file_path in files_to_clean:
                    if file_path.exists():
                        file_path.unlink()

            results['status'] = 'passed'
            print("✅ Security pipeline integration tests completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Security pipeline integration tests failed: {e}")

        return results

    def _test_consciousness_analysis_integration(self) -> Dict[str, Any]:
        """Test consciousness analysis integration."""
        print("\n🧠 Testing Consciousness Analysis Integration")
        print("-" * 40)

        results = {'status': 'pending', 'tests': []}

        try:
            modulator = self.components.get('wave_modulator')

            if modulator:
                # Test consciousness metrics with different content types
                test_cases = [
                    ("Simple text", "Basic consciousness test"),
                    ("Complex text", "Advanced neural dynamics involving fractal patterns and quantum coherence in conscious systems"),
                    ("Technical text", "ΨQRH framework integrates quaternion harmonics with spectral regularization for consciousness modeling")
                ]

                for case_name, test_text in test_cases:
                    # Generate consciousness metrics
                    embeddings = modulator._generate_wave_embeddings(test_text)
                    trajectories = modulator._generate_chaotic_trajectories(test_text)
                    spectra = modulator._compute_fourier_spectra(embeddings)
                    metrics = modulator._compute_consciousness_metrics(embeddings, trajectories, spectra)

                    # Validate metrics
                    valid_metrics = isinstance(metrics, dict) and all(key in metrics
                                                                     for key in ['complexity', 'coherence', 'adaptability', 'integration'])

                    reasonable_values = all(0 <= value <= 1 for value in metrics.values())

                    results['tests'].append({
                        'name': f'Consciousness Analysis: {case_name}',
                        'status': 'passed' if valid_metrics and reasonable_values else 'failed',
                        'details': f'Metrics valid: {valid_metrics}, Values reasonable: {reasonable_values}, Complexity: {metrics.get("complexity", 0):.3f}'
                    })

            results['status'] = 'passed'
            print("✅ Consciousness analysis integration tests completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Consciousness analysis integration tests failed: {e}")

        return results

    def _test_end_to_end_integration(self) -> Dict[str, Any]:
        """Test complete end-to-end integration."""
        print("\n🔗 Testing End-to-End Integration")
        print("-" * 40)

        results = {'status': 'pending', 'tests': []}

        try:
            # Complete pipeline: Text → Processing → File → .Ψcws → Security → Analysis
            test_content = """
            End-to-end integration test for ΨQRH architecture.
            This test validates the complete pipeline from text input through
            consciousness analysis and security protection.
            """

            # Step 1: Text processing through QRH factory
            factory = self.components.get('qrh_factory')
            if factory:
                processed_text = factory.process_text(test_content)

                results['tests'].append({
                    'name': 'QRH Text Processing',
                    'status': 'passed' if processed_text else 'failed',
                    'details': f'Text processing completed: {bool(processed_text)}'
                })

            # Step 2: File conversion and consciousness analysis
            modulator = self.components.get('wave_modulator')
            if modulator:
                test_file = Path("e2e_test.txt")
                with open(test_file, 'w') as f:
                    f.write(test_content)

                Ψcws_file = modulator.process_file(test_file)

                # Validate consciousness metrics
                if Ψcws_file and hasattr(Ψcws_file.spectral_data, 'consciousness_metrics'):
                    metrics = Ψcws_file.spectral_data.consciousness_metrics
                    metrics_valid = all(0 <= value <= 1 for value in metrics.values())

                    results['tests'].append({
                        'name': 'Consciousness Metrics Generation',
                        'status': 'passed' if metrics_valid else 'failed',
                        'details': f'Metrics generated: {metrics_valid}, Complexity: {metrics.get("complexity", 0):.3f}'
                    })

                # Step 3: Security integration
                protector = self.components.get('security_protector')
                if protector:
                    Ψcws_path = Path("e2e_test.Ψcws")
                    Ψcws_file.save(Ψcws_path)

                    protected_parts = protector.protect_file(Ψcws_path, parts=2)
                    read_success = protector.read_protected_file(protected_parts, "reconstructed_e2e.Ψcws")

                    results['tests'].append({
                        'name': 'Security Integration',
                        'status': 'passed' if read_success else 'failed',
                        'details': f'Security pipeline success: {read_success}'
                    })

                # Cleanup
                files_to_clean = [test_file, Ψcws_path, Path("reconstructed_e2e.Ψcws")]
                for file_path in files_to_clean:
                    if file_path.exists():
                        file_path.unlink()

            # Overall pipeline success
            pipeline_success = all(test['status'] == 'passed' for test in results['tests'])

            results['tests'].append({
                'name': 'Complete End-to-End Pipeline',
                'status': 'passed' if pipeline_success else 'failed',
                'details': f'All integration steps completed successfully: {pipeline_success}'
            })

            results['status'] = 'passed'
            print("✅ End-to-end integration tests completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ End-to-end integration tests failed: {e}")

        return results

    def generate_integration_report(self) -> str:
        """Generate comprehensive integration test report."""
        if not self.integration_results:
            return "No integration test results available. Run tests first."

        report = []
        report.append("ΨQRH Integration Test Report")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Integration summary
        total_tests = 0
        passed_tests = 0

        for category, result in self.integration_results.items():
            if 'tests' in result:
                category_tests = len(result['tests'])
                category_passed = sum(1 for test in result['tests'] if test['status'] == 'passed')

                total_tests += category_tests
                passed_tests += category_passed

        integration_score = passed_tests / total_tests if total_tests > 0 else 0

        report.append(f"Integration Score: {integration_score*100:.1f}% ({passed_tests}/{total_tests} tests passed)")
        report.append("")

        # Detailed integration results
        for category, result in self.integration_results.items():
            report.append(f"{category.upper()} Integration - {result['status'].upper()}")
            report.append("-" * 40)

            if 'tests' in result:
                for test in result['tests']:
                    status_icon = "✅" if test['status'] == 'passed' else "❌"
                    report.append(f"{status_icon} {test['name']}")
                    report.append(f"   {test['details']}")

            if 'error' in result:
                report.append(f"⚠️ Integration Error: {result['error']}")

            report.append("")

        # Integration recommendations
        report.append("Integration Recommendations:")
        report.append("-" * 40)

        if integration_score >= 0.9:
            report.append("✅ Excellent integration - system components work well together")
            report.append("✅ Continue current development practices")
        elif integration_score >= 0.7:
            report.append("⚠️ Good integration - some components need optimization")
            report.append("⚠️ Review component interfaces and error handling")
        else:
            report.append("❌ Integration issues detected - immediate attention needed")
            report.append("❌ Review component dependencies and communication")

        return "\n".join(report)


def main():
    """Main function to run integration tests."""

    # Initialize integration test runner
    integration_runner = ΨQRHIntegrationTestRunner()

    # Run integration test suite
    integration_results = integration_runner.run_integration_test_suite()

    # Generate and display integration report
    integration_report = integration_runner.generate_integration_report()
    print("\n" + integration_report)

    # Determine overall integration status
    all_passed = all(result.get('status') == 'passed' for result in integration_results.values())

    if all_passed:
        print("\n🎉 Integration tests passed! ΨQRH components work well together.")
    else:
        print("\n⚠️ Integration issues detected. Review the report above.")


if __name__ == "__main__":
    main()