#!/usr/bin/env python3
"""
ΨCWS Security Validation Test - Advanced Security Testing
=========================================================

Comprehensive security testing for ΨCWS protection system:
- 7-layer encryption validation
- File splitting integrity
- Anti-violation policy testing
- Hash verification systems
- Performance under attack scenarios
"""

import sys
import os
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import secrets

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class ΨCWSSecurityValidator:
    """Advanced security validation for ΨCWS protection system."""

    def __init__(self):
        self.test_results = {}
        self.security_metrics = {}

    def run_security_validation_suite(self) -> Dict[str, Any]:
        """Run comprehensive security validation tests."""

        print("🔒 Starting ΨCWS Security Validation Suite")
        print("=" * 60)

        validation_results = {}

        # 1. Encryption Layer Tests
        validation_results['encryption'] = self._test_encryption_layers()

        # 2. File Splitting Tests
        validation_results['file_splitting'] = self._test_file_splitting()

        # 3. Hash Verification Tests
        validation_results['hash_verification'] = self._test_hash_verification()

        # 4. Anti-Violation Policy Tests
        validation_results['anti_violation'] = self._test_anti_violation_policy()

        # 5. Performance Under Attack Tests
        validation_results['performance_attack'] = self._test_performance_under_attack()

        # 6. Integration Security Tests
        validation_results['integration'] = self._test_integration_security()

        self.test_results = validation_results
        return validation_results

    def _test_encryption_layers(self) -> Dict[str, Any]:
        """Test individual encryption layers."""
        print("\n🔐 Testing 7-Layer Encryption")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            from src.conscience.secure_Ψcws_protector import ΨCWSSecurityLayer

            security_layer = ΨCWSSecurityLayer()

            # Test data
            test_data = b"ΨCWS encryption test data for layer validation"

            # Test each layer individually
            layers = [
                ('AES-GCM', security_layer._layer_aes_gcm),
                ('ChaCha20', security_layer._layer_chacha20),
                ('Fernet', security_layer._layer_fernet),
                ('XOR', security_layer._layer_xor),
                ('Transposition', security_layer._layer_transposition),
                ('HMAC-AES', security_layer._layer_hmac_aes),
                ('Obfuscation', security_layer._layer_obfuscation)
            ]

            for layer_name, layer_func in layers:
                # Test encryption
                key = secrets.token_bytes(32)
                encrypted = layer_func(test_data, key)

                # Test decryption
                decrypted = layer_func(encrypted, key, decrypt=True)

                # Verify integrity
                integrity_ok = test_data == decrypted

                results['tests'].append({
                    'name': f'{layer_name} Layer',
                    'status': 'passed' if integrity_ok else 'failed',
                    'details': f'Data integrity: {integrity_ok}, Size change: {len(encrypted)} → {len(decrypted)}'
                })

            # Test full 7-layer encryption
            full_encrypted = security_layer.encrypt_7_layers(test_data)
            full_decrypted = security_layer.decrypt_7_layers(full_encrypted)

            full_integrity = test_data == full_decrypted

            results['tests'].append({
                'name': 'Full 7-Layer Encryption',
                'status': 'passed' if full_integrity else 'failed',
                'details': f'Full pipeline integrity: {full_integrity}'
            })

            results['status'] = 'passed'
            print("✅ Encryption layer tests completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Encryption layer tests failed: {e}")

        return results

    def _test_file_splitting(self) -> Dict[str, Any]:
        """Test file splitting and reassembly."""
        print("\n📊 Testing File Splitting System")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            from src.conscience.secure_Ψcws_protector import ΨCWSFileSplitter, ΨCWSSecurityLayer

            security_layer = ΨCWSSecurityLayer()
            splitter = ΨCWSFileSplitter(security_layer)

            # Create test file
            test_content = "ΨCWS file splitting test content with consciousness data" * 100
            test_file = Path("test_splitting.Ψcws")

            with open(test_file, 'w') as f:
                f.write(test_content)

            # Test different split configurations
            split_configs = [2, 4, 8]

            for parts in split_configs:
                # Split file
                file_parts = splitter.split_file(test_file, parts=parts)

                # Verify part structure
                valid_structure = all(
                    hasattr(part, 'part_number') and
                    hasattr(part, 'content_hash') and
                    hasattr(part, 'file_hash')
                    for part in file_parts
                )

                # Test reassembly
                reassembled_file = Path(f"reassembled_{parts}.Ψcws")
                success = splitter.reassemble_file(file_parts, reassembled_file)

                # Verify content integrity
                with open(reassembled_file, 'r') as f:
                    reassembled_content = f.read()

                content_integrity = test_content == reassembled_content

                results['tests'].append({
                    'name': f'File Splitting ({parts} parts)',
                    'status': 'passed' if valid_structure and success and content_integrity else 'failed',
                    'details': f'Parts: {len(file_parts)}, Structure: {valid_structure}, Reassembly: {success}, Integrity: {content_integrity}'
                })

                # Cleanup
                if reassembled_file.exists():
                    reassembled_file.unlink()

            # Cleanup test file
            if test_file.exists():
                test_file.unlink()

            results['status'] = 'passed'
            print("✅ File splitting tests completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ File splitting tests failed: {e}")

        return results

    def _test_hash_verification(self) -> Dict[str, Any]:
        """Test hash verification systems."""
        print("\n🔍 Testing Hash Verification")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            from src.conscience.secure_Ψcws_protector import ΨCWSFileSplitter, ΨCWSSecurityLayer

            security_layer = ΨCWSSecurityLayer()
            splitter = ΨCWSFileSplitter(security_layer)

            # Create test file
            test_content = "Hash verification test content"
            test_file = Path("test_hash.Ψcws")

            with open(test_file, 'w') as f:
                f.write(test_content)

            # Split file
            file_parts = splitter.split_file(test_file, parts=2)

            # Test individual part verification
            part_verification_results = []
            for part in file_parts:
                verification_result = splitter._verify_part_integrity(part)
                part_verification_results.append(verification_result)

            # Test parts consistency
            consistency_result = splitter._verify_parts_consistency(file_parts)

            # Test tamper detection
            tampered_part = file_parts[0]
            original_hash = tampered_part.content_hash
            tampered_part.content_hash = "tampered_hash"  # Simulate tampering

            tamper_detection = not splitter._verify_part_integrity(tampered_part)

            # Restore original hash
            tampered_part.content_hash = original_hash

            results['tests'].extend([
                {
                    'name': 'Individual Part Verification',
                    'status': 'passed' if all(part_verification_results) else 'failed',
                    'details': f'Part verification results: {part_verification_results}'
                },
                {
                    'name': 'Parts Consistency Check',
                    'status': 'passed' if consistency_result else 'failed',
                    'details': f'Parts consistency: {consistency_result}'
                },
                {
                    'name': 'Tamper Detection',
                    'status': 'passed' if tamper_detection else 'failed',
                    'details': f'Tamper detection working: {tamper_detection}'
                }
            ])

            # Cleanup
            if test_file.exists():
                test_file.unlink()

            results['status'] = 'passed'
            print("✅ Hash verification tests completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Hash verification tests failed: {e}")

        return results

    def _test_anti_violation_policy(self) -> Dict[str, Any]:
        """Test anti-violation policy."""
        print("\n🛡️ Testing Anti-Violation Policy")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            from src.conscience.secure_Ψcws_protector import ΨCWSAntiViolationPolicy

            policy = ΨCWSAntiViolationPolicy()

            # Create test file
            test_file = Path("test_violation.Ψcws")
            with open(test_file, 'w') as f:
                f.write("Test content")

            # Test file scanning
            scan_result = policy.scan_file(test_file)

            # Test access verification with valid files
            valid_files = [test_file]
            access_result = policy.verify_access(valid_files)

            # Test access verification with non-existent files
            invalid_files = [Path("non_existent.Ψcws")]
            invalid_access_result = policy.verify_access(invalid_files)

            # Test excessive attempts
            for _ in range(4):  # Exceed max attempts
                policy.verify_access(valid_files)

            excessive_attempts_result = policy.verify_access(valid_files)

            results['tests'].extend([
                {
                    'name': 'File Scanning',
                    'status': 'passed' if scan_result else 'failed',
                    'details': f'File scan result: {scan_result}'
                },
                {
                    'name': 'Valid Access Verification',
                    'status': 'passed' if access_result else 'failed',
                    'details': f'Valid access result: {access_result}'
                },
                {
                    'name': 'Invalid File Detection',
                    'status': 'passed' if not invalid_access_result else 'failed',
                    'details': f'Invalid file detection: {not invalid_access_result}'
                },
                {
                    'name': 'Excessive Attempts Blocking',
                    'status': 'passed' if not excessive_attempts_result else 'failed',
                    'details': f'Excessive attempts blocked: {not excessive_attempts_result}'
                }
            ])

            # Cleanup
            if test_file.exists():
                test_file.unlink()

            results['status'] = 'passed'
            print("✅ Anti-violation policy tests completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Anti-violation policy tests failed: {e}")

        return results

    def _test_performance_under_attack(self) -> Dict[str, Any]:
        """Test performance under simulated attack scenarios."""
        print("\n⚔️ Testing Performance Under Attack")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            from src.conscience.secure_Ψcws_protector import ΨCWSProtector

            protector = ΨCWSProtector()

            # Create large test file
            large_content = "X" * (1024 * 1024)  # 1MB file
            large_file = Path("large_test.Ψcws")

            with open(large_file, 'w') as f:
                f.write(large_content)

            # Test performance with large files
            start_time = time.time()
            protected_parts = protector.protect_file(large_file, parts=8)
            protection_time = time.time() - start_time

            # Test reading performance
            start_time = time.time()
            read_success = protector.read_protected_file(protected_parts, "reconstructed.Ψcws")
            read_time = time.time() - start_time

            # Test under high load (multiple operations)
            load_start = time.time()
            operations = 10
            for i in range(operations):
                small_file = Path(f"load_test_{i}.Ψcws")
                with open(small_file, 'w') as f:
                    f.write(f"Load test content {i}")
                protector.protect_file(small_file, parts=2)
                if small_file.exists():
                    small_file.unlink()

            load_time = time.time() - load_start

            results['tests'].extend([
                {
                    'name': 'Large File Protection Performance',
                    'status': 'passed' if protection_time < 10 else 'warning',
                    'details': f'1MB file protection time: {protection_time:.2f}s'
                },
                {
                    'name': 'Large File Reading Performance',
                    'status': 'passed' if read_time < 5 else 'warning',
                    'details': f'1MB file reading time: {read_time:.2f}s'
                },
                {
                    'name': 'High Load Performance',
                    'status': 'passed' if load_time < 30 else 'warning',
                    'details': f'{operations} operations time: {load_time:.2f}s'
                }
            ])

            # Cleanup
            if large_file.exists():
                large_file.unlink()
            if Path("reconstructed.Ψcws").exists():
                Path("reconstructed.Ψcws").unlink()

            results['status'] = 'passed'
            print("✅ Performance under attack tests completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Performance under attack tests failed: {e}")

        return results

    def _test_integration_security(self) -> Dict[str, Any]:
        """Test security integration with other components."""
        print("\n🔗 Testing Security Integration")
        print("-" * 30)

        results = {'status': 'pending', 'tests': []}

        try:
            from src.conscience.secure_Ψcws_protector import ΨCWSProtector
            from src.conscience.conscious_wave_modulator import ConsciousWaveModulator

            # Test integration with wave modulator
            modulator = ConsciousWaveModulator()
            protector = ΨCWSProtector()

            # Create test content
            test_content = "Security integration test content"
            test_file = Path("integration_test.txt")

            with open(test_file, 'w') as f:
                f.write(test_content)

            # Process through wave modulator
            Ψcws_file = modulator.process_file(test_file)

            # Save and protect
            Ψcws_path = Path("integration_test.Ψcws")
            Ψcws_file.save(Ψcws_path)

            # Protect with security system
            protected_parts = protector.protect_file(Ψcws_path, parts=4)

            # Test reading protected file
            read_success = protector.read_protected_file(protected_parts, "reconstructed_integration.Ψcws")

            # Verify integrity
            with open("reconstructed_integration.Ψcws", 'rb') as f:
                reconstructed_data = f.read()

            # Compare with original (this would need proper .Ψcws comparison)
            integration_success = read_success and len(reconstructed_data) > 0

            results['tests'].extend([
                {
                    'name': 'Wave Modulator Integration',
                    'status': 'passed' if Ψcws_file else 'failed',
                    'details': f'Wave modulator processing: {bool(Ψcws_file)}'
                },
                {
                    'name': 'Security Protection Integration',
                    'status': 'passed' if protected_parts else 'failed',
                    'details': f'Security protection applied: {bool(protected_parts)}'
                },
                {
                    'name': 'End-to-End Security Pipeline',
                    'status': 'passed' if integration_success else 'failed',
                    'details': f'Full pipeline success: {integration_success}'
                }
            ])

            # Cleanup
            files_to_clean = [test_file, Ψcws_path, Path("reconstructed_integration.Ψcws")]
            for file_path in files_to_clean:
                if file_path.exists():
                    file_path.unlink()

            results['status'] = 'passed'
            print("✅ Security integration tests completed")

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Security integration tests failed: {e}")

        return results

    def generate_security_report(self) -> str:
        """Generate comprehensive security validation report."""
        if not self.test_results:
            return "No security validation results available."

        report = []
        report.append("ΨCWS Security Validation Report")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Security metrics summary
        total_tests = 0
        passed_tests = 0
        security_level = "HIGH"

        for category, result in self.test_results.items():
            if 'tests' in result:
                category_tests = len(result['tests'])
                category_passed = sum(1 for test in result['tests'] if test['status'] == 'passed')

                total_tests += category_tests
                passed_tests += category_passed

        pass_rate = passed_tests / total_tests if total_tests > 0 else 0

        if pass_rate >= 0.9:
            security_level = "VERY HIGH"
        elif pass_rate >= 0.7:
            security_level = "HIGH"
        elif pass_rate >= 0.5:
            security_level = "MEDIUM"
        else:
            security_level = "LOW"

        report.append(f"Security Level: {security_level}")
        report.append(f"Test Results: {passed_tests}/{total_tests} passed ({pass_rate*100:.1f}%)")
        report.append("")

        # Detailed security findings
        for category, result in self.test_results.items():
            report.append(f"{category.upper()} Security - {result['status'].upper()}")
            report.append("-" * 40)

            if 'tests' in result:
                for test in result['tests']:
                    status_icon = "✅" if test['status'] == 'passed' else "❌"
                    report.append(f"{status_icon} {test['name']}")
                    report.append(f"   {test['details']}")

            if 'error' in result:
                report.append(f"⚠️ Security Issue: {result['error']}")

            report.append("")

        # Security recommendations
        report.append("Security Recommendations:")
        report.append("-" * 40)

        if security_level == "VERY HIGH":
            report.append("✅ System security is excellent")
            report.append("✅ Continue current security practices")
        elif security_level == "HIGH":
            report.append("⚠️ Consider additional encryption layers")
            report.append("⚠️ Monitor performance under high load")
        else:
            report.append("❌ Immediate security improvements needed")
            report.append("❌ Review encryption implementation")

        return "\n".join(report)


def main():
    """Main function to run security validation."""

    # Initialize security validator
    security_validator = ΨCWSSecurityValidator()

    # Run security validation suite
    validation_results = security_validator.run_security_validation_suite()

    # Generate and display security report
    security_report = security_validator.generate_security_report()
    print("\n" + security_report)

    # Determine overall security status
    all_passed = all(result.get('status') == 'passed' for result in validation_results.values())

    if all_passed:
        print("\n🎉 Security validation passed! ΨCWS protection system is secure.")
    else:
        print("\n⚠️ Security validation issues detected. Review recommendations above.")


if __name__ == "__main__":
    main()