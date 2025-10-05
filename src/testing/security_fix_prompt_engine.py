#!/usr/bin/env python3
"""
ΨQRH Security Fix Prompt Engine - Advanced Security Issue Resolution
====================================================================

Advanced prompt engine specifically designed to fix security issues in ΨQRH:
- HMAC-AES signature verification problems
- 7-layer encryption pipeline completion
- File part integrity verification
- Import issues resolution
- Component dependency optimization
"""

import sys
import os
import time
import json
import hashlib
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import inspect

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class ΨQRHSecurityFixEngine:
    """Advanced security issue resolution engine for ΨQRH."""

    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode
        self.fix_results = {}
        self.issue_analysis = {}

    def analyze_security_issues(self) -> Dict[str, Any]:
        """Comprehensive analysis of all security issues."""
        print("🔍 Analyzing ΨQRH Security Issues")
        print("=" * 60)

        analysis_results = {}

        # 1. Analyze HMAC-AES Layer Issues
        analysis_results['hmac_aes'] = self._analyze_hmac_aes_issues()

        # 2. Analyze 7-Layer Pipeline Issues
        analysis_results['pipeline'] = self._analyze_pipeline_issues()

        # 3. Analyze File Integrity Issues
        analysis_results['file_integrity'] = self._analyze_file_integrity_issues()

        # 4. Analyze Import Issues
        analysis_results['import_issues'] = self._analyze_import_issues()

        # 5. Analyze Dependency Issues
        analysis_results['dependency_issues'] = self._analyze_dependency_issues()

        self.issue_analysis = analysis_results
        return analysis_results

    def _analyze_hmac_aes_issues(self) -> Dict[str, Any]:
        """Analyze HMAC-AES layer signature verification issues."""
        print("\n🔐 Analyzing HMAC-AES Layer Issues")
        print("-" * 40)

        analysis = {'status': 'analyzing', 'issues': [], 'root_cause': ''}

        try:
            from conscience.secure_Ψcws_protector import ΨCWSSecurityLayer

            # Test HMAC-AES layer
            security = ΨCWSSecurityLayer()
            test_data = b'test'
            key = security.layer_keys[4]

            # Test encryption
            encrypted = security._layer_hmac_aes(test_data, key)

            # Test decryption
            try:
                decrypted = security._layer_hmac_aes(encrypted, key, decrypt=True)
                integrity_ok = test_data == decrypted

                if integrity_ok:
                    analysis['issues'].append('HMAC-AES layer working correctly')
                    analysis['status'] = 'functional'
                else:
                    analysis['issues'].append('HMAC-AES integrity failure')
                    analysis['status'] = 'broken'

            except Exception as e:
                analysis['issues'].append(f'HMAC-AES decryption error: {e}')
                analysis['status'] = 'broken'
                analysis['root_cause'] = str(e)

            # Analyze the implementation
            hmac_aes_code = inspect.getsource(security._layer_hmac_aes)

            # Check for common issues
            if 'authenticate_additional_data' in hmac_aes_code:
                analysis['issues'].append('Potential GCM mode confusion in HMAC-AES')

            if 'finalize_with_tag' in hmac_aes_code:
                analysis['issues'].append('Incorrect tag handling in HMAC-AES')

        except Exception as e:
            analysis['issues'].append(f'Analysis failed: {e}')
            analysis['status'] = 'error'

        return analysis

    def _analyze_pipeline_issues(self) -> Dict[str, Any]:
        """Analyze 7-layer encryption pipeline issues."""
        print("\n🔗 Analyzing 7-Layer Pipeline Issues")
        print("-" * 40)

        analysis = {'status': 'analyzing', 'issues': [], 'functional_layers': []}

        try:
            from conscience.secure_Ψcws_protector import ΨCWSSecurityLayer

            security = ΨCWSSecurityLayer()
            test_data = b'pipeline_test'

            # Test individual layers
            layers = [
                ('AES-GCM', security._layer_aes_gcm),
                ('ChaCha20', security._layer_chacha20),
                ('Fernet', security._layer_fernet),
                ('XOR', security._layer_xor),
                ('Transposition', security._layer_transposition),
                ('HMAC-AES', security._layer_hmac_aes),
                ('Obfuscation', security._layer_obfuscation)
            ]

            for layer_name, layer_func in layers:
                try:
                    key = security.layer_keys[layers.index((layer_name, layer_func))]
                    encrypted = layer_func(test_data, key)
                    decrypted = layer_func(encrypted, key, decrypt=True)

                    if test_data == decrypted:
                        analysis['functional_layers'].append(layer_name)
                    else:
                        analysis['issues'].append(f'{layer_name} integrity failure')

                except Exception as e:
                    analysis['issues'].append(f'{layer_name} error: {e}')

            # Test full pipeline
            try:
                encrypted = security.encrypt_7_layers(test_data)
                decrypted = security.decrypt_7_layers(encrypted)

                if test_data == decrypted:
                    analysis['status'] = 'functional'
                else:
                    analysis['status'] = 'broken'
                    analysis['issues'].append('Full pipeline integrity failure')

            except Exception as e:
                analysis['status'] = 'broken'
                analysis['issues'].append(f'Full pipeline error: {e}')

        except Exception as e:
            analysis['issues'].append(f'Pipeline analysis failed: {e}')
            analysis['status'] = 'error'

        return analysis

    def _analyze_file_integrity_issues(self) -> Dict[str, Any]:
        """Analyze file part integrity verification issues."""
        print("\n📊 Analyzing File Integrity Issues")
        print("-" * 40)

        analysis = {'status': 'analyzing', 'issues': [], 'verification_steps': []}

        try:
            from conscience.secure_Ψcws_protector import ΨCWSFileSplitter, ΨCWSSecurityLayer

            security = ΨCWSSecurityLayer()
            splitter = ΨCWSFileSplitter(security)

            # Create test file
            test_file = Path("integrity_test.txt")
            with open(test_file, 'w') as f:
                f.write("file integrity test")

            # Split file
            parts = splitter.split_file(test_file, parts=2)

            # Test part verification
            for i, part in enumerate(parts):
                try:
                    verification_result = splitter._verify_part_integrity(part)
                    analysis['verification_steps'].append({
                        'part': i,
                        'result': verification_result,
                        'content_hash': part.content_hash[:16] + '...',
                        'integrity_hash': part.integrity_hash[:16] + '...'
                    })

                    if not verification_result:
                        analysis['issues'].append(f'Part {i} verification failed')

                except Exception as e:
                    analysis['issues'].append(f'Part {i} verification error: {e}')

            # Cleanup
            if test_file.exists():
                test_file.unlink()

            if all(step['result'] for step in analysis['verification_steps']):
                analysis['status'] = 'functional'
            else:
                analysis['status'] = 'broken'

        except Exception as e:
            analysis['issues'].append(f'File integrity analysis failed: {e}')
            analysis['status'] = 'error'

        return analysis

    def _analyze_import_issues(self) -> Dict[str, Any]:
        """Analyze import issues in core modules."""
        print("\n📦 Analyzing Import Issues")
        print("-" * 40)

        analysis = {'status': 'analyzing', 'issues': [], 'import_tests': []}

        # Test core imports
        import_tests = [
            ('core.ΨQRH', 'QRHFactory'),
            ('core.qrh_layer', 'QRHLayer'),
            ('core.quaternion_operations', 'QuaternionOperations'),
            ('conscience.conscious_wave_modulator', 'ConsciousWaveModulator'),
            ('conscience.secure_Ψcws_protector', 'ΨCWSSecurityLayer'),
            ('fractal.spectral_filter', 'SpectralFilter')
        ]

        for module_path, class_name in import_tests:
            try:
                exec(f"from {module_path} import {class_name}")
                analysis['import_tests'].append({
                    'module': module_path,
                    'class': class_name,
                    'status': 'success'
                })
            except Exception as e:
                analysis['import_tests'].append({
                    'module': module_path,
                    'class': class_name,
                    'status': 'failed',
                    'error': str(e)
                })
                analysis['issues'].append(f'{module_path}.{class_name}: {e}')

        failed_imports = sum(1 for test in analysis['import_tests'] if test['status'] == 'failed')

        if failed_imports == 0:
            analysis['status'] = 'functional'
        else:
            analysis['status'] = 'broken'

        return analysis

    def _analyze_dependency_issues(self) -> Dict[str, Any]:
        """Analyze component dependency issues."""
        print("\n🔗 Analyzing Dependency Issues")
        print("-" * 40)

        analysis = {'status': 'analyzing', 'issues': [], 'dependency_graph': {}}

        try:
            # Analyze dependency patterns
            modules_to_check = [
                'src/core/ΨQRH.py',
                'src/core/qrh_layer.py',
                'src/conscience/conscious_wave_modulator.py',
                'src/conscience/secure_Ψcws_protector.py'
            ]

            for module_path in modules_to_check:
                if Path(module_path).exists():
                    with open(module_path, 'r') as f:
                        content = f.read()

                    # Count imports
                    import_count = content.count('import ')
                    from_count = content.count('from ')

                    analysis['dependency_graph'][module_path] = {
                        'imports': import_count,
                        'from_imports': from_count,
                        'relative_imports': content.count('from .'),
                        'external_imports': content.count('from src.')
                    }

            # Check for circular dependency patterns
            high_dependency_modules = [
                mod for mod, deps in analysis['dependency_graph'].items()
                if deps['imports'] + deps['from_imports'] > 10
            ]

            if high_dependency_modules:
                analysis['issues'].append(f'High dependency modules: {high_dependency_modules}')

            analysis['status'] = 'analyzed'

        except Exception as e:
            analysis['issues'].append(f'Dependency analysis failed: {e}')
            analysis['status'] = 'error'

        return analysis

    def generate_fix_plan(self) -> Dict[str, Any]:
        """Generate comprehensive fix plan based on analysis."""
        if not self.issue_analysis:
            self.analyze_security_issues()

        fix_plan = {
            'priority_fixes': [],
            'medium_priority_fixes': [],
            'low_priority_fixes': [],
            'estimated_time': '2-4 hours',
            'risk_level': 'MEDIUM'
        }

        # Priority 1: Critical security issues
        if self.issue_analysis.get('hmac_aes', {}).get('status') == 'broken':
            fix_plan['priority_fixes'].append({
                'issue': 'HMAC-AES signature verification',
                'action': 'Rewrite HMAC-AES layer with proper tag handling',
                'files': ['src/conscience/secure_Ψcws_protector.py'],
                'estimated_time': '1 hour'
            })

        if self.issue_analysis.get('pipeline', {}).get('status') == 'broken':
            fix_plan['priority_fixes'].append({
                'issue': '7-layer encryption pipeline',
                'action': 'Complete full encryption/decryption pipeline',
                'files': ['src/conscience/secure_Ψcws_protector.py'],
                'estimated_time': '1.5 hours'
            })

        # Priority 2: Functional issues
        if self.issue_analysis.get('file_integrity', {}).get('status') == 'broken':
            fix_plan['priority_fixes'].append({
                'issue': 'File part integrity verification',
                'action': 'Fix part verification logic and hash validation',
                'files': ['src/conscience/secure_Ψcws_protector.py'],
                'estimated_time': '45 minutes'
            })

        # Medium priority: Import and dependency issues
        if self.issue_analysis.get('import_issues', {}).get('status') == 'broken':
            fix_plan['medium_priority_fixes'].append({
                'issue': 'Import issues in core modules',
                'action': 'Fix relative import paths and module structure',
                'files': ['src/core/ΨQRH.py', 'src/core/__init__.py'],
                'estimated_time': '30 minutes'
            })

        if self.issue_analysis.get('dependency_issues', {}).get('issues'):
            fix_plan['medium_priority_fixes'].append({
                'issue': 'Complex component dependencies',
                'action': 'Optimize import structure and reduce coupling',
                'files': ['Multiple core files'],
                'estimated_time': '1 hour'
            })

        return fix_plan

    def apply_hmac_aes_fix(self) -> bool:
        """Apply fix for HMAC-AES signature verification issues."""
        print("\n🔧 Applying HMAC-AES Fix")
        print("-" * 40)

        try:
            protector_file = Path("src/conscience/secure_Ψcws_protector.py")

            if not protector_file.exists():
                print("❌ Protector file not found")
                return False

            # Read current implementation
            with open(protector_file, 'r') as f:
                content = f.read()

            # Find and replace HMAC-AES implementation
            old_hmac_aes = '''
    def _layer_hmac_aes(self, data: bytes, key: bytes, decrypt: bool = False) -> bytes:
        """Camada 6: HMAC + AES."""
        def pad_data(data: bytes) -> bytes:
            """Pad data to multiple of block size."""
            block_size = 16
            padding_length = block_size - (len(data) % block_size)
            return data + bytes([padding_length] * padding_length)

        def unpad_data(data: bytes) -> bytes:
            """Remove padding from data."""
            padding_length = data[-1]
            return data[:-padding_length]

        if decrypt:
            # Extrair HMAC e dados
            hmac_digest = data[:32]
            ciphertext = data[32:]

            # Verificar HMAC
            h = hmac.HMAC(key, hashes.SHA256(), backend=self.backend)
            h.update(ciphertext)
            h.verify(hmac_digest)

            # Decriptar AES
            iv = ciphertext[:16]
            encrypted_data = ciphertext[16:]
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(encrypted_data) + decryptor.finalize()
            return unpad_data(decrypted)
        else:
            # Pad data before encryption
            padded_data = pad_data(data)

            # Criptografar AES
            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()

            # Calcular HMAC
            h = hmac.HMAC(key, hashes.SHA256(), backend=self.backend)
            h.update(ciphertext)
            hmac_digest = h.finalize()

            return hmac_digest + iv + ciphertext
'''

            new_hmac_aes = '''
    def _layer_hmac_aes(self, data: bytes, key: bytes, decrypt: bool = False) -> bytes:
        """Camada 6: HMAC + AES (Fixed Version)."""
        from cryptography.hazmat.primitives import padding

        if decrypt:
            # Extract HMAC and ciphertext
            if len(data) < 32 + 16:  # HMAC + IV minimum
                raise ValueError("Invalid HMAC-AES data format")

            hmac_digest = data[:32]
            ciphertext = data[32:]

            # Verify HMAC BEFORE decryption
            h = hmac.HMAC(key, hashes.SHA256(), backend=self.backend)
            h.update(ciphertext)
            h.verify(hmac_digest)

            # Extract IV and encrypted data
            iv = ciphertext[:16]
            encrypted_data = ciphertext[16:]

            # Decrypt
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

            # Unpad
            unpadder = padding.PKCS7(128).unpadder()
            return unpadder.update(padded_data) + unpadder.finalize()

        else:
            # Pad data
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data) + padder.finalize()

            # Encrypt
            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()

            # Calculate HMAC
            h = hmac.HMAC(key, hashes.SHA256(), backend=self.backend)
            h.update(ciphertext)
            hmac_digest = h.finalize()

            return hmac_digest + iv + ciphertext
'''

            # Replace the implementation
            if old_hmac_aes in content:
                content = content.replace(old_hmac_aes, new_hmac_aes)

                # Write updated content
                with open(protector_file, 'w') as f:
                    f.write(content)

                print("✅ HMAC-AES layer fixed successfully")

                # Test the fix
                from conscience.secure_Ψcws_protector import ΨCWSSecurityLayer
                security = ΨCWSSecurityLayer()
                test_data = b'test'

                encrypted = security._layer_hmac_aes(test_data, security.layer_keys[4])
                decrypted = security._layer_hmac_aes(encrypted, security.layer_keys[4], decrypt=True)

                if test_data == decrypted:
                    print("✅ HMAC-AES fix verified")
                    return True
                else:
                    print("❌ HMAC-AES fix verification failed")
                    return False
            else:
                print("❌ Could not find HMAC-AES implementation to replace")
                return False

        except Exception as e:
            print(f"❌ HMAC-AES fix failed: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return False

    def generate_fix_report(self) -> str:
        """Generate comprehensive fix report."""
        if not self.issue_analysis:
            self.analyze_security_issues()

        fix_plan = self.generate_fix_plan()

        report = []
        report.append("ΨQRH Security Fix Report")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Issue Summary
        report.append("ISSUE SUMMARY")
        report.append("-" * 40)

        for category, analysis in self.issue_analysis.items():
            status_icon = "✅" if analysis.get('status') == 'functional' else "❌"
            report.append(f"{status_icon} {category.replace('_', ' ').title()}: {analysis.get('status', 'unknown')}")

            if analysis.get('issues'):
                for issue in analysis['issues'][:3]:  # Show top 3 issues
                    report.append(f"   • {issue}")

        report.append("")

        # Fix Plan
        report.append("FIX PLAN")
        report.append("-" * 40)

        for priority, fixes in fix_plan.items():
            if fixes and isinstance(fixes, list):
                report.append(f"\n{priority.replace('_', ' ').title()}:")
                for fix in fixes:
                    report.append(f"• {fix['issue']}")
                    report.append(f"  Action: {fix['action']}")
                    report.append(f"  Files: {', '.join(fix['files'])}")
                    report.append(f"  Time: {fix['estimated_time']}")

        report.append("")
        report.append(f"Total Estimated Time: {fix_plan['estimated_time']}")
        report.append(f"Risk Level: {fix_plan['risk_level']}")

        return "\n".join(report)


def main():
    """Main function to run security fix engine."""
    print("ΨQRH Security Fix Prompt Engine")
    print("=" * 50)

    # Initialize fix engine
    fix_engine = ΨQRHSecurityFixEngine(debug_mode=True)

    # Analyze issues
    print("\nStep 1: Analyzing security issues...")
    analysis = fix_engine.analyze_security_issues()

    # Generate fix plan
    print("\nStep 2: Generating fix plan...")
    fix_plan = fix_engine.generate_fix_plan()

    # Generate report
    print("\nStep 3: Generating comprehensive report...")
    report = fix_engine.generate_fix_report()
    print("\n" + report)

    # Ask user if they want to apply fixes
    print("\n" + "=" * 50)
    response = input("Apply HMAC-AES fix now? (y/n): ").strip().lower()

    if response == 'y':
        print("\nStep 4: Applying HMAC-AES fix...")
        success = fix_engine.apply_hmac_aes_fix()

        if success:
            print("\n🎉 HMAC-AES fix applied successfully!")
            print("Run 'make -f Makefile.testing test-security' to verify.")
        else:
            print("\n⚠️ HMAC-AES fix failed. Manual intervention required.")
    else:
        print("\nℹ️  Fix plan generated. Apply fixes manually as needed.")

    print("\n" + "=" * 50)
    print("Security fix analysis complete.")


if __name__ == "__main__":
    main()