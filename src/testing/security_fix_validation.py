#!/usr/bin/env python3
"""
ΨQRH Security Fix Validation - Verify All Security Issues Are Resolved
======================================================================

Comprehensive validation of all security fixes applied to ΨQRH system.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def validate_security_fixes():
    """Validate all security fixes have been successfully applied."""
    print("🔒 ΨQRH Security Fix Validation")
    print("=" * 60)

    validation_results = {}

    # 1. Validate HMAC-AES Layer Fix
    print("\n1. Validating HMAC-AES Layer Fix")
    print("-" * 40)

    try:
        from conscience.secure_Ψcws_protector import ΨCWSSecurityLayer
        security = ΨCWSSecurityLayer()
        test_data = b'test'

        # Test HMAC-AES layer
        encrypted = security._layer_hmac_aes(test_data, security.layer_keys[4])
        decrypted = security._layer_hmac_aes(encrypted, security.layer_keys[4], decrypt=True)

        hmac_aes_ok = test_data == decrypted
        validation_results['hmac_aes'] = hmac_aes_ok
        print(f"✅ HMAC-AES Layer: {'FIXED' if hmac_aes_ok else 'BROKEN'}")

    except Exception as e:
        validation_results['hmac_aes'] = False
        print(f"❌ HMAC-AES Layer: ERROR - {e}")

    # 2. Validate 7-Layer Pipeline
    print("\n2. Validating 7-Layer Pipeline")
    print("-" * 40)

    try:
        from conscience.secure_Ψcws_protector import ΨCWSSecurityLayer
        security = ΨCWSSecurityLayer()
        test_data = b'PSIQRH security test data'

        # Test full pipeline
        encrypted = security.encrypt_7_layers(test_data)
        decrypted = security.decrypt_7_layers(encrypted)

        pipeline_ok = test_data == decrypted
        validation_results['pipeline'] = pipeline_ok
        print(f"✅ 7-Layer Pipeline: {'FIXED' if pipeline_ok else 'BROKEN'}")
        print(f"   Original: {len(test_data)} bytes")
        print(f"   Encrypted: {len(encrypted)} bytes")
        print(f"   Decrypted: {len(decrypted)} bytes")

    except Exception as e:
        validation_results['pipeline'] = False
        print(f"❌ 7-Layer Pipeline: ERROR - {e}")

    # 3. Validate File Integrity Verification
    print("\n3. Validating File Integrity Verification")
    print("-" * 40)

    try:
        from conscience.secure_Ψcws_protector import ΨCWSFileSplitter, ΨCWSSecurityLayer
        security = ΨCWSSecurityLayer()
        splitter = ΨCWSFileSplitter(security)

        # Create test file
        test_file = Path("validation_test.txt")
        with open(test_file, 'w') as f:
            f.write("File integrity validation test")

        # Split and verify
        parts = splitter.split_file(test_file, parts=2)
        part0_ok = splitter._verify_part_integrity(parts[0])
        part1_ok = splitter._verify_part_integrity(parts[1])

        integrity_ok = part0_ok and part1_ok
        validation_results['file_integrity'] = integrity_ok
        print(f"✅ File Integrity: {'FIXED' if integrity_ok else 'BROKEN'}")
        print(f"   Part 0: {'VALID' if part0_ok else 'INVALID'}")
        print(f"   Part 1: {'VALID' if part1_ok else 'INVALID'}")

        # Cleanup
        if test_file.exists():
            test_file.unlink()

    except Exception as e:
        validation_results['file_integrity'] = False
        print(f"❌ File Integrity: ERROR - {e}")

    # 4. Validate Individual Layers
    print("\n4. Validating Individual Encryption Layers")
    print("-" * 40)

    try:
        from conscience.secure_Ψcws_protector import ΨCWSSecurityLayer
        security = ΨCWSSecurityLayer()
        test_data = b'test'

        layers = [
            ('AES-GCM', security._layer_aes_gcm),
            ('ChaCha20', security._layer_chacha20),
            ('Fernet', security._layer_fernet),
            ('XOR', security._layer_xor),
            ('Transposition', security._layer_transposition),
            ('HMAC-AES', security._layer_hmac_aes),
            ('Obfuscation', security._layer_obfuscation)
        ]

        layer_results = {}
        for layer_name, layer_func in layers:
            try:
                key = security.layer_keys[layers.index((layer_name, layer_func))]
                encrypted = layer_func(test_data, key)
                decrypted = layer_func(encrypted, key, decrypt=True)
                layer_ok = test_data == decrypted
                layer_results[layer_name] = layer_ok
                print(f"   {layer_name}: {'✅' if layer_ok else '❌'}")
            except Exception as e:
                layer_results[layer_name] = False
                print(f"   {layer_name}: ❌ ERROR")

        validation_results['individual_layers'] = layer_results
        all_layers_ok = all(layer_results.values())
        print(f"✅ Individual Layers: {'ALL FIXED' if all_layers_ok else 'SOME BROKEN'}")

    except Exception as e:
        validation_results['individual_layers'] = {}
        print(f"❌ Individual Layers: ERROR - {e}")

    # 5. Summary
    print("\n" + "=" * 60)
    print("SECURITY FIX VALIDATION SUMMARY")
    print("=" * 60)

    total_tests = len(validation_results)
    passed_tests = sum(1 for result in validation_results.values()
                      if result is True or (isinstance(result, dict) and all(result.values())))

    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")

    if passed_tests == total_tests:
        print("\n🎉 ALL SECURITY FIXES VALIDATED SUCCESSFULLY!")
        print("ΨQRH security system is now fully functional.")
        return True
    else:
        print("\n⚠️ SOME SECURITY ISSUES REMAIN")
        print("Review failed tests above for further fixes.")
        return False

def main():
    """Main validation function."""
    print("ΨQRH Security Fix Validation Tool")
    print("This tool validates all security fixes applied to the system.")
    print("=" * 60)

    success = validate_security_fixes()

    if success:
        print("\n✅ Security validation completed successfully!")
        print("The ΨQRH security system is ready for production use.")
    else:
        print("\n❌ Security validation found issues.")
        print("Please review the failed tests and apply additional fixes.")

if __name__ == "__main__":
    main()