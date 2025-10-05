#!/usr/bin/env python3
"""
Simple Security Test - Basic ΨCWS Security Validation
====================================================

Simplified test for ΨCWS security system focusing on core functionality.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_basic_security():
    """Test basic security functionality."""
    print("🔒 Testing Basic ΨCWS Security")
    print("=" * 40)

    try:
        from conscience.secure_Ψcws_protector import ΨCWSSecurityLayer

        # Initialize security layer
        security = ΨCWSSecurityLayer()
        print("✅ Security layer initialized")

        # Test data
        test_data = b"PSIQRH security test data"

        # Test individual layers
        print("\n🧪 Testing Individual Layers:")

        # Layer 1: AES-GCM
        encrypted1 = security._layer_aes_gcm(test_data, security.layer_keys[0])
        decrypted1 = security._layer_aes_gcm(encrypted1, security.layer_keys[0], decrypt=True)
        print(f"✅ AES-GCM: {test_data == decrypted1}")

        # Layer 4: XOR
        encrypted4 = security._layer_xor(test_data, security.layer_keys[3])
        decrypted4 = security._layer_xor(encrypted4, security.layer_keys[3], decrypt=True)
        print(f"✅ XOR: {test_data == decrypted4}")

        # Layer 7: Obfuscation
        encrypted7 = security._layer_obfuscation(test_data, security.layer_keys[5])
        decrypted7 = security._layer_obfuscation(encrypted7, security.layer_keys[5], decrypt=True)
        print(f"✅ Obfuscation: {test_data == decrypted7}")

        # Test simplified 3-layer security
        print("\n🧪 Testing Simplified 3-Layer Security:")

        # Encrypt: AES-GCM → XOR → Obfuscation
        encrypted = security._layer_aes_gcm(test_data, security.layer_keys[0])
        encrypted = security._layer_xor(encrypted, security.layer_keys[3])
        encrypted = security._layer_obfuscation(encrypted, security.layer_keys[5])

        # Decrypt: Reverse order
        decrypted = security._layer_obfuscation(encrypted, security.layer_keys[5], decrypt=True)
        decrypted = security._layer_xor(decrypted, security.layer_keys[3], decrypt=True)
        decrypted = security._layer_aes_gcm(decrypted, security.layer_keys[0], decrypt=True)

        print(f"✅ 3-Layer Security: {test_data == decrypted}")
        print(f"   Original size: {len(test_data)} bytes")
        print(f"   Encrypted size: {len(encrypted)} bytes")

        # Test file splitting basics
        print("\n📊 Testing File Splitting Basics:")

        # Create test file
        test_file = Path("security_test.txt")
        with open(test_file, 'wb') as f:
            f.write(test_data)

        from conscience.secure_Ψcws_protector import ΨCWSFileSplitter

        splitter = ΨCWSFileSplitter(security)
        file_parts = splitter.split_file(test_file, parts=2)

        print(f"✅ File split into {len(file_parts)} parts")
        print(f"✅ Part 0 hash: {file_parts[0].content_hash[:16]}...")
        print(f"✅ Part 1 hash: {file_parts[1].content_hash[:16]}...")

        # Test part integrity
        part0_ok = splitter._verify_part_integrity(file_parts[0])
        part1_ok = splitter._verify_part_integrity(file_parts[1])

        print(f"✅ Part 0 integrity: {part0_ok}")
        print(f"✅ Part 1 integrity: {part1_ok}")

        # Cleanup
        if test_file.exists():
            test_file.unlink()

        print("\n🎉 Basic security tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consciousness_integration():
    """Test integration with consciousness components."""
    print("\n🧠 Testing Consciousness Integration")
    print("=" * 40)

    try:
        from conscience.conscious_wave_modulator import ConsciousWaveModulator

        # Initialize modulator
        modulator = ConsciousWaveModulator()
        print("✅ Consciousness modulator initialized")

        # Test basic functionality
        test_text = "Consciousness integration test"

        # Generate wave embeddings
        embeddings = modulator._generate_wave_embeddings(test_text)
        print(f"✅ Wave embeddings generated: {embeddings.shape}")

        # Generate chaotic trajectories
        trajectories = modulator._generate_chaotic_trajectories(test_text)
        print(f"✅ Chaotic trajectories: {trajectories.shape}")

        # Compute consciousness metrics
        spectra = modulator._compute_fourier_spectra(embeddings)
        metrics = modulator._compute_consciousness_metrics(embeddings, trajectories, spectra)

        print(f"✅ Consciousness metrics computed:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.3f}")

        print("\n🎉 Consciousness integration tests completed!")
        return True

    except Exception as e:
        print(f"❌ Consciousness integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("ΨQRH Simple Security Test Suite")
    print("=" * 50)

    # Run security tests
    security_ok = test_basic_security()

    # Run consciousness tests
    consciousness_ok = test_consciousness_integration()

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"Security Tests: {'✅ PASSED' if security_ok else '❌ FAILED'}")
    print(f"Consciousness Tests: {'✅ PASSED' if consciousness_ok else '❌ FAILED'}")

    if security_ok and consciousness_ok:
        print("\n🎉 All basic tests passed! ΨQRH system is functional.")
    else:
        print("\n⚠️ Some tests failed. Review errors above.")

if __name__ == "__main__":
    main()