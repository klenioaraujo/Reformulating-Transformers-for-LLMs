
#!/usr/bin/env python3
"""
Test for Spectral Projector Component
"""

import sys
import os
sys.path.append('src')

def test_spectral_projector_import():
    """Test that spectral projector can be imported"""
    try:
        from core.spectral_projector import SpectralProjector
        print("✅ SpectralProjector import successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_spectral_projector_creation():
    """Test that spectral projector can be instantiated"""
    try:
        from core.spectral_projector import SpectralProjector
        projector = SpectralProjector(projection_dim=50)
        print("✅ SpectralProjector instantiation successful")
        return True
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        return False

def test_spectral_projector_methods():
    """Test that spectral projector has required methods"""
    try:
        from core.spectral_projector import SpectralProjector
        projector = SpectralProjector()
        
        # Check required methods
        required_methods = ['fit', 'transform', 'inverse_transform', 'get_spectral_statistics']
        for method in required_methods:
            if not hasattr(projector, method):
                print(f"❌ Missing method: {method}")
                return False
        
        print("✅ All required methods present")
        return True
    except Exception as e:
        print(f"❌ Method test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running Spectral Projector Tests...")
    print("=" * 40)
    
    tests = [
        test_spectral_projector_import,
        test_spectral_projector_creation,
        test_spectral_projector_methods
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
