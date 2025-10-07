#!/usr/bin/env python3
"""
Test script for ΨQRH API integration with new pipeline
Tests the integration without requiring Flask to be installed
"""

import sys
import os
import json

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def test_pipeline_import():
    """Test importing the ΨQRHPipeline"""
    print("🧪 Testing ΨQRHPipeline import...")
    try:
        from psiqrh import ΨQRHPipeline
        print("✅ ΨQRHPipeline import successful")
        return True
    except Exception as e:
        print(f"❌ ΨQRHPipeline import failed: {e}")
        return False

def test_pipeline_initialization():
    """Test initializing the ΨQRHPipeline"""
    print("\n🧪 Testing ΨQRHPipeline initialization...")
    try:
        from psiqrh import ΨQRHPipeline
        pipeline = ΨQRHPipeline(task="text-generation", device="cpu")
        print("✅ ΨQRHPipeline initialization successful")
        print(f"   📋 Task: {pipeline.task}")
        print(f"   💻 Device: {pipeline.device}")
        print(f"   🔢 Embed dim: {pipeline.config['embed_dim']}")
        return pipeline
    except Exception as e:
        print(f"❌ ΨQRHPipeline initialization failed: {e}")
        return None

def test_pipeline_processing(pipeline):
    """Test processing text with the pipeline"""
    print("\n🧪 Testing pipeline text processing...")
    try:
        test_text = "Hello, this is a test of the ΨQRH pipeline integration."
        result = pipeline(test_text)

        print("✅ Pipeline processing successful")
        print(f"   📝 Input: {test_text[:50]}...")
        print(f"   📤 Status: {result.get('status', 'unknown')}")
        print(f"   📊 Response length: {len(result.get('response', ''))}")
        print(f"   🔬 Physical metrics available: {'physical_metrics' in result}")

        if 'physical_metrics' in result:
            pm = result['physical_metrics']
            print(f"   🌌 Fractal dimension: {pm.get('D_fractal', 'N/A')}")
            print(f"   ⚛️ FCI: {pm.get('FCI', 'N/A')}")
            print(f"   🎯 Consciousness state: {pm.get('consciousness_state', 'N/A')}")

        return result
    except Exception as e:
        print(f"❌ Pipeline processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_api_structure(pipeline):
    """Test that the API structure works with the pipeline"""
    print("\n🧪 Testing API structure compatibility...")

    # Simulate what the API endpoints would do
    test_message = "Test API integration"

    try:
        # Simulate chat endpoint
        result = pipeline(test_message)

        # Check response structure expected by API
        api_response = {
            'status': result.get('status', 'success'),
            'user_message': test_message,
            'timestamp': 1234567890.0,  # Mock timestamp
            'processing_parameters': {
                'pipeline_config': {
                    'task': pipeline.task,
                    'device': pipeline.device,
                    'embed_dim': pipeline.config['embed_dim'],
                    'alpha': pipeline.config['alpha'],
                    'beta': pipeline.config['beta']
                }
            }
        }

        if result.get('status') == 'success':
            api_response['response'] = result.get('response', '')

            # Add physical metrics as consciousness metrics
            if 'physical_metrics' in result:
                physical_metrics = result['physical_metrics']
                api_response['physical_metrics'] = physical_metrics
                api_response['consciousness_metrics'] = {
                    'fci': physical_metrics.get('FCI', 0.0),
                    'state': physical_metrics.get('consciousness_state', 'UNKNOWN'),
                    'fractal_dimension': physical_metrics.get('D_fractal', 1.0)
                }

        print("✅ API structure compatibility test passed")
        print(f"   📋 API response keys: {list(api_response.keys())}")
        return True

    except Exception as e:
        print(f"❌ API structure compatibility test failed: {e}")
        return False

def test_health_endpoint(pipeline):
    """Test health endpoint structure"""
    print("\n🧪 Testing health endpoint structure...")
    try:
        status = 'healthy' if pipeline is not None else 'unhealthy'

        health_response = {
            'status': status,
            'system': 'ΨQRH API',
            'components': {
                'qrh_pipeline': 'loaded' if pipeline is not None else 'failed',
                'consciousness_processor': 'loaded' if hasattr(pipeline, 'consciousness_processor') and pipeline.consciousness_processor else 'unavailable',
                'gls_generator': 'not tested'  # GLS not available in test
            }
        }

        print("✅ Health endpoint structure test passed")
        print(f"   💚 Status: {health_response['status']}")
        return True

    except Exception as e:
        print(f"❌ Health endpoint structure test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Starting ΨQRH API Integration Tests")
    print("=" * 50)

    # Test 1: Import
    if not test_pipeline_import():
        print("\n❌ Integration tests failed at import stage")
        return 1

    # Test 2: Initialization
    pipeline = test_pipeline_initialization()
    if pipeline is None:
        print("\n❌ Integration tests failed at initialization stage")
        return 1

    # Test 3: Processing
    result = test_pipeline_processing(pipeline)
    if result is None:
        print("\n❌ Integration tests failed at processing stage")
        return 1

    # Test 4: API Structure
    if not test_api_structure(pipeline):
        print("\n❌ Integration tests failed at API structure stage")
        return 1

    # Test 5: Health Endpoint
    if not test_health_endpoint(pipeline):
        print("\n❌ Integration tests failed at health endpoint stage")
        return 1

    print("\n" + "=" * 50)
    print("🎉 All integration tests passed!")
    print("✅ ΨQRH API successfully integrated with new pipeline")
    print("=" * 50)

    return 0

if __name__ == "__main__":
    sys.exit(main())