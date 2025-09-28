#!/usr/bin/env python3
"""
Test main system initialization (dry-run mode)
Tests if key system components can be initialized without errors

Following isolation policy - all artifacts saved in data/
"""

import sys
import os
import json
import traceback
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

def test_production_system_init():
    """Test if production system can be initialized"""
    try:
        from src.core.production_system import ProductionSemanticQRH, ProductionConfig, ProductionMode

        # Try to create a minimal production config
        config = ProductionConfig(
            model_name="test_model",
            mode=ProductionMode.VALIDATION,
            batch_size=1,
            max_sequence_length=32
        )

        # This may fail due to dependencies, but we test import success
        return {
            'status': 'IMPORT_SUCCESS',
            'config_created': True,
            'error': None
        }

    except ImportError as e:
        return {
            'status': 'IMPORT_FAIL',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    except Exception as e:
        return {
            'status': 'CONFIG_FAIL',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def test_conceptual_engine_init():
    """Test conceptual engine components"""
    try:
        from src.conceptual.living_ecosystem_engine import LivingEcosystemEngine

        # Just test import - full init may require complex setup
        return {
            'status': 'IMPORT_SUCCESS',
            'error': None
        }

    except ImportError as e:
        return {
            'status': 'IMPORT_FAIL',
            'error': str(e)
        }
    except Exception as e:
        return {
            'status': 'INIT_FAIL',
            'error': str(e)
        }

def test_example_script():
    """Test if our example script works"""
    try:
        # Import and run basic functionality from our example
        import subprocess
        import tempfile

        # Create a simple test script
        test_script = '''
import sys
sys.path.insert(0, ".")

try:
    from src.core.qrh_layer import QRHLayer, QRHConfig
    config = QRHConfig(embed_dim=32)
    layer = QRHLayer(config)
    print("SUCCESS: Example script components working")
    exit(0)
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
'''

        # Write to temp file in data directory
        temp_path = os.path.join('data', 'test_logs', 'temp_test_script.py')
        with open(temp_path, 'w') as f:
            f.write(test_script)

        # Run the script
        result = subprocess.run([sys.executable, temp_path],
                              capture_output=True, text=True, timeout=30)

        # Clean up
        os.remove(temp_path)

        return {
            'status': 'SUCCESS' if result.returncode == 0 else 'FAIL',
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }

    except Exception as e:
        return {
            'status': 'FAIL',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def run_system_initialization_tests():
    """Run all system initialization tests"""

    print("🚀 ΨQRH System Initialization Tests")
    print("=" * 50)

    test_suite = [
        ('Production System', test_production_system_init),
        ('Conceptual Engine', test_conceptual_engine_init),
        ('Example Script', test_example_script)
    ]

    results = {}

    for test_name, test_func in test_suite:
        print(f"\n🔧 Testing {test_name}...")

        result = test_func()
        results[test_name] = result

        if result['status'] in ['SUCCESS', 'IMPORT_SUCCESS']:
            print(f"✅ {test_name}: {result['status']}")
        else:
            print(f"❌ {test_name}: {result['status']} - {result.get('error', 'Unknown error')}")

    # Generate report
    report = {
        'test_timestamp': datetime.now().isoformat(),
        'test_results': results,
        'summary': {
            'total_tests': len(test_suite),
            'successful_tests': sum(1 for r in results.values()
                                  if r['status'] in ['SUCCESS', 'IMPORT_SUCCESS']),
            'overall_status': 'PASS' if all(r['status'] in ['SUCCESS', 'IMPORT_SUCCESS']
                                          for r in results.values()) else 'PARTIAL'
        }
    }

    # Save report
    report_path = os.path.join('data', 'validation_reports',
                              f'system_initialization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📋 Summary: {report['summary']['successful_tests']}/{report['summary']['total_tests']} tests successful")
    print(f"📄 Report saved: {report_path}")

    return report

if __name__ == "__main__":
    run_system_initialization_tests()