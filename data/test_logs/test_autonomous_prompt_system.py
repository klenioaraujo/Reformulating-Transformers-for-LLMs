#!/usr/bin/env python3
"""
Test Script for Autonomous Prompt Generation System

Tests the autonomous prompt generation capabilities of the ΨQRH system
including file monitoring, template generation, and queue processing.

Classification: ΨQRH-Autonomous-Test-v1.0
"""

import sys
import os
import json
import time
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_autonomous_generator_initialization():
    """Test autonomous prompt generator initialization"""
    print("🧪 Testing Autonomous Generator Initialization")
    print("=" * 50)

    try:
        from src.cognitive.enhanced_agentic_runtime import create_enhanced_runtime

        # Create enhanced runtime (which includes autonomous generator)
        runtime = create_enhanced_runtime("development")
        print("✅ Enhanced runtime created with autonomous generator")

        # Check autonomous generator status
        status = runtime.get_runtime_status()
        autonomous_status = status.get("autonomous_generator")

        if autonomous_status:
            print(f"✅ Autonomous generator status: {autonomous_status}")
            print(f"   Templates loaded: {autonomous_status.get('templates_loaded', 0)}")
            print(f"   Ignored patterns: {len(autonomous_status.get('ignored_patterns', []))}")
            return True, runtime
        else:
            print("❌ Autonomous generator status not available")
            return False, None

    except Exception as e:
        print(f"❌ Autonomous generator initialization failed: {e}")
        return False, None

def test_file_monitoring_system(runtime):
    """Test file system monitoring and change detection"""
    print("\n🧪 Testing File System Monitoring")
    print("=" * 50)

    try:
        # Start the runtime (which starts autonomous monitoring)
        runtime.start()
        print("✅ Runtime started with file monitoring")

        # Wait a moment for monitoring to start
        time.sleep(2)

        # Check if monitoring is active
        status = runtime.get_runtime_status()
        autonomous_status = status.get("autonomous_generator")

        if autonomous_status and autonomous_status.get("running"):
            print("✅ File system monitoring is active")
            return True
        else:
            print("❌ File system monitoring is not active")
            return False

    except Exception as e:
        print(f"❌ File monitoring test failed: {e}")
        return False

def test_template_system():
    """Test prompt template loading and generation"""
    print("\n🧪 Testing Template System")
    print("=" * 50)

    try:
        from src.cognitive.autonomous_prompt_generator import PromptTemplate

        templates_dir = project_root / "construction_technical_manual" / "templates"
        template_system = PromptTemplate(templates_dir)

        print(f"✅ Template system initialized")
        print(f"   Templates loaded: {len(template_system.templates)}")

        # Test template generation
        variables = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "target_file": "src/core/test_component.py",
            "component_name": "test_component",
            "component_display_name": "Test Component",
            "priority": 80,
            "detection_time": datetime.utcnow().isoformat(),
            "detected_directory": "src/core",
            "component_classification": "ΨQRH-Test-Component-v1.0",
            "component_type": ".py"
        }

        prompt = template_system.generate_prompt("document_new_component", variables)

        if prompt:
            print("✅ Prompt generated from template successfully")
            print(f"   Generated prompt ID: {prompt.get('id', 'unknown')}")
            return True
        else:
            print("❌ Failed to generate prompt from template")
            return False

    except Exception as e:
        print(f"❌ Template system test failed: {e}")
        return False

def test_autonomous_prompt_generation(runtime):
    """Test actual autonomous prompt generation by creating a test file"""
    print("\n🧪 Testing Autonomous Prompt Generation")
    print("=" * 50)

    try:
        # Create test file in tests/
        test_dir = project_root / "tests"
        test_dir.mkdir(parents=True, exist_ok=True)

        test_file_path = test_dir / "test_new_layer.py"

        # Create the test file
        test_content = '''#!/usr/bin/env python3
"""
Test New Layer for Autonomous Prompt Generation

This is a test file created to trigger autonomous prompt generation
in the ΨQRH system. It should automatically generate documentation
and validation prompts.

Classification: ΨQRH-Test-Layer-v1.0
"""

import numpy as np
from typing import Optional, Dict, Any

class TestNewLayer:
    """Test layer for autonomous prompt generation testing"""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.weights = np.random.randn(dimension, dimension)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the test layer"""
        return np.dot(x, self.weights)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration"""
        return {
            "dimension": self.dimension,
            "layer_type": "test_new_layer"
        }
'''

        with open(test_file_path, 'w') as f:
            f.write(test_content)

        print(f"✅ Test file created: {test_file_path.relative_to(project_root)}")

        # Wait for the autonomous system to detect and process the change
        print("⏳ Waiting 12 seconds for autonomous prompt generation...")
        time.sleep(12)

        # Check if prompts were generated
        prompts_dir = runtime.prompt_engine.prompts_dir

        # Look for auto-generated prompts
        auto_prompts = []
        if prompts_dir.exists():
            for prompt_file in prompts_dir.glob("*.json"):
                try:
                    with open(prompt_file, 'r') as f:
                        prompt_data = json.load(f)

                    if prompt_data.get("generated_by") == "autonomous_prompt_generator":
                        auto_prompts.append(prompt_file.name)
                        print(f"✅ Found auto-generated prompt: {prompt_file.name}")

                except Exception as e:
                    print(f"⚠️ Could not read prompt file {prompt_file}: {e}")

        # Check autonomous generator queue status
        status = runtime.get_runtime_status()
        autonomous_status = status.get("autonomous_generator", {})
        queue_size = autonomous_status.get("queue_size", 0)

        print(f"   Autonomous generator queue size: {queue_size}")
        print(f"   Auto-generated prompts found: {len(auto_prompts)}")

        # Clean up test file
        if test_file_path.exists():
            test_file_path.unlink()
            print("✅ Test file cleaned up")

        # Success criteria: either prompts generated or queued
        if len(auto_prompts) > 0 or queue_size > 0:
            print("✅ Autonomous prompt generation working!")
            return True
        else:
            print("⚠️ No autonomous prompts generated or queued (may need more time)")
            return False

    except Exception as e:
        print(f"❌ Autonomous prompt generation test failed: {e}")
        return False

def test_priority_system():
    """Test priority calculation system"""
    print("\n🧪 Testing Priority System")
    print("=" * 50)

    try:
        from src.cognitive.autonomous_prompt_generator import PrioritySystem

        # Test different file priorities
        test_files = [
            (Path("src/core/psiqrh.py"), "modification", 90),
            (Path("src/cognitive/agent.py"), "creation", 95),
            (Path("src/conceptual/ecosystem.py"), "modification", 60),
            (Path("experiments/test.py"), "creation", 40),
            (Path("docs/readme.md"), "modification", 55)
        ]

        for file_path, change_type, expected_min in test_files:
            priority = PrioritySystem.calculate_priority(file_path, change_type)
            print(f"   {file_path} ({change_type}): priority {priority}")

            # Should be at least the expected minimum
            if priority >= expected_min:
                print(f"   ✅ Priority {priority} >= {expected_min}")
            else:
                print(f"   ❌ Priority {priority} < {expected_min}")
                return False

        print("✅ Priority system working correctly")
        return True

    except Exception as e:
        print(f"❌ Priority system test failed: {e}")
        return False

def test_change_detection():
    """Test file change detection with hashing"""
    print("\n🧪 Testing Change Detection System")
    print("=" * 50)

    try:
        from src.cognitive.autonomous_prompt_generator import ChangeDetector

        detector = ChangeDetector(project_root)
        print("✅ Change detector initialized")

        # Create a temporary test file
        test_dir = project_root / "data" / "test_temp"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / "change_test.py"

        # First write
        with open(test_file, 'w') as f:
            f.write("# Test content 1\nprint('hello')")

        # Detect first change (should be new)
        change1 = detector.detect_change(test_file)
        print(f"   First detection: changed={change1.get('changed')}, is_new={change1.get('is_new')}")

        if not change1.get("is_new"):
            print("❌ New file not detected as new")
            return False

        # Same content (should not detect change)
        change2 = detector.detect_change(test_file)
        print(f"   Same content: changed={change2.get('changed')}")

        if change2.get("changed"):
            print("❌ Same content incorrectly detected as changed")
            return False

        # Modify content
        with open(test_file, 'w') as f:
            f.write("# Test content 2\nprint('hello world')")

        change3 = detector.detect_change(test_file)
        print(f"   Modified content: changed={change3.get('changed')}")

        if not change3.get("changed"):
            print("❌ Modified content not detected as changed")
            return False

        # Clean up
        test_file.unlink()
        test_dir.rmdir()

        print("✅ Change detection system working correctly")
        return True

    except Exception as e:
        print(f"❌ Change detection test failed: {e}")
        return False

def run_autonomous_tests():
    """Run all autonomous prompt system tests"""
    print("🚀 ΨQRH Autonomous Prompt Generation Test Suite")
    print("=" * 60)

    results = {}
    runtime = None

    # Run tests
    try:
        results["initialization"], runtime = test_autonomous_generator_initialization()
        results["template_system"] = test_template_system()
        results["priority_system"] = test_priority_system()
        results["change_detection"] = test_change_detection()

        if runtime:
            results["file_monitoring"] = test_file_monitoring_system(runtime)
            results["autonomous_generation"] = test_autonomous_prompt_generation(runtime)

    finally:
        # Clean up runtime
        if runtime:
            runtime.stop()

    # Generate test report
    print("\n" + "=" * 60)
    print("📊 AUTONOMOUS SYSTEM TEST RESULTS")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():.<40} {status}")
        if success:
            passed += 1

    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\nSuccess Rate: {passed}/{total} ({success_rate:.1f}%)")

    # Save test report
    report = {
        "test_suite": "ΨQRH Autonomous Prompt Generation",
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
        "summary": {
            "passed": passed,
            "total": total,
            "success_rate": success_rate
        },
        "test_criteria": {
            "autonomous_detection_time": "≤10 seconds",
            "template_loading": "all templates loaded",
            "priority_calculation": "correct priorities assigned",
            "change_detection": "hash-based content changes",
            "file_monitoring": "watchdog active monitoring"
        }
    }

    # Save to data directory
    report_path = project_root / "data" / "validation_reports" / f"autonomous_system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Test report saved: {report_path}")

    if success_rate >= 80:
        print("🎉 Autonomous system tests PASSED!")
        return True
    else:
        print("⚠️ Autonomous system tests need attention")
        return False

if __name__ == "__main__":
    success = run_autonomous_tests()
    sys.exit(0 if success else 1)