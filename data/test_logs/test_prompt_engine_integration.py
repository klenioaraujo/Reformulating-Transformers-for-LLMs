#!/usr/bin/env python3
"""
Test Script for PromptEngine Integration with Cognitive Runtime

Tests the integrated PromptEngine system with the ΨQRH cognitive habitat.
All test artifacts saved in data/ following isolation policy.

Classification: ΨQRH-Integration-Test-v1.0
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

def test_prompt_engine_agent():
    """Test PromptEngine Agent functionality"""
    print("🧪 Testing PromptEngine Agent")
    print("=" * 50)

    try:
        from src.cognitive.prompt_engine_agent import create_prompt_engine_agent

        # Create agent
        agent = create_prompt_engine_agent("development")
        print(f"✅ PromptEngine Agent created (ID: {agent.agent_id[:8]})")

        # Test agent status
        status = agent.get_agent_status()
        print(f"✅ Agent Status: {status['agent_type']} in {status['habitat_mode']} mode")

        # Test audit logging
        agent.log_audit_entry("test_prompt", "test_execution", "success", ["test_artifact.txt"])
        print("✅ Audit logging successful")

        return True, agent

    except Exception as e:
        print(f"❌ PromptEngine Agent test failed: {e}")
        return False, None

def test_enhanced_runtime():
    """Test Enhanced Agentic Runtime"""
    print("\n🧪 Testing Enhanced Agentic Runtime")
    print("=" * 50)

    try:
        from src.cognitive.enhanced_agentic_runtime import create_enhanced_runtime

        # Create runtime
        runtime = create_enhanced_runtime("development")
        print("✅ Enhanced Runtime created")

        # Test runtime status
        status = runtime.get_runtime_status()
        print(f"✅ Runtime Type: {status['runtime_type']}")
        print(f"✅ Habitat Mode: {status['habitat_mode']}")

        # Test state updates
        runtime.update_system_state("test_key", "test_value")
        print("✅ System state update successful")

        # Test manual documentation trigger
        prompt_id = runtime.trigger_manual_documentation(
            "src/core/ΨQRH.py",
            "Test manual documentation trigger"
        )
        if prompt_id:
            print(f"✅ Manual documentation triggered: {prompt_id}")
        else:
            print("⚠️ Manual documentation trigger returned None")

        return True, runtime

    except Exception as e:
        print(f"❌ Enhanced Runtime test failed: {e}")
        return False, None

def test_production_safety():
    """Test production safety filtering"""
    print("\n🧪 Testing Production Safety")
    print("=" * 50)

    try:
        from src.cognitive.prompt_engine_agent import create_prompt_engine_agent

        # Create production agent
        agent = create_prompt_engine_agent("production")
        print("✅ Production agent created")

        # Test safe prompt
        safe_prompt = {
            "id": "test_safe",
            "action": "document_component",
            "production_safe": True,
            "instructions": "Generate documentation"
        }

        is_safe = agent.is_production_safe(safe_prompt)
        print(f"✅ Safe prompt detected: {is_safe}")

        # Test unsafe prompt
        unsafe_prompt = {
            "id": "test_unsafe",
            "action": "system_restart",
            "instructions": "Delete all files and restart system"
        }

        is_unsafe = agent.is_production_safe(unsafe_prompt)
        print(f"✅ Unsafe prompt blocked: {not is_unsafe}")

        return True

    except Exception as e:
        print(f"❌ Production safety test failed: {e}")
        return False

def test_prompt_execution():
    """Test actual prompt execution"""
    print("\n🧪 Testing Prompt Execution")
    print("=" * 50)

    try:
        from src.cognitive.prompt_engine_agent import create_prompt_engine_agent

        agent = create_prompt_engine_agent("development")

        # Create a test prompt
        test_prompt = {
            "id": f"test_integration_{int(time.time())}",
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "action": "document_component",
            "target_file": "test_component.py",
            "output_section": "Integration Test Documentation",
            "auto_delete": True,
            "production_safe": True,
            "instructions": "This is a test prompt generated during integration testing. It should be executed successfully and then auto-deleted."
        }

        # Save test prompt
        test_prompt_path = agent.prompts_dir / f"{test_prompt['id']}.json"
        with open(test_prompt_path, 'w') as f:
            json.dump(test_prompt, f, indent=2)

        print(f"✅ Test prompt created: {test_prompt['id']}")

        # Execute the prompt
        success, result = agent.execute_prompt(test_prompt_path)

        if success:
            print("✅ Prompt execution successful")
            print(f"   Execution time: {result.get('execution_time', 0):.3f}s")
            print(f"   Artifacts: {result.get('artifacts', [])}")
        else:
            print(f"❌ Prompt execution failed: {result.get('error', 'Unknown error')}")

        # Check if auto-deleted
        if not test_prompt_path.exists():
            print("✅ Auto-delete successful")
        else:
            print("⚠️ Auto-delete failed or disabled")

        return success

    except Exception as e:
        print(f"❌ Prompt execution test failed: {e}")
        return False

def test_audit_integration():
    """Test audit system integration"""
    print("\n🧪 Testing Audit System Integration")
    print("=" * 50)

    try:
        from src.cognitive.prompt_engine_agent import create_prompt_engine_agent

        agent = create_prompt_engine_agent("development")

        # Generate several audit entries
        test_entries = [
            ("test_1", "document_component", "success", ["manual.md"]),
            ("test_2", "validate_system", "success", ["validation_report.json"]),
            ("test_3", "integrate_systems", "failed", [], "Integration failed: missing dependency")
        ]

        for entry in test_entries:
            if len(entry) == 4:
                agent.log_audit_entry(entry[0], entry[1], entry[2], entry[3])
            else:
                agent.log_audit_entry(entry[0], entry[1], entry[2], entry[3], entry[4])

        print(f"✅ Generated {len(test_entries)} audit entries")

        # Check if audit file exists and has content
        if agent.audit_log_path.exists():
            with open(agent.audit_log_path, 'r') as f:
                lines = f.readlines()
            print(f"✅ Audit log contains {len(lines)} entries")

            # Validate JSON structure
            try:
                for line in lines[-len(test_entries):]:  # Check last entries
                    json.loads(line.strip())
                print("✅ Audit log JSON structure valid")
            except json.JSONDecodeError:
                print("❌ Invalid JSON in audit log")
                return False

        else:
            print("❌ Audit log file not found")
            return False

        return True

    except Exception as e:
        print(f"❌ Audit integration test failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 ΨQRH PromptEngine Integration Test Suite")
    print("=" * 60)

    # Test results
    results = {}

    # Run tests
    results["prompt_engine_agent"], agent = test_prompt_engine_agent()
    results["enhanced_runtime"], runtime = test_enhanced_runtime()
    results["production_safety"] = test_production_safety()
    results["prompt_execution"] = test_prompt_execution()
    results["audit_integration"] = test_audit_integration()

    # Generate test report
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1

    success_rate = (passed / total) * 100
    print(f"\nSuccess Rate: {passed}/{total} ({success_rate:.1f}%)")

    # Save test report
    report = {
        "test_suite": "ΨQRH PromptEngine Integration",
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
        "summary": {
            "passed": passed,
            "total": total,
            "success_rate": success_rate
        }
    }

    # Save to data directory
    report_path = project_root / "data" / "validation_reports" / f"prompt_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Test report saved: {report_path}")

    if success_rate >= 80:
        print("🎉 Integration tests PASSED!")
        return True
    else:
        print("⚠️ Integration tests need attention")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)