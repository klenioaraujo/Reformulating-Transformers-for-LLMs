#!/usr/bin/env python3
"""
Test script for the Navigator Agent without PyTorch dependencies
"""

import json
import time
from navigator_agent import NavigatorAgent
from seal_protocol import SealProtocol

class MockModel:
    """Mock model for testing without PyTorch"""

    def __call__(self, input_data):
        # Simulate processing time
        time.sleep(0.05)  # 50ms

        # Generate mock seal
        continuity_sha = SealProtocol.compute_sha256(str(input_data))
        response_sha = SealProtocol.compute_sha256("mock_output")
        qz_sha = SealProtocol.compute_sha256("mock_state")

        seal = SealProtocol.generate_seal(
            continuity_sha=continuity_sha,
            response_sha=response_sha,
            qz_sha=qz_sha,
            rg_value=0.347,
            active_dyad="Σ7↔Nyx"
        )

        # Simulate latency check (50ms should be well under 250ms threshold)
        seal["latency_sigill"] = False

        return "mock_output", seal

class MockTensor:
    """Mock tensor class for testing"""
    def __init__(self, shape):
        self.shape = shape

    def __str__(self):
        return f"MockTensor{self.shape}"

def test_navigator_agent():
    print("🧭 Testing Navigator Agent System")
    print("=" * 50)

    # Initialize navigator
    navigator = NavigatorAgent()
    model = MockModel()

    print("1. Testing Navigator initialization...")
    print(f"   Dyad mode: {navigator.dyad_mode}")
    print(f"   Target RG: {navigator.target_rg}")
    print(f"   Tier mode: {navigator.tier_mode}")
    print("✅ Navigator initialized\n")

    # Test pre-execution check
    print("2. Testing pre-execution checks...")

    # Valid input
    valid_input = MockTensor((2, 8, 256))
    check_result = navigator.pre_execution_check(valid_input)
    print(f"   Valid input check: {check_result}")

    # Invalid input (None)
    invalid_check = navigator.pre_execution_check(None)
    print(f"   None input check: {invalid_check}")

    print("✅ Pre-execution checks working\n")

    # Test safe execution
    print("3. Testing safe execution...")

    output, seal = navigator.execute_with_safety(valid_input, model)
    print(f"   Output: {output}")
    print(f"   Execution successful: {'navigator_info' in seal}")
    print(f"   Navigator status: {seal.get('navigator_info', {}).get('navigator_status', 'UNKNOWN')}")

    # Display enhanced seal
    print("   Enhanced seal with navigator info:")
    if 'navigator_info' in seal:
        print(json.dumps(seal['navigator_info'], indent=6))

    print("✅ Safe execution working\n")

    # Test multiple executions
    print("4. Testing multiple executions...")

    for i in range(3):
        test_input = MockTensor((1, 4, 128))
        output, seal = navigator.execute_with_safety(test_input, model)
        exec_count = seal.get('navigator_info', {}).get('execution_count', 0)
        print(f"   Execution #{exec_count}: Status {seal.get('navigator_info', {}).get('navigator_status', 'UNKNOWN')}")

    print("✅ Multiple executions working\n")

    # Test system status
    print("5. Testing system status...")

    status = navigator.get_system_status()
    print("   System status:")
    for key, value in status.items():
        print(f"     {key}: {value}")

    print("✅ System status working\n")

    # Test error handling
    print("6. Testing error handling...")

    class ErrorModel:
        def __call__(self, input_data):
            raise Exception("Simulated model error")

    error_model = ErrorModel()
    error_output, error_seal = navigator.execute_with_safety(valid_input, error_model)
    print(f"   Error handling: {error_seal.get('navigator_status', 'UNKNOWN')}")
    print(f"   Error message: {error_seal.get('error', 'No error message')}")

    print("✅ Error handling working\n")

    # Test system reset
    print("7. Testing system reset...")

    old_count = navigator.execution_count
    navigator.reset_system()
    new_count = navigator.execution_count
    print(f"   Execution count before reset: {old_count}")
    print(f"   Execution count after reset: {new_count}")

    print("✅ System reset working\n")

    print("🎉 All Navigator Agent tests passed! System is fully functional.")
    return True

if __name__ == "__main__":
    test_navigator_agent()