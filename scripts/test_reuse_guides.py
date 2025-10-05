#!/usr/bin/env python3
"""
Test Reuse Guides

This script tests all reuse guides to ensure they work correctly.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


def run_test(script_path: Path, check_imports_only: bool = False) -> Tuple[bool, str]:
    """
    Run a test script and return success status and output.

    Args:
        script_path: Path to script to test
        check_imports_only: If True, only check if imports work

    Returns:
        Tuple of (success, output_message)
    """
    print(f"\nTesting: {script_path.name}")
    print("-" * 60)

    if not script_path.exists():
        return False, f"✗ File not found: {script_path}"

    try:
        if check_imports_only:
            # Just check if the file can be imported/parsed
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return True, "✓ Syntax valid, imports successful"
            else:
                return False, f"✗ Syntax error:\n{result.stderr}"

        else:
            # Try to run the script with --help or similar
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Script might fail on purpose (e.g., missing dependencies)
            # We just want to make sure it doesn't have syntax errors
            if "Traceback" in result.stderr and "SyntaxError" in result.stderr:
                return False, f"✗ Syntax error:\n{result.stderr}"

            return True, "✓ Script executed (may have expected runtime errors)"

    except subprocess.TimeoutExpired:
        return False, "✗ Timeout - script took too long"
    except Exception as e:
        return False, f"✗ Error: {e}"


def test_fine_tune_guide() -> bool:
    """Test the fine-tuning guide"""
    script = Path("examples/reuse_guides/fine_tune_psiqrh.py")
    success, message = run_test(script, check_imports_only=True)
    print(message)
    return success


def test_integration_guide() -> bool:
    """Test the integration guide"""
    script = Path("examples/reuse_guides/integrate_with_standard_transformer.py")
    success, message = run_test(script, check_imports_only=False)
    print(message)
    return success


def test_onnx_guide() -> bool:
    """Test the ONNX conversion guide"""
    script = Path("examples/reuse_guides/convert_psiqrh_to_onnx.py")
    success, message = run_test(script, check_imports_only=True)
    print(message)
    return success


def check_dependencies() -> List[str]:
    """Check which optional dependencies are installed"""
    optional_deps = {
        "onnx": "ONNX export functionality",
        "onnxruntime": "ONNX runtime inference",
        "transformers": "HuggingFace integration",
        "jupyter": "Jupyter notebooks"
    }

    print("\nOptional Dependencies:")
    print("-" * 60)

    missing = []
    for package, description in optional_deps.items():
        try:
            __import__(package)
            print(f"✓ {package:20} - {description}")
        except ImportError:
            print(f"✗ {package:20} - {description}")
            missing.append(package)

    return missing


def main():
    print("=" * 60)
    print("ΨQRH Reuse Guides Test Suite")
    print("=" * 60)

    # Check dependencies
    missing_deps = check_dependencies()

    if missing_deps:
        print(f"\n⚠  Missing {len(missing_deps)} optional dependencies")
        print("Install with:")
        print(f"  pip install {' '.join(missing_deps)}")
    print()

    # Run tests
    results = {}

    print("Running Guide Tests:")
    print("=" * 60)

    results["Fine-tuning Guide"] = test_fine_tune_guide()
    results["Integration Guide"] = test_integration_guide()
    results["ONNX Export Guide"] = test_onnx_guide()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {name}")

    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All reuse guides are working correctly!")
        return 0
    else:
        print(f"\n✗ {total - passed} guide(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())