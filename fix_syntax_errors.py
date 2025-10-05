#!/usr/bin/env python3
"""
Fix Syntax Errors Script
=========================

This script repairs Python files that had their formatting damaged during
the emoji removal process. It fixes concatenated import statements and
properly formats Python code.
"""

import os
import sys
import re
from pathlib import Path

def fix_python_syntax(file_path: str) -> bool:
    """
    Fix syntax errors in a Python file by properly formatting concatenated statements.

    Args:
        file_path: Path to the Python file to fix

    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Fix concatenated import statements
        content = re.sub(r'import ([^@\n]+?)(\s*class|\s*def|\s*@|\s*from|\s*import)',
                        r'import \1\n\2', content)

        # Fix concatenated class definitions
        content = re.sub(r'(\w+):(\s*""".*?""")(\s*class|\s*def|\s*@)',
                        r'\1:\2\n\3', content, flags=re.DOTALL)

        # Fix concatenated function definitions
        content = re.sub(r'(\w+):(\s*""".*?""")(\s*def)',
                        r'\1:\2\n\3', content, flags=re.DOTALL)

        # Fix concatenated statements after class/def
        content = re.sub(r'(class [^:]+:)(\s*""".*?""")(\s*def|\s*@|\s*class)',
                        r'\1\2\n\3', content, flags=re.DOTALL)

        # Fix concatenated decorator statements
        content = re.sub(r'(@\w+)(\s*def|\s*class)', r'\1\n\2', content)

        # Ensure proper line breaks after pass statements
        content = re.sub(r'(\s+pass)(\s*class|\s*def|\s*@)', r'\1\n\2', content)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix syntax errors in core files."""
    print("Fixing Syntax Errors in Core Files")
    print("=" * 40)

    # Target directories to fix
    target_dirs = [
        "src/core",
        "src/fractal",
        "src/cognitive",
        "src/conceptual",
        "src/conscience"
    ]

    files_fixed = 0

    for target_dir in target_dirs:
        if os.path.exists(target_dir):
            print(f"\nChecking directory: {target_dir}")

            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        if fix_python_syntax(file_path):
                            files_fixed += 1

    print(f"\nFixed {files_fixed} files")

    # Test import of core modules
    print("\nTesting core module imports...")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))

        # Test basic imports
        from core.quaternion_operations import QuaternionOperations
        print("✓ QuaternionOperations imported successfully")

        from core.qrh_layer import QRHLayer, QRHConfig
        print("✓ QRHLayer imported successfully")

        print("\n✅ Core modules are functioning correctly")

    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)