#!/usr/bin/env python3
"""
Script to update ORCID in all project files

Usage:
    python scripts/update_orcid.py <your-orcid-id>

Example:
    python scripts/update_orcid.py 0000-0002-1234-5678

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file
"""

import sys
import re
from pathlib import Path


def validate_orcid(orcid: str) -> bool:
    """Validate ORCID format (0000-0000-0000-0000)"""
    pattern = r'^\d{4}-\d{4}-\d{4}-\d{3}[0-9X]$'
    return bool(re.match(pattern, orcid))


def update_file(filepath: Path, old_orcid: str, new_orcid: str) -> bool:
    """Update ORCID in a file"""
    try:
        content = filepath.read_text()
        if old_orcid in content:
            new_content = content.replace(old_orcid, new_orcid)
            filepath.write_text(new_content)
            print(f"✓ Updated: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"✗ Error updating {filepath}: {e}")
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/update_orcid.py <your-orcid-id>")
        print("Example: python scripts/update_orcid.py 0000-0002-1234-5678")
        print("\nGet your ORCID at: https://orcid.org/register")
        sys.exit(1)

    new_orcid = sys.argv[1]

    if not validate_orcid(new_orcid):
        print(f"✗ Invalid ORCID format: {new_orcid}")
        print("ORCID should be in format: 0000-0000-0000-0000")
        sys.exit(1)

    # Placeholder ORCID to replace
    old_orcid = "0000-0002-1234-5678"

    # Files to update
    files_to_update = [
        Path("metadata.yaml"),
    ]

    print(f"Updating ORCID from {old_orcid} to {new_orcid}")
    print("=" * 60)

    updated_count = 0
    for filepath in files_to_update:
        if filepath.exists():
            if update_file(filepath, old_orcid, new_orcid):
                updated_count += 1
        else:
            print(f"⚠ File not found: {filepath}")

    print("=" * 60)
    print(f"✓ Updated {updated_count} file(s)")
    print("\nNext steps:")
    print("1. Verify the changes in the updated files")
    print("2. Commit the changes to git")
    print("3. Update Zenodo record with correct ORCID")


if __name__ == "__main__":
    main()