#!/usr/bin/env python3
"""
Architectural Validator for ΨQRH System

Validates files against architectural rules and provides automatic correction
capabilities for the enhanced agentic runtime.

Classification: ΨQRH-Architectural-Validator-v1.0
"""

import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger("ArchitecturalValidator")

class ArchitecturalValidator:
    """
    Validates and enforces architectural rules for the ΨQRH system

    This validator ensures proper file organization, particularly the
    separation between production code (src/) and test code (tests/).
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations_log = []

        # Define architectural rules
        self.rules = {
            "test_files_separation": {
                "description": "Test files must be in tests/ directory only",
                "patterns": ["test_*.py", "*_test.py", "*test*.py"],
                "required_directory": "tests/",
                "forbidden_directories": ["src/", "experiments/", "./"],
                "severity": "critical",
                "auto_fix": True
            },
            "documentation_structure": {
                "description": "Documentation files should follow proper structure",
                "patterns": ["*.md", "*.rst"],
                "allowed_directories": ["docs/", "construction_technical_manual/", "./"],
                "forbidden_directories": ["src/core/", "src/fractal/"],
                "severity": "warning",
                "auto_fix": False
            }
        }

    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single file against all architectural rules"""
        violations = []
        file_name = file_path.name.lower()
        rel_path = file_path.relative_to(self.project_root)
        current_dir = str(rel_path.parent) + "/"

        for rule_name, rule_config in self.rules.items():
            # Check if file matches rule patterns
            matches_pattern = any(
                self._matches_pattern(file_name, pattern)
                for pattern in rule_config.get("patterns", [])
            )

            if matches_pattern:
                # Check forbidden directories
                forbidden_dirs = rule_config.get("forbidden_directories", [])
                is_in_forbidden = any(
                    current_dir.startswith(forbidden)
                    for forbidden in forbidden_dirs
                )

                if is_in_forbidden:
                    violations.append({
                        "rule": rule_name,
                        "severity": rule_config["severity"],
                        "description": rule_config["description"],
                        "message": f"File violates {rule_name}: {file_name} in {current_dir}",
                        "current_path": str(rel_path),
                        "required_directory": rule_config.get("required_directory"),
                        "auto_fix_available": rule_config.get("auto_fix", False),
                        "suggested_action": self._get_suggested_action(rule_name, file_path)
                    })

        return {
            "file_path": str(rel_path),
            "valid": len(violations) == 0,
            "violations": violations,
            "validation_timestamp": datetime.utcnow().isoformat()
        }

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches a pattern (with wildcards)"""
        if "*" in pattern:
            # Simple wildcard matching
            if pattern.startswith("*") and pattern.endswith("*"):
                return pattern[1:-1] in filename
            elif pattern.startswith("*"):
                return filename.endswith(pattern[1:])
            elif pattern.endswith("*"):
                return filename.startswith(pattern[:-1])
        else:
            return filename == pattern

    def _get_suggested_action(self, rule_name: str, file_path: Path) -> Dict[str, Any]:
        """Get suggested corrective action for a rule violation"""
        rule_config = self.rules.get(rule_name, {})

        if rule_name == "test_files_separation":
            required_dir = rule_config.get("required_directory", "tests/")
            new_path = self.project_root / required_dir / file_path.name

            return {
                "action": "move_file",
                "from": str(file_path.relative_to(self.project_root)),
                "to": str(new_path.relative_to(self.project_root)),
                "requires_import_updates": file_path.suffix == ".py"
            }

        return {"action": "manual_review", "message": "Manual review required"}

    def auto_fix_violation(self, violation: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically fix a violation if possible"""
        if not violation.get("auto_fix_available"):
            return {"success": False, "reason": "Auto-fix not available for this violation"}

        try:
            suggested_action = violation["suggested_action"]

            if suggested_action["action"] == "move_file":
                return self._move_file_with_import_fix(
                    violation["current_path"],
                    suggested_action["to"],
                    suggested_action.get("requires_import_updates", False)
                )

        except Exception as e:
            logger.error(f"Auto-fix failed for {violation['current_path']}: {e}")
            return {"success": False, "reason": f"Auto-fix error: {str(e)}"}

        return {"success": False, "reason": "Unknown auto-fix action"}

    def _move_file_with_import_fix(self, from_path: str, to_path: str, fix_imports: bool) -> Dict[str, Any]:
        """Move file and optionally fix imports"""
        source_file = self.project_root / from_path
        target_file = self.project_root / to_path

        if not source_file.exists():
            return {"success": False, "reason": "Source file not found"}

        # Create target directory if needed
        target_file.parent.mkdir(parents=True, exist_ok=True)

        # Read file content for import fixing
        content = source_file.read_text(encoding='utf-8')

        if fix_imports and content:
            # Add project root to Python path for moved test files
            import_header = '''import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

'''
            # Check if import header already exists
            if "sys.path.insert(0" not in content:
                # Find the first import statement
                lines = content.split('\n')
                insert_index = 0

                # Skip docstring and comments at the top
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        insert_index = i
                        break

                # Insert the import header
                lines.insert(insert_index, import_header.rstrip())
                content = '\n'.join(lines)

        # Write content to target file
        target_file.write_text(content, encoding='utf-8')

        # Remove source file
        source_file.unlink()

        logger.info(f"Successfully moved {from_path} to {to_path}")

        return {
            "success": True,
            "from": from_path,
            "to": to_path,
            "imports_fixed": fix_imports,
            "timestamp": datetime.utcnow().isoformat()
        }

    def scan_project_violations(self) -> Dict[str, Any]:
        """Scan entire project for architectural violations"""
        all_violations = []
        scanned_files = 0

        # Scan source files
        for pattern in ["**/*.py", "**/*.md", "**/*.rst"]:
            for file_path in self.project_root.glob(pattern):
                # Skip files in ignore patterns
                rel_path_str = str(file_path.relative_to(self.project_root))
                if any(ignore in rel_path_str for ignore in ['.git/', '__pycache__/', '.venv/']):
                    continue

                validation_result = self.validate_file(file_path)
                if not validation_result["valid"]:
                    all_violations.extend(validation_result["violations"])

                scanned_files += 1

        # Categorize violations by severity
        critical_violations = [v for v in all_violations if v["severity"] == "critical"]
        warning_violations = [v for v in all_violations if v["severity"] == "warning"]

        return {
            "scan_timestamp": datetime.utcnow().isoformat(),
            "files_scanned": scanned_files,
            "total_violations": len(all_violations),
            "critical_violations": len(critical_violations),
            "warning_violations": len(warning_violations),
            "violations": {
                "critical": critical_violations,
                "warning": warning_violations
            },
            "auto_fixable": len([v for v in all_violations if v.get("auto_fix_available")])
        }

    def fix_all_violations(self, severity_filter: Optional[str] = None) -> Dict[str, Any]:
        """Automatically fix all fixable violations"""
        scan_result = self.scan_project_violations()
        fixed_count = 0
        failed_fixes = []

        violations_to_fix = []
        if severity_filter:
            violations_to_fix = scan_result["violations"].get(severity_filter, [])
        else:
            violations_to_fix = (scan_result["violations"]["critical"] +
                               scan_result["violations"]["warning"])

        for violation in violations_to_fix:
            if violation.get("auto_fix_available"):
                fix_result = self.auto_fix_violation(violation)
                if fix_result["success"]:
                    fixed_count += 1
                    logger.info(f"Auto-fixed: {violation['current_path']}")
                else:
                    failed_fixes.append({
                        "violation": violation,
                        "reason": fix_result["reason"]
                    })

        return {
            "total_violations": len(violations_to_fix),
            "fixed_count": fixed_count,
            "failed_count": len(failed_fixes),
            "failed_fixes": failed_fixes,
            "timestamp": datetime.utcnow().isoformat()
        }

def create_architectural_validator(project_root: Path) -> ArchitecturalValidator:
    """Create and configure architectural validator"""
    return ArchitecturalValidator(project_root)