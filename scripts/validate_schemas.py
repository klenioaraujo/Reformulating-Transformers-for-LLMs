#!/usr/bin/env python3
"""
Validate JSON schemas and example reports

This script validates:
1. The JSON schema itself is valid
2. Example reports conform to the schema
3. Generated reports from tests conform to the schema

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def validate_schema_syntax(schema_path: Path) -> bool:
    """Validate that the schema itself is valid JSON Schema"""
    try:
        import jsonschema
        from jsonschema import Draft7Validator

        print(f"Validating schema: {schema_path}")

        with open(schema_path) as f:
            schema = json.load(f)

        # Check schema is valid
        Draft7Validator.check_schema(schema)
        print("  ✓ Schema syntax is valid")
        return True

    except ImportError:
        print("  ⚠ jsonschema not installed. Install with: pip install jsonschema")
        return False
    except jsonschema.SchemaError as e:
        print(f"  ✗ Schema validation failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_report_against_schema(report_path: Path, schema_path: Path) -> bool:
    """Validate a report file against the schema"""
    try:
        import jsonschema

        with open(schema_path) as f:
            schema = json.load(f)

        with open(report_path) as f:
            report = json.load(f)

        print(f"Validating report: {report_path.name}")

        # Validate
        jsonschema.validate(instance=report, schema=schema)
        print("  ✓ Report conforms to schema")
        return True

    except ImportError:
        print("  ⚠ jsonschema not installed")
        return False
    except jsonschema.ValidationError as e:
        print(f"  ✗ Validation failed: {e.message}")
        print(f"    Path: {' -> '.join(str(p) for p in e.path)}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def create_example_report() -> dict:
    """Create a valid example report for testing"""
    return {
        "$schema": "https://raw.githubusercontent.com/klenioaraujo/Reformulating-Transformers-for-LLMs/master/schemas/report_schema.json",
        "metadata": {
            "project": "ΨQRH Transformer",
            "version": "1.0.0",
            "doi": "https://zenodo.org/records/17171112",
            "license": "GPL-3.0-or-later"
        },
        "report_type": "energy_conservation",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0",
        "provenance": {
            "software_version": "1.0.0",
            "git_commit": "abc1234",
            "hardware": {
                "cpu": "Intel Xeon",
                "gpu": "NVIDIA RTX 3090",
                "ram": "32GB",
                "device": "cuda"
            },
            "execution_environment": {
                "python_version": "3.10.0",
                "pytorch_version": "2.0.0",
                "numpy_version": "1.26.4",
                "platform": "Linux",
                "os": "Ubuntu 22.04"
            },
            "input_data_hash": "sha256:abc123def456",
            "random_seed": 42,
            "execution_time": 123.45
        },
        "configuration": {
            "model_params": {
                "d_model": 512,
                "n_heads": 8,
                "n_layers": 6,
                "vocab_size": 50000,
                "max_seq_length": 512
            }
        },
        "energy_conservation": {
            "overall_conservation": 99.8,
            "layer_conservation": [
                {
                    "layer": 0,
                    "conservation": 99.9,
                    "input_energy": 1000.0,
                    "output_energy": 999.0
                },
                {
                    "layer": 1,
                    "conservation": 99.7,
                    "input_energy": 999.0,
                    "output_energy": 996.0
                }
            ],
            "deviation": 0.1,
            "status": "PASS"
        },
        "validation_summary": {
            "status": "PASS",
            "passed_tests": 10,
            "failed_tests": 0,
            "total_tests": 10,
            "success_rate": 100.0,
            "notes": ["All tests passed successfully"],
            "warnings": [],
            "errors": []
        }
    }


def main():
    print("=" * 60)
    print("ΨQRH JSON Schema Validation")
    print("=" * 60)
    print()

    # Paths
    schema_path = Path("schemas/report_schema.json")
    reports_dir = Path("data/validation_reports")

    # Check if schema exists
    if not schema_path.exists():
        print(f"✗ Schema not found: {schema_path}")
        sys.exit(1)

    # Validate schema syntax
    print("Step 1: Validating schema syntax")
    print("-" * 60)
    if not validate_schema_syntax(schema_path):
        sys.exit(1)
    print()

    # Create and validate example report
    print("Step 2: Validating example report")
    print("-" * 60)

    example_report = create_example_report()
    example_path = Path("tmp/example_report.json")
    example_path.parent.mkdir(parents=True, exist_ok=True)

    with open(example_path, 'w') as f:
        json.dump(example_report, f, indent=2)

    if not validate_report_against_schema(example_path, schema_path):
        sys.exit(1)
    print(f"  Example saved to: {example_path}")
    print()

    # Validate existing reports
    if reports_dir.exists():
        print("Step 3: Validating existing reports")
        print("-" * 60)

        report_files = list(reports_dir.glob("*.json"))
        if report_files:
            valid_count = 0
            for report_path in report_files:
                if validate_report_against_schema(report_path, schema_path):
                    valid_count += 1

            print()
            print(f"Summary: {valid_count}/{len(report_files)} reports valid")
        else:
            print("  No reports found in data/validation_reports/")
    else:
        print("Step 3: Skipping existing reports (directory not found)")

    print()
    print("=" * 60)
    print("✓ Schema validation completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Use the schema in your report generation code")
    print("2. Validate reports before saving:")
    print("   from scripts.validate_schemas import validate_report_against_schema")
    print("3. Add schema reference to all reports:")
    print('   report["$schema"] = "https://raw.githubusercontent.com/..."')


if __name__ == "__main__":
    main()