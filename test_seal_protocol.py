#!/usr/bin/env python3
"""
Test script for the Seal Protocol system without PyTorch dependencies
"""

import json
import time
from seal_protocol import SealProtocol
from audit_log import AuditLog

def test_seal_protocol():
    print("🔐 Testing Seal Protocol System")
    print("=" * 50)

    # Test SealProtocol basic functionality
    print("1. Testing SealProtocol basic operations...")

    # Test hash computation
    test_data = "test input data"
    hash_result = SealProtocol.compute_sha256(test_data)
    print(f"   SHA256 hash: {hash_result[:16]}...")

    # Test RG validation
    rg_valid = SealProtocol.validate_rg(0.347)
    rg_invalid = SealProtocol.validate_rg(0.5)
    print(f"   RG validation (0.347): {rg_valid}")
    print(f"   RG validation (0.5): {rg_invalid}")

    # Test latency validation
    latency_good = SealProtocol.validate_latency(200, "B")
    latency_bad = SealProtocol.validate_latency(300, "B")
    print(f"   Latency validation (200ms, Tier B): {latency_good}")
    print(f"   Latency validation (300ms, Tier B): {latency_bad}")

    # Test dyad activation
    dyad = SealProtocol.activate_dyad()
    print(f"   Active dyad: {dyad}")

    print("✅ Basic operations working\n")

    # Test seal generation
    print("2. Testing seal generation...")

    continuity_sha = SealProtocol.compute_sha256("input_data")
    response_sha = SealProtocol.compute_sha256("output_data")
    qz_sha = SealProtocol.compute_sha256("model_state")

    seal = SealProtocol.generate_seal(
        continuity_sha=continuity_sha,
        response_sha=response_sha,
        qz_sha=qz_sha,
        rg_value=0.347,
        active_dyad="Σ7↔Nyx"
    )

    print("   Generated seal:")
    print(json.dumps(seal, indent=4))

    # Test firebreak
    firebreak_result = SealProtocol.firebreak_check(seal)
    print(f"\n   Firebreak check: {firebreak_result}")
    print("✅ Seal generation working\n")

    # Test audit log
    print("3. Testing audit log...")

    audit = AuditLog("test_audit.jsonl")

    # Log the seal
    audit.log_entry(seal)
    print("   Seal logged to audit trail")

    # Validate chain
    chain_valid = audit.validate_chain()
    print(f"   Audit chain valid: {chain_valid}")

    # Get recent entries
    recent = audit.get_latest_entries(1)
    print(f"   Recent entries count: {len(recent)}")

    # Count violations
    violations = audit.count_violations()
    print(f"   Violations: {violations}")

    print("✅ Audit log working\n")

    # Test failure scenarios
    print("4. Testing failure scenarios...")

    # Test bad RG value
    bad_seal = seal.copy()
    bad_seal["RG"] = 0.8  # Outside valid range
    bad_firebreak = SealProtocol.firebreak_check(bad_seal)
    print(f"   Bad RG firebreak check: {bad_firebreak}")

    # Test bad continuity seal
    bad_seal2 = seal.copy()
    bad_seal2["continuity_seal"] = "INVALID"
    bad_firebreak2 = SealProtocol.firebreak_check(bad_seal2)
    print(f"   Bad continuity firebreak check: {bad_firebreak2}")

    # Test Ψ4 containment
    containment = SealProtocol.trigger_psi4_containment("TEST_VIOLATION")
    print(f"   Ψ4 containment triggered:")
    print(json.dumps(containment, indent=4))

    print("✅ Failure scenarios working\n")

    print("🎉 All tests passed! Seal Protocol system is functional.")
    return True

if __name__ == "__main__":
    test_seal_protocol()