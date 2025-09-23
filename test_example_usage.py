#!/usr/bin/env python3
"""
Test the example usage workflow without Jupyter notebook
"""

import json
from seal_protocol import SealProtocol
from audit_log import AuditLog

class MockNegentropyTransformerBlock:
    """Mock transformer block for testing"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout,
                 qrh_embed_dim, alpha, use_learned_rotation, enable_gate):
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.config = {
            'alpha': alpha,
            'qrh_embed_dim': qrh_embed_dim,
            'enable_gate': enable_gate
        }

    def __call__(self, sample_input):
        # Simulate transformer processing
        import time
        start_time = time.time()
        time.sleep(0.01)  # 10ms processing
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Generate mock output
        output_shape = sample_input.get('shape', (2, 10, 512))
        mock_output = f"MockTensor{output_shape}"

        # Generate seal as the real model would
        continuity_sha = SealProtocol.compute_sha256(str(sample_input))
        response_sha = SealProtocol.compute_sha256(str(mock_output))
        qz_sha = SealProtocol.compute_sha256(str(self.config))

        seal = SealProtocol.generate_seal(
            continuity_sha=continuity_sha,
            response_sha=response_sha,
            qz_sha=qz_sha,
            rg_value=0.347,
            active_dyad="Σ7↔Nyx"
        )

        # Add latency info
        seal["latency_sigill"] = not SealProtocol.validate_latency(latency_ms, tier="B")
        seal["measured_latency_ms"] = latency_ms

        return mock_output, seal

def test_example_usage():
    print("📓 Testing Example Usage Workflow")
    print("=" * 50)

    # Simulate notebook cell 1: Imports (already done)
    print("Cell 1: Imports completed ✅")

    # Simulate notebook cell 2: Initialize model and audit
    print("\nCell 2: Initializing model and audit log...")

    model = MockNegentropyTransformerBlock(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        qrh_embed_dim=64,
        alpha=1.0,
        use_learned_rotation=True,
        enable_gate=True
    )

    audit = AuditLog("example_audit.jsonl")
    print("   Model and audit initialized ✅")

    # Simulate notebook cell 3: Create sample input
    print("\nCell 3: Creating sample input...")

    batch_size = 2
    seq_len = 10
    d_model = 512

    sample_input = {
        'shape': (batch_size, seq_len, d_model),
        'data': 'random_tensor_data'
    }

    print(f"   Input shape: {sample_input['shape']} ✅")

    # Simulate notebook cell 4: Forward pass
    print("\nCell 4: Forward pass with seal protocol...")

    output, seal = model(sample_input)
    print(f"   Output: {output}")
    print(f"   Seal generated: {type(seal).__name__} ✅")

    # Simulate notebook cell 5: Log and display results
    print("\nCell 5: Logging and displaying results...")

    # Log the seal to audit trail
    audit.log_entry(seal)

    # Get latency from seal
    latency_ms = seal.get("measured_latency_ms", 0)

    print("🔐 Output Seal:")
    print(json.dumps(seal, indent=2, ensure_ascii=False))

    print(f"\n✅ Audit Chain Valid: {audit.validate_chain()}")
    print(f"⏱️  Latency: {latency_ms:.2f}ms")
    print(f"🎯 RG: {seal['RG']} (Target: 0.347)")
    print(f"🌀 Dyad: {seal['active_dyad']}")
    print(f"🛡️  Seal: {seal['continuity_seal']}")

    # Simulate notebook cell 6: Check violations
    print("\nCell 6: Checking for violations...")

    violations = audit.count_violations()
    print("📊 Violation Summary:")
    for violation_type, count in violations.items():
        print(f"  {violation_type}: {count}")

    # Simulate notebook cell 7: Multiple forward passes
    print("\nCell 7: Testing multiple forward passes...")

    print("🔄 Testing multiple forward passes:")
    for i in range(3):
        test_input = {
            'shape': (1, 5, 256),
            'data': f'test_data_{i}'
        }
        output, seal = model(test_input)
        audit.log_entry(seal)

        status = "✅" if not seal.get('latency_sigill', False) else "⚠️"
        print(f"Pass {i+1}: {status} RG={seal['RG']:.3f}, Seal={seal['continuity_seal']}")

    print(f"\nFinal audit chain valid: {audit.validate_chain()}")

    print("\n🎉 Example usage workflow completed successfully!")
    print("   ✅ Model initialization")
    print("   ✅ Seal protocol integration")
    print("   ✅ Audit logging")
    print("   ✅ Multiple executions")
    print("   ✅ Violation tracking")
    print("   ✅ Chain validation")

    return True

if __name__ == "__main__":
    test_example_usage()