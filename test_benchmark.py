#!/usr/bin/env python3
"""
Quick test script for ΨQRH benchmark functionality
"""

import torch
from generate_benchmark_data import create_model, count_parameters, create_tokenizer

def test_model_creation():
    """Test that models can be created and run forward pass"""
    print("Testing model creation and forward pass...")

    tokenizer = create_tokenizer()
    seq_len = 32  # Small for testing

    # Test baseline model
    baseline = create_model('baseline', tokenizer.vocab_size, seq_len)
    baseline_params = count_parameters(baseline)
    print(f"Baseline model: {baseline_params:,} parameters")

    # Test ΨQRH model
    psiqrh = create_model('psiqrh', tokenizer.vocab_size, seq_len)
    psiqrh_params = count_parameters(psiqrh)
    print(f"ΨQRH model: {psiqrh_params:,} parameters")

    # Test forward pass
    batch_size = 2
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))

    baseline.eval()
    psiqrh.eval()

    with torch.no_grad():
        baseline_out = baseline(input_ids)
        psiqrh_out = psiqrh(input_ids)

    print(f"Baseline output shape: {baseline_out.shape}")
    print(f"ΨQRH output shape: {psiqrh_out.shape}")

    assert baseline_out.shape == psiqrh_out.shape, "Output shapes should match"
    assert baseline_out.shape == (batch_size, seq_len, tokenizer.vocab_size), "Output shape incorrect"

    print("✅ Model creation and forward pass test passed!")

if __name__ == '__main__':
    test_model_creation()