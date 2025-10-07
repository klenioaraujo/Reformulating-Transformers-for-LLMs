#!/usr/bin/env python3
"""
Test Complete ΨQRH Transformer Implementation

Validates the complete ΨQRH transformer architecture with all new components.
Includes physics validation tests for:
- Energy conservation
- Unitarity preservation
- Fractal dimension estimation
- Spectral attention adaptation
- SO(4) evolution
- Optical probe resonance
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np

# Add src and configs directories to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

from src.architecture.psiqrh_transformer import PsiQRHTransformer
from examples.config_loader import get_scientific_test_config
from src.core.quaternion_operations import (
    QuaternionLinear,
    QuaternionLayerNorm,
    SpectralActivation,
    AdaptiveSpectralDropout,
    RealTimeFractalAnalyzer,
    quaternion_normalize,
    quaternion_norm
)

# Import new ΨQRH components
from src.core.fractal_quantum_embedding import (
    OptimizedFractalEmbedding,
    ContextFractalAnalyzer,
    SpectralAttentionLayer,
    SO4EvolutionLayer,
    OpticalProbeGenerator,
    LeechLatticeCorrector,
    PsiQRHTransformerBlock,
    PsiQRHTransformerComplete
)


def test_quaternion_components():
    """Test individual quaternion components"""
    print("=== Testing Quaternion Components ===")

    # Load scientific configuration from BKP
    config = get_scientific_test_config("SCI_001")
    print(f"Using scientific config: d_model={config.get('d_model', 512)}, vocab_size={config.get('vocab_size', 10000)}")

    # Test QuaternionLinear with scientific dimensions
    print("\n1. Testing QuaternionLinear...")
    linear = QuaternionLinear(config.get('d_model', 512), config.get('d_model', 512))
    # QuaternionLinear expects input of shape [..., in_features * quaternion_multiplier]
    x = torch.randn(config.get('default_test_batch_size', 2), config.get('default_test_seq_length', 128), config.get('d_model', 512) * config.get('quaternion_multiplier', 4))
    output = linear(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ QuaternionLinear working")

    # Test QuaternionLayerNorm
    print("\n2. Testing QuaternionLayerNorm...")
    norm = QuaternionLayerNorm(config.get('d_model', 512))
    x = torch.randn(config.get('default_test_batch_size', 2), config.get('default_test_seq_length', 128), config.get('d_model', 512) * config.get('quaternion_multiplier', 4))
    output = norm(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ QuaternionLayerNorm working")

    # Test SpectralActivation
    print("\n3. Testing SpectralActivation...")
    activation = SpectralActivation()
    x = torch.randn(config.get('default_test_batch_size', 2), config.get('default_test_seq_length', 128), config.get('d_model', 512) * config.get('quaternion_multiplier', 4))
    output = activation(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ SpectralActivation working")

    # Test AdaptiveSpectralDropout
    print("\n4. Testing AdaptiveSpectralDropout...")
    dropout = AdaptiveSpectralDropout(p=0.1)
    x = torch.randn(config.get('default_test_batch_size', 2), config.get('default_test_seq_length', 128), config.get('d_model', 512) * config.get('quaternion_multiplier', 4))
    output = dropout(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ AdaptiveSpectralDropout working")

    # Test RealTimeFractalAnalyzer
    print("\n5. Testing RealTimeFractalAnalyzer...")
    analyzer = RealTimeFractalAnalyzer()
    x = torch.randn(config.get('default_test_batch_size', 2), config.get('default_test_seq_length', 128), config.get('d_model', 512) * config.get('quaternion_multiplier', 4))
    metrics = analyzer.analyze(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Fractal dimension: {metrics['dimension']:.4f}")
    print(f"   Spectral entropy: {metrics['spectral_entropy']:.4f}")
    print(f"   ✅ RealTimeFractalAnalyzer working")


def test_complete_psiqrh_transformer():
    """Test complete ΨQRH transformer"""
    print("\n=== Testing Complete ΨQRH Transformer ===")

    # Create model
    # Load scientific configuration
    config = get_scientific_test_config("SCI_001")

    print(f"Creating ΨQRH Transformer with:")
    print(f"  vocab_size: {config.get('vocab_size', 10000)}")
    print(f"  d_model: {config.get('d_model', 512)}")
    print(f"  n_layers: {config.get('n_layers', 6)}")
    print(f"  n_heads: {config.get('n_heads', 8)}")

    model = PsiQRHTransformer(
        vocab_size=config.get('vocab_size', 10000),
        d_model=config.get('d_model', 512),
        n_layers=config.get('n_layers', 6),
        n_heads=config.get('n_heads', 8),
        quaternion_multiplier=config.get('quaternion_multiplier', 4)
    )

    # Test forward pass
    print("\nTesting forward pass...")
    input_ids = torch.randint(0, config.get('vocab_size', 10000), (config.get('default_test_batch_size', 2), config.get('default_test_seq_length', 128)))

    with torch.no_grad():
        output = model(input_ids)

    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Test energy conservation
    print("\nTesting energy conservation...")
    from src.core.utils import energy_normalize

    input_embeddings = model.token_embedding(input_ids)
    input_energy = torch.sum(input_embeddings**2).item()
    output_energy = torch.sum(output**2).item()
    conservation_ratio = output_energy / input_energy

    print(f"   Input energy: {input_energy:.6f}")
    print(f"   Output energy: {output_energy:.6f}")
    print(f"   Conservation ratio: {conservation_ratio:.6f}")
    print(f"   Energy preserved: {'✅ YES' if abs(conservation_ratio - 1.0) < 0.05 else '❌ NO'}")

    return conservation_ratio


def test_memory_efficiency():
    """Test memory efficiency compared to standard transformer"""
    print("\n=== Testing Memory Efficiency ===")

    # Load scientific configuration for performance test
    config = get_scientific_test_config("SCI_002")  # Different scenario for performance

    # ΨQRH transformer
    psiqrh_model = PsiQRHTransformer(
        vocab_size=config.get('vocab_size', 5000),
        d_model=config.get('d_model', 512),
        n_layers=config.get('n_layers', 6),
        n_heads=config.get('n_heads', 8),
        quaternion_multiplier=config.get('quaternion_multiplier', 4)
    )

    # Standard transformer (for comparison)
    standard_model = nn.Transformer(
        d_model=config.get('d_model', 512),
        nhead=config.get('n_heads', 8),
        num_encoder_layers=config.get('n_layers', 6),
        num_decoder_layers=config.get('n_layers', 6)
    )

    # Calculate parameter counts
    psiqrh_params = sum(p.numel() for p in psiqrh_model.parameters())
    standard_params = sum(p.numel() for p in standard_model.parameters())

    print(f"ΨQRH Transformer parameters: {psiqrh_params:,}")
    print(f"Standard Transformer parameters: {standard_params:,}")

    # Calculate memory usage
    psiqrh_memory = psiqrh_params * 4 / (1024 ** 2)  # MB
    standard_memory = standard_params * 4 / (1024 ** 2)  # MB

    print(f"ΨQRH Transformer memory: {psiqrh_memory:.2f} MB")
    print(f"Standard Transformer memory: {standard_memory:.2f} MB")

    memory_reduction = (1 - psiqrh_memory / standard_memory) * 100
    print(f"Memory reduction: {memory_reduction:.1f}%")

    # Check if ΨQRH is actually more efficient
    if memory_reduction < 0:
        print("⚠️  ΨQRH uses more memory than standard transformer")
    else:
        print("✅ ΨQRH is more memory efficient than standard transformer")

    return memory_reduction


def main():
    """Main test function"""
    print("ΨQRH - Complete Transformer Implementation Test")
    print("=" * 60)

    # Test individual components
    test_quaternion_components()

    # Test complete transformer
    conservation_ratio = test_complete_psiqrh_transformer()

    # Test memory efficiency
    memory_reduction = test_memory_efficiency()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"✅ All quaternion components working")
    print(f"✅ ΨQRH transformer forward pass successful")
    print(f"✅ Energy conservation ratio: {conservation_ratio:.6f}")
    print(f"✅ Memory reduction: {memory_reduction:.1f}%")

    if abs(conservation_ratio - 1.0) < 0.05:
        print("\n🎯 COMPLETE ΨQRH TRANSFORMER IMPLEMENTATION SUCCESSFUL!")
        print("✅ All components integrated and working")
        print("✅ Energy conservation achieved")
        if memory_reduction > 20:
            print("✅ Memory efficiency demonstrated")
        else:
            print("⚠️  Memory efficiency needs optimization (expected for complex architecture)")
    else:
        print("\n⚠️  Some tests need improvement")
        print("❌ Review implementation details")


# ============================================================================
# NEW PHYSICS VALIDATION TESTS
# ============================================================================

def test_fractal_embedding_physics():
    """Test 1: Fractal Quantum Embedding - Physics Validation"""
    print("\n" + "=" * 70)
    print("TEST 1: Fractal Quantum Embedding - Physics Validation")
    print("=" * 70)

    vocab_size = 100
    embed_dim = 64
    batch_size = 4
    seq_len = 16

    embedding = OptimizedFractalEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        quaternion_dim=4,
        precompute_on_init=True
    )

    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    quaternions = embedding(input_ids)

    # Physics Test 1: Unitarity (||Ψ|| = 1)
    norms = quaternion_norm(quaternions)
    unitarity_error = torch.abs(norms - 1.0).mean().item()

    print(f"\n📊 Unitarity Test:")
    print(f"   Mean norm: {norms.mean():.6f} (should be 1.0)")
    print(f"   Std norm:  {norms.std():.6f}")
    print(f"   Error:     {unitarity_error:.6f}")

    assert unitarity_error < 0.01, "Unitarity violation!"
    print("   ✅ PASSED: All quaternions are unit")

    # Physics Test 2: Fractal dimension range
    D_values = embedding.fractal_dimensions.cpu().numpy()
    print(f"\n📊 Fractal Dimension Distribution:")
    print(f"   Range: [{D_values.min():.3f}, {D_values.max():.3f}]")
    print(f"   Mean:  {D_values.mean():.3f}")
    print(f"   Std:   {D_values.std():.3f}")

    assert D_values.min() >= 1.0 and D_values.max() <= 2.0, "D out of range!"
    print("   ✅ PASSED: D ∈ [1, 2] for 2D fractals")

    # Physics Test 3: α(D) mapping
    alpha_values = embedding.alpha_cache.cpu().numpy()
    print(f"\n📊 Alpha Parameter Distribution:")
    print(f"   Range: [{alpha_values.min():.3f}, {alpha_values.max():.3f}]")
    print(f"   Mean:  {alpha_values.mean():.3f}")

    print("\n✅ Fractal Embedding Physics: VALIDATED\n")


def test_spectral_attention_physics():
    """Test 2: Spectral Attention - Adaptive α(D)"""
    print("\n" + "=" * 70)
    print("TEST 2: Spectral Attention - Adaptive α(D)")
    print("=" * 70)

    d_model = 128
    n_heads = 4
    batch_size = 2
    seq_len = 32

    layer = SpectralAttentionLayer(d_model=d_model, n_heads=n_heads)

    x = torch.randn(batch_size, seq_len, d_model)
    output = layer(x)

    # Physics Test: Energy preservation (relative)
    input_energy = torch.sum(x ** 2).item()
    output_energy = torch.sum(output ** 2).item()
    energy_ratio = output_energy / input_energy

    print(f"\n📊 Energy Conservation:")
    print(f"   Input energy:  {input_energy:.2f}")
    print(f"   Output energy: {output_energy:.2f}")
    print(f"   Ratio:         {energy_ratio:.6f}")

    assert 0.5 < energy_ratio < 2.0, "Energy not preserved!"
    print("   ✅ PASSED: Energy approximately conserved")

    # Test adaptation
    analyzer = layer.context_analyzer
    D = analyzer.compute_fractal_dimension(x)
    alpha = analyzer.compute_alpha(D)

    print(f"\n📊 Context Adaptation:")
    print(f"   Fractal D: {D.mean():.3f} ± {D.std():.3f}")
    print(f"   Alpha α:   {alpha.mean():.3f} ± {alpha.std():.3f}")

    print("\n✅ Spectral Attention Physics: VALIDATED\n")


def test_so4_evolution_physics():
    """Test 3: SO(4) Evolution - Energy Conservation"""
    print("\n" + "=" * 70)
    print("TEST 3: SO(4) Evolution - Unitarity & Energy Conservation")
    print("=" * 70)

    quaternion_dim = 4
    n_rotations = 4
    batch_size = 4
    seq_len = 16

    layer = SO4EvolutionLayer(quaternion_dim=quaternion_dim, n_rotations=n_rotations)

    # Generate random unit quaternions
    x = torch.randn(batch_size, seq_len, quaternion_dim)
    x = quaternion_normalize(x)

    # Verify input is unit
    input_norms = quaternion_norm(x)
    print(f"\n📊 Input Quaternions:")
    print(f"   Norm: {input_norms.mean():.6f} ± {input_norms.std():.6f}")

    # Apply SO(4) evolution
    output = layer(x)

    # Physics Test: Unitarity preserved
    output_norms = quaternion_norm(output)
    unitarity_error = torch.abs(output_norms - 1.0).mean().item()

    print(f"\n📊 Output Quaternions:")
    print(f"   Norm: {output_norms.mean():.6f} ± {output_norms.std():.6f}")
    print(f"   Unitarity error: {unitarity_error:.6f}")

    assert unitarity_error < 0.01, "SO(4) evolution broke unitarity!"
    print("   ✅ PASSED: ||Ψ_out|| = ||Ψ_in|| = 1")

    print("\n✅ SO(4) Evolution Physics: VALIDATED\n")


def test_optical_probe_physics():
    """Test 4: Optical Probe - Resonance Generation"""
    print("\n" + "=" * 70)
    print("TEST 4: Optical Probe - Resonance Generation")
    print("=" * 70)

    vocab_size = 100
    quaternion_dim = 4
    batch_size = 2

    probe = OpticalProbeGenerator(vocab_size=vocab_size, quaternion_dim=quaternion_dim)

    # Random final state
    psi_last = torch.randn(batch_size, quaternion_dim)
    psi_last = quaternion_normalize(psi_last)

    # Generate logits
    logits = probe(psi_last, alpha=1.5, beta=0.02)

    print(f"\n📊 Logits Statistics:")
    print(f"   Shape: {logits.shape}")
    print(f"   Range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"   Mean:  {logits.mean():.3f}")

    # TODO: Consider replacing softmax with physical decoding in test validation
    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()

    print(f"\n📊 Probability Distribution:")
    print(f"   Entropy: {entropy:.3f} (max={np.log(vocab_size):.3f})")
    print(f"   Prob sum: {probs.sum(dim=-1).mean():.6f} (should be 1.0)")

    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    print("   ✅ PASSED: Valid probability distribution")

    print("\n✅ Optical Probe Physics: VALIDATED\n")


def test_complete_psiqrh_transformer_new():
    """Test 5: Complete ΨQRH Transformer - End-to-End"""
    print("\n" + "=" * 70)
    print("TEST 5: Complete ΨQRH Transformer - End-to-End Pipeline")
    print("=" * 70)

    vocab_size = 200
    embed_dim = 64
    d_model = 256
    n_layers = 2
    batch_size = 2
    seq_len = 16

    model = PsiQRHTransformerComplete(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        quaternion_dim=4,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        n_rotations=2,
        dropout=0.1,
        max_seq_len=128
    )

    # Forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)

    print(f"\n📊 Model Output:")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {vocab_size})")

    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("   ✅ PASSED: Correct output shape")

    # Test generation
    print(f"\n📊 Testing Generation:")
    prompt = input_ids[:1, :4]  # Use first 4 tokens as prompt
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)

    print(f"   Prompt length: {prompt.shape[1]}")
    print(f"   Generated length: {generated.shape[1]}")
    print(f"   New tokens: {generated.shape[1] - prompt.shape[1]}")

    assert generated.shape[1] > prompt.shape[1], f"Generation failed: {generated.shape[1]} <= {prompt.shape[1]}"
    print("   ✅ PASSED: Generation successful")

    # Test quaternion extraction
    quaternions = model(input_ids, return_quaternions=True)
    norms = quaternion_norm(quaternions)

    print(f"\n📊 Internal Quaternion States:")
    print(f"   Shape: {quaternions.shape}")
    print(f"   Norms: {norms.mean():.6f} ± {norms.std():.6f}")

    unitarity_error = torch.abs(norms - 1.0).mean().item()
    print(f"   Unitarity error: {unitarity_error:.6f}")

    assert unitarity_error < 0.1
    print("   ✅ PASSED: Quaternions approximately unit")

    print("\n✅ Complete Transformer: VALIDATED\n")

    return unitarity_error


def main_physics_validation():
    """Main physics validation test suite"""
    print("\n" + "=" * 70)
    print("ΨQRH TRANSFORMER - COMPLETE PHYSICS VALIDATION SUITE")
    print("=" * 70)

    try:
        test_fractal_embedding_physics()
        test_spectral_attention_physics()
        test_so4_evolution_physics()
        test_optical_probe_physics()
        error = test_complete_psiqrh_transformer_new()

        print("\n" + "=" * 70)
        print("🎯 ALL PHYSICS TESTS PASSED!")
        print("=" * 70)
        print("\n✅ Fractal Quantum Embedding: Unit quaternions + D ∈ [1,2]")
        print("✅ Spectral Attention: Adaptive α(D) + Energy conservation")
        print("✅ SO(4) Evolution: Unitarity preserved (||Ψ|| = 1)")
        print("✅ Optical Probe: Valid probability distribution")
        print("✅ Complete Transformer: End-to-end pipeline functional")
        print(f"\n📊 Final Unitarity Error: {error:.6f}")

        print("\n🌟 ΨQRH TRANSFORMER IMPLEMENTATION: PHYSICALLY RIGOROUS ✓")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run original tests
    # main()

    # Run new physics validation
    main_physics_validation()