# 4D Unitary Layer: Reformulating Transformers for LLMs

## Overview

This implementation provides a comprehensive 4D Unitary (U) Layer that reformulates transformers for Large Language Models (LLMs) with advanced mathematical foundations. The layer combines:

- **Quaternion Algebra**: SO(4) rotations using two independent quaternions for complete 4D unitary transformations
- **Spectral Processing**: FFT-based negentropy filtering with numerical stability
- **Gate Mechanisms**: ABSTAIN/DELIVER/CLARIFY flow control based on numerical receipts
- **Transformer Integration**: Seamless integration with standard transformer architectures

## Mathematical Foundation

### Core Equation
```
Ψ_QRH = R_left · F^{-1} { F(k) · F { Ψ } } · R_right
```

Where:
- **Ψ**: Input quaternion state (4D embedding)
- **F{·}**: Forward Fourier Transform
- **F(k)**: Spectral filter with negentropy properties
- **F^{-1}{·}**: Inverse Fourier Transform
- **R_left, R_right**: Independent SO(4) rotation quaternions

### SO(4) Rotations
The implementation uses two independent unit quaternions for complete SO(4) group representation:
```
R_left = [cos(θ₁/2), sin(θ₁/2)cos(ω₁), sin(θ₁/2)sin(ω₁)cos(φ₁), sin(θ₁/2)sin(ω₁)sin(φ₁)]
R_right = [cos(θ₂/2), sin(θ₂/2)cos(ω₂), sin(θ₂/2)sin(ω₂)cos(φ₂), sin(θ₂/2)sin(ω₂)sin(φ₂)]
```

### Spectral Filtering
The negentropy filter uses stabilized activation functions:
```
F(k) = exp(i * α * GELU(log(|k| + ε)))
```
With numerical stability improvements and clamping for extreme values.

## Architecture Components

### 1. QuaternionOperations
Utility class for quaternion mathematics:
```python
from ΨQRH import QuaternionOperations

# Create unit quaternion from angles
q = QuaternionOperations.create_unit_quaternion(theta, omega, phi)

# Quaternion multiplication
result = QuaternionOperations.multiply(q1, q2)
```

### 2. SpectralFilter
Numerically stable spectral filtering:
```python
from ΨQRH import SpectralFilter

filter = SpectralFilter(alpha=1.0, use_stable_activation=True)
filtered = filter(k_magnitude)
```

### 3. QRHLayer
Core 4D unitary layer with SO(4) rotations:
```python
from ΨQRH import QRHLayer

layer = QRHLayer(
    embed_dim=64,
    alpha=1.0,
    use_learned_rotation=True,
    spatial_dims=(seq_len,)
)

output = layer(input_tensor)  # [batch, seq, 4*embed_dim]
```

### 4. GateController
Flow control mechanism with receipt-based decisions:
```python
from ΨQRH import GateController

controller = GateController(
    orthogonal_threshold=1e-6,
    energy_threshold=0.1,
    drift_threshold=0.1
)

receipts = controller.calculate_receipts(input_tensor, output_tensor, params)
decision = controller.decide_gate(receipts)  # 'ABSTAIN', 'DELIVER', or 'CLARIFY'
```

### 5. NegentropyTransformerBlock
Complete transformer block with 4D integration:
```python
from ΨQRH import NegentropyTransformerBlock

block = NegentropyTransformerBlock(
    d_model=512,
    nhead=8,
    qrh_embed_dim=128,
    enable_gate=True
)

output = block(input_tensor)  # [batch, seq, d_model]
```

## Usage Examples

### Basic Usage
```python
import torch
from ΨQRH import QRHLayer

# Create layer
layer = QRHLayer(embed_dim=64, use_learned_rotation=True)

# Process input (expects 4*embed_dim channels)
x = torch.randn(2, 32, 256)  # batch=2, seq=32, embed=4*64=256
output = layer(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

### Transformer Integration
```python
import torch.nn as nn
from ΨQRH import NegentropyTransformerBlock

class MyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Use 4D unitary blocks
        self.layers = nn.ModuleList([
            NegentropyTransformerBlock(
                d_model=d_model,
                nhead=8,
                qrh_embed_dim=d_model//4,
                enable_gate=True
            ) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)
```

### Advanced Configuration
```python
# Custom spectral filter with stability
from ΨQRH import SpectralFilter

custom_filter = SpectralFilter(
    alpha=1.5,
    epsilon=1e-10,
    use_stable_activation=True
)

# Multi-dimensional spatial filtering
layer = QRHLayer(
    embed_dim=128,
    alpha=1.0,
    spatial_dims=(32, 32),  # 2D spatial processing
    use_learned_rotation=True
)

# Gate controller with custom thresholds
gate = GateController(
    orthogonal_threshold=1e-7,
    energy_threshold=0.05,
    drift_threshold=0.02
)
```

## Key Features

### ✅ SO(4) Rotations
- Two independent quaternions for complete 4D rotation group
- Learnable rotation parameters
- Unit quaternion normalization

### ✅ Spectral Processing
- FFT-based frequency domain processing
- Negentropy filtering for information preservation
- Multi-dimensional spatial support
- Numerical stability with clamping and safe functions

### ✅ Gate Mechanism
- Receipt-based flow control
- Three decision states: ABSTAIN/DELIVER/CLARIFY
- Configurable thresholds
- Automatic fallback policies

### ✅ Transformer Integration
- Drop-in replacement for attention/FFN layers
- Linear projections for dimension matching
- Residual connections and layer normalization
- Mixed precision support

### ✅ Numerical Stability
- GELU activation for spectral filtering
- Gradient clamping and NaN detection
- Safe logarithm operations
- PyTorch autocast integration

## Performance Characteristics

### Complexity
- **Time**: O(n log n) due to FFT
- **Space**: O(n) for quaternion representations
- **Memory**: Linear scaling with sequence length

### Advantages over Standard Transformers
- **Efficiency**: O(n log n) vs O(n²) for attention
- **Expressiveness**: 4D unitary transformations
- **Stability**: Built-in numerical safeguards
- **Control**: Gate mechanism for adaptive processing

## Validation and Testing

Run the comprehensive test suite:
```bash
python test_4d_unitary_layer.py
```

Tests cover:
- Quaternion mathematical properties
- Spectral filter numerical stability
- SO(4) rotation correctness
- Gate mechanism decisions
- Transformer integration
- Performance benchmarks
- Edge case handling

## Mathematical Validation

### Energy Preservation
The layer preserves signal energy within reasonable bounds:
```
||Ux|| ≈ ||x||  (within factor of 2)
```

### Unitary Properties
SO(4) rotations maintain:
- Norm preservation: ||Rx|| = ||x||
- Orthogonality: R^T R = I
- Determinant: det(R) = 1

### Spectral Properties
The negentropy filter provides:
- Frequency-dependent attenuation
- Phase preservation
- Information content maintenance

## Future Extensions

### Hardware Acceleration
- CUDA kernels for spectral operations
- Custom FPGA implementations
- Optical computing integration

### Advanced Architectures
- Hierarchical 4D processing
- Multi-scale spectral analysis
- Adaptive filter learning

### Research Directions
- Theoretical analysis of expressiveness
- Convergence properties
- Optimal hyperparameter selection

## References

1. **Quaternion Mathematics**: "Quaternions and Rotation Sequences" by Jack B. Kuipers
2. **Spectral Methods**: "Fourier Analysis and Its Applications" by Gerald B. Folland
3. **Transformer Theory**: "Attention Is All You Need" by Vaswani et al.
4. **Negentropy**: "Independent Component Analysis" by Aapo Hyvärinen

## License

This implementation is provided under the MIT License. See LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Run the test suite before submitting
2. Add tests for new features
3. Update documentation
4. Follow the existing code style

## Citation

If you use this implementation in your research, please cite:
```
@misc{4d_unitary_layer,
  title={4D Unitary Layer: Reformulating Transformers with Quaternion Algebra and Spectral Processing},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/4d-unitary-layer}
}