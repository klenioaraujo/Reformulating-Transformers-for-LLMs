# ΨCWS Training System - ΨCWS Training System

## 📋 Overview

The ΨCWS system implements a complete training pipeline that converts:
```
TEXT → SPECTRUM → OUTPUT SPECTRUM → INPUT SPECTRUM → TEXT CONVERSION
```

**Architecture:**
- **Base:** Open-source models
- **Security:** 7 encryption layers
- **Pattern:** Scientific mask to ensure pattern
- **Processing:** Spectral conversion

## 🚀 How to Use

### 1. Parameter Configuration

```python
from Ψcws_training_parameters import ΨCWSTrainingParameters

# Default configuration
params = ΨCWSTrainingParameters()

# Predefined configuration
params = get_preset_config("large")  # small, medium, large, spectral_focus

# Optimize for hardware
params.optimize_for_hardware("gpu")  # gpu, cpu, tpu

# Validate parameters
is_valid, errors = params.validate_parameters()
```

### 2. Main Parameters

#### Training
```python
{
    "batch_size": 32,
    "learning_rate": 1e-4,
    "max_epochs": 100,
    "gradient_clip": 1.0,
    "optimizer": "AdamW",
    "scheduler": "cosine"
}
```

#### Model
```python
{
    "vocab_size": 50000,
    "embedding_dim": 512,
    "hidden_dim": 1024,
    "num_layers": 6,
    "num_heads": 8,
    "spectral_dim": 256
}
```

#### Spectral
```python
{
    "fft_bins": 128,
    "window_size": 64,
    "hop_length": 32,
    "n_mels": 80,
    "compression_method": "log"
}
```

#### Encryption
```python
{
    "encryption_layers": 7,
    "encryption_key_size": 32,
    "scientific_mask_enabled": True,
    "mask_pattern": "fractal_gaussian"
}
```

## 🔧 Processing Pipeline

### 1. Text → Spectrum Conversion
```python
from src.conscience.conscious_wave_modulator import ConsciousWaveModulator

# Configure modulator
config = {
    'embedding_dim': 256,
    'sequence_length': 64,
    'device': 'cpu'
}
modulator = ConsciousWaveModulator(config)

# Convert file
Ψcws_file = modulator.process_file("document.pdf")
Ψcws_file.save("output.Ψcws")
```

### 2. Protection with Encryption
```python
from src.conscience.secure_Ψcws_protector import create_secure_Ψcws_protector

# Create protector
protector = create_secure_Ψcws_protector()

# Protect file
protected_parts = protector.protect_file("output.Ψcws", parts=4)
```

### 3. Spectral Processing
```python
# Optimized spectral parameters
spectral_config = {
    'use_stft': True,
    'n_fft': 1024,
    'n_mels': 80,
    'compression_method': 'log'
}
```

## 🎯 Predefined Configurations

### `small` - Quick Test
- Batch size: 8
- Embedding: 256
- Layers: 4
- Epochs: 10

### `medium` - Development
- Batch size: 16
- Embedding: 384
- Layers: 6
- Epochs: 50

### `large` - Production
- Batch size: 32
- Embedding: 512
- Layers: 8
- Epochs: 100

### `spectral_focus` - Spectral Focus
- Spectral dim: 512
- FFT bins: 256
- Mel bands: 128
- MFCC enabled

## 🔒 Security System

### 7 Encryption Layers
1. **AES-256-GCM** - Symmetric encryption
2. **ChaCha20-Poly1305** - Stream encryption
3. **Fernet** - Authenticated encryption
4. **XOR-Custom** - Custom obfuscation
5. **Transposition** - Data transposition
6. **HMAC-AES** - Authentication + encryption
7. **Obfuscation** - Final obfuscation

### Scientific Mask
- Pattern: `fractal_gaussian`
- Entropy threshold: 0.8
- Ensures consistent mathematical pattern

## 📊 Training Metrics

### Consciousness
- **Complexity**: Embedding entropy
- **Coherence**: Trajectory autocorrelation
- **Adaptability**: Spectral diversity
- **Integration**: Cross-correlation

### Performance
- **Loss**: Cross-entropy
- **Accuracy**: Conversion accuracy
- **Spectral Fidelity**: Spectral fidelity
- **Encryption Security**: Encryption security

## 🛠️ Makefile Commands

### File Conversion
```bash
# Convert PDF to ΨCWS
make convert-pdf PDF=document.pdf

# ΨCWS statistics
make Ψcws-stats

# List ΨCWS files
make list-Ψcws
```

### Training
```bash
# Quick test
python3 train_Ψcws.py --preset small

# Complete training
python3 train_Ψcws.py --preset large --device gpu

# Spectral training
python3 train_Ψcws.py --preset spectral_focus
```

## 📁 File Structure

```
Ψcws_training_parameters.py    # Training parameters
src/conscience/
├── conscious_wave_modulator.py    # Text→spectrum conversion
├── secure_Ψcws_protector.py       # Security system
└── ...
data/Ψcws_cache/               # ΨCWS file cache
secure_parts/                  # Encrypted parts
```

## 🎯 Complete Example

```python
import torch
from Ψcws_training_parameters import ΨCWSTrainingParameters
from src.conscience.conscious_wave_modulator import ConsciousWaveModulator

# 1. Configure parameters
params = ΨCWSTrainingParameters()
params.optimize_for_hardware("gpu")

# 2. Convert text to spectrum
modulator = ConsciousWaveModulator({
    'embedding_dim': params.training_config.embedding_dim,
    'sequence_length': params.training_config.max_sequence_length
})

Ψcws_file = modulator.process_file("input.txt")

# 3. Protect with encryption
from src.conscience.secure_Ψcws_protector import create_secure_Ψcws_protector
protector = create_secure_Ψcws_protector()
protected_parts = protector.protect_file("input.Ψcws")

print("✅ ΨCWS pipeline configured successfully!")
```

## 🔍 Validation

```python
# Validate parameters
is_valid, errors = params.validate_parameters()
if is_valid:
    print("✅ Valid parameters")
else:
    print(f"❌ Errors: {errors}")

# Check hardware compatibility
print(f"Device: {params.training_config.device}")
print(f"Optimized batch size: {params.training_config.batch_size}")
```

## 📈 Optimizations

### For GPU
- Increased batch size
- Mixed precision enabled
- Reduced gradient accumulation

### For CPU
- Reduced batch size
- Mixed precision disabled
- Increased gradient accumulation

### For TPU
- Maximum batch size
- Mixed precision enabled
- Minimum accumulation

## 🐛 Troubleshooting

### Error: "embedding_dim not divisible by num_heads"
```python
# Solution: Adjust embedding_dim
params.training_config.embedding_dim = 512  # Divisible by 8
```

### Error: "No GPU available"
```python
# Solution: Use CPU
params.training_config.device = "cpu"
params.optimize_for_hardware("cpu")
```

### Error: "ΨCWS file corrupted"
```python
# Solution: Check encryption
from src.conscience.secure_Ψcws_protector import create_secure_Ψcws_protector
protector = create_secure_Ψcws_protector()
success = protector.read_protected_file(protected_parts)
```

## 📞 Support

For problems or questions:
- Check validation logs
- Consult predefined parameters
- Validate hardware compatibility
- Check ΨCWS file integrity