# ΨTWS Training Data Directory

This directory contains ΨTWS (Ψ Text-Wave-Spectrum) training data files with `.Ψtws` extension.

## File Structure

```
data/Ψtws/
├── README.md                    # This file (Portuguese)
├── Ψtws_README_ENGLISH.md      # English version
├── Ψtws_config.yaml            # Configuration file
├── Ψtws_loader.py              # File loader utility
├── training_data_1.Ψtws        # Training sample 1
├── training_data_2.Ψtws        # Training sample 2
├── validation_data.Ψtws        # Validation dataset
└── test_data.Ψtws              # Test dataset
```

## File Format

ΨTWS files contain spectral representations of text data in the format:
```
TEXT → SPECTRUM → OUTPUT SPECTRUM → INPUT SPECTRUM → TEXT CONVERSION
```

## Usage

ΨTWS files are used for training the ΨCWS system with the following pipeline:
- Text input processing
- Spectral transformation
- Encryption layers (7 layers)
- Scientific mask application
- Training and validation

## Security Features

- 7-layer encryption system
- Scientific mask patterns
- Anti-violation protection
- Secure key derivation

## Configuration

The `Ψtws_config.yaml` file contains all configuration parameters:
- Spectral processing settings
- Encryption layer configurations
- Training parameters
- Validation metrics
- Security policies

## Loading Files

Use the `Ψtws_loader.py` utility to load and process ΨTWS files:

```python
from Ψtws_loader import ΨTWSLoader

loader = ΨTWSLoader()
training_files = loader.load_training_files()
validation_files = loader.load_validation_files()
test_files = loader.load_test_files()
```

## File Contents

Each ΨTWS file contains:
- **Metadata**: File format, version, creation date
- **Input Text**: Original text content
- **Spectral Data**: Processing parameters and dimensions
- **Encryption Info**: 7-layer encryption configuration
- **Scientific Mask**: Mask pattern and entropy settings
- **Validation Hash**: File integrity verification

## Training Pipeline

1. **Text Input**: Raw text data
2. **Spectral Conversion**: Convert to spectral representation
3. **Encryption**: Apply 7-layer encryption
4. **Mask Application**: Apply scientific mask
5. **Training**: Use in ΨCWS model training
6. **Validation**: Test reconstruction accuracy

## File Validation

Each file includes validation hashes and integrity checks to ensure data security and prevent tampering.