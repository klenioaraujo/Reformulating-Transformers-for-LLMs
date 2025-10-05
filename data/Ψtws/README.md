# ΨTWS Training Data Directory

This directory contains ΨTWS (Ψ Text-Wave-Spectrum) training data files with `.Ψtws` extension.

## File Structure

```
data/Ψtws/
├── README.md
├── training_data_1.Ψtws
├── training_data_2.Ψtws
├── validation_data.Ψtws
└── test_data.Ψtws
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