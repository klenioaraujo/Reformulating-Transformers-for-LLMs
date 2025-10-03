# ΨQRH Reuse Guides

This directory contains practical guides for reusing and extending the ΨQRH Transformer in your own projects.

## 📚 Available Guides

### 1. Fine-tuning ΨQRH (`fine_tune_psiqrh.py`)

Learn how to fine-tune the ΨQRH transformer on your custom datasets and tasks.

**Key Features:**
- Load pre-trained ΨQRH models
- Implement custom datasets
- Freeze layers for transfer learning
- Save and manage checkpoints
- Monitor training progress

**Usage:**
```bash
python examples/reuse_guides/fine_tune_psiqrh.py
```

**Quick Example:**
```python
from fine_tune_psiqrh import load_pretrained_model, fine_tune

# Load model
model, config = load_pretrained_model("models/checkpoint.pt")

# Fine-tune on your data
fine_tune(
    model=model,
    train_dataloader=your_train_loader,
    val_dataloader=your_val_loader,
    epochs=3,
    learning_rate=1e-5
)
```

---

### 2. Integration with Standard Transformers (`integrate_with_standard_transformer.py`)

Integrate ΨQRH components into existing PyTorch or HuggingFace transformer models.

**Key Features:**
- Hybrid transformer layers (standard + ΨQRH)
- Replace specific layers in existing models
- Adapter pattern for compatibility
- HuggingFace integration examples

**Usage:**
```bash
python examples/reuse_guides/integrate_with_standard_transformer.py
```

**Quick Example:**
```python
from integrate_with_standard_transformer import integrate_qrh_into_huggingface_model
from transformers import GPT2LMHeadModel

# Load HuggingFace model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Replace middle layers with ΨQRH
model = integrate_qrh_into_huggingface_model(
    model,
    layer_indices=[4, 5, 6]  # Replace layers 4-6
)
```

---

### 3. ONNX Export (`convert_psiqrh_to_onnx.py`)

Export ΨQRH models to ONNX format for optimized cross-platform deployment.

**Key Features:**
- Export to ONNX with dynamic axes
- Verify exported models
- Benchmark inference performance
- TensorRT conversion (optional)

**Usage:**
```bash
python examples/reuse_guides/convert_psiqrh_to_onnx.py
```

**Quick Example:**
```python
from convert_psiqrh_to_onnx import export_to_onnx, verify_onnx_model

# Export model
onnx_path = export_to_onnx(
    model=your_model,
    output_path='models/psiqrh.onnx'
)

# Verify
verify_onnx_model(onnx_path)
```

---

## 🚀 Quick Start Workflow

1. **Start with Fine-tuning:**
   - Adapt ΨQRH to your specific task
   - Use transfer learning for faster convergence

2. **Integrate into Existing Pipelines:**
   - Enhance existing models with quaternion layers
   - Maintain compatibility with standard tools

3. **Deploy with ONNX:**
   - Export optimized models
   - Deploy across different platforms

---

## 📋 Prerequisites

### Required Dependencies
```bash
pip install torch>=2.0.0
pip install pyyaml
pip install numpy scipy
```

### Optional Dependencies

For HuggingFace integration:
```bash
pip install transformers
```

For ONNX export:
```bash
pip install onnx onnxruntime
pip install onnx-simplifier  # For graph optimization
```

For TensorRT (NVIDIA GPUs):
```bash
# Follow instructions at: https://developer.nvidia.com/tensorrt
```

---

## 💡 Use Cases

### Academic Research
- Experiment with quaternion representations
- Study energy conservation in neural networks
- Analyze fractal consciousness metrics

### Production Applications
- Fine-tune on domain-specific data
- Integrate into existing LLM pipelines
- Deploy optimized models for inference

### Model Development
- Prototype new architectures
- Benchmark against baselines
- Evaluate performance improvements

---

## 📖 Additional Resources

### Documentation
- [Main README](../../README.md)
- [API Documentation](../../docs/api/)
- [Configuration Guide](../../configs/README.md)

### Research Papers
- [ΨQRH Theory](../../docs/TRANSFORMER_REFORMULATION_PLAN.md)
- [Implementation Details](../../IMPLEMENTATION_SUMMARY.md)

### Examples
- [Basic Usage](../basic_usage.py)
- [Energy Conservation Tests](../energy_conservation_test.py)
- [Complete Integration](../test_complete_psiqrh.py)

---

## 🤝 Contributing

If you develop new reuse patterns or integration methods, please contribute:

1. Add your guide to this directory
2. Update this README
3. Submit a pull request

---

## 📝 License

All guides are licensed under GNU GPLv3.

```
Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file
```

---

## 📧 Support

- **Issues:** [GitHub Issues](https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs/issues)
- **Discussions:** [GitHub Discussions](https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs/discussions)
- **DOI:** https://zenodo.org/records/17171112

---

## 🔄 Version History

- **v1.0.0** (2025-09-30): Initial reuse guides release
  - Fine-tuning guide
  - Integration guide
  - ONNX export guide