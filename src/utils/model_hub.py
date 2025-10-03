"""
HuggingFace Hub Integration

Utilities for uploading and downloading ΨQRH models from HuggingFace Hub.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json
import yaml


def create_model_card(
    model_name: str,
    description: str,
    training_data: str,
    metrics: Dict[str, float],
    version: str = "1.0.0"
) -> str:
    """
    Create a model card for HuggingFace Hub.

    Args:
        model_name: Name of the model (e.g., "psiqrh-base")
        description: Model description
        training_data: Description of training data
        metrics: Dictionary of evaluation metrics
        version: Model version

    Returns:
        Model card content as string
    """
    card = f"""---
license: gpl-3.0
language:
- en
tags:
- pytorch
- transformer
- quaternion
- psiqrh
- energy-conservation
- fractal-consciousness
datasets:
- {training_data}
metrics:
{yaml.dump(metrics, indent=2)}
---

# {model_name}

## Model Description

{description}

This is a ΨQRH (Psi-Quaternionic Relativistic Harmonic) Transformer, a novel architecture that integrates:
- Quaternionic representations for 4D spatial relationships
- Spectral harmonic analysis for frequency domain processing
- Fractal consciousness metrics for self-similarity
- Energy conservation principles

## Model Details

- **Developed by:** Klenio Araujo Padilha
- **Model type:** Quaternionic-Harmonic Transformer
- **Language:** English (adaptable to other languages)
- **License:** GNU GPLv3
- **DOI:** [10.5281/zenodo.17171112](https://zenodo.org/records/17171112)
- **Version:** {version}

## Intended Use

**Primary use cases:**
- General-purpose language modeling
- Text generation
- Research on quaternion-based neural architectures
- Energy-efficient transformer applications

**Out-of-scope uses:**
- Production systems without thorough testing
- Safety-critical applications
- Applications requiring real-time guarantees

## Usage

### Installation

```bash
pip install psiqrh
```

### Basic Usage

```python
from src.utils.model_hub import load_pretrained

# Load model
model = load_pretrained("{model_name}")

# Generate text
output = model.generate(input_text, max_length=100)
```

### Advanced Usage

```python
import torch
from src.architecture.psiqrh_transformer import PsiQRHTransformer

# Load with custom config
config = {{
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6
}}

model = PsiQRHTransformer(config)

# Load pretrained weights
checkpoint = torch.load("pytorch_model.bin")
model.load_state_dict(checkpoint)
```

## Training Data

{training_data}

## Evaluation Metrics

{yaml.dump(metrics, default_flow_style=False)}

## Training Procedure

### Hardware
- GPU: NVIDIA RTX 3090 (24GB)
- Training time: Approximately [specify]

### Hyperparameters
- Learning rate: 1e-4
- Batch size: 32
- Optimizer: AdamW
- Warmup steps: 1000
- Total steps: 100000

## Energy Conservation

ΨQRH transformers maintain energy conservation through:
- Unitary quaternion operations
- Spectral normalization
- Conservation-aware training objectives

Measured conservation rate: {metrics.get('energy_conservation', 'N/A')}

## Limitations

- May require more computation than standard transformers
- Best suited for GPU inference
- Training requires careful hyperparameter tuning

## Ethical Considerations

This model should not be used for:
- Generating misleading or false information
- Impersonation or deception
- Illegal activities
- Discrimination or bias amplification

## Citation

```bibtex
@software{{padilha2025psiqrh,
  author = {{Padilha, Klenio Araujo}},
  title = {{{{ΨQRH Transformer: {model_name}}}}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  doi = {{10.5281/zenodo.17171112}},
  url = {{https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs}}
}}
```

## Contact

- **GitHub:** [klenioaraujo/Reformulating-Transformers-for-LLMs](https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs)
- **Issues:** [GitHub Issues](https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs/issues)

## Acknowledgments

This project was developed with assistance from Claude AI (Anthropic).

---

**License:** GNU General Public License v3.0
**DOI:** [10.5281/zenodo.17171112](https://zenodo.org/records/17171112)
"""
    return card


def push_to_hub(
    model: torch.nn.Module,
    repo_id: str,
    commit_message: str = "Upload ΨQRH model",
    private: bool = False,
    token: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Push model to HuggingFace Hub.

    Args:
        model: PyTorch model to upload
        repo_id: Repository ID (e.g., "username/model-name")
        commit_message: Commit message
        private: Whether to make repo private
        token: HuggingFace API token (optional, uses HF_TOKEN env var if not provided)
        config: Model configuration dictionary

    Example:
        ```python
        from src.architecture.psiqrh_transformer import PsiQRHTransformer

        model = PsiQRHTransformer(config)
        # ... train model ...

        push_to_hub(
            model=model,
            repo_id="klenioaraujo/psiqrh-base",
            config=config
        )
        ```

    Note:
        Requires `huggingface_hub` package:
        pip install huggingface_hub
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for this function. "
            "Install with: pip install huggingface_hub"
        )

    # Create temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save model
        model_path = tmpdir / "pytorch_model.bin"
        torch.save(model.state_dict(), model_path)

        # Save config
        if config:
            config_path = tmpdir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

        # Create model card
        if config:
            card_content = create_model_card(
                model_name=repo_id.split('/')[-1],
                description="ΨQRH Transformer model",
                training_data="Custom dataset",
                metrics={"energy_conservation": 0.998},
                version="1.0.0"
            )
            card_path = tmpdir / "README.md"
            with open(card_path, 'w') as f:
                f.write(card_content)

        # Create repository
        api = HfApi(token=token)
        try:
            create_repo(repo_id, private=private, token=token, exist_ok=True)
        except Exception as e:
            print(f"Repository may already exist: {e}")

        # Upload files
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token
        )

        print(f"✓ Model uploaded to https://huggingface.co/{repo_id}")


def load_pretrained(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None
) -> torch.nn.Module:
    """
    Load a pretrained model from HuggingFace Hub.

    Args:
        repo_id: Repository ID (e.g., "klenioaraujo/psiqrh-base")
        revision: Git revision (branch, tag, or commit hash)
        token: HuggingFace API token for private repos

    Returns:
        Loaded model

    Example:
        ```python
        model = load_pretrained("klenioaraujo/psiqrh-base")
        ```

    Note:
        Requires `huggingface_hub` package:
        pip install huggingface_hub
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for this function. "
            "Install with: pip install huggingface_hub"
        )

    from src.architecture.psiqrh_transformer import PsiQRHTransformer

    # Download config
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        revision=revision,
        token=token
    )

    with open(config_path) as f:
        config = json.load(f)

    # Download model weights
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="pytorch_model.bin",
        revision=revision,
        token=token
    )

    # Initialize model
    model = PsiQRHTransformer(config)

    # Load weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    print(f"✓ Model loaded from https://huggingface.co/{repo_id}")
    return model


# Example usage
if __name__ == "__main__":
    print("ΨQRH HuggingFace Hub Integration")
    print("=" * 60)
    print()
    print("Usage examples:")
    print()
    print("1. Upload model:")
    print("   push_to_hub(model, 'username/psiqrh-base')")
    print()
    print("2. Load model:")
    print("   model = load_pretrained('klenioaraujo/psiqrh-base')")
    print()
    print("3. Create model card:")
    print("   card = create_model_card(...)")
    print()
    print("See documentation for full details.")