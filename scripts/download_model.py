#!/usr/bin/env python3
"""
Model Download Script for ΨQRH Knowledge Distillation
===================================================

Downloads and caches Hugging Face models for use in knowledge distillation workflows.
This script is used internally by the Makefile and is not exposed as a public target.

Usage:
    python scripts/download_model.py --model_name gpt2
"""

import argparse
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def download_model(model_name: str):
    """
    Download and cache a Hugging Face model for distillation.

    Args:
        model_name: Name of the model to download (e.g., 'gpt2', 'microsoft/DialoGPT-medium')
    """
    print(f"📥 Downloading model '{model_name}' for ΨQRH distillation...")

    # Define local cache path
    cache_dir = Path("models/source") / model_name.replace('/', '_')
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download tokenizer
        print("   📝 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.save_pretrained(cache_dir)
        print("   ✅ Tokenizer downloaded and cached")

        # Download model
        print("   🤖 Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        model.save_pretrained(cache_dir)
        print("   ✅ Model downloaded and cached")

        # Save metadata
        metadata = {
            'model_name': model_name,
            'vocab_size': len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size,
            'model_type': model.config.model_type,
            'hidden_size': model.config.hidden_size,
            'num_layers': model.config.num_hidden_layers,
            'num_heads': model.config.num_attention_heads,
            'max_position_embeddings': model.config.max_position_embeddings
        }

        import json
        with open(cache_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Model '{model_name}' successfully cached at: {cache_dir}")
        print(f"   📊 Vocab size: {metadata['vocab_size']}")
        print(f"   🏗️  Hidden size: {metadata['hidden_size']}")
        print(f"   📚 Layers: {metadata['num_layers']}")

    except Exception as e:
        print(f"❌ Error downloading model '{model_name}': {e}")
        sys.exit(1)


def check_model_exists(model_name: str) -> bool:
    """
    Check if a model is already cached locally.

    Args:
        model_name: Name of the model to check

    Returns:
        True if model exists locally, False otherwise
    """
    cache_dir = Path("models/source") / model_name.replace('/', '_')
    metadata_file = cache_dir / 'metadata.json'

    if metadata_file.exists():
        print(f"✅ Model '{model_name}' already cached at: {cache_dir}")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description='Download Hugging Face models for ΨQRH distillation')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to download (e.g., gpt2, microsoft/DialoGPT-medium)')

    args = parser.parse_args()

    # Check if model already exists
    if check_model_exists(args.model_name):
        print("   ℹ️  Using cached model - no download needed")
        return

    # Download the model
    download_model(args.model_name)


if __name__ == "__main__":
    main()