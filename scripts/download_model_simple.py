#!/usr/bin/env python3
"""
Simple Model Download Script for ΨQRH Knowledge Distillation
==========================================================

Downloads and caches Hugging Face models WITHOUT transformers dependency.
Uses direct HTTP requests to Hugging Face Hub.

Usage:
    python scripts/download_model_simple.py --model_name gpt2
"""

import argparse
import os
import sys
import json
import requests
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def get_model_info(model_name: str):
    """Get model information from Hugging Face Hub API."""
    url = f"https://huggingface.co/api/models/{model_name}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching model info for '{model_name}': {e}")
        return None


def download_file(url: str, local_path: Path):
    """Download a file from URL to local path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False


def download_model_simple(model_name: str):
    """
    Download and cache a Hugging Face model WITHOUT transformers dependency.

    Args:
        model_name: Name of the model to download (e.g., 'gpt2', 'microsoft/DialoGPT-medium')
    """
    print(f"📥 Downloading model '{model_name}' for ΨQRH distillation (simple method)...")

    # Define local cache path
    cache_dir = Path("models/source") / model_name.replace('/', '_')
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Get model information
        model_info = get_model_info(model_name)
        if not model_info:
            print(f"❌ Could not get model info for '{model_name}'")
            return False

        print(f"   📋 Model info: {model_info.get('modelId', 'Unknown')}")
        print(f"   📊 Downloads: {model_info.get('downloads', 0):,}")

        # Download config.json
        print("   📝 Downloading config...")
        config_url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
        config_path = cache_dir / "config.json"
        if download_file(config_url, config_path):
            print("   ✅ Config downloaded")
        else:
            print("   ⚠️  Could not download config")

        # Download tokenizer files
        print("   🔤 Downloading tokenizer files...")
        tokenizer_files = [
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.json"
        ]

        for tokenizer_file in tokenizer_files:
            tokenizer_url = f"https://huggingface.co/{model_name}/resolve/main/{tokenizer_file}"
            tokenizer_path = cache_dir / tokenizer_file
            if download_file(tokenizer_url, tokenizer_path):
                print(f"   ✅ {tokenizer_file} downloaded")
            else:
                print(f"   ⚠️  Could not download {tokenizer_file}")

        # Try to download model weights (pytorch_model.bin or model.safetensors)
        print("   🤖 Checking for model weights...")
        model_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json"
        ]

        weights_found = False
        for model_file in model_files:
            model_url = f"https://huggingface.co/{model_name}/resolve/main/{model_file}"
            model_path = cache_dir / model_file
            if download_file(model_url, model_path):
                print(f"   ✅ {model_file} downloaded")
                weights_found = True
                break
            else:
                print(f"   ⚠️  Could not download {model_file}")

        if not weights_found:
            print("   ⚠️  No model weights found - will use spectral projection only")

        # Load config to get model parameters
        config_data = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)

        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_id': model_info.get('modelId', model_name),
            'tags': model_info.get('tags', []),
            'pipeline_tag': model_info.get('pipeline_tag', 'unknown'),
            'downloads': model_info.get('downloads', 0),
            'config': config_data,
            'vocab_size': config_data.get('vocab_size', 50257),  # GPT-2 default
            'hidden_size': config_data.get('hidden_size', 768),
            'num_layers': config_data.get('num_hidden_layers', 12),
            'num_heads': config_data.get('num_attention_heads', 12),
            'max_position_embeddings': config_data.get('max_position_embeddings', 1024),
            'weights_available': weights_found
        }

        with open(cache_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Model '{model_name}' successfully cached at: {cache_dir}")
        print(f"   📊 Vocab size: {metadata['vocab_size']}")
        print(f"   🏗️  Hidden size: {metadata['hidden_size']}")
        print(f"   📚 Layers: {metadata['num_layers']}")
        print(f"   🔧 Weights: {'✅ Available' if weights_found else '❌ Not available'}")

        return True

    except Exception as e:
        print(f"❌ Error downloading model '{model_name}': {e}")
        return False


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
    parser = argparse.ArgumentParser(description='Download Hugging Face models for ΨQRH distillation (simple method)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to download (e.g., gpt2, microsoft/DialoGPT-medium)')

    args = parser.parse_args()

    # Check if model already exists
    if check_model_exists(args.model_name):
        print("   ℹ️  Using cached model - no download needed")
        return

    # Download the model
    success = download_model_simple(args.model_name)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()