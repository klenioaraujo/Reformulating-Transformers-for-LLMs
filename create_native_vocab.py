#!/usr/bin/env python3
"""
Create Native Vocabulary Script
===============================

This script creates the native vocabulary file required by the ΨQRH pipeline.
It loads a real GPT-2 tokenizer from Hugging Face and extracts the complete
vocabulary mapping, saving it to data/native_vocab.json in the format expected
by psiqrh.py.

Usage:
    python create_native_vocab.py

Output:
    data/native_vocab.json - Complete GPT-2 vocabulary with token-to-ID mapping
"""

import os
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

def create_native_vocab():
    """
    Create the native vocabulary file from GPT-2 tokenizer.

    This function:
    1. Loads the GPT-2 tokenizer from Hugging Face
    2. Extracts the complete vocabulary (token -> ID mapping)
    3. Saves it to data/native_vocab.json in the expected format
    """
    print("🔬 Creating native vocabulary from GPT-2 tokenizer...")

    try:
        # Load GPT-2 tokenizer
        print("   📚 Loading GPT-2 tokenizer from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # Get vocabulary
        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)

        print(f"   ✅ Loaded GPT-2 vocabulary: {vocab_size} tokens")

        # Create the expected format for psiqrh.py
        vocab_data = {
            'vocab_size': vocab_size,
            'tokens': vocab,  # token -> id mapping
            'model_name': 'gpt2',
            'description': 'Complete GPT-2 vocabulary extracted for ΨQRH pipeline'
        }

        # Ensure data directory exists
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)

        # Save to data/native_vocab.json
        vocab_path = data_dir / 'native_vocab.json'
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

        print(f"   💾 Saved native vocabulary to: {vocab_path}")
        print(f"   📊 Vocabulary size: {vocab_size}")
        print(f"   🎯 Format: token -> ID mapping")

        # Verify the file was created correctly
        with open(vocab_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        loaded_vocab_size = loaded_data.get('vocab_size', 0)
        loaded_tokens = loaded_data.get('tokens', {})

        if loaded_vocab_size == vocab_size and len(loaded_tokens) == vocab_size:
            print("   ✅ Vocabulary file verification: PASSED")
            print("   🎉 Native vocabulary creation completed successfully!")
            return True
        else:
            print("   ❌ Vocabulary file verification: FAILED")
            return False

    except Exception as e:
        print(f"❌ Error creating native vocabulary: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("ΨQRH Native Vocabulary Creator")
    print("=" * 40)

    success = create_native_vocab()

    if success:
        print("\n✅ SUCCESS: Native vocabulary created successfully!")
        print("   The ΨQRH pipeline can now load the complete GPT-2 vocabulary.")
        print("   Run 'make vocab' to regenerate the vocabulary anytime.")
        return 0
    else:
        print("\n❌ FAILED: Could not create native vocabulary.")
        return 1

if __name__ == "__main__":
    sys.exit(main())