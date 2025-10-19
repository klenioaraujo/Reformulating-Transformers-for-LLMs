#!/usr/bin/env python3
"""
Quantum Prompt Generator for ΨQRH Pipeline

Generates quantum-themed prompts by substituting characters with quantum words,
following the same pattern as the original characters but with quantum references.
"""

import sys
import argparse
from quantum_vocab_mapping import create_quantum_prompt, get_quantum_reference

def generate_quantum_prompt(text: str, verbose: bool = False) -> str:
    """
    Generate a quantum prompt by substituting characters with quantum words.

    Args:
        text: Input text to transform
        verbose: Whether to show detailed mapping information

    Returns:
        Quantum-transformed prompt string
    """
    quantum_prompt = create_quantum_prompt(text)

    if verbose:
        print(f"\n📊 Detailed Character Mapping:")
        print("=" * 60)
        for i, char in enumerate(text):
            ref = get_quantum_reference(char)
            print(f"  Position {i:2d}: '{char}' → '{ref['word']}'")
            print(f"           Quantum Reference: {ref['quantum_reference']}")
            print(f"           Energy Level: {ref['energy_level']}")
            print("-" * 60)

    return quantum_prompt

def main():
    parser = argparse.ArgumentParser(description="Generate quantum-themed prompts for ΨQRH pipeline")
    parser.add_argument("text", help="Input text to transform into quantum prompt")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed mapping information")
    parser.add_argument("-o", "--output", help="Output file to save the quantum prompt")

    args = parser.parse_args()

    print(f"🔬 ΨQRH Quantum Prompt Generator")
    print(f"📝 Original text: {args.text}")

    # Generate quantum prompt
    quantum_prompt = generate_quantum_prompt(args.text, verbose=args.verbose)

    print(f"\n🎯 Quantum Prompt:")
    print(f"   {quantum_prompt}")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(quantum_prompt)
        print(f"\n💾 Quantum prompt saved to: {args.output}")

    # Generate usage instructions for ΨQRH pipeline
    print(f"\n🚀 Usage with ΨQRH Pipeline:")
    print(f"   python3 psiqrh.py \"{quantum_prompt}\"")

if __name__ == "__main__":
    main()