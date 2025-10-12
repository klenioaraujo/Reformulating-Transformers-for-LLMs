#!/usr/bin/env python3
"""
Create Training Data Tool
========================

Generates training data from raw text files by creating sliding window contexts
for next-character prediction tasks.
"""

import json
import argparse
from pathlib import Path


def create_training_data(input_file: str, output_file: str, context_window: int = 4):
    """
    Create training data from a text file using sliding window approach.

    Args:
        input_file: Path to input text file
        output_file: Path to output JSON file
        context_window: Size of the context window
    """
    # Read input text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    if len(text) < context_window + 1:
        raise ValueError(f"Input text too short. Need at least {context_window + 1} characters.")

    training_data = []

    # Create sliding windows
    for i in range(len(text) - context_window):
        context = text[i:i + context_window]
        target = text[i + context_window]
        combined = context + target

        training_pair = {
            "context": context,
            "target": target,
            "combined": combined
        }

        training_data.append(training_pair)

    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"Created {len(training_data)} training pairs in {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create training data from text file")
    parser.add_argument("--input", "-i", required=True, help="Input text file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--window", "-w", type=int, default=4, help="Context window size")

    args = parser.parse_args()

    create_training_data(args.input, args.output, args.window)


if __name__ == "__main__":
    main()