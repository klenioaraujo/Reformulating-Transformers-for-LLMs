#!/usr/bin/env python3
"""
Debug character mapping analysis
"""

import torch
import torch.fft as fft
import math
import numpy as np
from typing import Dict, Tuple, List

# Sample characters to analyze
input_chars = "The quick brown fox jumps over the lazy dog. This"
output_chars = "-A> JNB<D ;KHPG ?HQ CNFIL HO>K MA> E:SR =H@  -ABL"

print("🔍 CHARACTER MAPPING ANALYSIS")
print("=" * 50)

# Analyze the pattern
for i, (in_char, out_char) in enumerate(zip(input_chars, output_chars)):
    in_ascii = ord(in_char)
    out_ascii = ord(out_char)
    diff = out_ascii - in_ascii

    print(f"{i:2d}: '{in_char}' ({in_ascii:3d}) → '{out_char}' ({out_ascii:3d}) | diff: {diff:3d}")

print("\n📊 PATTERN ANALYSIS:")
print(f"Input length: {len(input_chars)}")
print(f"Output length: {len(output_chars)}")

# Check if there's a consistent transformation
print("\n🔍 LOOKING FOR TRANSFORMATION PATTERNS:")

# Test if it's a simple shift
for shift in range(-50, 51):
    matches = 0
    for in_char, out_char in zip(input_chars, output_chars):
        if in_char != ' ' and out_char != ' ':
            expected = chr((ord(in_char) + shift) % 128)
            if expected == out_char:
                matches += 1

    if matches > len(input_chars) * 0.5:  # More than 50% match
        print(f"Possible shift: {shift}, matches: {matches}/{len(input_chars)}")

# Test if it's a modulo operation
print("\n🔍 CHECKING MODULO OPERATIONS:")
for mod in [32, 64, 96, 128]:
    matches = 0
    for in_char, out_char in zip(input_chars, output_chars):
        if in_char != ' ' and out_char != ' ':
            expected = chr(ord(in_char) % mod)
            if expected == out_char:
                matches += 1

    if matches > len(input_chars) * 0.3:
        print(f"Possible modulo {mod}, matches: {matches}/{len(input_chars)}")

print("\n🔍 CHECKING BITWISE OPERATIONS:")
# Test bitwise operations
for op in ['&', '|', '^']:
    for mask in [0x7F, 0x3F, 0x1F, 0x0F]:
        matches = 0
        for in_char, out_char in zip(input_chars, output_chars):
            if in_char != ' ' and out_char != ' ':
                in_val = ord(in_char)
                if op == '&':
                    expected = chr(in_val & mask)
                elif op == '|':
                    expected = chr(in_val | mask)
                elif op == '^':
                    expected = chr(in_val ^ mask)

                if expected == out_char:
                    matches += 1

        if matches > len(input_chars) * 0.3:
            print(f"Possible {op} with mask 0x{mask:02X}, matches: {matches}/{len(input_chars)}")