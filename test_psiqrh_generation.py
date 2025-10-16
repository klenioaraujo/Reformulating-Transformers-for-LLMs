#!/usr/bin/env python3
"""
ΨQRH Language Model Generation and Testing

Test script for ΨQRH autoregressive generation with sampling controls.
Validates perplexity, generation quality, and physical properties.
"""

import torch
import torch.nn as nn
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

from psiqrh_llm import PsiQRHConfig, PsiQRHForCausalLM
from psiqrh_tokenizer import create_psiqrh_tokenizer


def load_model(
    model_path: Optional[str] = None,
    vocab_size: int = 50257
) -> Tuple[PsiQRHForCausalLM, PsiQRHTokenizer, torch.device]:
    """Load ΨQRH model and tokenizer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer
    tokenizer = create_psiqrh_tokenizer(vocab_size=vocab_size)

    # Initialize model
    config = PsiQRHConfig(vocab_size=vocab_size)
    model = PsiQRHForCausalLM(config).to(device)

    # Load checkpoint if provided
    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print("Using randomly initialized model")

    model.eval()
    return model, tokenizer, device


def compute_perplexity(
    model: PsiQRHForCausalLM,
    tokenizer: PsiQRHTokenizer,
    text: str,
    device: torch.device
) -> float:
    """Compute perplexity on given text"""
    model.eval()

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss

    perplexity = math.exp(loss.item())
    return perplexity


def generate_text(
    model: PsiQRHForCausalLM,
    tokenizer: PsiQRHTokenizer,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    device: torch.device = None
) -> str:
    """Generate text with ΨQRH model"""
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Remove prompt from generated text
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    return generated_text


def test_generation_quality(
    model: PsiQRHForCausalLM,
    tokenizer: PsiQRHTokenizer,
    device: torch.device
) -> Dict[str, Any]:
    """Test generation quality with various prompts"""
    test_prompts = [
        "Quantum entanglement is",
        "The fundamental principle of",
        "In quantum mechanics,",
        "The Schrödinger equation describes",
        "Fractal dimensions measure",
        "Spectral filtering allows",
        "Consciousness emerges from"
    ]

    results = {}

    print("Testing generation quality...")

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")

        # Generate with different settings
        generations = {}

        # Greedy (temperature=0)
        greedy_text = generate_text(
            model, tokenizer, prompt,
            max_length=30, temperature=0.0, do_sample=False
        )
        generations['greedy'] = greedy_text
        print(f"Greedy: '{greedy_text}'")

        # Sampling with temperature
        for temp in [0.7, 1.0, 1.5]:
            sampled_text = generate_text(
                model, tokenizer, prompt,
                max_length=30, temperature=temp, top_k=40, top_p=0.9
            )
            generations[f'temp_{temp}'] = sampled_text
            print(f"Temp {temp}: '{sampled_text}'")

        results[prompt] = generations

    return results


def test_perplexity(
    model: PsiQRHForCausalLM,
    tokenizer: PsiQRHTokenizer,
    device: torch.device
) -> Dict[str, float]:
    """Test perplexity on various text samples"""
    test_texts = [
        "Quantum entanglement is a fundamental property of quantum mechanics.",
        "The Schrödinger equation describes how quantum states evolve over time.",
        "Fractal dimensions provide a measure of complexity in geometric structures.",
        "Spectral filtering is used to analyze frequency components of signals.",
        "Consciousness emerges from complex interactions in neural networks."
    ]

    perplexities = {}

    print("\nTesting perplexity...")

    for text in test_texts:
        ppl = compute_perplexity(model, tokenizer, text, device)
        perplexities[text] = ppl
        print(".2f")

    avg_ppl = sum(perplexities.values()) / len(perplexities)
    perplexities['average'] = avg_ppl

    print(".2f")

    return perplexities


def test_physical_properties(
    model: PsiQRHForCausalLM,
    tokenizer: PsiQRHTokenizer,
    device: torch.device
) -> Dict[str, Any]:
    """Test physical properties of the model"""
    physics_results = {}

    # Test energy conservation
    test_text = "Quantum mechanics describes the behavior of matter and energy."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        # Energy conservation: ||logits|| should be reasonable
        logits_norm = torch.norm(outputs.logits).item()
        input_norm = torch.norm(inputs['input_ids'].float()).item()

        physics_results['energy_conservation_ratio'] = logits_norm / input_norm

        # Check for NaN/inf values
        has_nan = torch.isnan(outputs.logits).any().item()
        has_inf = torch.isinf(outputs.logits).any().item()

        physics_results['has_nan'] = has_nan
        physics_results['has_inf'] = has_inf

        # Test attention unitarity (approximate)
        if outputs.attentions and len(outputs.attentions) > 0:
            attn = outputs.attentions[0]  # First layer attention
            attn_sum = attn.sum(dim=-1)  # Should sum to 1
            unitarity_error = torch.mean(torch.abs(attn_sum - 1.0)).item()
            physics_results['attention_unitarity_error'] = unitarity_error
        else:
            physics_results['attention_unitarity_error'] = None

    print("\nPhysical Properties Test:")
    print(f"Energy conservation ratio: {physics_results['energy_conservation_ratio']:.3f}")
    print(f"Has NaN values: {physics_results['has_nan']}")
    print(f"Has Inf values: {physics_results['has_inf']}")
    if physics_results['attention_unitarity_error'] is not None:
        print(f"Attention unitarity error: {physics_results['attention_unitarity_error']:.6f}")

    return physics_results


def benchmark_generation_speed(
    model: PsiQRHForCausalLM,
    tokenizer: PsiQRHTokenizer,
    device: torch.device
) -> Dict[str, float]:
    """Benchmark generation speed"""
    prompt = "Quantum entanglement"
    num_runs = 10
    max_length = 50

    print(f"\nBenchmarking generation speed ({num_runs} runs)...")

    times = []

    for i in range(num_runs):
        start_time = time.time()

        generate_text(
            model, tokenizer, prompt,
            max_length=max_length,
            temperature=1.0, top_k=40
        )

        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    tokens_per_sec = max_length / avg_time

    results = {
        'avg_generation_time': avg_time,
        'tokens_per_second': tokens_per_sec,
        'total_runs': num_runs
    }

    print(".3f")
    print(".1f")

    return results


def run_comprehensive_test(
    model_path: Optional[str] = None,
    output_file: str = "psiqrh_test_results.json"
) -> Dict[str, Any]:
    """Run comprehensive testing suite"""
    print("ΨQRH Language Model - Comprehensive Testing Suite")
    print("=" * 60)

    # Load model
    model, tokenizer, device = load_model(model_path)

    # Run all tests
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': model_path or 'random_init',
        'device': str(device)
    }

    # 1. Generation quality test
    results['generation_quality'] = test_generation_quality(model, tokenizer, device)

    # 2. Perplexity test
    results['perplexity'] = test_perplexity(model, tokenizer, device)

    # 3. Physical properties test
    results['physical_properties'] = test_physical_properties(model, tokenizer, device)

    # 4. Speed benchmark
    results['benchmark'] = benchmark_generation_speed(model, tokenizer, device)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_file}")

    # Summary
    avg_ppl = results['perplexity']['average']
    print("\nSUMMARY:")
    print(".2f")
    print(f"Generation quality: {len(results['generation_quality'])} prompts tested")
    print(f"Physical properties: {'PASS' if not results['physical_properties']['has_nan'] and not results['physical_properties']['has_inf'] else 'ISSUES'}")
    print(".1f")

    return results


def interactive_generation():
    """Interactive generation mode"""
    print("ΨQRH Interactive Generation Mode")
    print("Type 'quit' to exit")
    print("-" * 40)

    model, tokenizer, device = load_model()

    while True:
        prompt = input("\nEnter prompt: ").strip()

        if prompt.lower() in ['quit', 'exit', 'q']:
            break

        if not prompt:
            continue

        # Generate with different settings
        print("\nGenerating...")

        # Greedy
        greedy = generate_text(model, tokenizer, prompt, temperature=0.0, do_sample=False)
        print(f"Greedy: {greedy}")

        # Creative
        creative = generate_text(model, tokenizer, prompt, temperature=1.2, top_k=40, top_p=0.9)
        print(f"Creative: {creative}")

        # Focused
        focused = generate_text(model, tokenizer, prompt, temperature=0.7, top_k=20, top_p=0.8)
        print(f"Focused: {focused}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ΨQRH Language Model Testing")
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--output', type=str, default='psiqrh_test_results.json', help='Output file for results')

    args = parser.parse_args()

    if args.interactive:
        interactive_generation()
    else:
        run_comprehensive_test(args.model_path, args.output)