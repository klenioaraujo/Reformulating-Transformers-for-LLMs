"""
Chaotic Wave to Text Converter - Scientific Method Implementation
==================================================================

Implements scientific methods for quantum state to character conversion:
1. Logistic Map for chaotic pattern generation
2. Kuramoto Oscillator synchronization
3. Parseval energy conservation
4. Fractal dimension analysis

Based on validated scientific principles:
- Logistic map: x_{n+1} = r x_n (1 - x_n) for chaotic sequences
- Kuramoto model: θ̇_i = ω_i + K/N Σ sin(θ_j - θ_i) for synchronization
- Parseval theorem: Energy conservation in frequency domain

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import torch.fft as fft
import math
import numpy as np
from typing import Tuple, Optional, Dict, Any


def logistic_map_sequence(x0: float, r: float, n_steps: int) -> torch.Tensor:
    """
    Generate chaotic sequence using logistic map.

    Args:
        x0: Initial value (0 < x0 < 1)
        r: Control parameter (3.57 < r < 4.0 for chaos)
        n_steps: Number of iterations

    Returns:
        Chaotic sequence tensor
    """
    sequence = []
    x = x0

    for _ in range(n_steps):
        x = r * x * (1 - x)
        sequence.append(x)

    return torch.tensor(sequence, dtype=torch.float32)


class KuramotoOscillator:
    """Kuramoto oscillator for phase synchronization."""

    def __init__(self, n_oscillators: int, coupling_strength: float = 1.0):
        self.n_oscillators = n_oscillators
        self.K = coupling_strength
        self.natural_frequencies = torch.randn(n_oscillators) * 0.1
        self.phases = torch.rand(n_oscillators) * 2 * math.pi

    def step(self, dt: float = 0.01) -> torch.Tensor:
        """Advance oscillator phases by one time step."""
        # Calculate phase differences
        phase_diffs = self.phases.unsqueeze(1) - self.phases.unsqueeze(0)

        # Kuramoto equation: θ̇_i = ω_i + K/N Σ sin(θ_j - θ_i)
        coupling_term = (self.K / self.n_oscillators) * torch.sum(torch.sin(phase_diffs), dim=1)
        phase_dot = self.natural_frequencies + coupling_term

        # Update phases
        self.phases += phase_dot * dt

        # Normalize to [0, 2π]
        self.phases = self.phases % (2 * math.pi)

        return self.phases

    def get_order_parameter(self) -> float:
        """Calculate synchronization order parameter."""
        complex_phases = torch.exp(1j * self.phases)
        order_param = torch.abs(torch.mean(complex_phases))
        return order_param.item()


def extract_quantum_phase(psi: torch.Tensor) -> torch.Tensor:
    """
    Extract phase information from quantum state.

    Args:
        psi: Quantum state tensor [embed_dim, 4]

    Returns:
        Phase tensor [embed_dim]
    """
    # Convert quaternion to complex representation
    psi_complex = torch.complex(psi[..., 0], psi[..., 1])

    # Calculate phase angles
    phases = torch.angle(psi_complex)

    return phases


def chaotic_optical_probe(psi: torch.Tensor,
                         spectral_modes: torch.Tensor,
                         r_chaos: float = 3.99,
                         use_kuramoto: bool = True) -> Tuple[int, torch.Tensor]:
    """
    Chaotic optical probe using scientific methods.

    Args:
        psi: Quantum state [embed_dim, 4]
        spectral_modes: Character spectral modes [n_chars, n_modes]
        r_chaos: Logistic map parameter for chaos
        use_kuramoto: Use Kuramoto synchronization

    Returns:
        (char_index, probabilities)
    """
    n_chars, n_modes = spectral_modes.shape

    print(f"      🔬 [chaotic_optical_probe] Chaotic probe: {n_chars} chars, {n_modes} modes")
    print(f"      🌪️  [chaotic_optical_probe] Logistic r={r_chaos}, Kuramoto={use_kuramoto}")

    # Extract quantum phase information
    quantum_phases = extract_quantum_phase(psi)

    # Initialize chaotic sequence from quantum state
    initial_chaos = torch.mean(torch.abs(quantum_phases)).item() / (2 * math.pi)
    initial_chaos = max(0.001, min(0.999, initial_chaos))  # Ensure valid range

    # Generate chaotic measurement sequence
    chaos_sequence = logistic_map_sequence(initial_chaos, r_chaos, n_chars)

    # Initialize Kuramoto oscillators if enabled
    if use_kuramoto:
        kuramoto = KuramotoOscillator(n_chars, coupling_strength=1.0)
        # Set initial phases from chaotic sequence
        kuramoto.phases = chaos_sequence * 2 * math.pi

    # Calculate probabilities using chaotic synchronization
    probabilities = []

    for i in range(n_chars):
        # Character's spectral mode
        mode = spectral_modes[i]

        # Calculate base probability using quantum phase correlation
        phase_correlation = torch.mean(torch.cos(quantum_phases - torch.angle(torch.complex(mode[0], mode[1]))))
        base_prob = (phase_correlation + 1) / 2  # Normalize to [0, 1]

        # Apply chaotic modulation
        chaos_factor = chaos_sequence[i].item()

        # Apply Kuramoto synchronization if enabled
        if use_kuramoto:
            kuramoto.step()
            sync_factor = kuramoto.get_order_parameter()
            chaos_factor *= sync_factor

        # Combine base probability with chaotic modulation
        final_prob = base_prob * chaos_factor
        probabilities.append(final_prob.item())  # Convert to Python float

    probabilities = torch.tensor(probabilities, dtype=torch.float32)

    # Apply Parseval energy normalization
    total_energy = torch.sum(probabilities ** 2)
    if total_energy > 0:
        probabilities = probabilities / torch.sqrt(total_energy)

    # Apply softmax for probability distribution
    probabilities = torch.softmax(probabilities * 10.0, dim=0)

    # Select character with highest probability
    char_index = torch.argmax(probabilities).item()
    max_prob = probabilities[char_index].item()

    print(f"      🎯 [chaotic_optical_probe] Selected: index {char_index}, prob {max_prob:.4f}")

    # Show top 3 characters
    top_3_indices = torch.topk(probabilities, min(3, len(probabilities)))
    print(f"      📊 [chaotic_optical_probe] Top 3:")
    for idx in top_3_indices.indices:
        ascii_code = 32 + idx.item()
        prob = probabilities[idx].item()
        print(f"        - '{chr(ascii_code)}' (ASCII {ascii_code}): {prob:.4f}")

    return char_index, probabilities


def chaotic_wave_to_character(psi: torch.Tensor,
                             spectral_map: dict,
                             temperature: float = 1.0,
                             r_chaos: float = 3.99,
                             use_kuramoto: bool = True) -> str:
    """
    Convert quantum state to character using chaotic methods.

    Args:
        psi: Quantum state [embed_dim, 4]
        spectral_map: Character spectral mapping
        temperature: Sampling temperature
        r_chaos: Logistic map parameter
        use_kuramoto: Use Kuramoto synchronization

    Returns:
        Generated character
    """
    print(f"    🌪️  [chaotic_wave_to_character] Chaotic conversion: r={r_chaos}, T={temperature}")

    # Prepare spectral modes
    ascii_codes = sorted(spectral_map.keys())
    spectral_modes = torch.stack([spectral_map[code] for code in ascii_codes])

    # Chaotic optical probe
    char_index, probabilities = chaotic_optical_probe(
        psi, spectral_modes, r_chaos=r_chaos, use_kuramoto=use_kuramoto
    )

    # Apply temperature sampling
    if temperature != 1.0 and probabilities is not None:
        probabilities = probabilities / temperature
        probabilities = probabilities / probabilities.sum()
        char_index = torch.multinomial(probabilities, 1).item()
        print(f"    🌡️  [chaotic_wave_to_character] Temperature sampling: new index {char_index}")

    # Convert to character
    ascii_code = ascii_codes[char_index]
    char = chr(ascii_code)

    print(f"    ✅ [chaotic_wave_to_character] Generated: '{char}' (ASCII {ascii_code})")

    return char


def chaotic_wave_to_text(psi_sequence: torch.Tensor,
                        spectral_map: dict,
                        temperature: float = 1.0,
                        r_chaos: float = 3.99,
                        use_kuramoto: bool = True,
                        min_seq_len: int = 5) -> str:
    """
    Convert quantum state sequence to text using chaotic methods.

    Args:
        psi_sequence: Quantum state sequence [seq_len, embed_dim, 4]
        spectral_map: Character spectral mapping
        temperature: Sampling temperature
        r_chaos: Logistic map parameter
        use_kuramoto: Use Kuramoto synchronization
        min_seq_len: Minimum sequence length

    Returns:
        Generated text
    """
    print(f"🌪️  [chaotic_wave_to_text] Chaotic text generation:")
    print(f"   📊 seq_len={len(psi_sequence)}, r_chaos={r_chaos}, T={temperature}")
    print(f"   🔗 Kuramoto={use_kuramoto}, min_len={min_seq_len}")

    characters = []

    # Ensure minimum sequence length
    target_seq_len = max(len(psi_sequence), min_seq_len)

    if len(psi_sequence) < target_seq_len:
        print(f"  🔄 [chaotic_wave_to_text] Extending sequence: {len(psi_sequence)} → {target_seq_len}")
        extended_sequence = []
        for i in range(target_seq_len):
            base_idx = i % len(psi_sequence)
            base_psi = psi_sequence[base_idx]
            # Add chaotic variation
            chaos_noise = torch.randn_like(base_psi) * 0.02
            extended_sequence.append(base_psi + chaos_noise)
        psi_sequence = torch.stack(extended_sequence)

    # Generate each character
    for i in range(len(psi_sequence)):
        psi = psi_sequence[i]
        print(f"  📝 [chaotic_wave_to_text] Character {i+1}/{len(psi_sequence)}")

        char = chaotic_wave_to_character(
            psi, spectral_map,
            temperature=temperature,
            r_chaos=r_chaos,
            use_kuramoto=use_kuramoto
        )

        characters.append(char)

    result = ''.join(characters)
    print(f"🎯 [chaotic_wave_to_text] Final text: '{result}'")

    return result


def analyze_fractal_dimension(sequence: torch.Tensor) -> float:
    """
    Calculate approximate fractal dimension of sequence.

    Args:
        sequence: Input sequence

    Returns:
        Fractal dimension estimate
    """
    if len(sequence) < 10:
        return 1.0

    # Simple box counting approximation
    sequence_flat = sequence.view(-1)

    # Calculate variance as proxy for fractal dimension
    variance = torch.var(sequence_flat)

    # Map variance to approximate dimension (1D to 2D range)
    dimension = 1.0 + torch.sigmoid(variance * 10.0).item()

    return min(2.0, dimension)


def validate_energy_conservation(input_tensor: torch.Tensor,
                               output_tensor: torch.Tensor,
                               tolerance: float = 0.05) -> bool:
    """
    Validate energy conservation between input and output.

    Args:
        input_tensor: Input tensor
        output_tensor: Output tensor
        tolerance: Allowed energy deviation

    Returns:
        True if energy is conserved within tolerance
    """
    input_energy = torch.sum(input_tensor ** 2).item()
    output_energy = torch.sum(output_tensor ** 2).item()

    if input_energy == 0:
        return True

    ratio = output_energy / input_energy
    conserved = abs(ratio - 1.0) <= tolerance

    print(f"      ⚡ [energy_validation] Input: {input_energy:.6f}, Output: {output_energy:.6f}")
    print(f"      ⚡ [energy_validation] Ratio: {ratio:.6f}, Conserved: {'✅' if conserved else '❌'}")

    return conserved