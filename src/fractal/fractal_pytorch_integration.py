#!/usr/bin/env python3
"""
Enhanced Fractal-PyTorch Integration for ΨQRH Framework
========================================================

This module provides a comprehensive integration of fractal analysis with PyTorch
for hybrid testing of the ΨQRH transformer architecture. It extends the existing
work to support real-time fractal dimension calculation and adaptive neural layers.
"""

import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
import math
from scipy.optimize import curve_fit
import time

# Import base classes from existing modules
from ..core.ΨQRH import QuaternionOperations, SpectralFilter, QRHLayer
from .needle_fractal_dimension import FractalGenerator, LaserPulseSimulator

class AdaptiveFractalQRHLayer(nn.Module):
    """
    Enhanced QRH Layer with Real-Time Fractal Adaptation

    This layer dynamically adjusts its spectral filter parameters based on
    the fractal properties of the input data, enabling data-aware processing.
    """

    def __init__(self,
                 embed_dim: int,
                 alpha_range: Tuple[float, float] = (0.5, 2.5),
                 fractal_analysis_freq: int = 100,
                 enable_adaptive_alpha: bool = True,
                 use_learned_rotation: bool = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.total_dim = 4 * embed_dim
        self.alpha_range = alpha_range
        self.fractal_analysis_freq = fractal_analysis_freq
        self.enable_adaptive_alpha = enable_adaptive_alpha
        self.analysis_counter = 0

        # Adaptive alpha parameter
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # Core components
        self.spectral_filter = SpectralFilter(alpha=1.0)
        self.v_proj = nn.Linear(self.total_dim, self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, self.total_dim)

        # Rotation parameters (learnable)
        if use_learned_rotation:
            self.theta = nn.Parameter(torch.tensor(0.1))
            self.omega = nn.Parameter(torch.tensor(0.05))
            self.phi = nn.Parameter(torch.tensor(0.02))
        else:
            self.register_buffer('theta', torch.tensor(0.1))
            self.register_buffer('omega', torch.tensor(0.05))
            self.register_buffer('phi', torch.tensor(0.02))

        # Fractal analysis components
        self.fractal_buffer = []
        self.last_fractal_dim = 1.585  # Default Sierpinski dimension

        # FFT frequency buffer
        self.register_buffer('freqs', None)

    def analyze_input_fractality(self, x: torch.Tensor) -> float:
        """
        Analyze the fractal dimension of input tensor using 2D projections

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]

        Returns:
            Estimated fractal dimension
        """
        # Convert to numpy for fractal analysis
        x_np = x.detach().cpu().numpy()
        batch_size, seq_len, _ = x_np.shape

        # Use first two embedding dimensions as 2D projection
        if x_np.shape[-1] < 2:
            return self.last_fractal_dim

        # Aggregate across batch for stable analysis
        points_2d = x_np[:, :, :2].reshape(-1, 2)

        # Simple box-counting dimension calculation
        try:
            # Normalize points to [0, 1]
            min_vals = np.min(points_2d, axis=0)
            max_vals = np.max(points_2d, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0  # Avoid division by zero

            points_norm = (points_2d - min_vals) / range_vals

            # Box counting
            scales = np.logspace(-2, -0.3, 10)  # Reduced for speed
            counts = []

            for scale in scales:
                size = max(2, int(1/scale))
                grid = np.zeros((size, size), dtype=bool)
                indices = (points_norm * (size - 1)).astype(int)
                indices = np.clip(indices, 0, size-1)
                grid[indices[:, 0], indices[:, 1]] = True
                counts.append(np.sum(grid))

            # Linear fit in log space
            valid = np.array(counts) > 0
            if np.sum(valid) < 3:
                return self.last_fractal_dim

            log_scales = np.log(1/scales[valid])
            log_counts = np.log(np.array(counts)[valid])

            coeffs = np.polyfit(log_scales, log_counts, 1)
            fractal_dim = coeffs[0]

            # Clamp to reasonable range
            fractal_dim = np.clip(fractal_dim, 1.0, 2.0)

            return fractal_dim

        except Exception:
            return self.last_fractal_dim

    def update_alpha_from_fractality(self, fractal_dim: float):
        """Update alpha parameter based on fractal dimension"""
        if not self.enable_adaptive_alpha:
            return

        # Map fractal dimension [1.0, 2.0] to alpha range
        alpha_min, alpha_max = self.alpha_range
        normalized = (fractal_dim - 1.0) / (2.0 - 1.0)
        new_alpha = alpha_min + normalized * (alpha_max - alpha_min)

        # Update parameter with exponential moving average
        momentum = 0.9
        with torch.no_grad():
            self.alpha.data = momentum * self.alpha.data + (1 - momentum) * new_alpha

        # Update spectral filter
        self.spectral_filter.alpha = self.alpha.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive fractal analysis

        Args:
            x: Input tensor [batch_size, seq_len, 4*embed_dim]

        Returns:
            Processed tensor with same shape
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Periodic fractal analysis during training
        if self.training and self.enable_adaptive_alpha:
            self.analysis_counter += 1
            if self.analysis_counter % self.fractal_analysis_freq == 0:
                fractal_dim = self.analyze_input_fractality(x)
                self.update_alpha_from_fractality(fractal_dim)
                self.last_fractal_dim = fractal_dim

        # Standard QRH processing with adaptive alpha
        V = self.v_proj(x)

        # Divide into quaternion components
        D = self.embed_dim
        Ψ_w = V[:, :, 0*D:1*D]
        Ψ_i = V[:, :, 1*D:2*D]
        Ψ_j = V[:, :, 2*D:3*D]
        Ψ_k = V[:, :, 3*D:4*D]

        # Complex representation for spectral processing
        Ψ_complex = Ψ_w + 1j * Ψ_i

        # FFT
        Ψ_fft = fft.fft(Ψ_complex, dim=1)

        # Spectral filtering with adaptive alpha
        if self.freqs is None or self.freqs.size(0) != seq_len:
            self.freqs = fft.fftfreq(seq_len, d=1.0, device=device)

        k = 2 * math.pi * self.freqs.view(1, seq_len, 1).expand(batch_size, -1, D)

        # Use current alpha value
        k_mag = torch.abs(k) + 1e-10
        phase = self.alpha * torch.arctan(torch.log(k_mag))
        F_k = torch.exp(1j * phase)

        Ψ_filtered = Ψ_fft * F_k

        # Inverse FFT
        Ψ_ifft_complex = fft.ifft(Ψ_filtered, dim=1)

        # Update quaternion components
        Ψ_new_w = torch.real(Ψ_ifft_complex)
        Ψ_new_i = torch.imag(Ψ_ifft_complex)

        # Reconstruct quaternion tensor
        Ψ_new = torch.cat([Ψ_new_w, Ψ_new_i, Ψ_j, Ψ_k], dim=-1)

        # Quaternion rotation
        Ψ_reshaped = Ψ_new.view(batch_size, seq_len, D, 4)
        R = QuaternionOperations.create_unit_quaternion(self.theta, self.omega, self.phi)
        R_expanded = R.view(1, 1, 1, 4)
        rotated = QuaternionOperations.multiply(R_expanded, Ψ_reshaped)

        # Final output
        Ψ_final = rotated.view(batch_size, seq_len, self.total_dim)
        output = self.out_proj(Ψ_final) + x

        return output


class FractalTransformer(nn.Module):
    """
    Complete Transformer with Fractal-Adaptive QRH Layers

    This model demonstrates the full integration of fractal analysis
    with transformer architecture for hybrid testing.
    """

    def __init__(self,
                 vocab_size: int = 1000,
                 embed_dim: int = 64,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 seq_len: int = 128,
                 enable_fractal_adaptation: bool = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, 4 * embed_dim)
        self.position_embedding = nn.Embedding(seq_len, 4 * embed_dim)

        # Fractal-adaptive QRH layers
        self.layers = nn.ModuleList([
            AdaptiveFractalQRHLayer(
                embed_dim=embed_dim,
                enable_adaptive_alpha=enable_fractal_adaptation,
                fractal_analysis_freq=50 + i * 25  # Staggered analysis
            ) for i in range(num_layers)
        ])

        # Output projection
        self.ln_final = nn.LayerNorm(4 * embed_dim)
        self.output_proj = nn.Linear(4 * embed_dim, vocab_size)

        # Fractal monitoring
        self.fractal_history = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fractal-adaptive transformer

        Args:
            input_ids: Token indices [batch_size, seq_len]

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Track fractal properties through layers
        layer_fractals = []

        # Process through fractal-adaptive layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if hasattr(layer, 'last_fractal_dim'):
                layer_fractals.append(layer.last_fractal_dim)

        # Store fractal history for analysis
        if layer_fractals:
            self.fractal_history.append(layer_fractals)

        # Final processing
        x = self.ln_final(x)
        logits = self.output_proj(x)

        return logits

    def forward_pre_norm(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        A modified forward pass that returns the tensor just before the final LayerNorm.
        This is used to extract the true, un-normalized output for analysis.

        Args:
            input_ids: Token indices [batch_size, seq_len]

        Returns:
            The pre-normalization tensor [batch_size, seq_len, 4 * embed_dim]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Process through fractal-adaptive layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

        # Return the tensor BEFORE the final normalization
        return x

    def forward_first_layer_output(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Processes the input through the embedding and ONLY the first QRH layer.
        This provides the most meaningful signal for structural analysis.

        Args:
            input_ids: Token indices [batch_size, seq_len]

        Returns:
            The tensor after processing by the first layer.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Process through the FIRST layer only
        if self.layers:
            x = self.layers[0](x)

        return x

    def get_fractal_analysis(self) -> Dict:
        """Return comprehensive fractal analysis of the model"""
        if not self.fractal_history:
            return {"status": "No fractal data available"}

        recent_fractals = np.array(self.fractal_history[-10:])  # Last 10 steps

        return {
            "mean_fractal_dim": np.mean(recent_fractals),
            "std_fractal_dim": np.std(recent_fractals),
            "fractal_evolution": recent_fractals.tolist(),
            "layer_wise_fractals": np.mean(recent_fractals, axis=0).tolist() if len(recent_fractals) > 0 else []
        }


class QuartzLightSimulator:
    """
    Conceptual Quartz-Light System Simulator

    Simulates optical processing using quartz crystal properties
    and laser pulse interactions for physical AGI validation.
    """

    def __init__(self,
                 crystal_size: Tuple[float, float, float] = (10e-3, 10e-3, 2e-3),  # 10x10x2 mm
                 refractive_index: float = 1.544,  # α-quartz at 589nm
                 laser_wavelength: float = 1064e-9,  # Nd:YAG laser
                 pulse_duration: float = 10e-12):  # 10 ps

        self.crystal_size = crystal_size
        self.n = refractive_index
        self.wavelength = laser_wavelength
        self.pulse_duration = pulse_duration

        # Derived properties
        self.k0 = 2 * np.pi / laser_wavelength
        self.omega = 2 * np.pi * 3e8 / laser_wavelength

    def calculate_optical_path_length(self, fractal_structure: np.ndarray) -> float:
        """
        Calculate optical path length through fractal-structured quartz

        Args:
            fractal_structure: 3D array representing fractal density

        Returns:
            Effective optical path length
        """
        # Simplified model: path length depends on fractal density
        effective_thickness = np.mean(fractal_structure) * self.crystal_size[2]
        return self.n * effective_thickness

    def simulate_pulse_propagation(self,
                                 input_quaternion: torch.Tensor,
                                 fractal_dim: float) -> torch.Tensor:
        """
        Simulate laser pulse propagation through fractal-structured quartz

        Args:
            input_quaternion: Input quaternion state [4]
            fractal_dim: Fractal dimension affecting propagation

        Returns:
            Output quaternion after propagation
        """
        # Convert to numpy for optical simulation
        q_in = input_quaternion.detach().cpu().numpy()

        # Model nonlinear effects based on fractal dimension
        # Higher fractal dimension -> more complex light scattering
        nonlinearity = (fractal_dim - 1.0) / 1.0  # Normalize to [0, 1]

        # Simulate phase accumulation
        phase_factor = self.k0 * self.calculate_optical_path_length(
            np.random.random((10, 10, 10)) * nonlinearity
        )

        # Apply Jones matrix transformation (simplified)
        # Real implementation would use full electro-optic tensor
        rotation_angle = phase_factor * nonlinearity

        # Quaternion rotation representing optical transformation
        R = np.array([
            np.cos(rotation_angle/2),
            np.sin(rotation_angle/2) * 0.577,  # Along (1,1,1) axis
            np.sin(rotation_angle/2) * 0.577,
            np.sin(rotation_angle/2) * 0.577
        ])

        # Quaternion multiplication: q_out = R * q_in
        w1, x1, y1, z1 = R
        w2, x2, y2, z2 = q_in

        q_out = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

        return torch.tensor(q_out, dtype=input_quaternion.dtype, device=input_quaternion.device)

    def validate_agi_properties(self, model: FractalTransformer) -> Dict:
        """
        Validate physical AGI properties through quartz-light simulation

        Args:
            model: Trained fractal transformer model

        Returns:
            Validation metrics dictionary
        """
        validation_results = {
            "coherence_measure": 0.0,
            "information_density": 0.0,
            "fractal_stability": 0.0,
            "optical_efficiency": 0.0
        }

        # Generate test quaternions
        test_quaternions = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),  # Identity
            torch.tensor([0.707, 0.707, 0.0, 0.0]),  # 90° rotation
            torch.tensor([0.5, 0.5, 0.5, 0.5])  # Complex rotation
        ]

        coherence_scores = []
        efficiency_scores = []

        for q_in in test_quaternions:
            # Simulate optical processing
            fractal_dim = np.random.uniform(1.2, 1.8)  # Realistic range
            q_out = self.simulate_pulse_propagation(q_in, fractal_dim)

            # Measure coherence (quaternion norm preservation)
            norm_in = torch.norm(q_in)
            norm_out = torch.norm(q_out)
            coherence = 1.0 - abs(norm_in - norm_out) / norm_in
            coherence_scores.append(coherence.item())

            # Measure efficiency (information preservation)
            efficiency = torch.dot(q_in, q_out) / (norm_in * norm_out)
            efficiency_scores.append(abs(efficiency.item()))

        validation_results["coherence_measure"] = np.mean(coherence_scores)
        validation_results["optical_efficiency"] = np.mean(efficiency_scores)

        # Analyze model fractal properties
        fractal_analysis = model.get_fractal_analysis()
        if "mean_fractal_dim" in fractal_analysis:
            validation_results["fractal_stability"] = 1.0 / (1.0 + fractal_analysis["std_fractal_dim"])
            validation_results["information_density"] = fractal_analysis["mean_fractal_dim"] / 2.0

        return validation_results


def run_comprehensive_test():
    """
    Run comprehensive test of fractal-PyTorch integration
    """
    print("=== Comprehensive Fractal-PyTorch Integration Test ===\n")

    # Test parameters
    vocab_size = 1000
    embed_dim = 32
    seq_len = 64
    batch_size = 2
    num_layers = 3

    print(f"Model Configuration:")
    print(f"  Embedding Dimension: {embed_dim}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Number of Layers: {num_layers}")
    print(f"  Vocabulary Size: {vocab_size}")

    # Create fractal-adaptive transformer
    model = FractalTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        seq_len=seq_len,
        enable_fractal_adaptation=True
    )

    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    # Generate test data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Training simulation
    print("\n--- Training Simulation ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(20):
        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids)

        # Dummy loss (next token prediction)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), targets.view(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: Loss = {loss.item():.4f}")

    # Fractal analysis
    print("\n--- Fractal Analysis ---")
    fractal_analysis = model.get_fractal_analysis()
    print(f"  Mean Fractal Dimension: {fractal_analysis.get('mean_fractal_dim', 'N/A'):.4f}")
    print(f"  Fractal Stability (1/std): {fractal_analysis.get('std_fractal_dim', 'N/A'):.4f}")

    # Quartz-light validation
    print("\n--- Quartz-Light System Validation ---")
    quartz_sim = QuartzLightSimulator()
    validation_results = quartz_sim.validate_agi_properties(model)

    print(f"  Coherence Measure: {validation_results['coherence_measure']:.4f}")
    print(f"  Information Density: {validation_results['information_density']:.4f}")
    print(f"  Fractal Stability: {validation_results['fractal_stability']:.4f}")
    print(f"  Optical Efficiency: {validation_results['optical_efficiency']:.4f}")

    # Performance comparison
    print("\n--- Performance Comparison ---")
    model.eval()

    # Standard model without fractal adaptation
    standard_model = FractalTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        seq_len=seq_len,
        enable_fractal_adaptation=False
    )

    with torch.no_grad():
        # Timing comparison
        start_time = time.time()
        for _ in range(10):
            _ = model(input_ids)
        fractal_time = time.time() - start_time

        start_time = time.time()
        for _ in range(10):
            _ = standard_model(input_ids)
        standard_time = time.time() - start_time

        print(f"  Fractal-Adaptive Model: {fractal_time:.4f}s")
        print(f"  Standard Model: {standard_time:.4f}s")
        print(f"  Overhead: {((fractal_time - standard_time) / standard_time * 100):.1f}%")

    print("\n=== Test Complete ===")

    return {
        "model": model,
        "fractal_analysis": fractal_analysis,
        "validation_results": validation_results,
        "performance": {
            "fractal_time": fractal_time,
            "standard_time": standard_time
        }
    }


if __name__ == "__main__":
    # Run comprehensive test
    results = run_comprehensive_test()

    # Generate visualization
    if results["fractal_analysis"].get("fractal_evolution"):
        plt.figure(figsize=(12, 8))

        fractal_evolution = np.array(results["fractal_analysis"]["fractal_evolution"])

        plt.subplot(2, 2, 1)
        for i in range(fractal_evolution.shape[1]):
            plt.plot(fractal_evolution[:, i], label=f'Layer {i+1}')
        plt.title('Fractal Dimension Evolution by Layer')
        plt.xlabel('Training Step')
        plt.ylabel('Fractal Dimension')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        validation = results["validation_results"]
        metrics = ['coherence_measure', 'information_density', 'fractal_stability', 'optical_efficiency']
        values = [validation[m] for m in metrics]
        plt.bar(range(len(metrics)), values)
        plt.xticks(range(len(metrics)), [m.replace('_', ' ').title() for m in metrics], rotation=45)
        plt.title('Quartz-Light Validation Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)

        plt.subplot(2, 2, 3)
        perf = results["performance"]
        plt.bar(['Fractal-Adaptive', 'Standard'], [perf['fractal_time'], perf['standard_time']])
        plt.title('Performance Comparison')
        plt.ylabel('Time (seconds)')

        plt.subplot(2, 2, 4)
        # Theoretical fractal dimension ranges
        sierpinski_dim = np.log(3)/np.log(2)
        cantor_dim = np.log(2)/np.log(3)
        plt.axhline(sierpinski_dim, color='r', linestyle='--', label='Sierpinski Triangle')
        plt.axhline(cantor_dim, color='b', linestyle='--', label='Cantor Set')
        plt.axhline(1.5, color='g', linestyle='--', label='Optimal Range')

        if fractal_evolution.size > 0:
            plt.hist(fractal_evolution.flatten(), bins=20, alpha=0.7, density=True)
        plt.title('Fractal Dimension Distribution')
        plt.xlabel('Fractal Dimension')
        plt.ylabel('Density')
        plt.legend()

        plt.tight_layout()
        plt.savefig('/home/padilha/trabalhos/Reformulating Transformers/fractal_pytorch_integration_results.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'fractal_pytorch_integration_results.png'")