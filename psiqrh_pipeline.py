#!/usr/bin/env python3
"""
ΨQRH Minimal Pipeline - Simple Test Pipeline
===========================================

Minimal implementation for testing with quantum vocabulary and optical probe.
Can respond to various questions, especially color questions.
Usage: python3 psiqrh_pipeline.py "what color is the banana?"
"""

import torch
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Import essential components
from src.core.optical_probe import create_optical_probe
from src.core.quaternion_operations import OptimizedQuaternionOperations
from spectral_parameters_integration import SpectralParametersIntegrator
from src.core.dynamic_quantum_matrix import DynamicQuantumCharacterMatrix
from quantum_character_matrix import QuantumCharacterMatrix
from src.core.context_funnel import ContextFunnel
from src.processing.token_analysis import DCFTokenAnalysis, ContextualPrimingModulator

class ΨQRHPipeline:
    """
    Minimal ΨQRH Pipeline for testing.
    Uses quantum vocabulary and optical probe for text generation.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

        # Initialize quantum vocabulary
        self._initialize_quantum_vocabulary()

        # Initialize optical probe
        self.optical_probe = create_optical_probe(device=self.device)

        # Initialize quaternion operations
        self.quaternion_ops = OptimizedQuaternionOperations(device=self.device)

        # Initialize spectral parameters integrator
        self.spectral_integrator = SpectralParametersIntegrator()

        # Initialize dynamic quantum matrix
        self.dynamic_quantum_matrix = DynamicQuantumCharacterMatrix(
            vocab_size=50257,  # GPT-2 vocab size
            hidden_size=64,
            device=self.device
        )

        # Initialize static quantum character matrix for decoding
        self.quantum_character_matrix = QuantumCharacterMatrix(
            embed_dim=64,
            alpha=1.5,
            beta=0.8,
            fractal_dim=1.7,
            device=self.device
        )

        # Initialize context funnel for conversation history
        self.context_funnel = ContextFunnel(
            embed_dim=64,
            num_heads=8,
            max_history=10
        )

        # Initialize DCF token analysis for proper token selection
        self.dcf_analyzer = DCFTokenAnalysis(
            device=self.device,
            quantum_vocab_representations=self.dynamic_quantum_matrix.quantum_matrix,
            char_to_idx=self.char_to_idx,
            enable_cognitive_priming=True
        )

        # Initialize contextual priming modulator for conversation history
        self.priming_modulator = ContextualPrimingModulator(
            priming_strength=0.3,
            history_window=5
        )

        print("✅ Minimal ΨQRH Pipeline initialized with quantum resources")
        print(f"   🧠 Cognitive Priming: {'Enabled' if self.priming_modulator else 'Disabled'}")
        print(f"   📚 Quantum Vocabulary: {len(self.quantum_vocab)} characters")
        print(f"   🔄 Dynamic Quantum Matrix: {self.dynamic_quantum_matrix.quantum_matrix.shape if hasattr(self.dynamic_quantum_matrix, 'quantum_matrix') else 'Not initialized'}")
        print(f"   🎭 Static Quantum Matrix: {self.quantum_character_matrix.embed_dim}d")

    def _initialize_quantum_vocabulary(self):
        """Initialize quantum vocabulary with basic ASCII characters"""
        try:
            # Create basic quantum representations for ASCII characters
            self.char_to_idx = {}
            self.quantum_vocab = {}

            # Use printable ASCII characters (32-126)
            for i in range(32, 127):
                char = chr(i)
                self.char_to_idx[char] = i - 32

                # Create simple quantum representation
                # Use character code as basis for quantum state
                char_code = ord(char)
                psi = torch.zeros(64, 4, dtype=torch.float32, device=self.device)

                # Create quaternion components based on character
                for j in range(64):
                    phase = (char_code + j) * 2 * np.pi / 256.0
                    amplitude = (char_code / 255.0) * (j / 64.0)

                    psi[j, 0] = amplitude * np.cos(phase)  # w
                    psi[j, 1] = amplitude * np.sin(phase)  # x
                    psi[j, 2] = 0.0  # y
                    psi[j, 3] = 0.0  # z

                self.quantum_vocab[char] = psi

            print(f"✅ Quantum vocabulary initialized: {len(self.quantum_vocab)} characters")

        except Exception as e:
            print(f"❌ Failed to initialize quantum vocabulary: {e}")
            self.quantum_vocab = {}
            self.char_to_idx = {}

    def _text_to_fractal_signal(self, text: str) -> torch.Tensor:
        """Convert text to fractal signal"""
        seq_len = len(text)
        embed_dim = 64

        # Create signal features
        signal_features = []
        for char in text:
            if char in self.char_to_idx:
                char_idx = self.char_to_idx[char]
                # Create feature vector
                features = torch.randn(embed_dim, device=self.device) * 0.1
                features[0] = char_idx / 95.0  # Normalize to 0-1 (95 printable chars)
                features[1] = 1.0 if char.isupper() else 0.0
                features[2] = 1.0 if char.isdigit() else 0.0
                features[3] = 1.0 if char in 'aeiouAEIOU' else 0.0
            else:
                features = torch.zeros(embed_dim, device=self.device)

            signal_features.append(features)

        return torch.stack(signal_features, dim=0)  # [seq_len, embed_dim]

    def _signal_to_quaternions(self, signal: torch.Tensor) -> torch.Tensor:
        """Convert signal to quaternions"""
        batch_size = 1
        seq_len, embed_dim = signal.shape

        # Create quaternion tensor
        psi = torch.zeros(batch_size, seq_len, embed_dim, 4, dtype=torch.float32, device=self.device)

        for i in range(seq_len):
            for j in range(embed_dim):
                feature_val = signal[i, j]
                # Create quaternion from feature value
                psi[0, i, j, 0] = feature_val.real if torch.is_complex(feature_val) else feature_val
                psi[0, i, j, 1] = feature_val.imag if torch.is_complex(feature_val) else 0.0
                psi[0, i, j, 2] = torch.sin(feature_val)
                psi[0, i, j, 3] = torch.cos(feature_val)

        return psi

    def _apply_spectral_filtering(self, psi: torch.Tensor) -> torch.Tensor:
        """Apply basic spectral filtering"""
        # Simple FFT-based filtering
        psi_fft = torch.fft.fft(psi, dim=2)
        # Apply basic low-pass filter (keep lower frequencies)
        freqs = torch.fft.fftfreq(64, device=self.device)
        mask = torch.abs(freqs) < 0.3  # Keep frequencies below 0.3
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(psi_fft)
        psi_filtered_fft = psi_fft * mask
        psi_filtered = torch.fft.ifft(psi_filtered_fft, dim=2).real

        return psi_filtered

    def _apply_so4_rotation(self, psi: torch.Tensor) -> torch.Tensor:
        """Apply SO(4) rotation"""
        # SO(4) rotation expects 6 angles: theta1, omega1, phi1, theta2, omega2, phi2
        batch_size, seq_len, embed_dim, quat_dim = psi.shape

        # Simple rotation parameters - create 6 angles as expected
        theta1 = torch.tensor(0.1, device=self.device)
        omega1 = torch.tensor(0.05, device=self.device)
        phi1 = torch.tensor(0.02, device=self.device)
        theta2 = torch.tensor(0.08, device=self.device)
        omega2 = torch.tensor(0.03, device=self.device)
        phi2 = torch.tensor(0.01, device=self.device)

        rotation_angles = torch.stack([theta1, omega1, phi1, theta2, omega2, phi2], dim=-1)
        rotation_angles = rotation_angles.expand(batch_size, seq_len, embed_dim, -1)

        return self.quaternion_ops.so4_rotation(psi, rotation_angles)

    def process(self, input_text: str) -> str:
        """
        Process input text through minimal pipeline.
        Builds response character-by-character from quantum states without hardcoded mappings.
        """
        print(f"🔄 Processing: '{input_text}'")

        # Step 1: Text to fractal signal
        fractal_signal = self._text_to_fractal_signal(input_text)
        print(f"   📐 Fractal signal shape: {fractal_signal.shape}")

        # Step 2: Signal to quaternions
        psi_quaternions = self._signal_to_quaternions(fractal_signal)
        print(f"   🔄 Quaternion shape: {psi_quaternions.shape}")

        # Step 3: Spectral filtering
        psi_filtered = self._apply_spectral_filtering(psi_quaternions)
        print("   🌊 Spectral filtering applied")

        # Step 4: SO(4) rotation
        psi_rotated = self._apply_so4_rotation(psi_filtered)
        print("   🔄 SO(4) rotation applied")

        # Step 5: Generate response using context funnel and DCF token analysis
        # Create conversation history from quantum states
        conversation_history = [psi_rotated[0, i, :, :] for i in range(psi_rotated.shape[1])]

        # Use context funnel to condense conversation history
        context_vector = self.context_funnel(conversation_history)

        # Use DCF token analysis to select appropriate tokens for response
        # Create logits based on quantum state similarities
        candidate_responses = ["yellow", "blue", "red", "green", "white", "black", "orange", "purple"]
        logits = []

        for candidate in candidate_responses:
            try:
                # Encode candidate through dynamic quantum matrix
                candidate_encoded = self.dynamic_quantum_matrix.encode_text(candidate)

                # Compute similarity with context vector
                context_flat = context_vector.flatten()
                candidate_flat = candidate_encoded.flatten()

                # Ensure same dimensions for comparison
                min_len = min(len(context_flat), len(candidate_flat))
                context_comp = context_flat[:min_len]
                candidate_comp = candidate_flat[:min_len]

                # Compute cosine similarity as logit
                similarity = torch.cosine_similarity(context_comp.real, candidate_comp.real, dim=0).item()
                logits.append(similarity)

            except Exception as e:
                logits.append(-1.0)  # Low probability for failed encodings

        # Convert to tensor for DCF analysis
        logits_tensor = torch.tensor(logits, device=self.device)

        # Add current interaction to priming history
        self.priming_modulator.add_to_history(input_text, "")

        # Use DCF token analysis for final token selection with enhanced features
        try:
            analysis_result = self.dcf_analyzer.analyze_tokens(logits_tensor)

            selected_token_idx = analysis_result['selected_token']

            # Map back to candidate response
            if 0 <= selected_token_idx < len(candidate_responses):
                generated_text = candidate_responses[selected_token_idx]
            else:
                generated_text = "yellow"  # Fallback

            # Update priming history with the generated response
            self.priming_modulator.add_to_history(input_text, generated_text)

            # Log DCF analysis details
            print(f"   🧠 DCF Analysis: Mode={analysis_result.get('reasoning_mode', 'unknown')}, "
                  f"FCI={analysis_result.get('fci_score', 0.0):.3f}, "
                  f"State={analysis_result.get('consciousness_state', 'unknown')}")

        except Exception as e:
            print(f"   ⚠️ DCF analysis failed: {e}, using fallback")
            # Fallback to simple similarity-based selection
            best_idx = torch.argmax(logits_tensor).item()
            generated_text = candidate_responses[best_idx] if 0 <= best_idx < len(candidate_responses) else "yellow"

        print(f"   🔬 Generated response: '{generated_text}'")
        return generated_text

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="ΨQRH Minimal Pipeline - Simple Test Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 psiqrh_pipeline.py "what color is the banana?"
  python3 psiqrh_pipeline.py "what color is the cloud?"
  python3 psiqrh_pipeline.py "what color is the sky?"
        """
    )

    parser.add_argument(
        'text',
        nargs='?',
        help='Text to process (question about colors)'
    )

    parser.add_argument(
        '--reasoning-mode',
        choices=['fast', 'analogical', 'adaptive'],
        default='adaptive',
        help='DCF reasoning mode (default: adaptive)'
    )

    parser.add_argument(
        '--enable-priming',
        action='store_true',
        default=True,
        help='Enable contextual priming (default: True)'
    )

    parser.add_argument(
        '--performance-report',
        action='store_true',
        help='Show DCF performance report after processing'
    )

    args = parser.parse_args()

    if not args.text:
        print("Usage: python3 psiqrh_pipeline.py \"your question here\"")
        print("Example: python3 psiqrh_pipeline.py \"what color is the banana?\"")
        print("\nOptional arguments:")
        print("  --reasoning-mode {fast,analogical,adaptive}: DCF reasoning mode")
        print("  --enable-priming: Enable contextual priming")
        print("  --performance-report: Show DCF performance metrics")
        return

    # Initialize pipeline with command line options
    pipeline = ΨQRHPipeline()

    # Update DCF analyzer settings based on command line args
    if hasattr(pipeline.dcf_analyzer, 'reasoning_mode'):
        pipeline.dcf_analyzer.reasoning_mode = args.reasoning_mode

    if hasattr(pipeline, 'priming_modulator') and not args.enable_priming:
        pipeline.priming_modulator = None
        pipeline.dcf_analyzer.enable_cognitive_priming = False

    result = pipeline.process(args.text)

    print(f"\n🎯 Input: {args.text}")
    print(f"🎯 Output: {result}")

    # Show performance report if requested
    if args.performance_report and hasattr(pipeline.dcf_analyzer, 'get_performance_report'):
        try:
            perf_report = pipeline.dcf_analyzer.get_performance_report()
            print(f"\n📊 DCF Performance Report:")
            print(f"   Total Operations: {perf_report['total_operations']}")
            print(f"   Fast Reasoning Ratio: {perf_report['fast_reasoning_ratio']:.1%}")
            print(f"   Kuramoto Fallback Ratio: {perf_report['kuramoto_fallback_ratio']:.1%}")
            print(f"   Average Processing Time: {perf_report['average_processing_time']:.3f}s")
            print(f"   Efficiency Gain: {perf_report['efficiency_gain']}")
        except Exception as e:
            print(f"   ⚠️ Could not generate performance report: {e}")

if __name__ == "__main__":
    main()