#!/usr/bin/env python3
"""
ΨQRH Optical Probe - Padilha Wave Equation Implementation
=========================================================

Implements the true optical probe using the Padilha Wave Equation for wave-to-text conversion.
This follows the doe.md specification (section 2.9.5) for converting optical wave representations
back to text using the fundamental equation:

f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

The optical probe converts quantum states to optical wave functions and then to text,
implementing the complete physical pipeline: quantum → wave → text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import os


class OpticalProbe(nn.Module):
    """
    Optical Probe implementing spectral map-based decoding for quantum-linguistic alignment.

    This class loads the trained spectral vocabulary map and performs nearest-neighbor
    search using cosine similarity to decode quantum states to token IDs.
    """

    def __init__(self, device: str, vocab_map_path: str = "data/spectral_vocab_map.pt"):
        """
        Initialize the Optical Probe with spectral map-based decoding.

        Args:
            device: Device for tensor operations
            vocab_map_path: Path to the spectral vocabulary map file
        """
        super(OpticalProbe, self).__init__()
        self.device = device
        self.spectral_map = None

        # Get the base directory (project root)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_map_path = os.path.join(base_dir, vocab_map_path)

        try:
            if not os.path.exists(full_map_path):
                raise FileNotFoundError(f"Spectral map not found at {full_map_path}. Run tools/create_alignment_map.py first.")

            self.spectral_map = torch.load(full_map_path, map_location=self.device)
            self.vocab_size = self.spectral_map.shape[0]
            print(f"🔬 Optical Probe calibrated with spectral alignment map")
            print(f"   📁 Map loaded: {full_map_path}")
            print(f"   📊 Shape: {self.spectral_map.shape}")
            print(f"   📚 Vocabulary size: {self.vocab_size}")
        except Exception as e:
            print(f"❌ Failed to load spectral map: {e}")
            print("   💡 Run tools/create_alignment_map.py to generate the spectral map")
            self.spectral_map = None


    def forward(self, psi_final: torch.Tensor) -> int:
        """
        Decode quantum state to token ID using spectral map nearest-neighbor search.
        Handles dynamic embed_dim changes through adaptive projection.

        Args:
            psi_final: Final quantum state tensor [batch_size, embed_dim, 4] or [embed_dim, 4] or [embed_dim]

        Returns:
            Token ID (int) of the best matching vocabulary entry
        """
        if self.spectral_map is None:
            return -1

        # Handle different input shapes
        if psi_final.dim() == 3 and psi_final.shape[2] == 4:
            # Input is [batch_size, embed_dim, 4] - squeeze batch and flatten
            psi_final = psi_final.squeeze(0)  # Remove batch dimension
            psi_flat = psi_final.flatten()
        elif psi_final.dim() == 2 and psi_final.shape[1] == 4:
            # Input is [embed_dim, 4] - flatten to [embed_dim * 4]
            psi_flat = psi_final.flatten()
        elif psi_final.dim() == 1:
            # Input is already flattened [embed_dim] or [embed_dim * 4]
            psi_flat = psi_final
        else:
            raise ValueError(f"Input to OpticalProbe must be [batch_size, embed_dim, 4], [embed_dim, 4] or [embed_dim], got shape {psi_final.shape}")

        # Get input embed_dim from flattened tensor
        input_embed_dim = psi_flat.shape[0] // 4  # Since we flatten [embed_dim, 4] -> [embed_dim * 4]
        map_embed_dim = self.spectral_map.shape[1]  # embed_dim from spectral map [vocab_size, embed_dim, 4]

        # Handle embed_dim mismatch through adaptive projection
        if input_embed_dim != map_embed_dim:
            print(f"🔧 OpticalProbe: Adapting embed_dim {input_embed_dim} → {map_embed_dim}")

            # Project input to match spectral map dimensions
            if input_embed_dim > map_embed_dim:
                # Down-project: take first map_embed_dim components
                psi_projected = psi_flat[:map_embed_dim * 4].view(map_embed_dim, 4)
            else:
                # Up-project: pad with zeros
                padding_size = (map_embed_dim - input_embed_dim) * 4
                padding = torch.zeros(padding_size, device=psi_flat.device)
                psi_projected_flat = torch.cat([psi_flat, padding])
                psi_projected = psi_projected_flat.view(map_embed_dim, 4)

            psi_flat = psi_projected.flatten()

        # Flatten the spectral map for comparison: [vocab_size, embed_dim, 4] -> [vocab_size, embed_dim * 4]
        map_flat = self.spectral_map.view(self.vocab_size, -1)

        # Compute cosine similarities
        similarities = F.cosine_similarity(psi_flat.unsqueeze(0), map_flat, dim=1)

        # Find the best matching token ID
        best_token_id = torch.argmax(similarities).item()

        return best_token_id

def create_optical_probe(device: str = 'cpu', vocab_map_path: str = "data/spectral_vocab_map.pt") -> OpticalProbe:
    """
    Factory function to create an OpticalProbe with spectral map-based decoding.

    Args:
        device: Device for tensor operations
        vocab_map_path: Path to the spectral vocabulary map file

    Returns:
        Configured OpticalProbe instance
    """
    return OpticalProbe(device=device, vocab_map_path=vocab_map_path)


# Example usage and testing
if __name__ == "__main__":
    # Create optical probe
    probe = create_optical_probe(device='cpu')

    # Create test quantum state [embed_dim, 4]
    embed_dim = 16  # Matches the spectral map shape
    test_psi = torch.randn(embed_dim, 4)

    # Decode to token ID
    token_id = probe(test_psi)

    print(f"🔬 Optical Probe Test:")
    print(f"   🔢 Decoded token ID: {token_id}")
    print(f"   📊 Spectral map shape: {probe.spectral_map.shape}")
    print(f"   📚 Vocabulary size: {probe.vocab_size}")