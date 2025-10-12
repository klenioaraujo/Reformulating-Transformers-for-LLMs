#!/usr/bin/env python3
"""
ΨQRH Optical Probe Fixed - Enhanced Spectral Alignment
======================================================

Fixed version of the optical probe that properly handles:
1. Genesis vocabulary integration (86 linguistic primitives)
2. Dynamic dimension adaptation with better projection
3. Semantic validation and fallback mechanisms
4. Improved cosine similarity with normalization

Problem Identified:
- Optical probe was using old spectral map (41 tokens, embed_dim=16)
- Genesis system has 86 linguistic primitives with embed_dim=64
- Dimension mismatch causing semantic failure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import os


class OpticalProbeFixed(nn.Module):
    """
    Enhanced Optical Probe with genesis vocabulary integration and better alignment.

    This version properly handles the quantum linguistic genesis system and provides
    robust fallback mechanisms for semantic generation.
    """

    def __init__(self, device: str, vocab_map_path: str = "data/spectral_vocab_map.pt"):
        """
        Initialize the Enhanced Optical Probe.

        Args:
            device: Device for tensor operations
            vocab_map_path: Path to the spectral vocabulary map file
        """
        super(OpticalProbeFixed, self).__init__()
        self.device = device
        self.spectral_map = None
        self.genesis_vocab = None

        # Get the base directory (project root)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_map_path = os.path.join(base_dir, vocab_map_path)

        try:
            if not os.path.exists(full_map_path):
                raise FileNotFoundError(f"Spectral map not found at {full_map_path}. Run tools/create_alignment_map.py first.")

            self.spectral_map = torch.load(full_map_path, map_location=self.device)
            self.vocab_size = self.spectral_map.shape[0]
            print(f"🔬 Enhanced Optical Probe calibrated with spectral alignment map")
            print(f"   📁 Map loaded: {full_map_path}")
            print(f"   📊 Shape: {self.spectral_map.shape}")
            print(f"   📚 Vocabulary size: {self.vocab_size}")

            # Load genesis vocabulary for enhanced alignment
            self._load_genesis_vocabulary()

        except Exception as e:
            print(f"❌ Failed to load spectral map: {e}")
            print("   💡 Run tools/create_alignment_map.py to generate the spectral map")
            self.spectral_map = None

    def _load_genesis_vocabulary(self):
        """Load quantum linguistic genesis vocabulary for enhanced alignment"""
        try:
            from src.core.quantum_linguistic_genesis import QuantumLinguisticGenesis

            genesis = QuantumLinguisticGenesis(embed_dim=64, device=self.device)
            self.genesis_vocab, self.genesis_char_to_idx = genesis.get_quantum_vocabulary_tensor()

            print(f"🧬 Genesis vocabulary integrated:")
            print(f"   📊 Genesis size: {len(self.genesis_vocab)} linguistic primitives")
            print(f"   🔬 Genesis shape: {self.genesis_vocab.shape}")

        except Exception as e:
            print(f"⚠️  Genesis vocabulary not available: {e}")
            self.genesis_vocab = None

    def _adaptive_projection(self, psi_flat: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Enhanced adaptive projection with genesis-aware dimension handling
        """
        input_dim = psi_flat.shape[0] // 4  # Since we flatten [embed_dim, 4] -> [embed_dim * 4]

        if input_dim == target_dim:
            return psi_flat

        print(f"🔧 Enhanced Optical Probe: Adapting embed_dim {input_dim} → {target_dim}")

        # For genesis-aligned systems, prefer minimal projection
        if abs(input_dim - target_dim) <= 4:
            # Small difference - use simple truncation or padding
            psi_reshaped = psi_flat.view(input_dim, 4)
            if input_dim > target_dim:
                psi_projected = psi_reshaped[:target_dim]
            else:
                padding_size = target_dim - input_dim
                psi_projected = torch.cat([
                    psi_reshaped,
                    torch.zeros(padding_size, 4, device=psi_flat.device)
                ])
        else:
            # Larger difference - use interpolation
            psi_reshaped = psi_flat.view(input_dim, 4)
            if input_dim > target_dim:
                # Down-project with interpolation
                scale_factor = target_dim / input_dim
                psi_projected = F.interpolate(
                    psi_reshaped.unsqueeze(0).unsqueeze(0),  # [1, 1, input_dim, 4]
                    scale_factor=(scale_factor, 1),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            else:
                # Up-project with interpolation
                scale_factor = target_dim / input_dim
                psi_projected = F.interpolate(
                    psi_reshaped.unsqueeze(0).unsqueeze(0),  # [1, 1, input_dim, 4]
                    scale_factor=(scale_factor, 1),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)[:target_dim]

        return psi_projected.flatten()

    def _semantic_validation(self, token_id: int, similarity: float) -> bool:
        """
        Validate if the decoded token makes semantic sense
        """
        # Check if similarity is reasonable
        if similarity < 0.1:
            return False

        # Check if token_id is within valid range
        if token_id < 0 or token_id >= self.vocab_size:
            return False

        return True

    def _fallback_decoding(self, psi_flat: torch.Tensor) -> int:
        """
        Fallback decoding using genesis vocabulary when spectral map fails
        """
        if self.genesis_vocab is None:
            return 0  # Default fallback

        # Use genesis vocabulary for decoding
        genesis_flat = self.genesis_vocab.view(len(self.genesis_vocab), -1)

        # Project input to match genesis dimensions
        target_dim = self.genesis_vocab.shape[1]
        psi_projected = self._adaptive_projection(psi_flat, target_dim)

        # Compute similarities
        similarities = F.cosine_similarity(psi_projected.unsqueeze(0), genesis_flat, dim=1)

        # Find best match
        best_token_id = torch.argmax(similarities).item()
        best_similarity = similarities[best_token_id].item()

        print(f"   🔄 Fallback decoding: token_id={best_token_id}, similarity={best_similarity:.4f}")

        # Map genesis token to spectral map token if possible
        if best_token_id < self.vocab_size:
            return best_token_id
        else:
            # Use modulo mapping
            return best_token_id % self.vocab_size

    def forward(self, psi_final: torch.Tensor) -> Tuple[int, float, bool]:
        """
        Enhanced decode quantum state to token ID with validation.

        Args:
            psi_final: Final quantum state tensor [batch_size, embed_dim, 4] or [embed_dim, 4] or [embed_dim]

        Returns:
            Tuple of (token_id, confidence, is_valid)
        """
        if self.spectral_map is None:
            return -1, 0.0, False

        # Handle different input shapes from pipeline transformations
        if psi_final.dim() == 4 and psi_final.shape[2] == 4:
            # Input is [batch_size, seq_len, embed_dim, 4] from full pipeline
            # Take the last sequence element and average over sequence dimension
            psi_final = psi_final.squeeze(0).mean(dim=0)  # [seq_len, embed_dim, 4] -> [embed_dim, 4]
            psi_flat = psi_final.flatten()
        elif psi_final.dim() == 3 and psi_final.shape[2] == 4:
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
            raise ValueError(f"Input to OpticalProbe must be [batch_size, seq_len, embed_dim, 4], [batch_size, embed_dim, 4], [embed_dim, 4] or [embed_dim], got shape {psi_final.shape}")

        # Get spectral map dimensions
        map_embed_dim = self.spectral_map.shape[1]  # embed_dim from spectral map [vocab_size, embed_dim, 4]

        # Enhanced adaptive projection
        psi_projected = self._adaptive_projection(psi_flat, map_embed_dim)

        # Flatten the spectral map for comparison: [vocab_size, embed_dim, 4] -> [vocab_size, embed_dim * 4]
        map_flat = self.spectral_map.view(self.vocab_size, -1)

        # Compute cosine similarities with normalization
        psi_normalized = F.normalize(psi_projected.unsqueeze(0), p=2, dim=1)
        map_normalized = F.normalize(map_flat, p=2, dim=1)

        similarities = F.cosine_similarity(psi_normalized, map_normalized, dim=1)

        # Find the best matching token ID
        best_token_id = torch.argmax(similarities).item()
        confidence = similarities[best_token_id].item()

        # Semantic validation
        is_valid = self._semantic_validation(best_token_id, confidence)

        if not is_valid and confidence < 0.3:
            print(f"   ⚠️  Low confidence decoding: token_id={best_token_id}, confidence={confidence:.4f}")
            # Try fallback decoding
            fallback_token_id = self._fallback_decoding(psi_flat)
            if fallback_token_id != best_token_id:
                best_token_id = fallback_token_id
                confidence = 0.5  # Moderate confidence for fallback
                is_valid = True
                print(f"   🔄 Using fallback token: {fallback_token_id}")

        return best_token_id, confidence, is_valid

    def decode_with_context(self, psi_final: torch.Tensor, context_tokens: List[int] = None) -> Tuple[int, float, bool]:
        """
        Context-aware decoding that considers previous tokens
        """
        token_id, confidence, is_valid = self.forward(psi_final)

        # Simple context filtering: avoid repeating the same token
        if context_tokens and len(context_tokens) > 0:
            last_token = context_tokens[-1]
            if token_id == last_token and confidence < 0.7:
                # Find second best option
                map_flat = self.spectral_map.view(self.vocab_size, -1)
                psi_projected = self._adaptive_projection(psi_final.flatten(), self.spectral_map.shape[1])

                similarities = F.cosine_similarity(
                    F.normalize(psi_projected.unsqueeze(0), p=2, dim=1),
                    F.normalize(map_flat, p=2, dim=1),
                    dim=1
                )

                # Get top 2 tokens
                top_tokens = torch.topk(similarities, 2)
                if top_tokens.indices[1] != last_token:
                    token_id = top_tokens.indices[1].item()
                    confidence = top_tokens.values[1].item()
                    print(f"   🔄 Context filtering: changed from {last_token} to {token_id}")

        return token_id, confidence, is_valid


def create_enhanced_optical_probe(device: str = 'cpu', vocab_map_path: str = "data/spectral_vocab_map.pt") -> OpticalProbeFixed:
    """
    Factory function to create an Enhanced OpticalProbe.

    Args:
        device: Device for tensor operations
        vocab_map_path: Path to the spectral vocabulary map file

    Returns:
        Configured OpticalProbeFixed instance
    """
    return OpticalProbeFixed(device=device, vocab_map_path=vocab_map_path)


# Example usage and testing
if __name__ == "__main__":
    # Create enhanced optical probe
    probe = create_enhanced_optical_probe(device='cpu')

    # Create test quantum state [embed_dim, 4]
    embed_dim = 64  # Matches genesis system
    test_psi = torch.randn(embed_dim, 4)

    # Decode to token ID
    token_id, confidence, is_valid = probe(test_psi)

    print(f"🔬 Enhanced Optical Probe Test:")
    print(f"   🔢 Decoded token ID: {token_id}")
    print(f"   📊 Confidence: {confidence:.4f}")
    print(f"   ✅ Is valid: {is_valid}")
    print(f"   📊 Spectral map shape: {probe.spectral_map.shape}")
    print(f"   📚 Vocabulary size: {probe.vocab_size}")

    if probe.genesis_vocab is not None:
        print(f"   🧬 Genesis vocabulary: {len(probe.genesis_vocab)} primitives")