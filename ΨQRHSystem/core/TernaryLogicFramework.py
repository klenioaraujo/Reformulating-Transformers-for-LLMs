#!/usr/bin/env python3
"""
Ternary Logic Framework for ΨQRH System
========================================

This module implements ternary logic operations to replace binary logic in the ΨQRH system.
Instead of binary (0,1), we use ternary states: (-1, 0, 1) representing:
- -1: False/Inactive
-  0: Neutral/Undefined
-  1: True/Active

This allows for more nuanced quantum-like processing with superposition states.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass


@dataclass
class TernaryState:
    """Represents a ternary quantum state"""
    value: int  # -1, 0, or 1
    confidence: float  # [0, 1]
    superposition: torch.Tensor  # Probability distribution over [-1, 0, 1]

    def __post_init__(self):
        if self.value not in [-1, 0, 1]:
            raise ValueError(f"Ternary value must be -1, 0, or 1, got {self.value}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


class TernaryLogicFramework:
    """
    Framework for ternary logic operations in ΨQRH system.

    Provides ternary equivalents of common logical operations and
    quantum-like processing capabilities.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

        # Ternary truth table for basic operations
        self.ternary_and_table = torch.tensor([
            [-1, -1, -1],  # -1 AND [-1, 0, 1]
            [-1,  0,  0],  #  0 AND [-1, 0, 1]
            [-1,  0,  1]   #  1 AND [-1, 0, 1]
        ], device=device)

        self.ternary_or_table = torch.tensor([
            [-1,  0,  1],  # -1 OR [-1, 0, 1]
            [ 0,  0,  1],  #  0 OR [-1, 0, 1]
            [ 1,  1,  1]   #  1 OR [-1, 0, 1]
        ], device=device)

        self.ternary_not_table = torch.tensor([-1, 0, 1], device=device)  # NOT: -1->1, 0->0, 1->-1

        print("🔺 Ternary Logic Framework initialized")

    def ternary_and(self, a: Union[int, torch.Tensor], b: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """Ternary AND operation"""
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            # Vectorized operation
            a_idx = (a + 1).long()  # Convert -1,0,1 to 0,1,2
            b_idx = (b + 1).long()
            return self.ternary_and_table[a_idx, b_idx]
        else:
            # Scalar operation
            a_idx = a + 1  # -1->0, 0->1, 1->2
            b_idx = b + 1
            return self.ternary_and_table[a_idx, b_idx].item()

    def ternary_or(self, a: Union[int, torch.Tensor], b: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """Ternary OR operation"""
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            a_idx = (a + 1).long()
            b_idx = (b + 1).long()
            return self.ternary_or_table[a_idx, b_idx]
        else:
            a_idx = a + 1
            b_idx = b + 1
            return self.ternary_or_table[a_idx, b_idx].item()

    def ternary_not(self, a: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """Ternary NOT operation"""
        if isinstance(a, torch.Tensor):
            a_idx = (a + 1).long()
            return self.ternary_not_table[a_idx]
        else:
            a_idx = a + 1
            return self.ternary_not_table[a_idx].item()

    def ternary_xor(self, a: Union[int, torch.Tensor], b: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """Ternary XOR operation (exclusive or with neutral state)"""
        # XOR: different values, neutral if both neutral
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            # For tensors, implement element-wise
            result = torch.zeros_like(a)
            mask_eq = (a == b)
            mask_neutral = (a == 0) & (b == 0)
            result[mask_eq & ~mask_neutral] = 0  # Same non-neutral values -> neutral
            result[mask_neutral] = 0  # Both neutral -> neutral
            result[~mask_eq] = torch.where(a == 0, b, torch.where(b == 0, a, -a))  # Different values
            return result
        else:
            if a == b:
                return 0 if a != 0 else 0  # Same values -> neutral
            elif a == 0:
                return b
            elif b == 0:
                return a
            else:
                return -a  # Different non-neutral values -> negation of first

    def ternary_to_binary(self, ternary: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """Convert ternary to binary (collapse superposition)"""
        if isinstance(ternary, torch.Tensor):
            # For tensors, convert based on magnitude
            return torch.where(ternary >= 0, 1, 0)
        else:
            return 1 if ternary >= 0 else 0

    def binary_to_ternary(self, binary: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """Convert binary to ternary"""
        if isinstance(binary, torch.Tensor):
            return torch.where(binary == 1, 1, -1)
        else:
            return 1 if binary == 1 else -1

    def create_superposition(self, probs: Optional[torch.Tensor] = None) -> TernaryState:
        """Create a ternary superposition state"""
        if probs is None:
            # Uniform superposition
            probs = torch.ones(3, device=self.device) / 3

        # Sample from distribution
        sampled_value = torch.multinomial(probs, 1).item() - 1  # 0,1,2 -> -1,0,1

        return TernaryState(
            value=sampled_value,
            confidence=probs[sampled_value + 1].item(),
            superposition=probs
        )

    def measure_state(self, state: TernaryState, collapse: bool = True) -> int:
        """Measure a ternary state (collapse superposition if requested)"""
        if not collapse:
            return state.value

        # Collapse based on probabilities
        sampled_idx = torch.multinomial(state.superposition, 1).item()
        return sampled_idx - 1  # 0,1,2 -> -1,0,1

    def ternary_majority_vote(self, states: List[Union[int, TernaryState]]) -> int:
        """Ternary majority vote operation"""
        values = []
        for state in states:
            if isinstance(state, TernaryState):
                values.append(state.value)
            else:
                values.append(state)

        # Count votes for each ternary value
        counts = { -1: 0, 0: 0, 1: 0 }
        for v in values:
            counts[v] += 1

        # Return value with most votes
        return max(counts, key=counts.get)

    def ternary_consensus(self, states: List[TernaryState], threshold: float = 0.7) -> Optional[int]:
        """Ternary consensus operation with confidence threshold"""
        if not states:
            return None

        # Calculate weighted consensus
        weighted_sum = 0
        total_weight = 0

        for state in states:
            weight = state.confidence
            weighted_sum += state.value * weight
            total_weight += weight

        if total_weight == 0:
            return 0

        consensus_value = weighted_sum / total_weight

        # Check if consensus meets threshold
        if abs(consensus_value) >= threshold:
            return 1 if consensus_value > 0 else -1
        else:
            return 0  # Neutral if no strong consensus


class TernaryQuantumEmbedding(nn.Module):
    """
    Ternary quantum embedding layer for ΨQRH system.

    Embeds input tokens into ternary quantum space instead of binary.
    """

    def __init__(self, vocab_size: int, embed_dim: int, ternary_logic: TernaryLogicFramework):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.ternary_logic = ternary_logic

        # Embedding layers for ternary components
        self.embedding_neg1 = nn.Embedding(vocab_size, embed_dim)  # For -1 states
        self.embedding_zero = nn.Embedding(vocab_size, embed_dim)   # For 0 states
        self.embedding_pos1 = nn.Embedding(vocab_size, embed_dim)   # For 1 states

        # Superposition weights
        self.superposition_weights = nn.Parameter(torch.randn(vocab_size, 3))  # Weights for [-1, 0, 1]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ternary embedding

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Ternary embeddings [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings for each ternary state
        emb_neg1 = self.embedding_neg1(input_ids)  # [batch, seq, embed]
        emb_zero = self.embedding_zero(input_ids)
        emb_pos1 = self.embedding_pos1(input_ids)

        # Get superposition probabilities
        superposition_logits = self.superposition_weights[input_ids]  # [batch, seq, 3]
        superposition_probs = torch.softmax(superposition_logits, dim=-1)

        # Create ternary embedding as weighted combination
        ternary_embedding = (
            superposition_probs[..., 0:1] * emb_neg1 +
            superposition_probs[..., 1:2] * emb_zero +
            superposition_probs[..., 2:3] * emb_pos1
        )

        return ternary_embedding


class TernaryValidationFramework:
    """
    Validation framework for ternary logic operations.

    Ensures ternary operations maintain physical and mathematical consistency.
    """

    def __init__(self, ternary_logic: TernaryLogicFramework):
        self.ternary_logic = ternary_logic

    def validate_ternary_operations(self) -> Dict[str, bool]:
        """Validate basic ternary operations"""
        results = {}

        # Test AND operation
        results['and_commutative'] = self._test_commutativity(self.ternary_logic.ternary_and)
        results['and_associative'] = self._test_associativity(self.ternary_logic.ternary_and)

        # Test OR operation
        results['or_commutative'] = self._test_commutativity(self.ternary_logic.ternary_or)
        results['or_associative'] = self._test_associativity(self.ternary_logic.ternary_or)

        # Test NOT operation
        results['not_involution'] = self._test_not_involution()

        # Test De Morgan's laws
        results['de_morgan'] = self._test_de_morgan()

        return results

    def _test_commutativity(self, operation) -> bool:
        """Test if operation is commutative"""
        test_values = [-1, 0, 1]
        for a in test_values:
            for b in test_values:
                if operation(a, b) != operation(b, a):
                    return False
        return True

    def _test_associativity(self, operation) -> bool:
        """Test if operation is associative"""
        test_values = [-1, 0, 1]
        for a in test_values:
            for b in test_values:
                for c in test_values:
                    if operation(operation(a, b), c) != operation(a, operation(b, c)):
                        return False
        return True

    def _test_not_involution(self) -> bool:
        """Test if NOT(NOT(x)) = x"""
        test_values = [-1, 0, 1]
        for x in test_values:
            if self.ternary_logic.ternary_not(self.ternary_logic.ternary_not(x)) != x:
                return False
        return True

    def _test_de_morgan(self) -> bool:
        """Test De Morgan's laws for ternary logic"""
        test_values = [-1, 0, 1]
        for a in test_values:
            for b in test_values:
                # NOT(a AND b) = NOT(a) OR NOT(b)
                lhs1 = self.ternary_logic.ternary_not(self.ternary_logic.ternary_and(a, b))
                rhs1 = self.ternary_logic.ternary_or(
                    self.ternary_logic.ternary_not(a),
                    self.ternary_logic.ternary_not(b)
                )
                if lhs1 != rhs1:
                    return False

                # NOT(a OR b) = NOT(a) AND NOT(b)
                lhs2 = self.ternary_logic.ternary_not(self.ternary_logic.ternary_or(a, b))
                rhs2 = self.ternary_logic.ternary_and(
                    self.ternary_logic.ternary_not(a),
                    self.ternary_logic.ternary_not(b)
                )
                if lhs2 != rhs2:
                    return False

        return True

    def validate_quantum_consistency(self, ternary_embedding: torch.Tensor) -> Dict[str, float]:
        """Validate quantum consistency of ternary embeddings"""
        # Check normalization
        norms = torch.norm(ternary_embedding, dim=-1)
        norm_std = torch.std(norms).item()

        # Check superposition properties
        mean_superposition = torch.mean(ternary_embedding, dim=(0, 1))
        superposition_entropy = -torch.sum(mean_superposition * torch.log(mean_superposition + 1e-10))

        return {
            'norm_std': norm_std,
            'superposition_entropy': superposition_entropy.item(),
            'is_normalized': norm_std < 0.1  # Low standard deviation indicates good normalization
        }