"""
Hierarchical Gate System for ΨQRH Transformer

Implements a multi-level gate mechanism for controlling information flow
and resonance patterns in the transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class ResonanceConfig:
    """Configuration for hierarchical gate system"""
    num_levels: int = 4
    gate_dim: int = 512
    resonance_threshold: float = 0.7
    coherence_threshold: float = 0.8
    gate_activation: str = "sigmoid"
    enable_adaptive_resonance: bool = True
    enable_coherence_analysis: bool = True
    enable_energy_balancing: bool = True


class HierarchicalGateSystem(nn.Module):
    """
    Hierarchical gate system that controls information flow through multiple levels
    of abstraction, enabling resonance and coherence analysis.
    """

    def __init__(self, config: ResonanceConfig):
        super().__init__()
        self.config = config

        # Level-specific gate controllers
        self.level_gates = nn.ModuleList([
            GateController(config.gate_dim, config.gate_activation)
            for _ in range(config.num_levels)
        ])

        # Resonance detectors for each level
        self.resonance_detectors = nn.ModuleList([
            ResonanceDetector(config.gate_dim)
            for _ in range(config.num_levels)
        ])

        # Coherence analyzer
        if config.enable_coherence_analysis:
            self.coherence_analyzer = CoherenceAnalyzer(config.gate_dim)

        # Energy balancer
        if config.enable_energy_balancing:
            self.energy_balancer = EnergyBalancer(config.gate_dim)

        # Cross-level attention for resonance propagation
        self.cross_level_attention = CrossLevelAttention(config.gate_dim)

        # Adaptive resonance controller
        if config.enable_adaptive_resonance:
            self.adaptive_resonance = AdaptiveResonanceController(config)

        # Statistics tracking
        self.resonance_stats = {
            'level_resonance': [0.0] * config.num_levels,
            'global_coherence': 0.0,
            'energy_balance': 1.0
        }

    def process_through_hierarchy(self,
                                 input_tensor: torch.Tensor,
                                 processed_tensor: torch.Tensor,
                                 rotation_params: Optional[Dict] = None) -> Dict:
        """
        Process tensor through hierarchical gate system.

        Args:
            input_tensor: Original input tensor
            processed_tensor: Tensor after ΨQRH processing
            rotation_params: Optional rotation parameters from QRH layer

        Returns:
            Dictionary containing processed output and analysis results
        """
        batch_size, seq_len, d_model = processed_tensor.shape

        results = {
            'level_outputs': [],
            'gate_activations': [],
            'resonance_scores': [],
            'coherence_metrics': {},
            'energy_metrics': {}
        }

        # Initialize level inputs
        current_input = processed_tensor

        # Process through each level
        for level in range(self.config.num_levels):
            # Apply level-specific gate
            level_output, gate_activation = self.level_gates[level](current_input)

            # Detect resonance at this level
            resonance_score = self.resonance_detectors[level](level_output)

            # Store results
            results['level_outputs'].append(level_output)
            results['gate_activations'].append(gate_activation)
            results['resonance_scores'].append(resonance_score)

            # Update current input for next level
            current_input = level_output

        # Apply cross-level attention for resonance propagation
        if len(results['level_outputs']) > 1:
            cross_level_output = self.cross_level_attention(results['level_outputs'])
            results['cross_level_output'] = cross_level_output
        else:
            results['cross_level_output'] = results['level_outputs'][0]

        # Analyze coherence across levels
        if self.config.enable_coherence_analysis:
            coherence_metrics = self.coherence_analyzer.analyze_coherence(
                results['level_outputs'], results['resonance_scores']
            )
            results['coherence_metrics'] = coherence_metrics

        # Balance energy across levels
        if self.config.enable_energy_balancing:
            balanced_output, energy_metrics = self.energy_balancer.balance_energy(
                results['level_outputs'], results['cross_level_output']
            )
            results['balanced_output'] = balanced_output
            results['energy_metrics'] = energy_metrics
        else:
            results['balanced_output'] = results['cross_level_output']

        # Apply adaptive resonance if enabled
        if self.config.enable_adaptive_resonance:
            adaptive_results = self.adaptive_resonance.adapt_resonance(
                results['balanced_output'],
                results['resonance_scores'],
                rotation_params
            )
            results.update(adaptive_results)

        # Update statistics
        self._update_resonance_stats(results)

        return results

    def _update_resonance_stats(self, results: Dict) -> None:
        """Update resonance statistics"""
        # Update level resonance
        for i, score in enumerate(results['resonance_scores']):
            if i < len(self.resonance_stats['level_resonance']):
                alpha = 0.1  # Smoothing factor
                self.resonance_stats['level_resonance'][i] = (
                    alpha * score.item() +
                    (1 - alpha) * self.resonance_stats['level_resonance'][i]
                )

        # Update global coherence
        if 'coherence_metrics' in results:
            coherence = results['coherence_metrics'].get('global_coherence', 0.0)
            alpha = 0.1
            self.resonance_stats['global_coherence'] = (
                alpha * coherence +
                (1 - alpha) * self.resonance_stats['global_coherence']
            )

        # Update energy balance
        if 'energy_metrics' in results:
            balance = results['energy_metrics'].get('balance_score', 1.0)
            alpha = 0.1
            self.resonance_stats['energy_balance'] = (
                alpha * balance +
                (1 - alpha) * self.resonance_stats['energy_balance']
            )

    def get_hierarchy_health_report(self, results: Dict) -> Dict:
        """Generate health report for the hierarchical gate system"""
        health_report = {
            'overall_hierarchy_health': 1.0,
            'level_health': [],
            'resonance_health': [],
            'coherence_health': 1.0,
            'energy_health': 1.0
        }

        # Analyze level health
        for i, (gate_act, resonance) in enumerate(zip(
            results['gate_activations'], results['resonance_scores']
        )):
            level_health = self._compute_level_health(gate_act, resonance)
            health_report['level_health'].append(level_health)

        # Analyze resonance health
        resonance_health = self._compute_resonance_health(results['resonance_scores'])
        health_report['resonance_health'] = resonance_health

        # Analyze coherence health
        if 'coherence_metrics' in results:
            coherence = results['coherence_metrics'].get('global_coherence', 0.0)
            health_report['coherence_health'] = coherence

        # Analyze energy health
        if 'energy_metrics' in results:
            balance = results['energy_metrics'].get('balance_score', 1.0)
            health_report['energy_health'] = balance

        # Compute overall health
        health_scores = [
            np.mean(health_report['level_health']) if health_report['level_health'] else 1.0,
            np.mean(health_report['resonance_health']) if health_report['resonance_health'] else 1.0,
            health_report['coherence_health'],
            health_report['energy_health']
        ]

        health_report['overall_hierarchy_health'] = np.mean(health_scores)

        return health_report

    def _compute_level_health(self, gate_activation: torch.Tensor, resonance: torch.Tensor) -> float:
        """Compute health score for a single level"""
        # Gate activation should be neither too low nor too high
        gate_mean = gate_activation.mean().item()
        gate_health = 1.0 - abs(gate_mean - 0.5) * 2.0  # Best at 0.5

        # Resonance should be above threshold
        resonance_health = min(1.0, resonance.item() / self.config.resonance_threshold)

        # Combined health score
        return 0.6 * gate_health + 0.4 * resonance_health

    def _compute_resonance_health(self, resonance_scores: List[torch.Tensor]) -> List[float]:
        """Compute health scores for resonance at each level"""
        health_scores = []
        for score in resonance_scores:
            # Resonance should be above threshold
            health = min(1.0, score.item() / self.config.resonance_threshold)
            health_scores.append(health)

        return health_scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward pass for integration with other modules.

        Args:
            x: Input tensor

        Returns:
            Processed tensor through hierarchy
        """
        results = self.process_through_hierarchy(x, x)
        return results.get('balanced_output', x)


class GateController(nn.Module):
    """Individual gate controller for a specific hierarchy level"""

    def __init__(self, gate_dim: int, activation: str = "sigmoid"):
        super().__init__()
        self.gate_dim = gate_dim

        # Gate computation
        self.gate_linear = nn.Linear(gate_dim, gate_dim)
        self.gate_norm = nn.LayerNorm(gate_dim)

        # Activation function
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unsupported gate activation: {activation}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gate to input tensor.

        Args:
            x: Input tensor

        Returns:
            Tuple of (gated_output, gate_activation)
        """
        # Compute gate activation
        gate_input = self.gate_norm(x)
        gate_logits = self.gate_linear(gate_input)
        gate_activation = self.activation(gate_logits)

        # Apply gate
        gated_output = x * gate_activation

        return gated_output, gate_activation


class ResonanceDetector(nn.Module):
    """Detects resonance patterns at a specific hierarchy level"""

    def __init__(self, gate_dim: int):
        super().__init__()
        self.gate_dim = gate_dim

        # Resonance computation
        self.resonance_linear = nn.Linear(gate_dim, 1)
        self.resonance_norm = nn.LayerNorm(gate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute resonance score for input tensor.

        Args:
            x: Input tensor

        Returns:
            Resonance score tensor
        """
        # Compute resonance features
        resonance_input = self.resonance_norm(x)
        resonance_features = self.resonance_linear(resonance_input)

        # Global resonance score (average across sequence)
        resonance_score = torch.sigmoid(resonance_features.mean())

        return resonance_score


class CoherenceAnalyzer(nn.Module):
    """Analyzes coherence across hierarchy levels"""

    def __init__(self, gate_dim: int):
        super().__init__()
        self.gate_dim = gate_dim

        # Coherence computation
        self.coherence_projection = nn.Linear(gate_dim, gate_dim // 4)

    def analyze_coherence(self,
                         level_outputs: List[torch.Tensor],
                         resonance_scores: List[torch.Tensor]) -> Dict:
        """
        Analyze coherence across hierarchy levels.

        Args:
            level_outputs: Outputs from each level
            resonance_scores: Resonance scores from each level

        Returns:
            Dictionary of coherence metrics
        """
        if len(level_outputs) < 2:
            return {'global_coherence': 1.0, 'level_coherence': []}

        metrics = {
            'global_coherence': 0.0,
            'level_coherence': [],
            'resonance_coherence': 0.0
        }

        # Compute pairwise coherence between levels
        coherence_scores = []
        for i in range(len(level_outputs)):
            for j in range(i + 1, len(level_outputs)):
                coherence = self._compute_pairwise_coherence(
                    level_outputs[i], level_outputs[j]
                )
                coherence_scores.append(coherence)

        # Global coherence (average of pairwise)
        if coherence_scores:
            metrics['global_coherence'] = torch.mean(torch.stack(coherence_scores)).item()

        # Level-specific coherence
        for i, output in enumerate(level_outputs):
            level_coherence = self._compute_self_coherence(output)
            metrics['level_coherence'].append(level_coherence.item())

        # Resonance coherence
        if resonance_scores:
            resonance_tensor = torch.stack(resonance_scores)
            metrics['resonance_coherence'] = 1.0 - torch.std(resonance_tensor).item()

        return metrics

    def _compute_pairwise_coherence(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute coherence between two tensors"""
        # Project to lower dimension
        x1_proj = self.coherence_projection(x1)
        x2_proj = self.coherence_projection(x2)

        # Compute cosine similarity
        similarity = F.cosine_similarity(x1_proj, x2_proj, dim=-1)
        return similarity.mean()

    def _compute_self_coherence(self, x: torch.Tensor) -> torch.Tensor:
        """Compute self-coherence of a tensor"""
        # Compute correlation between different positions
        x_normalized = F.normalize(x, p=2, dim=-1)
        correlation = torch.matmul(x_normalized, x_normalized.transpose(-2, -1))

        # Average correlation (excluding diagonal)
        batch_size, seq_len, _ = x.shape
        mask = 1 - torch.eye(seq_len, device=x.device)
        coherence = (correlation * mask.unsqueeze(0)).sum(dim=(-2, -1))
        coherence /= (seq_len * (seq_len - 1))

        return coherence.mean()


class EnergyBalancer(nn.Module):
    """Balances energy distribution across hierarchy levels"""

    def __init__(self, gate_dim: int):
        super().__init__()
        self.gate_dim = gate_dim

        # Energy balancing parameters
        self.target_energy_ratio = 1.0

    def balance_energy(self,
                      level_outputs: List[torch.Tensor],
                      cross_level_output: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Balance energy distribution across levels.

        Args:
            level_outputs: Outputs from each level
            cross_level_output: Combined output from cross-level attention

        Returns:
            Tuple of (balanced_output, energy_metrics)
        """
        # Compute energy at each level
        level_energies = [torch.norm(output, p=2).item() for output in level_outputs]
        total_energy = sum(level_energies)

        # Compute energy ratios
        if total_energy > 0:
            energy_ratios = [energy / total_energy for energy in level_energies]
        else:
            energy_ratios = [1.0 / len(level_energies)] * len(level_energies)

        # Compute balance score (closer to uniform is better)
        target_ratio = 1.0 / len(level_energies)
        balance_score = 1.0 - sum(abs(ratio - target_ratio) for ratio in energy_ratios) / 2.0

        # Apply energy balancing if needed
        if balance_score < 0.8:  # Threshold for balancing
            balanced_output = self._apply_energy_balancing(cross_level_output, energy_ratios)
        else:
            balanced_output = cross_level_output

        # Energy metrics
        energy_metrics = {
            'level_energies': level_energies,
            'energy_ratios': energy_ratios,
            'balance_score': balance_score,
            'total_energy': total_energy
        }

        return balanced_output, energy_metrics

    def _apply_energy_balancing(self, x: torch.Tensor, energy_ratios: List[float]) -> torch.Tensor:
        """Apply energy balancing to tensor"""
        target_ratio = 1.0 / len(energy_ratios)

        # Compute scaling factors
        scaling_factors = [target_ratio / ratio if ratio > 0 else 1.0 for ratio in energy_ratios]
        avg_scaling = sum(scaling_factors) / len(scaling_factors)

        # Apply scaling
        balanced_output = x * avg_scaling

        return balanced_output


class CrossLevelAttention(nn.Module):
    """Attention mechanism for cross-level resonance propagation"""

    def __init__(self, gate_dim: int):
        super().__init__()
        self.gate_dim = gate_dim

        # Cross-level attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=gate_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, level_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply cross-level attention.

        Args:
            level_outputs: List of outputs from different levels

        Returns:
            Combined output with cross-level attention
        """
        if len(level_outputs) == 1:
            return level_outputs[0]

        # Stack level outputs
        stacked_outputs = torch.stack(level_outputs, dim=1)  # [batch, levels, seq_len, dim]
        batch_size, num_levels, seq_len, dim = stacked_outputs.shape

        # Reshape for attention
        reshaped = stacked_outputs.view(batch_size, num_levels * seq_len, dim)

        # Apply self-attention across levels
        attended, _ = self.cross_attention(reshaped, reshaped, reshaped)

        # Reshape back and average across levels
        attended_reshaped = attended.view(batch_size, num_levels, seq_len, dim)
        combined = attended_reshaped.mean(dim=1)

        return combined


class AdaptiveResonanceController(nn.Module):
    """Adaptively controls resonance based on input characteristics"""

    def __init__(self, config: ResonanceConfig):
        super().__init__()
        self.config = config

        # Adaptive resonance parameters
        self.resonance_gain = nn.Parameter(torch.ones(1))
        self.coherence_weight = nn.Parameter(torch.ones(1))

    def adapt_resonance(self,
                       balanced_output: torch.Tensor,
                       resonance_scores: List[torch.Tensor],
                       rotation_params: Optional[Dict] = None) -> Dict:
        """
        Adapt resonance based on current state.

        Args:
            balanced_output: Balanced output from hierarchy
            resonance_scores: Resonance scores from each level
            rotation_params: Optional rotation parameters from QRH

        Returns:
            Dictionary with adapted output and control metrics
        """
        results = {
            'adapted_output': balanced_output,
            'resonance_gain': self.resonance_gain.item(),
            'coherence_weight': self.coherence_weight.item()
        }

        # Compute average resonance
        if resonance_scores:
            avg_resonance = torch.mean(torch.stack(resonance_scores)).item()
        else:
            avg_resonance = 0.0

        # Adapt resonance gain based on average resonance
        target_gain = 1.0 + (avg_resonance - self.config.resonance_threshold)
        target_gain = max(0.5, min(2.0, target_gain))

        # Smooth update of resonance gain
        with torch.no_grad():
            alpha = 0.1
            self.resonance_gain.data = (
                alpha * target_gain + (1 - alpha) * self.resonance_gain.data
            )

        # Apply resonance gain
        adapted_output = balanced_output * self.resonance_gain

        # Update coherence weight based on rotation parameters if available
        if rotation_params is not None:
            coherence_weight = self._compute_coherence_weight(rotation_params)
            with torch.no_grad():
                self.coherence_weight.data = (
                    alpha * coherence_weight + (1 - alpha) * self.coherence_weight.data
                )

        results['adapted_output'] = adapted_output
        results['avg_resonance'] = avg_resonance

        return results

    def _compute_coherence_weight(self, rotation_params: Dict) -> float:
        """Compute coherence weight based on rotation parameters"""
        # Extract rotation angles
        theta_left = rotation_params.get('theta_left', 0.0)
        omega_left = rotation_params.get('omega_left', 0.0)
        phi_left = rotation_params.get('phi_left', 0.0)

        # Compute rotation complexity
        rotation_complexity = (
            abs(theta_left) + abs(omega_left) + abs(phi_left)
        ) / 3.0

        # Higher complexity = higher coherence requirement
        coherence_weight = 1.0 + rotation_complexity

        return min(2.0, max(0.5, coherence_weight))