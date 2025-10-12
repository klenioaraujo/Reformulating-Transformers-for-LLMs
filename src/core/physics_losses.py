#!/usr/bin/env python3
"""
Physics-Based Loss Functions for ΨQRH Training
==============================================

Loss functions that optimize for physical dynamics and consciousness metrics
rather than simple output comparison.

Key Principles:
1. Optimize trajectory quality, not just final states
2. Maximize consciousness metrics (FCI, sync order, cluster coherence)
3. Penalize poor physical dynamics (instability, fragmentation)
4. Reward semantic achievement through physical reasoning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class SyncQualityLoss(nn.Module):
    """Loss function for Kuramoto synchronization quality."""

    def __init__(self, target_sync_order: float = 0.8, stability_weight: float = 0.3):
        super().__init__()
        self.target_sync_order = target_sync_order
        self.stability_weight = stability_weight

    def forward(self, kuramoto_results: Dict) -> torch.Tensor:
        """
        Compute loss based on synchronization quality.

        Args:
            kuramoto_results: Dictionary with sync metrics
                - final_sync_order: Final synchronization order parameter
                - synchronization_order: History of sync orders over time
                - phase_coherence: Phase coherence metric

        Returns:
            sync_loss: Loss tensor
        """
        # Final sync order loss
        final_sync = kuramoto_results.get('final_sync_order', 0.5)
        sync_order_loss = torch.abs(torch.tensor(final_sync - self.target_sync_order))

        # Stability loss (variance in sync order over time)
        sync_history = kuramoto_results.get('synchronization_order', [])
        if len(sync_history) > 1:
            stability_loss = torch.tensor(np.var(sync_history))
        else:
            stability_loss = torch.tensor(0.5)

        # Phase coherence loss
        phase_coherence = kuramoto_results.get('phase_coherence', 0.5)
        coherence_loss = torch.abs(torch.tensor(phase_coherence - 1.0))

        # Combined loss
        total_loss = (
            sync_order_loss * 0.5 +
            stability_loss * self.stability_weight +
            coherence_loss * 0.2
        )

        return total_loss


class ClusterCoherenceLoss(nn.Module):
    """Loss function for cluster coherence and emergence."""

    def __init__(self, target_coherence: float = 0.8, emergence_weight: float = 0.4):
        super().__init__()
        self.target_coherence = target_coherence
        self.emergence_weight = emergence_weight

    def forward(self, cluster_analysis: Dict) -> torch.Tensor:
        """
        Compute loss based on cluster quality.

        Args:
            cluster_analysis: Dictionary with cluster metrics
                - clusters: List of cluster information
                - dominant_cluster: Information about dominant cluster
                - total_clusters: Number of clusters found

        Returns:
            cluster_loss: Loss tensor
        """
        clusters = cluster_analysis.get('clusters', [])

        if not clusters:
            # Penalize lack of clustering
            return torch.tensor(1.0)

        # Cluster coherence loss
        order_params = [c.get('order_parameter', 0.5) for c in clusters]
        mean_coherence = np.mean(order_params)
        coherence_loss = torch.abs(torch.tensor(mean_coherence - self.target_coherence))

        # Cluster emergence loss (prefer moderate number of well-defined clusters)
        n_clusters = len(clusters)
        if n_clusters == 1:
            emergence_loss = torch.tensor(0.7)  # Penalize single cluster
        elif n_clusters > 5:
            emergence_loss = torch.tensor(0.5)  # Penalize too many clusters
        else:
            emergence_loss = torch.tensor(0.2)  # Reward moderate clustering

        # Dominant cluster quality
        dominant_cluster = cluster_analysis.get('dominant_cluster', {})
        dominant_size = dominant_cluster.get('size', 1)
        dominant_order = dominant_cluster.get('order_parameter', 0.5)

        # Penalize if dominant cluster is too small or has low order
        if dominant_size < 2:
            dominance_loss = torch.tensor(0.8)
        else:
            dominance_loss = torch.abs(torch.tensor(dominant_order - 1.0))

        # Combined loss
        total_loss = (
            coherence_loss * 0.4 +
            emergence_loss * self.emergence_weight +
            dominance_loss * 0.2
        )

        return total_loss


class ConsciousnessGapLoss(nn.Module):
    """Loss function for consciousness metrics gap."""

    def __init__(self, target_fci: float = 0.7, state_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.target_fci = target_fci

        # Default state weights (higher = better)
        self.state_weights = state_weights or {
            'EMERGENCE': 0.9,
            'MEDITATION': 0.7,
            'ANALYSIS': 0.5,
            'COMA': 0.1,
            'UNKNOWN': 0.3
        }

    def forward(self, consciousness_results: Dict) -> torch.Tensor:
        """
        Compute loss based on consciousness metrics.

        Args:
            consciousness_results: Dictionary with consciousness metrics
                - fci: Fractal Consciousness Index
                - state: Consciousness state
                - clz: Consciousness level

        Returns:
            consciousness_loss: Loss tensor
        """
        # FCI gap loss
        fci = consciousness_results.get('fci', 0.5)
        fci_loss = torch.abs(torch.tensor(fci - self.target_fci))

        # State quality loss
        state = consciousness_results.get('state', 'UNKNOWN')
        state_score = self.state_weights.get(state, 0.3)
        state_loss = torch.tensor(1.0 - state_score)

        # CLZ loss (if available)
        clz = consciousness_results.get('clz', 0.5)
        clz_loss = torch.abs(torch.tensor(clz - 0.8))

        # Combined loss
        total_loss = (
            fci_loss * 0.5 +
            state_loss * 0.3 +
            clz_loss * 0.2
        )

        return total_loss


class TrajectoryComplexityLoss(nn.Module):
    """Loss function for trajectory complexity and dynamics quality."""

    def __init__(self, target_complexity: float = 0.6, convergence_weight: float = 0.3):
        super().__init__()
        self.target_complexity = target_complexity
        self.convergence_weight = convergence_weight

    def forward(self, kuramoto_results: Dict) -> torch.Tensor:
        """
        Compute loss based on trajectory complexity.

        Args:
            kuramoto_results: Dictionary with trajectory metrics
                - synchronization_order: History of sync orders
                - oscillator_phases: Phase evolution over time

        Returns:
            trajectory_loss: Loss tensor
        """
        sync_history = kuramoto_results.get('synchronization_order', [])

        if len(sync_history) < 3:
            return torch.tensor(0.8)  # Penalize insufficient trajectory data

        # Complexity loss (measure of interesting dynamics)
        differences = np.diff(sync_history)
        complexity = np.std(differences) * 2.0
        complexity_loss = torch.abs(torch.tensor(complexity - self.target_complexity))

        # Convergence loss (should converge, not oscillate wildly)
        final_sync = sync_history[-1] if sync_history else 0.5
        initial_sync = sync_history[0] if sync_history else 0.5

        # Penalize if final sync is worse than initial
        if final_sync < initial_sync:
            convergence_loss = torch.tensor(0.8)
        else:
            # Reward convergence improvement
            improvement = final_sync - initial_sync
            convergence_loss = torch.tensor(max(0.0, 0.5 - improvement))

        # Oscillation penalty
        max_oscillation = np.max(np.abs(differences))
        oscillation_penalty = torch.tensor(min(1.0, max_oscillation * 5.0))

        # Combined loss
        total_loss = (
            complexity_loss * 0.4 +
            convergence_loss * self.convergence_weight +
            oscillation_penalty * 0.2
        )

        return total_loss


class SemanticAchievementLoss(nn.Module):
    """Loss function for semantic reasoning achievement."""

    def __init__(self, success_reward: float = 0.2, partial_credit: bool = True):
        super().__init__()
        self.success_reward = success_reward
        self.partial_credit = partial_credit

    def forward(self, generated_text: str, target_text: str,
                consciousness_results: Dict) -> torch.Tensor:
        """
        Compute loss based on semantic achievement.

        Args:
            generated_text: Generated text from pipeline
            target_text: Target text for semantic reasoning
            consciousness_results: Consciousness metrics for weighting

        Returns:
            semantic_loss: Loss tensor
        """
        # Base achievement check
        target_achieved = target_text.lower() in generated_text.lower()

        if target_achieved:
            # Success - small reward (we want to focus on dynamics, not just outputs)
            base_loss = torch.tensor(-self.success_reward)
        else:
            # Failure - moderate penalty
            base_loss = torch.tensor(0.5)

        # Consciousness-weighted loss
        fci = consciousness_results.get('fci', 0.5)
        state = consciousness_results.get('state', 'UNKNOWN')

        # Higher consciousness should lead to better results
        consciousness_factor = 1.0 - fci  # Higher FCI reduces penalty

        # State-specific adjustments
        if state == 'EMERGENCE' and not target_achieved:
            # Emergence without achievement is concerning
            consciousness_factor *= 1.2
        elif state == 'COMA' and target_achieved:
            # Achievement in coma state is suspicious
            consciousness_factor *= 1.5

        final_loss = base_loss * consciousness_factor

        # Ensure loss is positive (except for small rewards)
        return torch.clamp(final_loss, min=-0.3, max=2.0)


class PhysicsBasedCombinedLoss(nn.Module):
    """Combined physics-based loss function for ΨQRH training."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()

        # Default weights for different loss components
        self.weights = weights or {
            'sync_quality': 0.25,
            'cluster_coherence': 0.20,
            'consciousness_gap': 0.25,
            'trajectory_complexity': 0.15,
            'semantic_achievement': 0.15
        }

        # Initialize individual loss functions
        self.sync_loss = SyncQualityLoss()
        self.cluster_loss = ClusterCoherenceLoss()
        self.consciousness_loss = ConsciousnessGapLoss()
        self.trajectory_loss = TrajectoryComplexityLoss()
        self.semantic_loss = SemanticAchievementLoss()

    def forward(self, pipeline_results: Dict, target_text: str) -> torch.Tensor:
        """
        Compute combined physics-based loss.

        Args:
            pipeline_results: Complete pipeline results with all metrics
            target_text: Target text for semantic reasoning

        Returns:
            total_loss: Combined physics-based loss tensor
        """
        # Extract components from pipeline results
        kuramoto_results = pipeline_results.get('kuramoto_metrics', {})
        cluster_analysis = pipeline_results.get('cluster_analysis', {})
        consciousness_results = pipeline_results.get('consciousness_metrics', {})
        generated_text = pipeline_results.get('generated_text', '')

        # Compute individual losses
        sync_component = self.sync_loss(kuramoto_results)
        cluster_component = self.cluster_loss(cluster_analysis)
        consciousness_component = self.consciousness_loss(consciousness_results)
        trajectory_component = self.trajectory_loss(kuramoto_results)
        semantic_component = self.semantic_loss(generated_text, target_text, consciousness_results)

        # Combine with weights
        total_loss = (
            sync_component * self.weights['sync_quality'] +
            cluster_component * self.weights['cluster_coherence'] +
            consciousness_component * self.weights['consciousness_gap'] +
            trajectory_component * self.weights['trajectory_complexity'] +
            semantic_component * self.weights['semantic_achievement']
        )

        return total_loss


# Utility functions for physics-based training

def compute_physics_metrics(pipeline_results: Dict) -> Dict[str, float]:
    """
    Compute comprehensive physics metrics from pipeline results.

    Args:
        pipeline_results: Pipeline results dictionary

    Returns:
        physics_metrics: Dictionary of computed metrics
    """
    kuramoto_results = pipeline_results.get('kuramoto_metrics', {})
    cluster_analysis = pipeline_results.get('cluster_analysis', {})
    consciousness_results = pipeline_results.get('consciousness_metrics', {})

    metrics = {
        # Synchronization metrics
        'sync_order': kuramoto_results.get('final_sync_order', 0.5),
        'sync_stability': _compute_sync_stability(kuramoto_results),
        'phase_coherence': kuramoto_results.get('phase_coherence', 0.5),

        # Cluster metrics
        'cluster_coherence': _compute_cluster_coherence(cluster_analysis),
        'cluster_emergence': _compute_cluster_emergence(cluster_analysis),
        'dominant_cluster_quality': _compute_dominant_cluster_quality(cluster_analysis),

        # Consciousness metrics
        'fci': consciousness_results.get('fci', 0.5),
        'consciousness_state': consciousness_results.get('state', 'UNKNOWN'),
        'consciousness_score': _state_to_score(consciousness_results.get('state', 'UNKNOWN')),

        # Combined quality score
        'physics_quality_score': 0.0  # Will be computed below
    }

    # Compute combined quality score
    quality_score = (
        metrics['sync_order'] * 0.25 +
        metrics['sync_stability'] * 0.15 +
        metrics['cluster_coherence'] * 0.20 +
        metrics['cluster_emergence'] * 0.15 +
        metrics['fci'] * 0.25
    )
    metrics['physics_quality_score'] = quality_score

    return metrics


def _compute_sync_stability(kuramoto_results: Dict) -> float:
    """Compute synchronization stability from history."""
    sync_history = kuramoto_results.get('synchronization_order', [])
    if len(sync_history) < 2:
        return 0.5
    variance = np.var(sync_history)
    return max(0.0, 1.0 - variance * 10.0)


def _compute_cluster_coherence(cluster_analysis: Dict) -> float:
    """Compute average cluster coherence."""
    clusters = cluster_analysis.get('clusters', [])
    if not clusters:
        return 0.5
    order_params = [c.get('order_parameter', 0.5) for c in clusters]
    return float(np.mean(order_params))


def _compute_cluster_emergence(cluster_analysis: Dict) -> float:
    """Compute cluster emergence quality."""
    clusters = cluster_analysis.get('clusters', [])
    if len(clusters) <= 1:
        return 0.3  # Limited emergence with single cluster

    # Measure diversity in cluster sizes
    sizes = [c.get('size', 1) for c in clusters]
    if len(sizes) > 1:
        diversity = np.std(sizes) / np.mean(sizes)
        return min(1.0, diversity * 2.0)
    else:
        return 0.5


def _compute_dominant_cluster_quality(cluster_analysis: Dict) -> float:
    """Compute quality of dominant cluster."""
    dominant_cluster = cluster_analysis.get('dominant_cluster', {})
    size = dominant_cluster.get('size', 1)
    order = dominant_cluster.get('order_parameter', 0.5)

    # Quality combines size and order
    if size >= 3 and order > 0.8:
        return 0.9
    elif size >= 2 and order > 0.6:
        return 0.7
    else:
        return 0.4


def _state_to_score(state: str) -> float:
    """Convert consciousness state to numerical score."""
    state_scores = {
        'EMERGENCE': 0.9,
        'MEDITATION': 0.7,
        'ANALYSIS': 0.5,
        'COMA': 0.1,
        'UNKNOWN': 0.3
    }
    return state_scores.get(state, 0.3)


if __name__ == "__main__":
    # Test the loss functions
    print("🧪 Testing Physics-Based Loss Functions...")

    # Sample data
    test_results = {
        'kuramoto_metrics': {
            'final_sync_order': 0.75,
            'synchronization_order': [0.5, 0.6, 0.7, 0.75],
            'phase_coherence': 0.8
        },
        'cluster_analysis': {
            'clusters': [
                {'order_parameter': 0.9, 'size': 3},
                {'order_parameter': 0.7, 'size': 2}
            ],
            'dominant_cluster': {'order_parameter': 0.9, 'size': 3}
        },
        'consciousness_metrics': {
            'fci': 0.65,
            'state': 'MEDITATION',
            'clz': 0.6
        },
        'generated_text': 'The sky is blue'
    }

    # Test combined loss
    combined_loss = PhysicsBasedCombinedLoss()
    loss_value = combined_loss(test_results, 'blue')
    print(f"📊 Combined Physics Loss: {loss_value:.4f}")

    # Test metrics computation
    metrics = compute_physics_metrics(test_results)
    print(f"📈 Physics Metrics: {metrics}")
