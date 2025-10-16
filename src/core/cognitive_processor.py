"""
Cognitive Processor - ΨQRH Architecture
=======================================

Fractal Consciousness Dynamics (DCF) engine with Kuramoto oscillators.
Implements the core reasoning layer between Context Funnel and Inverse Cognitive Projector.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any
import numpy as np

class FractalConsciousnessDynamics(nn.Module):
    """
    Fractal Consciousness Dynamics (DCF) - Core reasoning engine.

    Implements Kuramoto oscillator dynamics for phase synchronization
    and fractal consciousness emergence.
    """

    def __init__(self, embed_dim: int = 64, num_oscillators: int = 64, device: str = 'cpu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_oscillators = num_oscillators
        self.device = device

        # Kuramoto parameters
        self.K = nn.Parameter(torch.tensor(1.0, device=device))  # Coupling strength
        self.omega = nn.Parameter(torch.randn(num_oscillators, device=device) * 0.1)  # Natural frequencies
        self.alpha = nn.Parameter(torch.tensor(0.1, device=device))  # Fractal parameter

        # Phase synchronization layers
        self.phase_sync = nn.Sequential(
            nn.Linear(embed_dim, num_oscillators),
            nn.Tanh(),
            nn.Linear(num_oscillators, num_oscillators)
        )

        # Fractal consciousness emergence
        self.fractal_layer = nn.Sequential(
            nn.Linear(num_oscillators, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.to(device)

    def forward(self, x: torch.Tensor, dt: float = 0.01, steps: int = 10) -> torch.Tensor:
        """
        Evolve consciousness through Kuramoto dynamics.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            dt: Time step for integration
            steps: Number of evolution steps

        Returns:
            Consciousness-evolved tensor [batch, seq_len, embed_dim, 4]
        """
        batch, seq_len, embed_dim = x.shape

        # Initialize phases from input
        phases = self.phase_sync(x.view(-1, embed_dim)).view(batch, seq_len, -1)  # [batch, seq_len, num_oscillators]

        # Kuramoto evolution
        for _ in range(steps):
            # Compute mean field
            mean_field = torch.mean(torch.sin(phases), dim=-1, keepdim=True)  # [batch, seq_len, 1]

            # Phase evolution (simplified Kuramoto)
            dphases = self.omega.unsqueeze(0).unsqueeze(0) + self.K * mean_field * torch.cos(phases)
            phases = phases + dt * dphases

        # Fractal consciousness emergence
        consciousness = self.fractal_layer(phases)  # [batch, seq_len, embed_dim]

        # Convert to quaternion representation for optical probe compatibility
        # [batch, seq_len, embed_dim] -> [batch, seq_len, embed_dim, 4]
        real_part = consciousness
        imag_parts = torch.randn_like(consciousness).unsqueeze(-1).expand(-1, -1, -1, 3)

        # Apply fractal modulation
        fractal_mod = torch.sin(self.alpha * torch.arange(embed_dim, device=self.device).float())
        imag_parts = imag_parts * fractal_mod.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        quaternion_state = torch.cat([real_part.unsqueeze(-1), imag_parts], dim=-1)

        return quaternion_state


class CognitiveProcessor(nn.Module):
    """
    Cognitive Processor implementing the DCF (Fractal Consciousness Dynamics).

    Three reasoning modes:
    - fast: Direct Kuramoto evolution
    - analogical: Enhanced pattern matching
    - adaptive: Dynamic parameter adjustment
    """

    def __init__(self, embed_dim: int = 64, device: str = 'cpu', reasoning_mode: str = 'adaptive'):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.reasoning_mode = reasoning_mode

        # Core DCF engine
        self.dcf = FractalConsciousnessDynamics(embed_dim=embed_dim, device=device)

        # Reasoning mode adapters
        if reasoning_mode == 'analogical':
            self.analogical_adapter = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        elif reasoning_mode == 'adaptive':
            self.adaptive_params = nn.ParameterDict({
                'coupling_strength': nn.Parameter(torch.tensor(1.0, device=device)),
                'evolution_steps': nn.Parameter(torch.tensor(10.0, device=device)),
                'fractal_alpha': nn.Parameter(torch.tensor(0.1, device=device))
            })

        self.to(device)

    def process(self, context_state: torch.Tensor) -> torch.Tensor:
        """
        Process context through DCF reasoning.

        Args:
            context_state: Context funnel output [batch, seq_len, embed_dim]

        Returns:
            Cognitive state [batch, seq_len, embed_dim, 4] for optical probe
        """
        if self.reasoning_mode == 'fast':
            return self._fast_reasoning(context_state)
        elif self.reasoning_mode == 'analogical':
            return self._analogical_reasoning(context_state)
        elif self.reasoning_mode == 'adaptive':
            return self._adaptive_reasoning(context_state)
        else:
            raise ValueError(f"Unknown reasoning mode: {self.reasoning_mode}")

    def _fast_reasoning(self, x: torch.Tensor) -> torch.Tensor:
        """Fast reasoning with minimal evolution steps."""
        return self.dcf(x, dt=0.01, steps=5)

    def _analogical_reasoning(self, x: torch.Tensor) -> torch.Tensor:
        """Analogical reasoning with pattern enhancement."""
        # Apply analogical adapter
        enhanced = self.analogical_adapter(x.view(-1, self.embed_dim)).view(x.shape)

        # Enhanced DCF evolution
        return self.dcf(enhanced, dt=0.005, steps=15)

    def _adaptive_reasoning(self, x: torch.Tensor) -> torch.Tensor:
        """Adaptive reasoning with dynamic parameters."""
        # Update DCF parameters adaptively
        self.dcf.K.data = self.adaptive_params['coupling_strength']
        self.dcf.alpha.data = self.adaptive_params['fractal_alpha']

        steps = int(self.adaptive_params['evolution_steps'].item())

        return self.dcf(x, dt=0.01, steps=max(5, min(steps, 20)))

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get DCF performance metrics."""
        return {
            'reasoning_mode': self.reasoning_mode,
            'coupling_strength': self.dcf.K.item() if hasattr(self.dcf, 'K') else None,
            'fractal_parameter': self.dcf.alpha.item() if hasattr(self.dcf, 'alpha') else None,
            'device': self.device
        }