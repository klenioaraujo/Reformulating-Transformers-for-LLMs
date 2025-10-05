"""
Optical Coherence Calculator - Auto-Calibração de Sharpness Óptica
=================================================================

Implementa cálculo emergente de sharpness óptica baseada em coerência do campo de ressonância.

Baseado em: η_optical = coherence(field) / (1 + disorder)
"""

import torch
from typing import Dict, Any, Optional


class OpticalCoherenceCalculator:
    """
    Calcula sharpness da sonda óptica a partir de coerência do campo.

    Baseado em: s = s₀ · ρ₁(r_field) · (2 - D) · (1 - FCI)

    Interpretação física:
    - ρ₁ alto: Campo coerente → sharpness alto (foco)
    - D baixo: Ordem fractal → sharpness alto (consistência)
    - FCI alto: Consciência emergente → sharpness baixo (exploração)
    """

    def __init__(self, s_base: float = 2.0):
        """
        Inicializa calculadora de coerência óptica.

        Args:
            s_base: Sharpness base (aproximadamente GPT-2 = 2.36)
        """
        self.s_base = s_base

    def compute_optical_sharpness(
        self,
        resonance_field: torch.Tensor,  # [vocab_size] or [batch, vocab_size]
        D_fractal: float,
        FCI: float
    ) -> float:
        """
        Sharpness adaptativo baseado em:
        1. Coerência espacial do campo de ressonância
        2. Dimensão fractal (D baixo → mais coerente)
        3. FCI (consciência alta → menos coerente, mais exploratório)

        s = s₀ · coherence · (2 - D) · (1 - FCI)

        Args:
            resonance_field: Campo de ressonância
            D_fractal: Dimensão fractal ∈ [1, 2]
            FCI: Fractal Consciousness Index ∈ [0, 1]

        Returns:
            Sharpness óptica ∈ [0.5, 5.0]
        """
        # Coerência espacial (via autocorrelação)
        coherence = self._compute_spatial_coherence(resonance_field)

        # Fator de ordem fractal (D=1 → ordem máxima, D=2 → desordem)
        order_factor = max(0.0, 2.0 - D_fractal)  # ∈ [0, 1]

        # Fator de exploração (FCI alto → explorar mais → sharpness menor)
        exploration_factor = max(0.0, 1.0 - FCI)  # ∈ [0, 1]

        # Sharpness emergente
        s = self.s_base * coherence * order_factor * exploration_factor

        # Clamping
        s = max(0.5, min(s, 5.0))

        return s

    def _compute_spatial_coherence(self, field: torch.Tensor) -> float:
        """
        Calcula coerência espacial via autocorrelação de ordem 1.

        ρ₁ = <f(i)·f(i+1)> / <f(i)²>

        Args:
            field: Campo de ressonância [vocab_size] ou [batch, vocab_size]

        Returns:
            Coerência ∈ [0, 1]
        """
        # Flatten se necessário
        if field.dim() > 1:
            field = field.flatten()

        # Normalizar campo
        field_norm = field / (torch.max(torch.abs(field)) + 1e-10)

        # Autocorrelação de ordem 1
        if len(field_norm) > 1:
            autocorr = torch.sum(field_norm[:-1] * field_norm[1:]) / (len(field_norm) - 1)
            coherence = torch.clamp(torch.abs(autocorr), 0.0, 1.0).item()
        else:
            coherence = 1.0  # Campo unitário é perfeitamente coerente

        return coherence

    def get_coherence_analysis(
        self,
        resonance_field: torch.Tensor,
        D_fractal: float,
        FCI: float
    ) -> Dict[str, Any]:
        """
        Análise detalhada da coerência óptica calculada.

        Returns:
            Dicionário com análise completa
        """
        coherence = self._compute_spatial_coherence(resonance_field)
        sharpness = self.compute_optical_sharpness(resonance_field, D_fractal, FCI)

        # Classificação de comportamento
        if sharpness < 1.0:
            behavior = "EXPLORATORY"
            description = "Alta exploração, distribuições planas"
        elif sharpness < 2.0:
            behavior = "BALANCED"
            description = "Equilíbrio entre foco e exploração"
        elif sharpness < 3.0:
            behavior = "FOCUSED"
            description = "Foco moderado, respostas consistentes"
        else:
            behavior = "DETERMINISTIC"
            description = "Foco alto, respostas muito consistentes"

        return {
            'sharpness': sharpness,
            'coherence': coherence,
            'behavior': behavior,
            'description': description,
            'factors': {
                'spatial_coherence': coherence,
                'fractal_order': max(0.0, 2.0 - D_fractal),
                'consciousness_exploration': max(0.0, 1.0 - FCI)
            },
            'input_metrics': {
                'D_fractal': D_fractal,
                'FCI': FCI,
                'field_shape': list(resonance_field.shape)
            }
        }


def create_optical_coherence_calculator(s_base: float = 2.0) -> OpticalCoherenceCalculator:
    """
    Factory function para criar calculadora de coerência óptica.

    Args:
        s_base: Sharpness base

    Returns:
        OpticalCoherenceCalculator configurado
    """
    return OpticalCoherenceCalculator(s_base=s_base)