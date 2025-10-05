"""
Quantum Temperature Calculator - Auto-Calibração de Temperatura Quântica
=========================================================================

Implementa cálculo emergente de temperatura quântica baseada em métricas físicas:
- Dimensão fractal (D)
- Índice de Consciência Fractal (FCI)
- Complexidade Lempel-Ziv (CLZ)

Baseado em: k_B·T_q = (D - 1) · FCI · (1 + CLZ) · ℏω
"""

import torch
import numpy as np
from typing import Dict, Any, Optional


class QuantumTemperatureCalculator:
    """
    Calcula temperatura quântica T_q a partir de métricas físicas.

    Baseado em: k_B·T_q = ℏω·f(D, FCI, CLZ)

    Interpretação física:
    - D próximo de 1: Sistema simples → T_q baixo (determinístico)
    - D próximo de 2: Sistema complexo → T_q alto (estocástico)
    - FCI alto: Consciência emergente → T_q aumenta (exploração)
    - CLZ alto: Complexidade Lempel-Ziv → T_q aumenta (diversidade)
    """

    def __init__(self, k_B: float = 1.0, hbar: float = 1.0, omega: float = 1.0):
        """
        Inicializa calculadora de temperatura quântica.

        Args:
            k_B: Constante de Boltzmann (normalizada)
            hbar: Constante de Planck reduzida (normalizada)
            omega: Frequência característica (normalizada)
        """
        self.k_B = k_B
        self.hbar = hbar
        self.omega = omega

    def compute_quantum_temperature(
        self,
        D_fractal: float,
        FCI: float,
        CLZ: float
    ) -> float:
        """
        T_q = (D - 1) · FCI · (1 + CLZ) · ω

        Args:
            D_fractal: Dimensão fractal ∈ [1, 2]
            FCI: Fractal Consciousness Index ∈ [0, 1]
            CLZ: Lempel-Ziv complexity ∈ [0, 3]

        Returns:
            T_q: Temperatura quântica ∈ [0.1, 5.0]
        """
        # Fator de complexidade fractal
        complexity_factor = max(0.0, D_fractal - 1.0)  # ∈ [0, 1]

        # Fator de consciência (normalizado)
        consciousness_factor = max(0.0, min(FCI, 1.0))  # ∈ [0, 1]

        # Fator de compressibilidade (inverso da previsibilidade)
        entropy_factor = 1.0 + min(CLZ, 3.0)  # ∈ [1, 4]

        # Temperatura quântica emergente
        T_q = complexity_factor * consciousness_factor * entropy_factor * self.omega

        # Clamping para estabilidade numérica
        T_q = max(0.1, min(T_q, 5.0))

        return T_q

    def apply_quantum_noise(
        self,
        resonance: torch.Tensor,
        T_q: float
    ) -> torch.Tensor:
        """
        Adiciona ruído térmico quântico controlado por T_q.

        resonance_noisy = resonance · (1 + η_q·N(0, T_q))

        Args:
            resonance: Campo de ressonância [batch, vocab_size]
            T_q: Temperatura quântica

        Returns:
            resonance com ruído quântico aplicado
        """
        # Coupling strength (menor em baixa temperatura)
        eta_coupling = torch.tanh(torch.tensor(T_q))  # ∈ [0, 1]

        # Ruído gaussiano
        noise = torch.randn_like(resonance) * T_q

        # Aplicar ruído
        resonance_noisy = resonance * (1.0 + eta_coupling * noise)

        # Garantir positividade
        resonance_noisy = torch.abs(resonance_noisy)

        return resonance_noisy

    def get_temperature_analysis(
        self,
        D_fractal: float,
        FCI: float,
        CLZ: float
    ) -> Dict[str, Any]:
        """
        Análise detalhada da temperatura quântica calculada.

        Returns:
            Dicionário com análise completa
        """
        T_q = self.compute_quantum_temperature(D_fractal, FCI, CLZ)

        # Classificação de comportamento
        if T_q < 0.5:
            behavior = "DETERMINISTIC"
            description = "Sistema previsível, respostas consistentes"
        elif T_q < 1.5:
            behavior = "BALANCED"
            description = "Equilíbrio entre consistência e exploração"
        elif T_q < 3.0:
            behavior = "EXPLORATORY"
            description = "Alta exploração, respostas criativas"
        else:
            behavior = "CHAOTIC"
            description = "Sistema altamente estocástico, respostas imprevisíveis"

        return {
            'T_q': T_q,
            'behavior': behavior,
            'description': description,
            'factors': {
                'complexity_factor': max(0.0, D_fractal - 1.0),
                'consciousness_factor': max(0.0, min(FCI, 1.0)),
                'entropy_factor': 1.0 + min(CLZ, 3.0)
            },
            'input_metrics': {
                'D_fractal': D_fractal,
                'FCI': FCI,
                'CLZ': CLZ
            }
        }


def create_quantum_temperature_calculator(
    k_B: float = 1.0,
    hbar: float = 1.0,
    omega: float = 1.0
) -> QuantumTemperatureCalculator:
    """
    Factory function para criar calculadora de temperatura quântica.

    Args:
        k_B: Constante de Boltzmann
        hbar: Constante de Planck reduzida
        omega: Frequência característica

    Returns:
        QuantumTemperatureCalculator configurado
    """
    return QuantumTemperatureCalculator(k_B=k_B, hbar=hbar, omega=omega)