"""Calculador de Coerência Óptica para o sistema ΨQRH."""

import torch
import numpy as np
from typing import Dict, Any, Optional


class OpticalCoherenceCalculator:
    """
    Calcula a coerência óptica baseada na análise de padrões espectrais.
    A coerência óptica mede a capacidade do sistema de manter relações de fase consistentes.
    """

    def __init__(self, config=None):
        """
        Inicializa o calculador de coerência óptica.

        Args:
            config: Configuração opcional para o calculador
        """
        self.config = config or {}
        self.coherence_threshold = self.config.get('coherence_threshold', 0.7)
        print("✅ Optical Coherence Calculator initialized.")

    def compute_optical_coherence(self, spectral_data: Dict[str, Any]) -> float:
        """
        Computa a coerência óptica a partir de dados espectrais.

        Args:
            spectral_data: Dados espectrais contendo magnitude, phase, etc.

        Returns:
            Coerência óptica normalizada [0.0, 1.0]
        """
        try:
            if not spectral_data or 'magnitude' not in spectral_data:
                return 0.5  # Coerência padrão

            magnitude = spectral_data['magnitude']
            if isinstance(magnitude, list):
                magnitude = torch.tensor(magnitude, dtype=torch.float32)

            if magnitude.numel() == 0:
                return 0.5

            # Coerência baseada na concentração espectral
            # Sistemas coerentes têm energia concentrada em poucas frequências
            energy_concentration = self._compute_energy_concentration(magnitude)

            # Coerência baseada na estabilidade de fase
            phase_coherence = spectral_data.get('phase_coherence', 0.5)
            if isinstance(phase_coherence, list):
                phase_coherence = torch.tensor(phase_coherence).mean().item()

            # Coerência baseada na relação sinal-ruído espectral
            snr_coherence = self._compute_spectral_snr(magnitude)

            # Combinação ponderada
            coherence = 0.4 * energy_concentration + 0.4 * phase_coherence + 0.2 * snr_coherence

            return max(0.0, min(1.0, coherence))

        except Exception as e:
            print(f"⚠️  Error computing optical coherence: {e}")
            return 0.5  # Coerência padrão

    def _compute_energy_concentration(self, magnitude: torch.Tensor) -> float:
        """Computa a concentração de energia espectral."""
        try:
            # Usar índice de Gini para medir concentração
            sorted_magnitude = torch.sort(magnitude, descending=True).values
            n = len(sorted_magnitude)

            if n <= 1:
                return 1.0

            # Índice de Gini
            cumulative = torch.cumsum(sorted_magnitude, dim=0)
            gini = (2 * torch.sum(cumulative) - torch.sum(sorted_magnitude)) / (n * torch.sum(sorted_magnitude) + 1e-9)

            # Converter Gini para concentração (menor Gini = maior concentração)
            concentration = 1.0 - gini.item()

            return max(0.0, min(1.0, concentration))

        except Exception:
            return 0.5

    def _compute_spectral_snr(self, magnitude: torch.Tensor) -> float:
        """Computa a relação sinal-ruído espectral."""
        try:
            if magnitude.numel() < 2:
                return 0.5

            # SNR baseado na razão entre o pico máximo e a média
            peak_value = torch.max(magnitude)
            mean_value = torch.mean(magnitude)

            if mean_value > 0:
                snr = (peak_value / mean_value).item()
                # Normalizar para [0, 1]
                snr_normalized = min(snr / 10.0, 1.0)  # Assumindo SNR máximo de 10
                return snr_normalized
            else:
                return 0.0

        except Exception:
            return 0.5

    def get_coherence_analysis(self, spectral_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fornece análise detalhada da coerência óptica.

        Returns:
            Dicionário com coerência e análise detalhada
        """
        coherence = self.compute_optical_coherence(spectral_data)

        # Classificar nível de coerência
        if coherence < 0.3:
            level = "low"
            description = "Sistema com baixa coerência óptica"
        elif coherence < 0.7:
            level = "medium"
            description = "Sistema com coerência óptica moderada"
        else:
            level = "high"
            description = "Sistema altamente coerente"

        return {
            'coherence': coherence,
            'level': level,
            'description': description,
            'threshold_met': coherence >= self.coherence_threshold
        }