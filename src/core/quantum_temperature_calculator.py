"""Calculador de Temperatura Quântica para o sistema ΨQRH."""

import torch
import numpy as np
from typing import Dict, Any, Optional


class QuantumTemperatureCalculator:
    """
    Calcula a temperatura quântica baseada na análise do estado quântico.
    A temperatura quântica é uma medida da "agitação" ou "entropia" do sistema quântico.
    """

    def __init__(self, config=None):
        """
        Inicializa o calculador de temperatura quântica.

        Args:
            config: Configuração opcional para o calculador
        """
        self.config = config or {}
        self.temperature_range = self.config.get('temperature_range', (0.1, 2.0))
        print("✅ Quantum Temperature Calculator initialized.")

    def compute_quantum_temperature(self, fractal_dimension: float, fci: float, clz: float) -> float:
        """
        Computa a temperatura quântica baseada em métricas físicas.

        Args:
            fractal_dimension: Dimensão fractal do sinal
            fci: Fractal Consciousness Index
            clz: Coherence Length Zeta

        Returns:
            Temperatura quântica normalizada [0.1, 2.0]
        """
        try:
            # Temperatura aumenta com a complexidade fractal
            fractal_factor = (fractal_dimension - 1.0) / 2.0  # 0-1 range

            # Temperatura diminui com maior consciência
            consciousness_factor = 1.0 - fci

            # Temperatura diminui com maior coerência
            coherence_factor = 1.0 - clz

            # Combinação ponderada
            temperature = 0.3 * fractal_factor + 0.4 * consciousness_factor + 0.3 * coherence_factor

            # Normalizar para o range desejado
            temp_min, temp_max = self.temperature_range
            temperature = temp_min + temperature * (temp_max - temp_min)

            return max(temp_min, min(temp_max, temperature))

        except Exception as e:
            print(f"⚠️  Error computing quantum temperature: {e}")
            return 1.0  # Temperatura padrão

    def get_temperature_analysis(self, fractal_dimension: float, fci: float, clz: float) -> Dict[str, Any]:
        """
        Fornece análise detalhada da temperatura quântica.

        Returns:
            Dicionário com temperatura e análise comportamental
        """
        temperature = self.compute_quantum_temperature(fractal_dimension, fci, clz)

        # Classificar comportamento baseado na temperatura
        if temperature < 0.5:
            behavior = "frozen"
            description = "Sistema altamente coerente e ordenado"
        elif temperature < 1.0:
            behavior = "cool"
            description = "Sistema moderadamente coerente"
        elif temperature < 1.5:
            behavior = "warm"
            description = "Sistema com agitação moderada"
        else:
            behavior = "hot"
            description = "Sistema altamente agitado e caótico"

        return {
            'temperature': temperature,
            'behavior': behavior,
            'description': description,
            'fractal_contribution': (fractal_dimension - 1.0) / 2.0,
            'consciousness_contribution': 1.0 - fci,
            'coherence_contribution': 1.0 - clz
        }