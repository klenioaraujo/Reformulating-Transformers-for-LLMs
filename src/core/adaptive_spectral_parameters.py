"""Parâmetros Espectrais Adaptativos para o sistema ΨQRH."""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple


class AdaptiveSpectralParameters:
    """
    Gerencia parâmetros espectrais adaptativos baseados no estado do sistema quântico.
    Ajusta dinamicamente α (filtragem espectral) e β (não-linearidade) com base em
    temperatura quântica, coerência óptica e dimensão fractal.
    """

    def __init__(self, config=None):
        """
        Inicializa os parâmetros espectrais adaptativos.

        Args:
            config: Configuração opcional para os parâmetros
        """
        self.config = config or {}

        # Ranges padrão para os parâmetros
        self.alpha_range = self.config.get('alpha_range', (0.1, 3.0))
        self.beta_range = self.config.get('beta_range', (0.1, 2.0))

        # Pesos para combinação dos fatores
        self.weights = {
            'fractal': 0.4,
            'temperature': 0.3,
            'coherence': 0.3
        }

        print("✅ Adaptive Spectral Parameters initialized.")

    def update_parameters(self, fractal_dimension: float,
                         quantum_temperature: float,
                         optical_coherence: float) -> Dict[str, float]:
        """
        Atualiza os parâmetros espectrais baseados no estado atual do sistema.

        Args:
            fractal_dimension: Dimensão fractal atual
            quantum_temperature: Temperatura quântica atual
            optical_coherence: Coerência óptica atual

        Returns:
            Dicionário com parâmetros α e β atualizados
        """
        try:
            # Normalizar entradas para o range [0, 1]
            fractal_norm = self._normalize_fractal_dimension(fractal_dimension)
            temp_norm = self._normalize_temperature(quantum_temperature)
            coherence_norm = optical_coherence  # Já está em [0, 1]

            # Calcular α (filtragem espectral)
            alpha = self._compute_alpha(fractal_norm, temp_norm, coherence_norm)

            # Calcular β (não-linearidade)
            beta = self._compute_beta(fractal_norm, temp_norm, coherence_norm)

            return {
                'alpha': alpha,
                'beta': beta,
                'alpha_range': self.alpha_range,
                'beta_range': self.beta_range
            }

        except Exception as e:
            print(f"⚠️  Error updating spectral parameters: {e}")
            return self._get_default_parameters()

    def _normalize_fractal_dimension(self, fractal_dimension: float) -> float:
        """Normaliza a dimensão fractal para [0, 1]."""
        # Dimensões fractais típicas: 1.0 (linha) a 2.0+ (superfície)
        # Normalizar para [0, 1] onde 1.5 = 0.5
        normalized = (fractal_dimension - 1.0) / 2.0
        return max(0.0, min(1.0, normalized))

    def _normalize_temperature(self, quantum_temperature: float) -> float:
        """Normaliza a temperatura quântica para [0, 1]."""
        # Temperaturas típicas: 0.1 (congelada) a 2.0 (quente)
        # Normalizar para [0, 1]
        normalized = (quantum_temperature - 0.1) / 1.9
        return max(0.0, min(1.0, normalized))

    def _compute_alpha(self, fractal_norm: float, temp_norm: float, coherence_norm: float) -> float:
        """
        Computa o parâmetro α (filtragem espectral).

        α controla a força da filtragem F(k) = exp(i α · arctan(ln(|k| + ε)))
        - Maior α = filtragem mais forte
        - α aumenta com complexidade fractal
        - α diminui com alta temperatura (mais ruído)
        - α aumenta com alta coerência (melhor sinal)
        """
        # Base alpha
        base_alpha = 1.0

        # Contribuições dos fatores
        fractal_contrib = fractal_norm * 1.0      # +1.0 para alta complexidade
        temp_contrib = (1.0 - temp_norm) * 0.5    # -0.5 para alta temperatura
        coherence_contrib = coherence_norm * 0.5  # +0.5 para alta coerência

        # Combinação ponderada
        alpha = base_alpha + self.weights['fractal'] * fractal_contrib + \
                self.weights['temperature'] * temp_contrib + \
                self.weights['coherence'] * coherence_contrib

        # Aplicar range
        alpha_min, alpha_max = self.alpha_range
        return max(alpha_min, min(alpha_max, alpha))

    def _compute_beta(self, fractal_norm: float, temp_norm: float, coherence_norm: float) -> float:
        """
        Computa o parâmetro β (não-linearidade).

        β controla a não-linearidade na equação de Padilha
        - Maior β = não-linearidade mais forte
        - β aumenta com complexidade fractal
        - β aumenta com temperatura (mais caos)
        - β diminui com alta coerência (mais ordem)
        """
        # Base beta
        base_beta = 0.5

        # Contribuições dos fatores
        fractal_contrib = fractal_norm * 0.8      # +0.8 para alta complexidade
        temp_contrib = temp_norm * 0.6            # +0.6 para alta temperatura
        coherence_contrib = (1.0 - coherence_norm) * 0.4  # -0.4 para alta coerência

        # Combinação ponderada
        beta = base_beta + self.weights['fractal'] * fractal_contrib + \
               self.weights['temperature'] * temp_contrib + \
               self.weights['coherence'] * coherence_contrib

        # Aplicar range
        beta_min, beta_max = self.beta_range
        return max(beta_min, min(beta_max, beta))

    def _get_default_parameters(self) -> Dict[str, float]:
        """Retorna parâmetros padrão em caso de erro."""
        return {
            'alpha': 1.0,
            'beta': 0.5,
            'alpha_range': self.alpha_range,
            'beta_range': self.beta_range
        }

    def get_parameter_analysis(self, fractal_dimension: float,
                              quantum_temperature: float,
                              optical_coherence: float) -> Dict[str, Any]:
        """
        Fornece análise detalhada dos parâmetros espectrais.

        Returns:
            Dicionário com parâmetros e análise de contribuições
        """
        params = self.update_parameters(fractal_dimension, quantum_temperature, optical_coherence)

        return {
            'parameters': params,
            'contributions': {
                'fractal_dimension': self._normalize_fractal_dimension(fractal_dimension),
                'quantum_temperature': self._normalize_temperature(quantum_temperature),
                'optical_coherence': optical_coherence
            },
            'weights': self.weights,
            'description': self._get_parameter_description(params['alpha'], params['beta'])
        }

    def _get_parameter_description(self, alpha: float, beta: float) -> str:
        """Gera descrição dos parâmetros atuais."""
        if alpha > 2.0 and beta > 1.5:
            return "Sistema altamente complexo e não-linear"
        elif alpha > 2.0:
            return "Sistema com forte filtragem espectral"
        elif beta > 1.5:
            return "Sistema altamente não-linear"
        elif alpha < 0.5 and beta < 0.5:
            return "Sistema simples e linear"
        else:
            return "Sistema com parâmetros moderados"