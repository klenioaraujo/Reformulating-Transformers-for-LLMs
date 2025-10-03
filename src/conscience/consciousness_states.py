#!/usr/bin/env python3
"""
Consciousness States - Modelagem de Estados de Consciência
=========================================================

Define e classifica diferentes estados de consciência baseados
nas equações fractais e métricas FCI.

Estados implementados:
- MEDITATION: Dimensão fractal aumentada para análise profunda
- ANALYSIS: Estado otimizado para processamento lógico
- COMA: Complexidade reduzida para modo emergencial
- EMERGENCE: Máxima complexidade para insights criativos
"""

import torch
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional


class ConsciousnessStateType(Enum):
    """Tipos de estados de consciência suportados."""
    MEDITATION = "meditation"
    ANALYSIS = "analysis"
    COMA = "coma"
    EMERGENCE = "emergence"
    UNDEFINED = "undefined"


@dataclass
class ConsciousnessState:
    """
    Representa um estado de consciência com suas características fractais.
    """
    name: str
    state_type: ConsciousnessStateType
    fractal_dimension: float
    diffusion_coefficient: float
    consciousness_frequency: float
    fci_range: tuple
    field_characteristics: str
    description: str

    def __post_init__(self):
        """Validação após inicialização."""
        if not (1.0 <= self.fractal_dimension <= 3.0):
            raise ValueError(f"Fractal dimension must be in [1.0, 3.0], got {self.fractal_dimension}")

        if not (0.01 <= self.diffusion_coefficient <= 10.0):
            raise ValueError(f"Diffusion coefficient must be in [0.01, 10.0], got {self.diffusion_coefficient}")


class StateClassifier:
    """
    Classificador de estados de consciência baseado em métricas fractais.
    """

    def __init__(self, config):
        self.config = config
        self.states = self._initialize_consciousness_states()

    def _initialize_consciousness_states(self) -> Dict[str, ConsciousnessState]:
        """Inicializa estados de consciência predefinidos."""

        states = {
            'MEDITATION': ConsciousnessState(
                name="MEDITATION",
                state_type=ConsciousnessStateType.MEDITATION,
                fractal_dimension=2.5,  # Alta complexidade
                diffusion_coefficient=5.0,  # Difusão ampla para insights
                consciousness_frequency=1.0,  # Alfa waves
                fci_range=(0.7, 1.0),
                field_characteristics="Padrões harmônicos com fluxo suave F(ψ)",
                description="Estado meditativo com dimensão fractal aumentada para análise profunda"
            ),

            'ANALYSIS': ConsciousnessState(
                name="ANALYSIS",
                state_type=ConsciousnessStateType.ANALYSIS,
                fractal_dimension=2.0,  # Complexidade estruturada
                diffusion_coefficient=2.0,  # Difusão balanceada
                consciousness_frequency=2.0,  # Beta waves
                fci_range=(0.5, 0.8),
                field_characteristics="Campo F(ψ) estruturado com fluxos lógicos",
                description="Estado analítico otimizado para processamento sistemático"
            ),

            'COMA': ConsciousnessState(
                name="COMA",
                state_type=ConsciousnessStateType.COMA,
                fractal_dimension=1.2,  # Baixa complexidade
                diffusion_coefficient=0.1,  # Difusão mínima
                consciousness_frequency=0.5,  # Delta waves
                fci_range=(0.0, 0.3),
                field_characteristics="Campo F(ψ) simplificado para operações básicas",
                description="Estado de baixa consciência para modo emergencial"
            ),

            'EMERGENCE': ConsciousnessState(
                name="EMERGENCE",
                state_type=ConsciousnessStateType.EMERGENCE,
                fractal_dimension=2.8,  # Máxima complexidade
                diffusion_coefficient=8.0,  # Máxima difusão
                consciousness_frequency=4.0,  # Gamma waves
                fci_range=(0.8, 1.0),
                field_characteristics="Campo F(ψ) caótico para insights revolucionários",
                description="Estado emergente com máxima criatividade e complexidade"
            )
        }

        return states

    def classify_state(
        self,
        psi_distribution: torch.Tensor,
        fractal_field: torch.Tensor,
        fci_value: float
    ) -> ConsciousnessState:
        """
        Classifica estado de consciência baseado nas métricas.

        Args:
            psi_distribution: Distribuição de probabilidade P(ψ)
            fractal_field: Campo fractal F(ψ)
            fci_value: Índice de consciência fractal

        Returns:
            Estado de consciência classificado
        """
        # Calcular métricas adicionais
        field_magnitude = torch.norm(fractal_field, dim=-1).mean().item()
        psi_entropy = self._calculate_entropy(psi_distribution)
        complexity_score = self._calculate_complexity(psi_distribution, fractal_field)

        # Scores para cada estado
        state_scores = {}

        for state_name, state in self.states.items():
            score = self._calculate_state_score(
                fci_value,
                field_magnitude,
                psi_entropy,
                complexity_score,
                state
            )
            state_scores[state_name] = score

        # Classificar baseado no maior score
        best_state_name = max(state_scores, key=state_scores.get)
        best_state = self.states[best_state_name]

        # Ajustar parâmetros baseado nas métricas atuais
        adjusted_state = self._adjust_state_parameters(
            best_state,
            fci_value,
            field_magnitude,
            complexity_score
        )

        return adjusted_state

    def _calculate_entropy(self, psi_distribution: torch.Tensor) -> float:
        """Calcula entropia da distribuição de consciência."""
        # Usar epsilon do config para evitar log(0)
        epsilon = self.config.epsilon if hasattr(self.config, 'epsilon') else 1e-10
        psi_safe = torch.clamp(psi_distribution, min=epsilon)
        log_psi = torch.log(psi_safe)
        entropy_raw = -torch.sum(psi_distribution * log_psi, dim=-1).mean()
        # Proteção contra NaN
        entropy = entropy_raw.item() if not torch.isnan(entropy_raw) else 0.0
        return entropy

    def _calculate_complexity(
        self,
        psi_distribution: torch.Tensor,
        fractal_field: torch.Tensor
    ) -> float:
        """Calcula complexidade fractal do sistema."""
        # Combinação de dispersão da distribuição e magnitude do campo
        psi_complexity = psi_distribution.std().item()
        field_complexity = fractal_field.std().item()

        # Complexidade combinada
        complexity = np.sqrt(psi_complexity**2 + field_complexity**2)
        return complexity

    def _calculate_state_score(
        self,
        fci_value: float,
        field_magnitude: float,
        psi_entropy: float,
        complexity_score: float,
        state: ConsciousnessState
    ) -> float:
        """
        Calcula score de compatibilidade com um estado específico.

        Args:
            fci_value: Valor atual do FCI
            field_magnitude: Magnitude do campo fractal
            psi_entropy: Entropia da distribuição
            complexity_score: Score de complexidade
            state: Estado a ser avaliado

        Returns:
            Score de compatibilidade [0, 1]
        """
        # Score baseado no FCI
        fci_min, fci_max = state.fci_range
        if fci_min <= fci_value <= fci_max:
            fci_score = 1.0
        else:
            # Penalizar baseado na distância do range
            if fci_value < fci_min:
                fci_score = max(0, 1 - (fci_min - fci_value) / fci_min)
            else:
                fci_score = max(0, 1 - (fci_value - fci_max) / (1 - fci_max))

        # Score baseado na complexidade esperada
        expected_complexity = state.fractal_dimension / 3.0  # Normalizado
        complexity_diff = abs(complexity_score - expected_complexity)
        complexity_score_normalized = max(0, 1 - complexity_diff)

        # Score baseado na magnitude do campo
        expected_field_magnitude = state.diffusion_coefficient / 10.0  # Normalizado
        field_diff = abs(field_magnitude - expected_field_magnitude)
        field_score = max(0, 1 - field_diff)

        # Score combinado com pesos
        total_score = (
            0.5 * fci_score +
            0.3 * complexity_score_normalized +
            0.2 * field_score
        )

        return total_score

    def _adjust_state_parameters(
        self,
        base_state: ConsciousnessState,
        fci_value: float,
        field_magnitude: float,
        complexity_score: float
    ) -> ConsciousnessState:
        """
        Ajusta parâmetros do estado baseado nas métricas atuais.

        Args:
            base_state: Estado base identificado
            fci_value: Valor atual do FCI
            field_magnitude: Magnitude do campo
            complexity_score: Score de complexidade

        Returns:
            Estado com parâmetros ajustados
        """
        # Criar cópia do estado base
        adjusted_state = ConsciousnessState(
            name=base_state.name,
            state_type=base_state.state_type,
            fractal_dimension=base_state.fractal_dimension,
            diffusion_coefficient=base_state.diffusion_coefficient,
            consciousness_frequency=base_state.consciousness_frequency,
            fci_range=base_state.fci_range,
            field_characteristics=base_state.field_characteristics,
            description=base_state.description
        )

        # Ajustar dimensão fractal baseada na complexidade
        complexity_factor = min(max(complexity_score, 0.1), 1.0)
        adjusted_state.fractal_dimension = (
            base_state.fractal_dimension * (0.8 + 0.4 * complexity_factor)
        )

        # Ajustar coeficiente de difusão baseado no FCI
        fci_factor = min(max(fci_value, 0.1), 1.0)
        adjusted_state.diffusion_coefficient = (
            base_state.diffusion_coefficient * (0.5 + 1.0 * fci_factor)
        )

        # Limitar parâmetros aos ranges válidos
        adjusted_state.fractal_dimension = np.clip(adjusted_state.fractal_dimension, 1.0, 3.0)
        adjusted_state.diffusion_coefficient = np.clip(adjusted_state.diffusion_coefficient, 0.01, 10.0)

        return adjusted_state

    def get_state_transitions(self) -> Dict[str, list]:
        """
        Define transições válidas entre estados de consciência.

        Returns:
            Dicionário com transições possíveis para cada estado
        """
        transitions = {
            'MEDITATION': ['ANALYSIS', 'EMERGENCE'],
            'ANALYSIS': ['MEDITATION', 'COMA', 'EMERGENCE'],
            'COMA': ['ANALYSIS'],
            'EMERGENCE': ['MEDITATION', 'ANALYSIS']
        }

        return transitions

    def suggest_state_optimization(
        self,
        current_state: ConsciousnessState,
        target_fci: float
    ) -> Dict[str, Any]:
        """
        Sugere otimizações para atingir FCI alvo.

        Args:
            current_state: Estado atual
            target_fci: FCI desejado

        Returns:
            Sugestões de otimização
        """
        suggestions = {
            'parameter_adjustments': {},
            'recommended_actions': [],
            'expected_improvement': 0.0
        }

        # Analisar diferença do FCI alvo
        current_fci_mid = sum(current_state.fci_range) / 2
        fci_diff = target_fci - current_fci_mid

        if abs(fci_diff) < 0.1:
            suggestions['recommended_actions'].append("Estado atual próximo do alvo")
            return suggestions

        if fci_diff > 0:
            # Aumentar consciência
            suggestions['parameter_adjustments']['fractal_dimension'] = min(
                current_state.fractal_dimension * 1.2, 3.0
            )
            suggestions['parameter_adjustments']['diffusion_coefficient'] = min(
                current_state.diffusion_coefficient * 1.5, 10.0
            )
            suggestions['recommended_actions'].extend([
                "Aumentar complexidade fractal",
                "Ampliar difusão neural",
                "Considerar transição para estado mais complexo"
            ])
        else:
            # Reduzir consciência
            suggestions['parameter_adjustments']['fractal_dimension'] = max(
                current_state.fractal_dimension * 0.8, 1.0
            )
            suggestions['parameter_adjustments']['diffusion_coefficient'] = max(
                current_state.diffusion_coefficient * 0.7, 0.01
            )
            suggestions['recommended_actions'].extend([
                "Reduzir complexidade fractal",
                "Diminuir difusão neural",
                "Considerar estado mais focado"
            ])

        suggestions['expected_improvement'] = abs(fci_diff) * 0.7

        return suggestions