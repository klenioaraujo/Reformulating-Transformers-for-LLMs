"""
Quantum Reinforcement Learner - OPÇÃO 5 do Sistema de Calibração ΨQRH
====================================================================

Aprendizagem por reforço quântico leve (hill-climbing).
Ajusta parâmetros baseado na qualidade da saída emergente.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import random
import math
from typing import Dict, List, Any, Optional, Tuple
from collections import deque


class QuantumReinforcementLearner:
    """
    OPÇÃO 5: Aprendizagem por reforço quântico leve

    Usa hill-climbing simples para otimizar parâmetros quânticos.
    Baseado em scores de qualidade sem deep learning.
    """

    def __init__(self, exploration_rate: float = 0.1):
        # Histórico de parâmetros e scores
        self.parameter_history = deque(maxlen=100)  # Últimos 100
        self.quality_scores = deque(maxlen=100)

        # Melhores parâmetros encontrados
        self.best_params = {
            'alpha': 1.0,
            'beta': 0.5,
            'temperature': 1.0,
            'fractal_weight': 1.0,
            'coherence_weight': 1.0
        }
        self.best_score = 0.0

        # Taxa de exploração para hill-climbing
        self.exploration_rate = exploration_rate

        # Estatísticas
        self.total_iterations = 0
        self.improvements_found = 0

        print("🎯 [QuantumReinforcementLearner] Inicializado - Opção 5 ativada")

    def reinforce_parameters(self, current_params: Dict[str, float],
                           quality_score: float) -> Dict[str, float]:
        """
        Reforça parâmetros baseado na qualidade da saída.

        Args:
            current_params: Parâmetros atuais
            quality_score: Score de qualidade (0.0 a 1.0)

        Returns:
            Novos parâmetros sugeridos
        """
        self.total_iterations += 1

        # Registra na história
        self.parameter_history.append(current_params.copy())
        self.quality_scores.append(quality_score)

        # Atualiza melhores parâmetros se necessário
        if quality_score > self.best_score:
            self.best_score = quality_score
            self.best_params = current_params.copy()
            self.improvements_found += 1

            print(f"🎯 [QuantumReinforcementLearner] Novo melhor score: {quality_score:.3f}")
            print(f"   Parâmetros: {self.best_params}")

            # Explora vizinhança dos melhores parâmetros
            return self._generate_neighbor_params(current_params)
        else:
            # Mantém tendência para melhores parâmetros
            return self._drift_toward_best(current_params)

    def get_optimal_parameters(self) -> Dict[str, float]:
        """
        Retorna os melhores parâmetros encontrados até agora.
        """
        return self.best_params.copy()

    def _generate_neighbor_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Gera parâmetros próximos aos atuais (hill-climbing).

        Explora vizinhança com pequenas variações aleatórias.
        """
        new_params = {}

        for key, value in params.items():
            if random.random() < self.exploration_rate:
                # Exploração: variação aleatória
                if key in ['alpha', 'beta', 'temperature']:
                    # Variação relativa para parâmetros principais
                    variation = random.uniform(-0.2, 0.2) * value
                else:
                    # Variação absoluta para pesos
                    variation = random.uniform(-0.3, 0.3)

                new_value = value + variation

                # Garante limites razoáveis
                if key in ['alpha', 'beta']:
                    new_value = max(0.1, min(new_value, 3.0))
                elif key == 'temperature':
                    new_value = max(0.1, min(new_value, 2.0))
                else:  # pesos
                    new_value = max(0.1, min(new_value, 2.0))

                new_params[key] = new_value
            else:
                # Exploração direcionada baseada no gradiente
                gradient = self._estimate_gradient(key)
                step_size = 0.05  # Passo pequeno

                new_value = value + gradient * step_size
                new_params[key] = self._clamp_parameter(key, new_value)

        return new_params

    def _drift_toward_best(self, current_params: Dict[str, float]) -> Dict[str, float]:
        """
        Desloca gradualmente em direção aos melhores parâmetros.
        """
        new_params = {}

        for key, current_value in current_params.items():
            best_value = self.best_params.get(key, current_value)

            # Movimento gradual em direção ao melhor
            drift_factor = 0.1  # 10% em direção ao melhor
            new_value = current_value + (best_value - current_value) * drift_factor

            # Adiciona pequena variação aleatória
            noise = random.uniform(-0.02, 0.02) * current_value
            new_value += noise

            new_params[key] = self._clamp_parameter(key, new_value)

        return new_params

    def _estimate_gradient(self, param_key: str) -> float:
        """
        Estima gradiente baseado no histórico recente.
        """
        if len(self.parameter_history) < 5:
            return 0.0  # Sem histórico suficiente

        # Últimos 5 pontos
        recent_params = list(self.parameter_history)[-5:]
        recent_scores = list(self.quality_scores)[-5:]

        # Calcula correlação simples entre parâmetro e score
        param_values = [p.get(param_key, 0) for p in recent_params]

        # Gradiente aproximado
        if len(param_values) >= 2:
            # Diferença entre melhor e pior score recente
            best_idx = recent_scores.index(max(recent_scores))
            worst_idx = recent_scores.index(min(recent_scores))

            param_diff = param_values[best_idx] - param_values[worst_idx]
            score_diff = recent_scores[best_idx] - recent_scores[worst_idx]

            if abs(score_diff) > 0.01:  # Evita divisão por zero
                gradient = param_diff / score_diff
                return gradient * 0.1  # Escala pequena

        return 0.0

    def _clamp_parameter(self, key: str, value: float) -> float:
        """
        Garante que o parâmetro esteja dentro de limites razoáveis.
        """
        limits = {
            'alpha': (0.1, 3.0),
            'beta': (0.1, 2.0),
            'temperature': (0.1, 2.0),
            'fractal_weight': (0.1, 2.0),
            'coherence_weight': (0.1, 2.0),
            'similarity_weight': (0.1, 2.0)
        }

        min_val, max_val = limits.get(key, (0.1, 2.0))
        return max(min_val, min(max_val, value))

    def apply_quantum_guidance(self, current_params: Dict[str, float],
                              psi_stats: Dict[str, float]) -> Dict[str, float]:
        """
        Aplica orientação quântica aos parâmetros baseado nas estatísticas.

        Args:
            current_params: Parâmetros atuais
            psi_stats: Estatísticas do estado quântico

        Returns:
            Parâmetros ajustados
        """
        adjusted_params = current_params.copy()

        # Ajustes baseados nas propriedades quânticas
        coherence = psi_stats.get('finite', True)  # Simplificação
        complexity = abs(psi_stats.get('std', 0.5))

        # Estados coerentes → aumenta temperatura para diversidade
        if coherence:
            adjusted_params['temperature'] = min(1.5, current_params.get('temperature', 1.0) * 1.1)

        # Estados complexos → aumenta alpha para maior resolução
        if complexity > 0.8:
            adjusted_params['alpha'] = min(2.5, current_params.get('alpha', 1.0) * 1.05)

        # Estados simples → diminui beta para estabilidade
        elif complexity < 0.3:
            adjusted_params['beta'] = max(0.2, current_params.get('beta', 0.5) * 0.95)

        return adjusted_params

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do processo de aprendizado.
        """
        avg_score = sum(self.quality_scores) / max(len(self.quality_scores), 1)
        improvement_rate = self.improvements_found / max(self.total_iterations, 1)

        return {
            'total_iterations': self.total_iterations,
            'best_score': self.best_score,
            'average_score': avg_score,
            'improvements_found': self.improvements_found,
            'improvement_rate': improvement_rate,
            'best_params': self.best_params.copy(),
            'exploration_rate': self.exploration_rate
        }

    def reset_learning(self) -> None:
        """
        Reseta o processo de aprendizado (útil para novos contextos).
        """
        self.parameter_history.clear()
        self.quality_scores.clear()
        self.best_score = 0.0
        self.total_iterations = 0
        self.improvements_found = 0

        print("🔄 [QuantumReinforcementLearner] Aprendizado resetado")


# Função de interface para integração
def create_quantum_reinforcement_learner(exploration_rate: float = 0.1) -> QuantumReinforcementLearner:
    """
    Factory function para criar instância do learner de reforço quântico.
    """
    return QuantumReinforcementLearner(exploration_rate)


# Teste das implementações
if __name__ == "__main__":
    # Teste básico
    learner = create_quantum_reinforcement_learner()

    # Parâmetros iniciais
    params = {
        'alpha': 1.0,
        'beta': 0.5,
        'temperature': 1.0,
        'fractal_weight': 1.0
    }

    # Simula algumas iterações de aprendizado
    scores_and_params = [
        (0.6, {'alpha': 1.2, 'beta': 0.6, 'temperature': 0.9, 'fractal_weight': 1.1}),
        (0.8, {'alpha': 1.1, 'beta': 0.4, 'temperature': 1.1, 'fractal_weight': 0.9}),
        (0.5, {'alpha': 0.9, 'beta': 0.7, 'temperature': 1.2, 'fractal_weight': 1.2}),
        (0.9, {'alpha': 1.3, 'beta': 0.3, 'temperature': 0.8, 'fractal_weight': 1.0}),
    ]

    print("Simulando aprendizado por reforço...")
    for i, (score, test_params) in enumerate(scores_and_params):
        new_params = learner.reinforce_parameters(test_params, score)
        print(f"Iteração {i+1}: Score {score:.1f} → Novos params: {new_params}")

    # Estatísticas finais
    stats = learner.get_learning_stats()
    print(f"\nEstatísticas finais: {stats}")

    # Melhores parâmetros encontrados
    best = learner.get_optimal_parameters()
    print(f"Melhores parâmetros: {best}")