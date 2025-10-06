"""
Adaptive Calibration Engine - OPÇÃO 2 do Sistema de Calibração ΨQRH
===================================================================

Auto-calibração adaptativa baseada em padrões quânticos emergentes.
Aprende quais caracteres emergem naturalmente dos padrões quânticos.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple


class AdaptiveCalibrationEngine:
    """
    OPÇÃO 2: Auto-calibração adaptativa baseada em padrões quânticos emergentes

    Aprende correlação entre estados quânticos e caracteres gerados.
    Ajusta pesos baseado em histórico de sucesso sem treinamento.
    """

    def __init__(self):
        # Histórico de padrões quânticos e seus sucessos/fracassos
        self.quantum_patterns = {}
        self.success_patterns = defaultdict(lambda: defaultdict(int))
        self.failure_patterns = defaultdict(lambda: defaultdict(int))

        # Estatísticas de performance
        self.total_attempts = 0
        self.successful_attempts = 0

        # Cache para evitar recálculo
        self.pattern_cache = {}

        print("🔄 [AdaptiveCalibrationEngine] Inicializado - Opção 2 ativada")

    def learn_quantum_patterns(self, psi_state: torch.Tensor, generated_char: str,
                              success_score: float, input_text: str = None) -> None:
        """
        Aprende correlação entre estados quânticos e caracteres gerados.

        Args:
            psi_state: Estado quântico [embed_dim, 4]
            generated_char: Caractere gerado
            success_score: Score de sucesso (0.0 a 1.0)
            input_text: Texto de entrada (opcional)
        """
        self.total_attempts += 1

        # Extrai assinatura do padrão quântico
        pattern_key = self._extract_quantum_signature(psi_state)

        # Registra sucesso ou fracasso
        if success_score > 0.7:  # Sucesso
            self.success_patterns[pattern_key][generated_char] += 1
            self.successful_attempts += 1
        else:  # Fracasso
            self.failure_patterns[pattern_key][generated_char] += 1

        # Limpa cache periodicamente para evitar crescimento excessivo
        if self.total_attempts % 1000 == 0:
            self._cleanup_old_patterns()

        # Logging periódico
        if self.total_attempts % 100 == 0:
            success_rate = self.successful_attempts / self.total_attempts
            print(f"🔄 [AdaptiveCalibrationEngine] Performance: {success_rate:.3f} "
                  f"({self.successful_attempts}/{self.total_attempts})")

    def get_adaptive_weight(self, psi_state: torch.Tensor, candidate_char: str) -> float:
        """
        Retorna peso adaptativo baseado no histórico para este padrão quântico.

        Args:
            psi_state: Estado quântico atual
            candidate_char: Caractere candidato

        Returns:
            Peso adaptativo [0.5, 1.5]
        """
        pattern_key = self._extract_quantum_signature(psi_state)

        # Busca histórico para este padrão
        successes = self.success_patterns[pattern_key].get(candidate_char, 0)
        failures = self.failure_patterns[pattern_key].get(candidate_char, 0)

        if successes + failures == 0:
            # Sem histórico - peso neutro
            return 1.0

        # Calcula taxa de sucesso
        success_rate = successes / (successes + failures)

        # Converte para peso: sucesso alto → peso alto, sucesso baixo → peso baixo
        # Range: [0.5, 1.5] para evitar pesos extremos
        weight = 0.5 + success_rate

        return min(1.5, max(0.5, weight))

    def get_best_char_for_pattern(self, psi_state: torch.Tensor) -> Optional[str]:
        """
        Retorna o melhor caractere para este padrão quântico baseado no histórico.

        Args:
            psi_state: Estado quântico

        Returns:
            Melhor caractere ou None se sem histórico
        """
        pattern_key = self._extract_quantum_signature(psi_state)

        # Combina sucessos e fracassos para calcular scores
        char_scores = {}

        for char in self.success_patterns[pattern_key]:
            successes = self.success_patterns[pattern_key][char]
            failures = self.failure_patterns[pattern_key].get(char, 0)
            total = successes + failures

            if total >= 3:  # Mínimo de tentativas para confiabilidade
                success_rate = successes / total
                char_scores[char] = success_rate

        if not char_scores:
            return None

        # Retorna caractere com melhor score
        best_char = max(char_scores.keys(), key=lambda c: char_scores[c])
        return best_char

    def apply_adaptive_calibration(self, psi_state: torch.Tensor, char_probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Aplica calibração adaptativa às probabilidades de caracteres.

        Args:
            psi_state: Estado quântico
            char_probabilities: Probabilidades originais {char: prob}

        Returns:
            Probabilidades calibradas
        """
        calibrated_probs = {}

        for char, prob in char_probabilities.items():
            # Obtém peso adaptativo
            adaptive_weight = self.get_adaptive_weight(psi_state, char)

            # Aplica peso à probabilidade
            calibrated_prob = prob * adaptive_weight

            # Garante que não seja zero
            calibrated_prob = max(calibrated_prob, 1e-6)

            calibrated_probs[char] = calibrated_prob

        # Renormaliza para somar 1
        total = sum(calibrated_probs.values())
        if total > 0:
            calibrated_probs = {char: prob / total for char, prob in calibrated_probs.items()}

        return calibrated_probs

    def _extract_quantum_signature(self, psi_state: torch.Tensor) -> str:
        """
        Extrai assinatura única do estado quântico para indexação.

        Usa estatísticas principais para criar uma chave de identificação.
        """
        # Calcula estatísticas principais
        flat_psi = psi_state.flatten()

        # Estatísticas robustas
        mean_val = flat_psi.mean().item()
        std_val = flat_psi.std().item()
        min_val = flat_psi.min().item()
        max_val = flat_psi.max().item()

        # Quartis para distribuição
        sorted_psi = torch.sort(flat_psi).values
        q25 = sorted_psi[int(0.25 * len(sorted_psi))].item()
        q75 = sorted_psi[int(0.75 * len(sorted_psi))].item()

        # Cria assinatura arredondada para agrupamento
        signature = ",".join([
            ".2f",
            ".2f",
            ".2f",
            ".2f",
            ".2f",
            ".2f"
        ])

        return signature

    def _cleanup_old_patterns(self) -> None:
        """
        Limpa padrões antigos com pouco histórico para evitar crescimento excessivo.
        """
        print("🧹 [AdaptiveCalibrationEngine] Limpando padrões antigos...")

        patterns_to_remove = []

        for pattern_key in self.success_patterns.keys():
            total_attempts = sum(self.success_patterns[pattern_key].values()) + \
                           sum(self.failure_patterns[pattern_key].values())

            # Remove padrões com menos de 5 tentativas totais
            if total_attempts < 5:
                patterns_to_remove.append(pattern_key)

        for pattern_key in patterns_to_remove:
            if pattern_key in self.success_patterns:
                del self.success_patterns[pattern_key]
            if pattern_key in self.failure_patterns:
                del self.failure_patterns[pattern_key]

        print(f"🧹 [AdaptiveCalibrationEngine] Removidos {len(patterns_to_remove)} padrões antigos")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de performance da calibração adaptativa.
        """
        total_patterns = len(self.success_patterns)
        total_char_mappings = sum(len(chars) for chars in self.success_patterns.values())

        success_rate = self.successful_attempts / max(self.total_attempts, 1)

        return {
            'total_attempts': self.total_attempts,
            'successful_attempts': self.successful_attempts,
            'success_rate': success_rate,
            'total_patterns': total_patterns,
            'total_char_mappings': total_char_mappings,
            'avg_mappings_per_pattern': total_char_mappings / max(total_patterns, 1)
        }


# Função de interface para integração
def create_adaptive_calibration_engine() -> AdaptiveCalibrationEngine:
    """
    Factory function para criar instância do engine de calibração adaptativa.
    """
    return AdaptiveCalibrationEngine()


# Teste das implementações
if __name__ == "__main__":
    # Teste básico
    engine = create_adaptive_calibration_engine()

    # Estados de teste
    psi1 = torch.randn(64, 4)
    psi2 = torch.randn(64, 4)

    # Simula aprendizado
    engine.learn_quantum_patterns(psi1, 'a', 0.9)
    engine.learn_quantum_patterns(psi1, 'a', 0.8)
    engine.learn_quantum_patterns(psi1, 'b', 0.3)
    engine.learn_quantum_patterns(psi2, 'x', 0.95)

    # Testa pesos adaptativos
    weight_a = engine.get_adaptive_weight(psi1, 'a')
    weight_b = engine.get_adaptive_weight(psi1, 'b')

    print(f"Peso adaptativo para 'a': {weight_a:.3f}")
    print(f"Peso adaptativo para 'b': {weight_b:.3f}")

    # Testa melhor caractere
    best_char = engine.get_best_char_for_pattern(psi1)
    print(f"Melhor caractere para padrão 1: {best_char}")

    # Estatísticas
    stats = engine.get_performance_stats()
    print(f"Estatísticas: {stats}")