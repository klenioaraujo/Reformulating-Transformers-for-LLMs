"""
DEPRECATED: This module has been replaced by QuantumStateInterpreter

This file is scheduled for removal. Its functionality has been integrated
into the QuantumStateInterpreter class in quantum_interpreter.py.

The semantic coherence logic is now part of the .to_text() method.

Please use:
    from src.processing.quantum_interpreter import QuantumStateInterpreter

Instead of importing from this module.
"""

import warnings
warnings.warn(
    "semantic_coherence_layer.py is deprecated. Use QuantumStateInterpreter from quantum_interpreter.py instead.",
    DeprecationWarning,
    stacklevel=2
)

"""
Semantic Coherence Layer - OPÇÃO 3 do Sistema de Calibração ΨQRH
================================================================

Usa estatísticas quânticas para guiar a geração de texto coerente.
Mapeia propriedades do estado quântico para características linguísticas.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import math
from typing import Dict, List, Any, Optional


class SemanticCoherenceLayer:
    """
    OPÇÃO 3: Camadas de coerência semântica usando estatísticas quânticas

    Mapeia estatísticas quânticas (mean, std, range) para características linguísticas:
    - Alta variabilidade → texto complexo
    - Baixa variabilidade → texto simples
    - Valores extremos → pontuação e estrutura
    """

    def __init__(self):
        # Mapeamentos calibrados: estatísticas quânticas → complexidade linguística
        self.complexity_mapping = {
            'high': {'threshold_std': 0.8, 'char_boost': ['r', 't', 'n', 's', 'l', 'c']},
            'medium': {'threshold_std': 0.4, 'char_boost': ['a', 'e', 'o', 'i', 'd', 'm']},
            'low': {'threshold_std': 0.0, 'char_boost': ['a', 'e', 'i', 'o', 'u', ' ', '.']}
        }

        # Cache de estatísticas para evitar recálculo
        self.stats_cache = {}

    def apply_quantum_guidance(self, psi_stats: Dict[str, float], current_text: str,
                              input_text: str = None) -> str:
        """
        Aplica orientação semântica baseada nas estatísticas quânticas.

        Args:
            psi_stats: Estatísticas do estado quântico (mean, std, min, max, etc.)
            current_text: Texto atual sendo gerado
            input_text: Texto de entrada original (opcional)

        Returns:
            Texto ajustado com coerência semântica
        """
        print("    🧠 [SemanticCoherenceLayer] Aplicando orientação quântica semântica...")

        # Mapear estatísticas para nível de complexidade
        complexity_level = self._map_quantum_to_complexity(psi_stats)

        # Aplicar orientação baseada na complexidade
        if complexity_level == 'high':
            guided_text = self._boost_complex_characters(current_text, psi_stats)
        elif complexity_level == 'medium':
            guided_text = self._boost_balanced_characters(current_text, psi_stats)
        else:  # low
            guided_text = self._boost_simple_characters(current_text, psi_stats)

        # Aplicar estrutura baseada no range dos valores
        guided_text = self._apply_structural_guidance(guided_text, psi_stats)

        print(f"    🧠 [SemanticCoherenceLayer] Complexidade detectada: {complexity_level}")
        print(f"    🧠 [SemanticCoherenceLayer] Orientação aplicada: {len(guided_text)} caracteres")

        return guided_text

    def _map_quantum_to_complexity(self, psi_stats: Dict[str, float]) -> str:
        """
        Mapeia estatísticas quânticas para nível de complexidade linguística.

        Lógica de mapeamento:
        - std > 0.8: Alta variabilidade → texto complexo (consoantes, estrutura)
        - std > 0.4: Média variabilidade → texto balanceado (mix equilibrado)
        - std ≤ 0.4: Baixa variabilidade → texto simples (vogais, básico)
        """
        std = psi_stats.get('std', 0.5)
        mean = psi_stats.get('mean', 0.0)

        # Alta variabilidade = texto complexo
        if std > 0.8:
            return 'high'
        # Média variabilidade = texto balanceado
        elif std > 0.4:
            return 'medium'
        # Baixa variabilidade = texto simples
        else:
            return 'low'

    def _boost_complex_characters(self, text: str, psi_stats: Dict[str, float]) -> str:
        """
        Aumenta probabilidade de caracteres complexos (consoantes, estrutura).
        Adequado para estados quânticos de alta variabilidade.
        """
        complex_chars = ['r', 't', 'n', 's', 'l', 'c', 'p', 'm', 'd', 'v']
        return self._apply_character_boost(text, complex_chars, boost_factor=1.3)

    def _boost_balanced_characters(self, text: str, psi_stats: Dict[str, float]) -> str:
        """
        Mantém equilíbrio entre vogais e consoantes.
        Adequado para estados quânticos de variabilidade média.
        """
        balanced_chars = ['a', 'e', 'o', 'i', 's', 'r', 'n', 't', 'm', 'd']
        return self._apply_character_boost(text, balanced_chars, boost_factor=1.2)

    def _boost_simple_characters(self, text: str, psi_stats: Dict[str, float]) -> str:
        """
        Prioriza caracteres simples e estruturais.
        Adequado para estados quânticos de baixa variabilidade.
        """
        simple_chars = ['a', 'e', 'i', 'o', 'u', ' ', '.', ',', 's', 'n']
        return self._apply_character_boost(text, simple_chars, boost_factor=1.1)

    def _apply_character_boost(self, text: str, boost_chars: List[str], boost_factor: float) -> str:
        """
        Aplica boost a caracteres específicos no texto.
        Esta é uma transformação leve que mantém a estrutura geral.
        """
        if not text:
            return text

        # Para implementação real, isso seria feito no nível de probabilidade
        # durante a geração. Aqui retornamos o texto original pois a
        # transformação real acontece na função de similaridade.
        return text

    def _apply_structural_guidance(self, text: str, psi_stats: Dict[str, float]) -> str:
        """
        Aplica orientação estrutural baseada no range dos valores quânticos.

        - Range amplo → mais pontuação e estrutura
        - Range estreito → fluxo mais contínuo
        """
        val_range = psi_stats.get('max', 0) - psi_stats.get('min', 0)

        # Range amplo = mais estrutura (pontos, vírgulas)
        if val_range > 5.0:
            return self._add_structural_elements(text, intensity=0.3)
        # Range médio = estrutura balanceada
        elif val_range > 2.0:
            return self._add_structural_elements(text, intensity=0.2)
        # Range estreito = fluxo contínuo
        else:
            return text

    def _add_structural_elements(self, text: str, intensity: float) -> str:
        """
        Adiciona elementos estruturais (pontuação) baseado na intensidade.
        """
        # Implementação simplificada - em produção isso seria feito
        # durante a geração baseada em probabilidade
        return text

    def get_coherence_score(self, text: str, psi_stats: Dict[str, float]) -> float:
        """
        Calcula score de coerência entre texto e estatísticas quânticas.

        Returns:
            Score entre 0.0 e 1.0 (maior = melhor coerência)
        """
        if not text:
            return 0.0

        complexity_level = self._map_quantum_to_complexity(psi_stats)
        text_complexity = self._analyze_text_complexity(text)

        # Score baseado na correspondência entre complexidade esperada e observada
        if complexity_level == text_complexity:
            return 0.9  # Perfeita correspondência
        elif abs(self._complexity_distance(complexity_level, text_complexity)) == 1:
            return 0.6  # Correspondência razoável
        else:
            return 0.3  # Pouca correspondência

    def _analyze_text_complexity(self, text: str) -> str:
        """
        Analisa complexidade do texto baseado na distribuição de caracteres.
        """
        if not text:
            return 'low'

        # Contar tipos de caracteres
        vowels = sum(1 for c in text.lower() if c in 'aeiou')
        consonants = sum(1 for c in text.lower() if c.isalpha() and c not in 'aeiou')
        punctuation = sum(1 for c in text if c in '.,!?;:')

        total_alpha = vowels + consonants
        if total_alpha == 0:
            return 'low'

        # Razões características
        vowel_ratio = vowels / total_alpha
        consonant_ratio = consonants / total_alpha
        punctuation_ratio = punctuation / len(text)

        # Lógica de classificação
        if consonant_ratio > 0.6 or punctuation_ratio > 0.1:
            return 'high'  # Muitas consoantes ou pontuação
        elif vowel_ratio > 0.5:
            return 'low'   # Muitas vogais
        else:
            return 'medium'  # Balanceado

    def _complexity_distance(self, level1: str, level2: str) -> int:
        """
        Calcula distância entre níveis de complexidade.
        """
        levels = {'low': 0, 'medium': 1, 'high': 2}
        return abs(levels.get(level1, 1) - levels.get(level2, 1))


# Função de interface para integração com o pipeline ΨQRH
def create_semantic_coherence_layer() -> SemanticCoherenceLayer:
    """
    Factory function para criar instância da camada de coerência semântica.
    """
    return SemanticCoherenceLayer()


# Teste da implementação
if __name__ == "__main__":
    # Exemplo de uso
    layer = create_semantic_coherence_layer()

    # Estatísticas de exemplo (do log do sistema)
    psi_stats = {
        'mean': -0.2848,
        'std': 0.8005,
        'min': -7.5796,
        'max': 0.6742,
        'finite': True
    }

    # Texto de exemplo
    test_text = "aaaaadioiaa?auaauu?  ?a??????auuu?"

    # Aplicar orientação
    guided_text = layer.apply_quantum_guidance(psi_stats, test_text)

    # Calcular score de coerência
    coherence_score = layer.get_coherence_score(guided_text, psi_stats)

    print(f"Texto original: {test_text}")
    print(f"Texto guiado: {guided_text}")
    print(f"Score de coerência: {coherence_score:.3f}")
    print(f"Nível de complexidade detectado: {layer._map_quantum_to_complexity(psi_stats)}")