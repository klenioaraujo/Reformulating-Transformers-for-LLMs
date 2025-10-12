#!/usr/bin/env python3
"""
Sistema Híbrido Quântico-Clássico para ΨQRH
=============================================

Resolve o divórcio entre física quântica avançada e geração linguística limitada.
Implementa transição de fase crítica entre regimes quântico e clássico.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import math
import re


class CriticalPhaseTransition:
    """
    Teoria de Transição de Fase Linguística

    Controla a transição entre:
    - Fase desordenada (fonemas quânticos aleatórios)
    - Fase ordenada (linguagem estruturada clássica)
    """

    def __init__(self, critical_temperature: float = 1.0):
        self.T_c = critical_temperature
        self.order_parameter = None
        self.correlation_length = 1.0

    def compute_linguistic_order_parameter(self, quantum_state: torch.Tensor,
                                        linguistic_context: str) -> float:
        """
        Parâmetro de ordem que mede transição entre caos e estrutura

        Baseado na teoria de Landau-Ginzburg para transições de fase
        """
        # Medir coerência quântica do estado
        quantum_coherence = self._quantum_coherence(quantum_state)

        # Medir estrutura linguística esperada do contexto
        linguistic_structure = self._expected_structure(linguistic_context)

        # Parâmetro de ordem crítico
        order_param = quantum_coherence * linguistic_structure

        # Fator de correlação exponencial
        correlation = math.exp(-self.correlation_length / max(len(linguistic_context), 1))

        return float(order_param * correlation)

    def should_trigger_phase_transition(self, T_quantum: float,
                                      order_param: float,
                                      context_length: int = 0) -> bool:
        """
        Decidir quando transicionar do regime quântico para linguístico

        Critérios baseados na física de transições de fase:
        - Temperatura quântica abaixo da crítica
        - Parâmetro de ordem acima do threshold
        - Contexto suficiente disponível
        """
        temperature_condition = T_quantum < self.T_c
        order_condition = order_param > 0.6  # Threshold reduzido para maior sensibilidade
        context_condition = context_length > 3  # Mínimo de contexto

        return temperature_condition and order_condition and context_condition

    def _quantum_coherence(self, quantum_state: torch.Tensor) -> float:
        """Medir coerência quântica do estado"""
        # Handle different tensor dimensions
        if quantum_state.dim() == 4:  # [batch, seq, embed, 4] - quaternion format
            # Flatten to compute coherence across all dimensions
            flat_state = quantum_state.flatten()
        elif quantum_state.dim() == 2:  # [batch, features]
            flat_state = quantum_state.flatten()
        else:  # 1D tensor
            flat_state = quantum_state

        # Coerência como |⟨ψ|ψ⟩|² / ||ψ||⁴
        norm = torch.norm(flat_state)
        if norm > 0:
            coherence = torch.abs(torch.dot(flat_state.conj(), flat_state)) / (norm ** 4)
            return float(torch.clamp(coherence, 0, 1))
        return 0.0

    def _expected_structure(self, context: str) -> float:
        """Medir estrutura linguística esperada"""
        if not context:
            return 0.0

        # Fatores de estrutura
        length_factor = min(len(context) / 20, 1.0)  # Contextos mais longos têm mais estrutura
        word_factor = len(context.split()) / max(len(context) / 5, 1)  # Razão palavra/caractere
        punctuation_factor = len(re.findall(r'[.!?]', context)) / max(len(context) / 50, 1)

        # Combinação ponderada
        structure = 0.4 * length_factor + 0.4 * word_factor + 0.2 * punctuation_factor

        return min(structure, 1.0)


class QuantumClassicalInterface:
    """
    Interface Quântico-Clássica com Mapeamento Adiabático

    Preserva invariantes topológicos durante a transição de fase
    """

    def __init__(self, adiabatic_speed: float = 0.1):
        self.adiabatic_speed = adiabatic_speed
        self.ground_states = {}

    def adiabatic_mapping(self, quantum_state: torch.Tensor,
                         classical_template: str) -> str:
        """
        Mapeamento adiabático preservando topologia
        """
        # Extrair invariantes topológicos
        topological_invariants = self._extract_topological_invariants(quantum_state)

        # Mapear preservando topologia
        linguistic_structure = self._topology_preserving_map(
            topological_invariants, classical_template
        )

        # Evolução adiabática
        final_output = self._adiabatic_evolution(linguistic_structure)

        return final_output

    def _extract_topological_invariants(self, quantum_state: torch.Tensor) -> Dict[str, float]:
        """Extrair invariantes que sobrevivem à transição quântico-clássica"""
        return {
            'winding_number': self._compute_winding_number(quantum_state),
            'berry_phase': self._compute_berry_phase(quantum_state),
            'entanglement_entropy': self._entanglement_entropy(quantum_state),
            'symmetry_measure': self._detect_symmetries(quantum_state)
        }

    def _compute_winding_number(self, state: torch.Tensor) -> float:
        """Número de enrolamento topológico"""
        # Simplificado: baseado na fase total
        phase = torch.angle(state).flatten()
        if len(phase) > 1:
            phase_diff = torch.diff(phase)
            winding = torch.sum(torch.abs(phase_diff) > torch.pi).float()
            return float(winding / len(phase))
        return 0.0

    def _compute_berry_phase(self, state: torch.Tensor) -> float:
        """Fase de Berry (geometria quântica)"""
        # Simplificado: curvatura da fase
        phase = torch.angle(state)
        if phase.numel() > 1:
            curvature = torch.var(phase)
            return float(torch.clamp(curvature, 0, 2*torch.pi))
        return 0.0

    def _entanglement_entropy(self, state: torch.Tensor) -> float:
        """Entropia de emaranhamento"""
        # Para estado puro, entropia de von Neumann
        if state.numel() > 1:
            probs = torch.abs(state.flatten())**2
            probs = probs / torch.sum(probs)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            return float(entropy)
        return 0.0

    def _detect_symmetries(self, state: torch.Tensor) -> float:
        """Medir simetrias do estado"""
        # Simetria de reflexão simples
        if state.numel() > 2:
            left_half = state[:len(state)//2]
            right_half = state[len(state)//2:]
            symmetry = 1.0 - torch.mean(torch.abs(left_half - right_half.flip(0)))
            return float(symmetry)
        return 0.0

    def _topology_preserving_map(self, invariants: Dict[str, float],
                               template: str) -> str:
        """Mapear preservando topologia"""
        # Usar invariantes para modificar template
        symmetry_factor = invariants.get('symmetry_measure', 0.5)

        # Aplicar transformações baseadas em simetria
        if symmetry_factor > 0.7:
            # Alta simetria: preservar estrutura
            return template
        elif symmetry_factor > 0.4:
            # Simetria média: modificar ligeiramente
            return self._apply_symmetric_modifications(template)
        else:
            # Baixa simetria: transformar significativamente
            return self._apply_asymmetric_transformations(template)

    def _apply_symmetric_modifications(self, text: str) -> str:
        """Modificações que preservam simetria"""
        # Exemplo: adicionar palavras simétricas
        words = text.split()
        if len(words) >= 2:
            # Inserir palavra no centro
            mid = len(words) // 2
            words.insert(mid, "quantum")
        return ' '.join(words)

    def _apply_asymmetric_transformations(self, text: str) -> str:
        """Transformações assimétricas"""
        # Exemplo: reordenar baseado em complexidade
        words = text.split()
        if len(words) > 1:
            # Reordenar por comprimento
            words.sort(key=len, reverse=True)
        return ' '.join(words)

    def _adiabatic_evolution(self, structure: str) -> str:
        """Evolução adiabática final"""
        # Aplicar refinamentos graduais
        evolved = structure

        # Correção gradual de erros
        evolved = self._correct_grammar_gradually(evolved)
        evolved = self._improve_coherence_gradually(evolved)

        return evolved

    def _correct_grammar_gradually(self, text: str) -> str:
        """Correção gramatical gradual"""
        # Correções básicas
        corrections = {
            r'\ba\b': 'a',
            r'\ban\b': 'an',
            r'\bthe\b': 'the',
        }

        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _improve_coherence_gradually(self, text: str) -> str:
        """Melhoria gradual de coerência"""
        # Adicionar conectores se apropriado
        words = text.split()
        if len(words) > 3:
            # Inserir conectores
            connectors = ['and', 'or', 'but', 'so', 'because']
            insert_positions = [i for i in range(1, len(words)-1, 2)]

            for pos in reversed(insert_positions[:2]):  # Máximo 2 conectores
                connector = np.random.choice(connectors)
                words.insert(pos, connector)

        return ' '.join(words)


class QuantumConstrainedLinguisticProcessor:
    """
    Processador Linguístico com Restrições Quânticas

    Geração de texto informada por restrições derivadas da física quântica
    """

    def __init__(self):
        self.phoneme_inventory = self._load_articulatory_phonemes()
        self.quantum_constraints = QuantumConstraints()
        self.structure_generator = LinguisticStructureGenerator()

    def generate_quantum_informed_text(self, context: str,
                                     quantum_constraints: Dict) -> str:
        """
        Geração de texto usando restrições quânticas
        """
        # 1. Analisar contexto com restrições quânticas
        context_analysis = self._analyze_context_with_quantum_constraints(
            context, quantum_constraints
        )

        # 2. Gerar estrutura baseada em simetrias quânticas
        syntactic_structure = self._quantum_informed_syntax(
            context_analysis, quantum_constraints
        )

        # 3. Preencher com fonemas condicionados
        phoneme_sequence = self._quantum_articulatory_phonemes(
            syntactic_structure, quantum_constraints
        )

        # 4. Aplicar correções de coerência
        final_text = self._apply_quantum_coherence_corrections(
            phoneme_sequence, quantum_constraints
        )

        return final_text

    def _load_articulatory_phonemes(self) -> Dict[str, List[str]]:
        """Carregar inventário fonêmico articulatório"""
        return {
            'vowels': ['a', 'e', 'i', 'o', 'u', 'ə'],
            'consonants': ['m', 'n', 'p', 't', 'k', 's', 'l', 'r'],
            'liquids': ['w', 'j', 'h'],
            'punctuation': [' ', '.', ',', '!', '?']
        }

    def _analyze_context_with_quantum_constraints(self, context: str,
                                                constraints: Dict) -> Dict:
        """Análise de contexto informada por restrições quânticas"""
        analysis = {
            'length': len(context),
            'words': len(context.split()),
            'complexity': self._compute_context_complexity(context),
            'quantum_influence': constraints.get('symmetry_measure', 0.5)
        }

        # Modificar análise baseada em restrições quânticas
        if constraints.get('entanglement_entropy', 0) > 1.0:
            analysis['complexity'] *= 1.5  # Aumentar complexidade esperada

        return analysis

    def _quantum_informed_syntax(self, analysis: Dict, constraints: Dict) -> Dict:
        """Gerar estrutura sintática informada por restrições quânticas"""
        # Estrutura básica
        structure = {
            'sentence_type': 'simple',
            'word_count': max(3, min(analysis['words'] + 2, 10)),
            'complexity_level': analysis['complexity']
        }

        # Modificar baseado em simetria quântica
        symmetry = constraints.get('symmetry_measure', 0.5)
        if symmetry > 0.7:
            structure['sentence_type'] = 'compound'
            structure['word_count'] = int(structure['word_count'] * 1.5)
        elif symmetry < 0.3:
            structure['sentence_type'] = 'complex'
            structure['complexity_level'] *= 1.3

        return structure

    def _quantum_articulatory_phonemes(self, structure: Dict,
                                     constraints: Dict) -> List[str]:
        """Selecionar fonemas respeitando restrições quânticas"""
        phonemes = []

        # Gerar baseado na estrutura
        for i in range(structure['word_count']):
            # Selecionar tipo de fonema baseado em posição
            if i % 4 == 0:  # Início de palavra
                candidates = self.phoneme_inventory['consonants']
            elif i % 4 == 1:  # Meio de palavra
                candidates = self.phoneme_inventory['vowels']
            elif i % 4 == 2:  # Meio/final
                candidates = self.phoneme_inventory['liquids'] + self.phoneme_inventory['consonants']
            else:  # Final
                candidates = self.phoneme_inventory['punctuation']

            # Filtrar por restrições quânticas
            quantum_valid = [
                p for p in candidates
                if self.quantum_constraints.respects_symmetry(p, constraints)
            ]

            if quantum_valid:
                # Selecionar baseado em amplitude quântica
                chosen = self._select_by_quantum_amplitude(quantum_valid, constraints)
            else:
                chosen = np.random.choice(candidates)

            phonemes.append(chosen)

        return phonemes

    def _select_by_quantum_amplitude(self, candidates: List[str],
                                   constraints: Dict) -> str:
        """Selecionar fonema baseado em amplitude quântica"""
        # Usar entropia como critério de seleção
        entropy = constraints.get('entanglement_entropy', 1.0)

        if entropy > 1.5:
            # Alta entropia: preferir consoantes complexas
            preference = ['k', 's', 't', 'p']
        elif entropy > 1.0:
            # Entropia média: vogais
            preference = ['a', 'e', 'i', 'o', 'u']
        else:
            # Baixa entropia: sons simples
            preference = ['m', 'n', ' ', '.']

        # Interseção com candidatos
        valid_preferred = [p for p in preference if p in candidates]

        if valid_preferred:
            return np.random.choice(valid_preferred)
        else:
            return np.random.choice(candidates)

    def _apply_quantum_coherence_corrections(self, phonemes: List[str],
                                           constraints: Dict) -> str:
        """Aplicar correções de coerência quântica"""
        text = ''.join(phonemes)

        # Correções baseadas em invariantes quânticos
        symmetry = constraints.get('symmetry_measure', 0.5)

        if symmetry > 0.8:
            # Alta simetria: adicionar estrutura simétrica
            text = self._add_symmetric_structure(text)
        elif symmetry < 0.2:
            # Baixa simetria: adicionar conectores
            text = self._add_connectors(text)

        return text

    def _add_symmetric_structure(self, text: str) -> str:
        """Adicionar estrutura simétrica"""
        words = text.split()
        if len(words) >= 3:
            # Adicionar palavra simétrica no centro
            mid = len(words) // 2
            words.insert(mid, "quantum")
        return ' '.join(words)

    def _add_connectors(self, text: str) -> str:
        """Adicionar conectores para melhorar fluxo"""
        words = text.split()
        connectors = ['and', 'or', 'but', 'so']

        if len(words) > 4:
            # Inserir conector
            pos = len(words) // 2
            connector = np.random.choice(connectors)
            words.insert(pos, connector)

        return ' '.join(words)

    def _compute_context_complexity(self, context: str) -> float:
        """Computar complexidade do contexto"""
        if not context:
            return 0.0

        # Métricas de complexidade
        length = len(context)
        unique_chars = len(set(context))
        word_count = len(context.split())

        # Complexidade combinada
        complexity = (unique_chars / max(length, 1)) * (word_count / max(length / 5, 1))

        return min(complexity, 1.0)


class QuantumConstraints:
    """Restrições derivadas da física quântica"""

    def respects_symmetry(self, phoneme: str, constraints: Dict) -> bool:
        """Verificar se fonema respeita simetrias quânticas"""
        symmetry = constraints.get('symmetry_measure', 0.5)

        # Regras simples baseadas em simetria
        if symmetry > 0.7:
            # Alta simetria: preferir fonemas "equilibrados"
            return phoneme in ['a', 'e', 'i', 'm', 'n', ' ']
        elif symmetry < 0.3:
            # Baixa simetria: permitir mais variedade
            return True
        else:
            # Simetria média: equilíbrio
            return len(phoneme) <= 1  # Preferir caracteres únicos


class LinguisticStructureGenerator:
    """Gerador de estruturas linguísticas"""

    def generate_structure(self, analysis: Dict) -> Dict:
        """Gerar estrutura linguística baseada na análise"""
        return {
            'type': 'sentence',
            'complexity': analysis.get('complexity', 0.5),
            'length': analysis.get('words', 5)
        }


class HybridQuantumClassicalSystem:
    """
    Sistema Híbrido Quântico-Clássico

    Combina física quântica avançada com processamento linguístico clássico
    para resolver o divórcio entre física e linguística.
    """

    def __init__(self):
        self.phase_controller = CriticalPhaseTransition()
        self.interface = QuantumClassicalInterface()
        self.linguistic_processor = QuantumConstrainedLinguisticProcessor()

        # Métricas de desempenho
        self.performance_metrics = {
            'quantum_calls': 0,
            'classical_calls': 0,
            'hybrid_calls': 0,
            'average_quality': 0.0
        }

    def hybrid_text_generation(self, input_text: str,
                             quantum_features: Optional[Dict] = None) -> str:
        """
        Geração de texto híbrida com decisão dinâmica de método

        Args:
            input_text: Texto de entrada
            quantum_features: Características quânticas (opcional)

        Returns:
            Texto gerado usando abordagem apropriada
        """
        # Simular características quânticas se não fornecidas
        if quantum_features is None:
            quantum_features = self._simulate_quantum_features(input_text)

        # Extrair parâmetros críticos
        T_quantum = quantum_features.get('quantum_temperature', 1.0)
        quantum_state = quantum_features.get('quantum_state', torch.randn(10))

        # Computar parâmetro de ordem
        order_param = self.phase_controller.compute_linguistic_order_parameter(
            quantum_state, input_text
        )

        # Decidir método baseado na física
        context_length = len(input_text.split())

        if self.phase_controller.should_trigger_phase_transition(
            T_quantum, order_param, context_length):

            # MODO HÍBRIDO: física quântica + linguística clássica
            self.performance_metrics['hybrid_calls'] += 1
            return self._hybrid_mode(quantum_features, input_text)

        elif T_quantum < 0.3:
            # MODO QUÂNTICO PURO: baixa temperatura
            self.performance_metrics['quantum_calls'] += 1
            return self._pure_quantum_mode(quantum_features)

        else:
            # MODO CLÁSSICO: alta temperatura/desordenado
            self.performance_metrics['classical_calls'] += 1
            return self._classical_fallback(input_text)

    def _hybrid_mode(self, quantum_features: Dict, context: str) -> str:
        """Modo híbrido: combinação ótima de física e linguística"""
        # Extrair invariantes quânticos
        quantum_invariants = self._extract_quantum_invariants(quantum_features)

        # Usar invariantes para condicionar geração clássica
        conditioned_output = self.linguistic_processor.generate_quantum_informed_text(
            context,
            quantum_constraints=quantum_invariants
        )

        # Aplicar mapeamento adiabático final
        quantum_state = quantum_features.get('quantum_state', torch.randn(10))
        final_output = self.interface.adiabatic_mapping(quantum_state, conditioned_output)

        return final_output

    def _pure_quantum_mode(self, quantum_features: Dict) -> str:
        """Modo quântico puro para estados de baixa temperatura"""
        # Gerar baseado apenas em características quânticas
        quantum_state = quantum_features.get('quantum_state', torch.randn(10))

        # Mapeamento direto para texto simples
        text_length = max(5, int(quantum_features.get('coherence', 0.5) * 20))

        # Gerar texto baseado em padrões quânticos
        base_words = ['quantum', 'field', 'state', 'wave', 'particle', 'energy']
        selected_words = np.random.choice(base_words, size=min(text_length//2, len(base_words)))

        return ' '.join(selected_words)

    def _classical_fallback(self, context: str) -> str:
        """ZERO FALLBACK POLICY: Sistema deve falhar claramente"""
        raise RuntimeError("Hybrid quantum-classical system failed - ZERO FALLBACK POLICY: No classical fallback allowed")

    def _simulate_quantum_features(self, text: str) -> Dict:
        """Simular características quânticas para teste"""
        length = len(text)
        complexity = len(set(text)) / max(length, 1)

        return {
            'quantum_temperature': 0.8 - 0.4 * complexity,  # Temperatura baseada em complexidade
            'coherence': complexity,
            'quantum_state': torch.randn(max(10, length//2)),
            'symmetry_measure': 0.5 + 0.3 * np.sin(length / 10),
            'entanglement_entropy': complexity * 2.0
        }

    def _extract_quantum_invariants(self, features: Dict) -> Dict:
        """Extrair invariantes quânticos para interface"""
        return {
            'symmetry_measure': features.get('symmetry_measure', 0.5),
            'entanglement_entropy': features.get('entanglement_entropy', 1.0),
            'coherence': features.get('coherence', 0.5),
            'quantum_temperature': features.get('quantum_temperature', 1.0)
        }

    def get_performance_metrics(self) -> Dict:
        """Retornar métricas de desempenho do sistema híbrido"""
        return self.performance_metrics.copy()

    def update_quality_metric(self, quality_score: float):
        """Atualizar métrica de qualidade média"""
        total_calls = (self.performance_metrics['quantum_calls'] +
                      self.performance_metrics['classical_calls'] +
                      self.performance_metrics['hybrid_calls'])

        if total_calls > 0:
            self.performance_metrics['average_quality'] = (
                (self.performance_metrics['average_quality'] * (total_calls - 1) + quality_score)
                / total_calls
            )


# Função de compatibilidade
def create_hybrid_system() -> HybridQuantumClassicalSystem:
    """
    Factory function para criar sistema híbrido

    Returns:
        Sistema híbrido quântico-clássico configurado
    """
    return HybridQuantumClassicalSystem()


if __name__ == "__main__":
    # Teste do sistema híbrido
    print("🔬 Testando Sistema Híbrido Quântico-Clássico...")

    system = create_hybrid_system()

    test_inputs = [
        "hello world",
        "quantum mechanics",
        "prove that root two is irrational",
        "the system works perfectly"
    ]

    for input_text in test_inputs:
        print(f"\n📝 Entrada: '{input_text}'")

        # Simular características quânticas
        quantum_features = system._simulate_quantum_features(input_text)
        print(f"🔬 Características quânticas: T={quantum_features['quantum_temperature']:.2f}, "
              f"simetria={quantum_features['symmetry_measure']:.2f}")

        # Gerar texto híbrido
        output = system.hybrid_text_generation(input_text, quantum_features)
        print(f"📤 Saída híbrida: '{output}'")

    # Métricas finais
    metrics = system.get_performance_metrics()
    print(f"\n📊 Métricas finais: {metrics}")

    print("✅ Sistema híbrido quântico-clássico inicializado com sucesso!")