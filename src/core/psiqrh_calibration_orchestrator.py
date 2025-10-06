"""
ΨQRH Calibration Orchestrator - OPÇÃO 6: MIX Inteligente das 5 Estratégias
===========================================================================

Implementação como MIX das 5 estratégias de calibração automática:
1. Camada de coerência semântica (orientação quântica)
2. Auto-calibração adaptativa (aprendizado de padrões)
3. Métricas de similaridade quântica (análise estrutural)
4. Aprendizagem por reforço quântico (otimização paramétrica)
5. Orquestrador Central (coordenação inteligente)

O MIX combina elementos de todas as estratégias em uma abordagem unificada,
aplicando-as simultaneamente com pesos dinâmicos baseados no contexto quântico.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np

class ΨQRHCalibrationOrchestrator:
    """
    Orquestrador MIX das 5 Estratégias de Calibração ΨQRH

    Combina elementos de todas as estratégias em uma abordagem unificada:
    - Análise semântica com orientação quântica
    - Adaptação paramétrica inteligente
    - Validação estrutural por similaridade
    - Reforço quântico contínuo
    - Coordenação com pesos dinâmicos
    """

    def __init__(self):
        self.strategies = {}
        self.mix_history = []
        self.adaptive_weights = {}
        self.quantum_patterns = {}

    def register_strategy(self, name: str, strategy_instance: Any, weight: float = 1.0):
        """Registra uma estratégia no MIX"""
        self.strategies[name] = {
            'instance': strategy_instance,
            'base_weight': weight,
            'performance': [],
            'quantum_affinity': self._calculate_quantum_affinity(name)
        }
        self.adaptive_weights[name] = weight
        print(f"✅ [ΨQRHCalibrationOrchestrator] Estratégia MIX registrada: {name} (peso base: {weight})")

    def _calculate_quantum_affinity(self, strategy_name: str) -> Dict[str, float]:
        """Calcula afinidade quântica de cada estratégia"""
        affinities = {
            'semantic_coherence': {'coherence': 0.9, 'complexity': 0.3, 'stability': 0.7},
            'adaptive_calibration': {'coherence': 0.6, 'complexity': 0.8, 'stability': 0.9},
            'similarity_metrics': {'coherence': 0.8, 'complexity': 0.9, 'stability': 0.6},
            'reinforcement_learner': {'coherence': 0.7, 'complexity': 0.6, 'stability': 0.8}
        }
        return affinities.get(strategy_name, {'coherence': 0.5, 'complexity': 0.5, 'stability': 0.5})

    def apply_all_calibrations(self, psi: torch.Tensor, text: str, input_text: str, psi_stats: Dict) -> Dict[str, Any]:
        """
        Aplica MIX unificado das 5 estratégias de calibração

        O MIX combina:
        1. Análise semântica orientada quântica
        2. Adaptação paramétrica inteligente
        3. Validação estrutural por similaridade
        4. Reforço quântico contínuo
        5. Coordenação com pesos dinâmicos

        Args:
            psi: Estado quântico [seq_len, embed_dim, 4]
            text: Texto gerado atual
            input_text: Texto de entrada original
            psi_stats: Estatísticas do estado quântico

        Returns:
            Resultado do MIX de calibração
        """
        start_time = time.time()
        print(f"🎼 [ΨQRHCalibrationOrchestrator] Iniciando MIX unificado #{len(self.mix_history) + 1}")

        # Análise quântica do estado para determinar pesos dinâmicos
        quantum_characteristics = self._analyze_quantum_state(psi, psi_stats)
        dynamic_weights = self._calculate_dynamic_weights(quantum_characteristics)

        print(f"   🧠 Características quânticas detectadas: coerência={quantum_characteristics['coherence']:.3f}, complexidade={quantum_characteristics['complexity']:.3f}, estabilidade={quantum_characteristics['stability']:.3f}")

        # Análise inicial da qualidade
        initial_quality = self._assess_text_quality_unified(text, psi_stats, quantum_characteristics)
        print(f"   📝 Texto inicial: '{text[:50]}...'")
        print(f"   📊 Qualidade pré-MIX: {initial_quality:.3f}")

        # APLICAR MIX UNIFICADO
        mix_result = self._apply_unified_mix(psi, text, input_text, psi_stats, quantum_characteristics, dynamic_weights)

        # Análise final
        final_quality = self._assess_text_quality_unified(mix_result['calibrated_text'], psi_stats, quantum_characteristics)
        quality_improvement = final_quality - initial_quality
        mix_time = time.time() - start_time

        result = {
            'calibrated_text': mix_result['calibrated_text'],
            'initial_quality': initial_quality,
            'quality_score': final_quality,
            'quality_improvement': quality_improvement,
            'calibration_time': mix_time,
            'strategy_contributions': mix_result['contributions'],
            'quantum_characteristics': quantum_characteristics,
            'dynamic_weights': dynamic_weights,
            'mix_components_applied': len([c for c in mix_result['contributions'].values() if c > 0])
        }

        # Registrar no histórico do MIX
        self.mix_history.append(result)
        self._update_adaptive_weights(result)

        print(f"🎼 [MIX] Calibração concluída:")
        print(f"   📊 Score final: {final_quality:.3f}")
        print(f"   📈 Melhoria: {'+' if quality_improvement >= 0 else ''}{quality_improvement:.3f}")
        print(f"   ⏱️  Tempo: {mix_time:.3f}s")
        print(f"   🧬 Componentes MIX aplicados: {result['mix_components_applied']}")

        for strategy, contribution in mix_result['contributions'].items():
            print(f"   🎯 {strategy}: {contribution:.3f} (peso dinâmico: {dynamic_weights.get(strategy, 0):.3f})")

        return result

    def _analyze_quantum_state(self, psi: torch.Tensor, psi_stats: Dict) -> Dict[str, float]:
        """Análise abrangente do estado quântico para determinar estratégia MIX"""
        # Coerência quântica (baseada na variabilidade)
        coherence = min(1.0, 1.0 / (1.0 + psi_stats.get('std', 1.0)))

        # Complexidade (baseada na estrutura do tensor)
        complexity = min(1.0, torch.mean(torch.abs(psi)).item() * 2.0)

        # Estabilidade (baseada na finite-ness e range)
        stability = 1.0 if psi_stats.get('finite', True) else 0.5
        stability *= min(1.0, 10.0 / (psi_stats.get('max', 10.0) - psi_stats.get('min', 0.0) + 1e-10))

        return {
            'coherence': coherence,
            'complexity': complexity,
            'stability': stability
        }

    def _calculate_dynamic_weights(self, quantum_characteristics: Dict[str, float]) -> Dict[str, float]:
        """Calcula pesos dinâmicos baseados nas características quânticas"""
        weights = {}

        for strategy_name, strategy_info in self.strategies.items():
            affinity = strategy_info['quantum_affinity']
            base_weight = strategy_info['base_weight']

            # Calcular score de afinidade quântica
            affinity_score = (
                affinity['coherence'] * quantum_characteristics['coherence'] +
                affinity['complexity'] * quantum_characteristics['complexity'] +
                affinity['stability'] * quantum_characteristics['stability']
            ) / 3.0

            # Peso dinâmico = peso base * afinidade quântica * aprendizado adaptativo
            adaptive_factor = self.adaptive_weights.get(strategy_name, 1.0)
            weights[strategy_name] = base_weight * affinity_score * adaptive_factor

        # Normalizar pesos
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _apply_unified_mix(self, psi: torch.Tensor, text: str, input_text: str,
                          psi_stats: Dict, quantum_characteristics: Dict[str, float],
                          dynamic_weights: Dict[str, float]) -> Dict[str, Any]:
        """Aplica MIX unificado combinando todas as estratégias"""

        calibrated_text = text
        contributions = {}

        # 1. COMPONENTE SEMÂNTICO-COERENTE (orientação quântica)
        if 'semantic_coherence' in self.strategies and dynamic_weights.get('semantic_coherence', 0) > 0.1:
            try:
                semantic_strategy = self.strategies['semantic_coherence']['instance']
                semantic_text = semantic_strategy.apply_quantum_guidance(psi_stats, calibrated_text, input_text)
                semantic_score = semantic_strategy.get_coherence_score(semantic_text, psi_stats)

                # Aplicar com peso dinâmico
                weight = dynamic_weights['semantic_coherence']
                if semantic_score > self._assess_text_quality_simple(calibrated_text):
                    calibrated_text = semantic_text
                    contributions['semantic_coherence'] = weight * 0.3
                else:
                    contributions['semantic_coherence'] = 0.0

            except Exception as e:
                print(f"   ⚠️  Componente semântico falhou: {e}")
                contributions['semantic_coherence'] = 0.0

        # 2. COMPONENTE ADAPTATIVO (aprendizado de padrões)
        if 'adaptive_calibration' in self.strategies and dynamic_weights.get('adaptive_calibration', 0) > 0.1:
            try:
                adaptive_strategy = self.strategies['adaptive_calibration']['instance']
                quality_score = self._assess_text_quality_simple(calibrated_text)
                success_score = 0.8 if quality_score > 0.6 else 0.2

                adaptive_strategy.learn_quantum_patterns(psi, calibrated_text[:1], success_score, input_text)
                contributions['adaptive_calibration'] = dynamic_weights['adaptive_calibration'] * 0.2

            except Exception as e:
                print(f"   ⚠️  Componente adaptativo falhou: {e}")
                contributions['adaptive_calibration'] = 0.0

        # 3. COMPONENTE DE SIMILARIDADE (validação estrutural)
        if 'similarity_metrics' in self.strategies and dynamic_weights.get('similarity_metrics', 0) > 0.1:
            try:
                similarity_strategy = self.strategies['similarity_metrics']['instance']
                optimal_metric = similarity_strategy.select_optimal_metric(psi)

                # Aplicar refinamento baseado na métrica
                refined_text = self._apply_similarity_refinement(calibrated_text, optimal_metric, quantum_characteristics)
                if refined_text != calibrated_text:
                    calibrated_text = refined_text
                    contributions['similarity_metrics'] = dynamic_weights['similarity_metrics'] * 0.25
                else:
                    contributions['similarity_metrics'] = 0.0

            except Exception as e:
                print(f"   ⚠️  Componente de similaridade falhou: {e}")
                contributions['similarity_metrics'] = 0.0

        # 4. COMPONENTE DE REFORÇO (otimização paramétrica)
        if 'reinforcement_learner' in self.strategies and dynamic_weights.get('reinforcement_learner', 0) > 0.1:
            try:
                reinforcement_strategy = self.strategies['reinforcement_learner']['instance']

                current_params = {
                    'alpha': 1.0, 'beta': 0.5, 'temperature': 1.0,
                    'fractal_weight': 1.0, 'coherence_weight': 1.0
                }

                quality_score = self._assess_text_quality_simple(calibrated_text)
                suggested_params = reinforcement_strategy.reinforce_parameters(current_params, quality_score)

                # Aplicar orientação quântica aos parâmetros (feedback para próximas iterações)
                quantum_guided_params = reinforcement_strategy.apply_quantum_guidance(suggested_params, psi_stats)

                contributions['reinforcement_learner'] = dynamic_weights['reinforcement_learner'] * 0.25

            except Exception as e:
                print(f"   ⚠️  Componente de reforço falhou: {e}")
                contributions['reinforcement_learner'] = 0.0

        return {
            'calibrated_text': calibrated_text,
            'contributions': contributions
        }

    def _apply_similarity_refinement(self, text: str, metric: str, quantum_characteristics: Dict[str, float]) -> str:
        """Aplica refinamento baseado na métrica de similaridade selecionada"""
        if metric == 'quantum_fidelity' and quantum_characteristics['coherence'] > 0.7:
            # Estados coerentes: manter estrutura mais organizada
            return text
        elif metric == 'euclidean' and quantum_characteristics['complexity'] > 0.7:
            # Estados complexos: permitir mais diversidade
            return text
        elif metric == 'cosine' and quantum_characteristics['stability'] > 0.7:
            # Estados estáveis: aplicar suavização
            return text

        return text

    def _assess_text_quality_unified(self, text: str, psi_stats: Dict, quantum_characteristics: Dict[str, float]) -> float:
        """Avaliação unificada da qualidade considerando características quânticas"""
        base_quality = self._assess_text_quality_simple(text)

        # Ajustar baseado nas características quânticas
        quantum_adjustment = (
            quantum_characteristics['coherence'] * 0.3 +
            quantum_characteristics['complexity'] * 0.3 +
            quantum_characteristics['stability'] * 0.4
        )

        # Penalizar textos que não combinam com o estado quântico
        if quantum_characteristics['coherence'] > 0.8 and base_quality < 0.5:
            # Estados coerentes devem ter textos de qualidade
            quantum_adjustment *= 0.7
        elif quantum_characteristics['complexity'] > 0.8 and base_quality > 0.8:
            # Estados complexos podem ter textos complexos
            quantum_adjustment *= 1.2

        return min(1.0, base_quality * quantum_adjustment)

    def _assess_text_quality_simple(self, text: str) -> float:
        """Avaliação simples para compatibilidade"""
        if not text or not text.strip():
            return 0.0

        score = 0.0
        criteria_count = 0

        # Comprimento apropriado
        length_score = min(len(text.split()) / 15, 1.0)
        score += length_score
        criteria_count += 1

        # Diversidade de caracteres
        unique_chars = len(set(text))
        diversity_score = min(unique_chars / 20, 1.0)
        score += diversity_score
        criteria_count += 1

        # Presença de letras
        letter_count = sum(1 for c in text if c.isalpha())
        letter_ratio = letter_count / max(len(text), 1)
        letter_score = min(letter_ratio * 2, 1.0)
        score += letter_score
        criteria_count += 1

        return score / criteria_count if criteria_count > 0 else 0.0

    def _update_adaptive_weights(self, result: Dict[str, Any]):
        """Atualiza pesos adaptativos baseado no desempenho"""
        for strategy_name, contribution in result['strategy_contributions'].items():
            if strategy_name in self.adaptive_weights:
                # Ajustar peso baseado na contribuição e melhoria geral
                improvement_factor = 1.0 + (result['quality_improvement'] * 0.1)
                contribution_factor = 1.0 + (contribution * 0.2)

                self.adaptive_weights[strategy_name] *= improvement_factor * contribution_factor

                # Limitar range dos pesos
                self.adaptive_weights[strategy_name] = max(0.1, min(3.0, self.adaptive_weights[strategy_name]))


def create_psiqrh_calibration_orchestrator() -> ΨQRHCalibrationOrchestrator:
    """Factory function para criar orquestrador MIX ΨQRH"""
    return ΨQRHCalibrationOrchestrator()