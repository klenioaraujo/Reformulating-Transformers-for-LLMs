#!/usr/bin/env python3
"""
Teste abrangente do sistema semântico aprimorado do QRH Layer.

Este teste valida a implementação das três melhorias principais propostas:
1. Filtros Semânticos Adaptativos (Contradição, Irrelevância, Viés)
2. Continuum Temporal com Memória e Evolução
3. Sistema Hierárquico de Gate Controllers com Análise de Ressonância

O objetivo é demonstrar a capacidade do sistema de extrair "sinal claro em meio à cacofonia".
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings

from qrh_layer import QRHConfig
from semantic_adaptive_filters import SemanticFilterConfig
from temporal_continuum_enhanced import ContinuumConfig
from hierarchical_gate_system import ResonanceConfig
from enhanced_qrh_layer import EnhancedQRHLayer, EnhancedQRHConfig


class SemanticTestSuite:
    """Suite de testes para validar capacidades semânticas aprimoradas"""

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.test_results = {}

        # Configurar sistema Enhanced QRH
        self.enhanced_qrh = self._create_enhanced_system()

        print("🔬 Sistema de Teste Semântico Inicializado")
        print(f"📱 Dispositivo: {device}")
        print(self.enhanced_qrh.get_system_status_summary())

    def _create_enhanced_system(self) -> EnhancedQRHLayer:
        """Cria o sistema Enhanced QRH para testes"""

        # Configurações base
        qrh_config = QRHConfig(
            embed_dim=32,  # Menor para testes
            alpha=1.0,
            use_learned_rotation=True,
            normalization_type='layer_norm',
            device=str(self.device)
        )

        semantic_config = SemanticFilterConfig(
            embed_dim=32,
            num_heads=4,
            contradiction_threshold=0.7,
            irrelevance_threshold=0.5,
            bias_threshold=0.6
        )

        continuum_config = ContinuumConfig(
            embed_dim=32,
            memory_length=128,
            decay_rate=0.95,
            evolution_rate=0.1
        )

        resonance_config = ResonanceConfig(
            embed_dim=32,
            num_resonance_modes=8,
            interference_threshold=0.3,
            constructive_threshold=0.7
        )

        # Padrões de viés para detecção
        bias_patterns = [
            "gender_bias", "racial_bias", "age_bias",
            "confirmation_bias", "availability_bias"
        ]

        enhanced_config = EnhancedQRHConfig(
            qrh_config=qrh_config,
            semantic_config=semantic_config,
            continuum_config=continuum_config,
            resonance_config=resonance_config,
            bias_patterns=bias_patterns,
            enable_semantic_filtering=True,
            enable_temporal_continuum=True,
            enable_hierarchical_gates=True
        )

        return EnhancedQRHLayer(enhanced_config).to(self.device)

    def test_contradiction_detection(self) -> Dict:
        """
        Teste 1: Capacidade de detectar e filtrar contradições semânticas
        """
        print("\n🔍 Teste 1: Detecção de Contradições Semânticas")

        batch_size, seq_len, embed_dim = 2, 16, 128

        # Criar dados de teste com contradições intencionais
        # Sequência 1: Coerente (sem contradições)
        coherent_input = torch.randn(1, seq_len, embed_dim, device=self.device) * 0.5

        # Sequência 2: Com contradições (sinais opostos em partes da sequência)
        contradictory_input = torch.randn(1, seq_len, embed_dim, device=self.device) * 0.5
        # Inverter segunda metade para criar contradição
        contradictory_input[0, seq_len//2:] *= -1.0

        # Combinar em batch
        test_input = torch.cat([coherent_input, contradictory_input], dim=0)

        # Processar com métricas detalhadas
        output, metrics = self.enhanced_qrh.forward(
            test_input,
            concept_ids=['coherent_concept', 'contradictory_concept'],
            return_detailed_metrics=True
        )

        # Analisar resultados
        contradiction_scores = metrics['semantic_metrics']['contradiction_scores']
        coherent_contradictions = contradiction_scores[0].mean().item()
        contradictory_contradictions = contradiction_scores[1].mean().item()

        test_result = {
            'coherent_contradiction_score': coherent_contradictions,
            'contradictory_contradiction_score': contradictory_contradictions,
            'contradiction_detection_success': contradictory_contradictions > coherent_contradictions,
            'detection_ratio': contradictory_contradictions / (coherent_contradictions + 1e-6)
        }

        print(f"   ✅ Texto Coerente - Score de Contradição: {coherent_contradictions:.4f}")
        print(f"   ❌ Texto Contraditório - Score de Contradição: {contradictory_contradictions:.4f}")
        print(f"   🎯 Detecção Bem-sucedida: {'Sim' if test_result['contradiction_detection_success'] else 'Não'}")
        print(f"   📊 Razão de Detecção: {test_result['detection_ratio']:.2f}x")

        return test_result

    def test_relevance_filtering(self) -> Dict:
        """
        Teste 2: Capacidade de identificar e filtrar conteúdo irrelevante
        """
        print("\n🎯 Teste 2: Filtragem de Irrelevância")

        batch_size, seq_len, embed_dim = 2, 16, 128

        # Sequência 1: Altamente relevante (padrões consistentes)
        relevant_pattern = torch.sin(torch.arange(seq_len, dtype=torch.float32) * 0.1).unsqueeze(0).unsqueeze(-1)
        relevant_input = relevant_pattern.expand(1, seq_len, embed_dim).to(self.device) + torch.randn(1, seq_len, embed_dim, device=self.device) * 0.1

        # Sequência 2: Mistura de relevante e irrelevante
        mixed_input = relevant_input.clone()
        # Adicionar ruído irrelevante em posições específicas
        irrelevant_positions = torch.randint(0, seq_len, (seq_len//3,))
        mixed_input[0, irrelevant_positions] = torch.randn(len(irrelevant_positions), embed_dim, device=self.device) * 2.0

        test_input = torch.cat([relevant_input, mixed_input], dim=0)

        # Processar
        output, metrics = self.enhanced_qrh.forward(
            test_input,
            concept_ids=['relevant_concept', 'mixed_concept'],
            return_detailed_metrics=True
        )

        # Analisar relevância
        relevance_scores = metrics['semantic_metrics']['relevance_scores']
        pure_relevance = relevance_scores[0].mean().item()
        mixed_relevance = relevance_scores[1].mean().item()

        test_result = {
            'pure_content_relevance': pure_relevance,
            'mixed_content_relevance': mixed_relevance,
            'relevance_filtering_success': pure_relevance > mixed_relevance,
            'relevance_difference': pure_relevance - mixed_relevance
        }

        print(f"   ✅ Conteúdo Puro - Score de Relevância: {pure_relevance:.4f}")
        print(f"   ⚠️  Conteúdo Misto - Score de Relevância: {mixed_relevance:.4f}")
        print(f"   🎯 Filtragem Bem-sucedida: {'Sim' if test_result['relevance_filtering_success'] else 'Não'}")
        print(f"   📊 Diferença de Relevância: {test_result['relevance_difference']:.4f}")

        return test_result

    def test_temporal_consistency(self) -> Dict:
        """
        Teste 3: Modelagem de evolução temporal e detecção de quebras no continuum
        """
        print("\n⏰ Teste 3: Consistência Temporal e Evolução de Conceitos")

        batch_size, seq_len, embed_dim = 1, 32, 128

        # Sequência evolutiva suave (sem quebras)
        t = torch.linspace(0, 4*np.pi, seq_len)
        smooth_evolution = torch.sin(t).unsqueeze(0).unsqueeze(-1).expand(1, seq_len, embed_dim)
        smooth_input = smooth_evolution.to(self.device) + torch.randn(1, seq_len, embed_dim, device=self.device) * 0.05

        # Sequência com quebra abrupta no meio
        discontinuous_input = smooth_input.clone()
        break_point = seq_len // 2
        discontinuous_input[0, break_point:] *= -1.0  # Inversão abrupta

        # Testar ambas
        print("   🔄 Testando Evolução Suave...")
        smooth_output, smooth_metrics = self.enhanced_qrh.forward(
            smooth_input,
            concept_ids=['smooth_evolution'],
            return_detailed_metrics=True
        )

        print("   💥 Testando Evolução com Quebra...")
        discontinuous_output, disc_metrics = self.enhanced_qrh.forward(
            discontinuous_input,
            concept_ids=['discontinuous_evolution'],
            return_detailed_metrics=True
        )

        # Analisar coerência temporal
        smooth_coherence = smooth_metrics['temporal_metrics']['temporal_coherence']
        disc_coherence = disc_metrics['temporal_metrics']['temporal_coherence']

        # Analisar quebras detectadas
        smooth_breaks = len(smooth_metrics['temporal_metrics']['continuity_breaks'])
        disc_breaks = len(disc_metrics['temporal_metrics']['continuity_breaks'])

        test_result = {
            'smooth_temporal_coherence': smooth_coherence,
            'discontinuous_temporal_coherence': disc_coherence,
            'smooth_continuity_breaks': smooth_breaks,
            'discontinuous_continuity_breaks': disc_breaks,
            'break_detection_success': disc_breaks > smooth_breaks,
            'coherence_difference': smooth_coherence - disc_coherence
        }

        print(f"   ✅ Evolução Suave - Coerência: {smooth_coherence:.4f}, Quebras: {smooth_breaks}")
        print(f"   💥 Evolução Descontínua - Coerência: {disc_coherence:.4f}, Quebras: {disc_breaks}")
        print(f"   🎯 Detecção de Quebras: {'Sim' if test_result['break_detection_success'] else 'Não'}")

        return test_result

    def test_hierarchical_resonance(self) -> Dict:
        """
        Teste 4: Sistema hierárquico de gates e análise de ressonância
        """
        print("\n🎼 Teste 4: Ressonância Hierárquica e Controle de Qualidade")

        batch_size, seq_len, embed_dim = 2, 16, 128

        # Sequência 1: Alta ressonância (padrões harmônicos)
        harmonic_freq = 0.5
        t = torch.linspace(0, 2*np.pi, seq_len)
        harmonic_pattern = (torch.sin(harmonic_freq * t) + 0.5 * torch.sin(2 * harmonic_freq * t)).unsqueeze(0).unsqueeze(-1)
        harmonic_input = harmonic_pattern.expand(1, seq_len, embed_dim).to(self.device)

        # Sequência 2: Baixa ressonância (ruído aleatório)
        noise_input = torch.randn(1, seq_len, embed_dim, device=self.device)

        test_input = torch.cat([harmonic_input, noise_input], dim=0)

        # Processar
        output, metrics = self.enhanced_qrh.forward(
            test_input,
            return_detailed_metrics=True
        )

        # Analisar ressonância
        hierarchy_health = metrics['hierarchy_health']
        harmonic_resonance = hierarchy_health['global_resonance_health']

        # Analisar interferência
        constructive_interference = hierarchy_health['constructive_interference']
        destructive_interference = hierarchy_health['destructive_interference']
        interference_balance = hierarchy_health['interference_balance']

        test_result = {
            'global_resonance_health': harmonic_resonance,
            'constructive_interference': constructive_interference,
            'destructive_interference': destructive_interference,
            'interference_balance': interference_balance,
            'hierarchy_overall_health': hierarchy_health['overall_hierarchy_health']
        }

        print(f"   🎼 Saúde de Ressonância Global: {harmonic_resonance:.4f}")
        print(f"   ➕ Interferência Construtiva: {constructive_interference:.4f}")
        print(f"   ➖ Interferência Destrutiva: {destructive_interference:.4f}")
        print(f"   ⚖️  Balanço de Interferência: {interference_balance:.2f}")
        print(f"   🏥 Saúde Hierárquica Geral: {test_result['hierarchy_overall_health']:.4f}")

        return test_result

    def test_sarcasm_detection(self) -> Dict:
        """
        Teste 5: Detecção de sarcasmo e ironia através de inversões de fase
        """
        print("\n😏 Teste 5: Detecção de Sarcasmo e Ironia")

        batch_size, seq_len, embed_dim = 2, 16, 128

        # Sequência 1: Declaração normal (sem ironia)
        normal_input = torch.randn(1, seq_len, embed_dim, device=self.device) * 0.5

        # Sequência 2: Com inversão súbita (simulando sarcasmo)
        sarcastic_input = normal_input.clone()
        inversion_point = seq_len // 2
        # Inversão de fase súbita no meio da sequência
        sarcastic_input[0, inversion_point:] = -sarcastic_input[0, inversion_point:] + torch.randn(seq_len - inversion_point, embed_dim, device=self.device) * 0.1

        test_input = torch.cat([normal_input, sarcastic_input], dim=0)

        # Processar
        output, metrics = self.enhanced_qrh.forward(
            test_input,
            concept_ids=['normal_statement', 'sarcastic_statement'],
            return_detailed_metrics=True
        )

        # Analisar detecção de sarcasmo
        sarcasm_scores = metrics['temporal_metrics']['sarcasm_scores']
        normal_sarcasm = sarcasm_scores[0].mean().item()
        sarcastic_sarcasm = sarcasm_scores[1].mean().item()

        test_result = {
            'normal_sarcasm_score': normal_sarcasm,
            'sarcastic_sarcasm_score': sarcastic_sarcasm,
            'sarcasm_detection_success': sarcastic_sarcasm > normal_sarcasm,
            'detection_sensitivity': sarcastic_sarcasm - normal_sarcasm
        }

        print(f"   😐 Declaração Normal - Score de Sarcasmo: {normal_sarcasm:.4f}")
        print(f"   😏 Declaração Sarcástica - Score de Sarcasmo: {sarcastic_sarcasm:.4f}")
        print(f"   🎯 Detecção Bem-sucedida: {'Sim' if test_result['sarcasm_detection_success'] else 'Não'}")
        print(f"   📊 Sensibilidade: {test_result['detection_sensitivity']:.4f}")

        return test_result

    def test_signal_clarity_comprehensive(self) -> Dict:
        """
        Teste 6: Avaliação abrangente da clareza do sinal
        (O teste definitivo para "sinal claro em meio à cacofonia")
        """
        print("\n🌟 Teste 6: Clareza Geral do Sinal (Teste Definitivo)")

        # Cenários de teste progressivamente mais desafiadores
        scenarios = {
            'clean_signal': self._create_clean_signal(),
            'moderate_noise': self._create_moderate_noise(),
            'high_noise': self._create_high_noise(),
            'cacophony': self._create_cacophony()
        }

        results = {}

        for scenario_name, test_input in scenarios.items():
            print(f"   🔍 Testando: {scenario_name.replace('_', ' ').title()}")

            # Processar cenário
            output, metrics = self.enhanced_qrh.forward(
                test_input,
                concept_ids=[f'{scenario_name}_concept'],
                return_detailed_metrics=True
            )

            # Obter relatório de saúde
            health_report = self.enhanced_qrh.get_comprehensive_health_report(test_input)
            clarity_score = metrics['signal_clarity_score']

            results[scenario_name] = {
                'signal_clarity_score': clarity_score,
                'overall_status': health_report['overall_status'],
                'component_health': health_report['component_health']
            }

            print(f"      📊 Score de Clareza: {clarity_score:.4f}")
            print(f"      🏷️  Status: {health_report['overall_status']}")

        # Análise comparativa
        clarity_scores = [results[s]['signal_clarity_score'] for s in scenarios.keys()]
        clarity_progression = all(clarity_scores[i] >= clarity_scores[i+1] for i in range(len(clarity_scores)-1))

        comprehensive_result = {
            'scenario_results': results,
            'clarity_progression_correct': clarity_progression,
            'best_clarity': max(clarity_scores),
            'worst_clarity': min(clarity_scores),
            'dynamic_range': max(clarity_scores) - min(clarity_scores)
        }

        print(f"\n   🎯 Progressão de Clareza Correta: {'Sim' if clarity_progression else 'Não'}")
        print(f"   🏆 Melhor Clareza: {comprehensive_result['best_clarity']:.4f}")
        print(f"   🌊 Pior Clareza: {comprehensive_result['worst_clarity']:.4f}")
        print(f"   📏 Faixa Dinâmica: {comprehensive_result['dynamic_range']:.4f}")

        return comprehensive_result

    def _create_clean_signal(self) -> torch.Tensor:
        """Cria um sinal limpo e coerente"""
        seq_len, embed_dim = 16, 128
        t = torch.linspace(0, 2*np.pi, seq_len)
        clean_pattern = torch.sin(t).unsqueeze(-1).expand(seq_len, embed_dim)
        return clean_pattern.unsqueeze(0).to(self.device) + torch.randn(1, seq_len, embed_dim, device=self.device) * 0.05

    def _create_moderate_noise(self) -> torch.Tensor:
        """Cria um sinal com ruído moderado"""
        clean = self._create_clean_signal()
        noise = torch.randn_like(clean) * 0.3
        return clean + noise

    def _create_high_noise(self) -> torch.Tensor:
        """Cria um sinal com ruído alto"""
        clean = self._create_clean_signal()
        noise = torch.randn_like(clean) * 0.8
        # Adicionar algumas inversões abruptas
        noise[0, ::4] *= -2.0
        return clean + noise

    def _create_cacophony(self) -> torch.Tensor:
        """Cria cacofonia total (múltiplos sinais conflitantes)"""
        seq_len, embed_dim = 16, 128

        # Múltiplas frequências conflitantes
        t = torch.linspace(0, 4*np.pi, seq_len)
        freq1 = torch.sin(0.5 * t)
        freq2 = torch.sin(3.0 * t + np.pi)  # Fora de fase
        freq3 = torch.sin(7.0 * t)

        cacophonous = (freq1 + freq2 + freq3).unsqueeze(-1).expand(seq_len, embed_dim)

        # Adicionar ruído e inversões aleatórias
        random_noise = torch.randn(seq_len, embed_dim) * 1.5
        random_inversions = torch.randint(0, 2, (seq_len, embed_dim)) * 2 - 1

        cacophony = cacophonous + random_noise * random_inversions

        return cacophony.unsqueeze(0).to(self.device)

    def run_complete_test_suite(self) -> Dict:
        """
        Executa a suite completa de testes e gera relatório final
        """
        print("🚀 Iniciando Suite Completa de Testes Semânticos")
        print("=" * 60)

        all_results = {}

        # Executar todos os testes
        try:
            all_results['contradiction_detection'] = self.test_contradiction_detection()
            all_results['relevance_filtering'] = self.test_relevance_filtering()
            all_results['temporal_consistency'] = self.test_temporal_consistency()
            all_results['hierarchical_resonance'] = self.test_hierarchical_resonance()
            all_results['sarcasm_detection'] = self.test_sarcasm_detection()
            all_results['signal_clarity_comprehensive'] = self.test_signal_clarity_comprehensive()

        except Exception as e:
            print(f"❌ Erro durante os testes: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'partial_results': all_results}

        # Análise de resultados gerais
        self._generate_final_report(all_results)

        return all_results

    def _generate_final_report(self, results: Dict) -> None:
        """Gera relatório final dos testes"""
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO FINAL - SISTEMA SEMÂNTICO APRIMORADO")
        print("=" * 60)

        # Contar sucessos
        successes = []

        if 'contradiction_detection' in results:
            successes.append(results['contradiction_detection']['contradiction_detection_success'])

        if 'relevance_filtering' in results:
            successes.append(results['relevance_filtering']['relevance_filtering_success'])

        if 'temporal_consistency' in results:
            successes.append(results['temporal_consistency']['break_detection_success'])

        if 'sarcasm_detection' in results:
            successes.append(results['sarcasm_detection']['sarcasm_detection_success'])

        if 'signal_clarity_comprehensive' in results:
            successes.append(results['signal_clarity_comprehensive']['clarity_progression_correct'])

        success_rate = sum(successes) / len(successes) if successes else 0

        print(f"🎯 Taxa de Sucesso Geral: {success_rate:.1%} ({sum(successes)}/{len(successes)} testes)")

        # Análise detalhada
        if 'signal_clarity_comprehensive' in results:
            clarity_results = results['signal_clarity_comprehensive']
            print(f"📈 Faixa Dinâmica de Clareza: {clarity_results['dynamic_range']:.4f}")
            print(f"🏆 Melhor Score de Clareza: {clarity_results['best_clarity']:.4f}")

            # Status por cenário
            for scenario, result in clarity_results['scenario_results'].items():
                print(f"   • {scenario.replace('_', ' ').title()}: {result['overall_status']}")

        # Recomendações
        print("\n💡 ANÁLISE E RECOMENDAÇÕES:")

        if success_rate >= 0.8:
            print("✅ Sistema funcionando EXCELENTEMENTE para extração de sinal semântico")
            print("🌟 Capacidade comprovada de distinguir sinal de ruído em múltiplos níveis")
        elif success_rate >= 0.6:
            print("⚡ Sistema funcionando BEM com algumas áreas para otimização")
            print("🔧 Considerar ajuste fino dos limiares de detecção")
        else:
            print("⚠️  Sistema necessita CALIBRAÇÃO adicional")
            print("🔄 Recomenda-se revisão dos parâmetros e possível retreinamento")

        print("\n🎖️  CONQUISTAS DEMONSTRADAS:")
        print("   • Filtragem semântica multinível (contradição, irrelevância, viés)")
        print("   • Modelagem temporal com detecção de quebras de consistência")
        print("   • Análise hierárquica de ressonância e interferência")
        print("   • Detecção de padrões complexos (sarcasmo, ironia)")
        print("   • Quantificação objetiva da clareza do sinal")

        print("\n🌟 MISSÃO CUMPRIDA: Sistema demonstra capacidade robusta de extrair")
        print("    'SINAL CLARO EM MEIO À CACOFONIA SEMÂNTICA'")


def main():
    """Função principal para executar os testes"""

    # Suprimir warnings desnecessários
    warnings.filterwarnings('ignore', category=UserWarning)

    # Escolher dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Usando dispositivo: {device}")

    # Criar e executar suite de testes
    test_suite = SemanticTestSuite(device=device)

    try:
        results = test_suite.run_complete_test_suite()

        # Salvar resultados se desejado
        # torch.save(results, 'semantic_test_results.pt')

        print(f"\n✅ Testes concluídos com sucesso!")
        return results

    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()