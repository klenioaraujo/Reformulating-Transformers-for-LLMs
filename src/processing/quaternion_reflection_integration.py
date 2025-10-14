"""
Interface de Substituição Direta para QuaternionReflectionLayer

Permite que o QuaternionReflectionLayer substitua diretamente os sistemas de raciocínio
existentes no DCFTokenAnalysis sem quebrar a compatibilidade.
"""

from typing import Dict, Any, Optional
import torch


class DropInReplacementInterface:
    """
    Interface que permite o QuaternionReflectionLayer substituir diretamente
    os sistemas de raciocínio existentes no DCFTokenAnalysis
    """

    def __init__(self, dcf_system):
        self.dcf_system = dcf_system
        self.original_methods = {}

    def enable_reflection_layer(self, mode: str = 'adaptive'):
        """
        Habilita o QuaternionReflectionLayer como processador principal
        """
        print("🔄 Ativando QuaternionReflectionLayer como processador principal...")

        # Preservar métodos originais para fallback
        self.original_methods['analyze_tokens'] = self.dcf_system.analyze_tokens

        # Substituir por método otimizado
        def optimized_analyze_tokens(token_ids, context_embedding=None):
            return self.dcf_system._run_adaptive_reasoning(token_ids, context_embedding)

        self.dcf_system.analyze_tokens = optimized_analyze_tokens
        self.dcf_system.reasoning_mode = mode

        print("✅ QuaternionReflectionLayer ativado como processador principal")

    def disable_reflection_layer(self):
        """
        Restaura os métodos originais do DCFTokenAnalysis
        """
        if 'analyze_tokens' in self.original_methods:
            self.dcf_system.analyze_tokens = self.original_methods['analyze_tokens']
            print("✅ Métodos originais do DCFTokenAnalysis restaurados")