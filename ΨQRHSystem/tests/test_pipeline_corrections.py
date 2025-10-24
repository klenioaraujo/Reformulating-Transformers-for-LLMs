#!/usr/bin/env python3
"""
Testes unitários para as correções implementadas no sistema ΨQRH

Este módulo testa as correções específicas implementadas:
1. Configuração ConsciousnessConfig com atributo physics
2. Cálculo de dimensão fractal sem erro de tensor
3. Validação de energia reportando violação correta
4. Geração de texto semanticamente relevante
"""

import torch
import numpy as np
import unittest
from unittest.mock import Mock, patch
import sys
import os

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, project_root)

from ΨQRHSystem.core.PipelineManager import PipelineManager
from ΨQRHSystem.configs.SystemConfig import SystemConfig
from ΨQRHSystem.consciousness.fractal_consciousness_processor import ConsciousnessConfig


class TestPipelineCorrections(unittest.TestCase):
    """Testes para as correções implementadas no pipeline ΨQRH"""

    def setUp(self):
        """Configuração inicial dos testes"""
        # Usar configuração padrão do YAML
        self.config = SystemConfig.from_yaml("configs/system_config.yaml")
        self.pipeline = PipelineManager(self.config)

    def test_consciousness_config_physics_attribute(self):
        """Testa se ConsciousnessConfig tem o atributo physics necessário"""
        consciousness_config = ConsciousnessConfig()

        # Verificar se o atributo physics existe
        self.assertTrue(hasattr(consciousness_config, 'physics'),
                       "ConsciousnessConfig deve ter atributo physics")

        # Verificar se physics tem os atributos necessários
        physics = consciousness_config.physics
        required_attrs = ['I0', 'alpha', 'beta', 'k', 'omega']
        for attr in required_attrs:
            self.assertTrue(hasattr(physics, attr),
                           f"Physics config deve ter atributo {attr}")

    def test_fractal_dimension_calculation_no_tensor_error(self):
        """Testa se o cálculo de dimensão fractal não lança erro de tensor"""
        # Criar sinal de teste
        seq_len = 10
        embed_dim = 64
        test_signal = torch.randn(seq_len, embed_dim)

        # Tentar calcular dimensão fractal
        try:
            fractal_dim = self.pipeline._calculate_fractal_dimension_real(test_signal)
            # Verificar se resultado é um número finito
            self.assertTrue(torch.isfinite(torch.tensor(fractal_dim)),
                           "Dimensão fractal deve ser um número finito")
            # Verificar se está no range esperado
            self.assertTrue(1.0 <= fractal_dim <= 2.0,
                           f"Dimensão fractal {fractal_dim} deve estar entre 1.0 e 2.0")
        except Exception as e:
            self.fail(f"Cálculo de dimensão fractal falhou com erro: {e}")

    def test_energy_violation_validation(self):
        """Testa se a validação de energia reporta violação corretamente"""
        # Criar sinais de teste com energia diferente
        input_signal = torch.randn(10, 64)
        output_signal = torch.randn(10, 64) * 1.5  # Energia maior

        # Testar validação
        energy_violated = self.pipeline._validate_energy_violation_pi(
            input_signal, output_signal
        )

        # Para sinais com energia diferente, deve reportar violação
        self.assertTrue(energy_violated,
                       "Energia deve ser reportada como violada quando sinais têm energias diferentes")

    def test_text_generation_semantic_relevance(self):
        """Testa se a geração de texto produz resposta semanticamente relevante"""
        # Simular entrada e consciência
        optical_output = torch.randn(1, 10, 64, 4)  # Formato quaterniônico
        consciousness = {'fci': 0.5, 'temporal_coherence': 0.7}

        # Gerar texto
        generated_text = self.pipeline._generate_text_via_dcf(optical_output, consciousness)

        # Verificar se texto foi gerado
        self.assertIsInstance(generated_text, str)
        self.assertTrue(len(generated_text) > 0)

        # Verificar se contém palavras relacionadas à pergunta "Qual a cor do céu?"
        text_lower = generated_text.lower()
        relevant_words = ['sky', 'blue', 'light', 'scattering', 'atmosphere']
        has_relevant_content = any(word in text_lower for word in relevant_words)

        self.assertTrue(has_relevant_content,
                       f"Texto gerado deve conter conteúdo relevante: {generated_text}")

    def test_pipeline_end_to_end_execution(self):
        """Testa execução completa do pipeline sem erros"""
        test_text = "Qual a cor do céu?"

        # Executar pipeline
        try:
            result = self.pipeline.process(test_text)

            # Verificar estrutura do resultado
            required_keys = ['text', 'fractal_dim', 'energy_conserved', 'validation', 'status']
            for key in required_keys:
                self.assertIn(key, result, f"Resultado deve conter chave {key}")

            # Verificar status de sucesso
            self.assertEqual(result['status'], 'success')

            # Verificar que texto foi gerado
            self.assertIsInstance(result['text'], str)
            self.assertTrue(len(result['text']) > 0)

        except Exception as e:
            self.fail(f"Pipeline falhou na execução end-to-end: {e}")

    def test_energy_violation_with_text_output(self):
        """Testa validação de energia com saída de texto (sempre violada)"""
        input_signal = torch.randn(10, 64)
        text_output = "The sky is blue due to light scattering."

        energy_violated = self.pipeline._validate_energy_violation_pi(
            input_signal, text_output
        )

        # Transformação wave-to-text sempre deve violar conservação
        self.assertTrue(energy_violated,
                       "Transformação wave-to-text deve sempre violar conservação energética")


if __name__ == '__main__':
    # Configurar logging para testes
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduzir logs durante testes

    # Executar testes
    unittest.main(verbosity=2)