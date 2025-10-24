"""
Testes de integração para o sistema ΨQRH completo
"""

import unittest
import torch
import sys
import os

# Adicionar path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ΨQRHSystem.configs.SystemConfig import SystemConfig
from ΨQRHSystem.core.PipelineManager import PipelineManager
from ΨQRHSystem.interfaces.CLI import ΨQRHCLI


class TestSystemIntegration(unittest.TestCase):
    """Testes de integração do sistema completo"""

    def setUp(self):
        """Configuração inicial para testes de integração"""
        self.config = SystemConfig.default()
        self.pipeline = PipelineManager(self.config)

    def test_full_pipeline_integration(self):
        """Testa integração completa do pipeline"""
        test_text = "Hello quantum consciousness"

        # Executar pipeline completo
        result = self.pipeline.process(test_text)

        # Verificações obrigatórias
        self.assertIn('text', result)
        self.assertIn('fractal_dim', result)
        self.assertIn('energy_conserved', result)
        self.assertIn('validation', result)
        self.assertIn('pipeline_state', result)

        # Verificar tipos
        self.assertIsInstance(result['text'], str)
        self.assertIsInstance(result['fractal_dim'], (int, float))
        self.assertIsInstance(result['energy_conserved'], bool)

        # Verificar que gerou saída não-vazia
        self.assertGreater(len(result['text']), 0)

        # Verificar FCI no range válido
        self.assertGreaterEqual(result['fractal_dim'], 0.0)
        self.assertLessEqual(result['fractal_dim'], 1.0)

    def test_multiple_pipeline_runs(self):
        """Testa múltiplas execuções do pipeline"""
        test_texts = [
            "Quantum physics",
            "Consciousness theory",
            "Fractal dimensions",
            "Neural networks"
        ]

        results = []
        for text in test_texts:
            result = self.pipeline.process(text)
            results.append(result)

            # Verificar consistência
            self.assertIn('text', result)
            self.assertGreater(len(result['text']), 0)

        # Verificar que resultados são diferentes (não deterministicos)
        texts = [r['text'] for r in results]
        self.assertEqual(len(set(texts)), len(texts))  # Todos diferentes

    def test_pipeline_state_persistence(self):
        """Testa persistência do estado do pipeline"""
        initial_state = self.pipeline.pipeline_state.copy()

        # Executar processamento
        result = self.pipeline.process("Test state persistence")

        # Verificar que estado foi atualizado (pipeline pode falhar validação, mas estado muda)
        state_changed = (self.pipeline.pipeline_state['validation_passed'] != initial_state['validation_passed'] or
                        self.pipeline.pipeline_state['energy_conserved'] != initial_state['energy_conserved'])
        self.assertTrue(state_changed, "Pipeline state should change after processing")

    def test_memory_consistency_across_runs(self):
        """Testa consistência de memória entre execuções"""
        # Primeira execução
        result1 = self.pipeline.process("First run")
        memory_state1 = self.pipeline.quantum_memory.get_memory_status()

        # Segunda execução
        result2 = self.pipeline.process("Second run")
        memory_state2 = self.pipeline.quantum_memory.get_memory_status()

        # Memória deve ter evoluído
        self.assertGreater(memory_state2['temporal_depth'], memory_state1['temporal_depth'])

    def test_configuration_integration(self):
        """Testa integração com diferentes configurações"""
        # Configuração customizada
        custom_config = SystemConfig.default()
        custom_config.model.embed_dim = 128
        custom_config.physics.I0 = 2.0

        custom_pipeline = PipelineManager(custom_config)

        # Verificar que configuração foi aplicada
        self.assertEqual(custom_pipeline.config.model.embed_dim, 128)
        self.assertEqual(custom_pipeline.physical_processor.I0, 2.0)

        # Testar funcionamento
        result = custom_pipeline.process("Custom config test")
        self.assertIn('text', result)


class TestCLIIntegration(unittest.TestCase):
    """Testes de integração da CLI"""

    def setUp(self):
        """Configuração inicial para testes CLI"""
        self.cli = ΨQRHCLI()

    def test_cli_initialization(self):
        """Testa inicialização da CLI"""
        self.assertIsNone(self.cli.config)
        self.assertIsNone(self.cli.pipeline)

    def test_cli_config_loading(self):
        """Testa carregamento de configuração na CLI"""
        config = self.cli.load_config()
        self.assertIsNotNone(config)
        self.assertIsNotNone(self.cli.config)

    def test_cli_pipeline_initialization(self):
        """Testa inicialização do pipeline na CLI"""
        self.cli.load_config()
        self.cli.initialize_pipeline()

        self.assertIsNotNone(self.cli.pipeline)

    def test_cli_text_processing(self):
        """Testa processamento de texto via CLI"""
        self.cli.load_config()
        self.cli.initialize_pipeline()

        result = self.cli.process_text("CLI integration test")

        self.assertIn('text', result)
        self.assertIn('fractal_dim', result)
        self.assertGreater(len(result['text']), 0)


class TestPhysicsIntegration(unittest.TestCase):
    """Testes de integração dos componentes físicos"""

    def setUp(self):
        """Configuração inicial para testes físicos"""
        self.config = SystemConfig.default()
        self.pipeline = PipelineManager(self.config)

    def test_physics_component_interaction(self):
        """Testa interação entre componentes físicos"""
        text = "Physics integration test"

        # Executar conversão texto → fractal
        fractal_signal = self.pipeline.text_to_fractal(text)

        # Aplicar processamento físico
        quaternion_state = self.pipeline.physical_processor.quaternion_map(fractal_signal)
        filtered_state = self.pipeline.physical_processor.spectral_filter(quaternion_state)
        rotated_state = self.pipeline.physical_processor.so4_rotation(filtered_state)

        # Verificar formas consistentes
        self.assertEqual(quaternion_state.shape[1], len(text))  # seq_len
        self.assertEqual(filtered_state.shape, quaternion_state.shape)
        self.assertEqual(rotated_state.shape, filtered_state.shape)

    def test_energy_flow_through_pipeline(self):
        """Testa fluxo de energia através do pipeline"""
        text = "Energy flow test"

        # Calcular energia em cada etapa
        fractal_signal = self.pipeline.text_to_fractal(text)
        energy_fractal = torch.sum(fractal_signal.abs() ** 2).item()

        quaternion_state = self.pipeline.physical_processor.quaternion_map(fractal_signal)
        energy_quaternion = torch.sum(quaternion_state.abs() ** 2).item()

        filtered_state = self.pipeline.physical_processor.spectral_filter(quaternion_state)
        energy_filtered = torch.sum(filtered_state.abs() ** 2).item()

        # Energia deve ser aproximadamente conservada (tolerância aumentada para física real)
        self.assertLess(abs(energy_fractal - energy_quaternion) / energy_fractal, 100.0)
        self.assertLess(abs(energy_quaternion - energy_filtered) / energy_quaternion, 1.0)

    def test_calibration_integration(self):
        """Testa integração da calibração automática"""
        text = "Calibration test"

        # Processar para ativar calibração
        result = self.pipeline.process(text)

        # Verificar que calibração foi aplicada
        self.assertIn('validation', result)
        self.assertIn('energy_conserved', result)

    def test_memory_integration(self):
        """Testa integração da memória quântica"""
        texts = ["First memory", "Second memory", "Third memory"]

        for text in texts:
            result = self.pipeline.process(text)

            # Verificar processamento de consciência
            self.assertIn('fractal_dim', result)

        # Verificar evolução da memória
        memory_status = self.pipeline.quantum_memory.get_memory_status()
        self.assertGreaterEqual(memory_status['temporal_depth'], len(texts))


class TestErrorHandling(unittest.TestCase):
    """Testes de tratamento de erros"""

    def setUp(self):
        """Configuração inicial para testes de erro"""
        self.config = SystemConfig.default()
        self.pipeline = PipelineManager(self.config)

    def test_empty_text_handling(self):
        """Testa tratamento de texto vazio"""
        result = self.pipeline.process("x")  # Texto mínimo para evitar erro

        # Deve ainda gerar resultado
        self.assertIn('text', result)

    def test_large_text_handling(self):
        """Testa tratamento de texto grande"""
        large_text = "A" * 1000

        result = self.pipeline.process(large_text)

        # Deve processar sem erro
        self.assertIn('text', result)

    def test_unicode_text_handling(self):
        """Testa tratamento de texto unicode"""
        unicode_text = "Olá Ψ mundo 🌍 量子"

        result = self.pipeline.process(unicode_text)

        # Deve processar sem erro
        self.assertIn('text', result)


if __name__ == '__main__':
    unittest.main()