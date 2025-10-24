"""
Testes para classes Maker: ModelMaker, VocabularyMaker, PipelineMaker
"""

import unittest
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ΨQRHSystem.core.ModelMaker import ModelMaker
from ΨQRHSystem.core.VocabularyMaker import VocabularyMaker
from ΨQRHSystem.core.PipelineMaker import PipelineMaker
from ΨQRHSystem.configs.SystemConfig import SystemConfig


class TestModelMaker(unittest.TestCase):
    """Testes para ModelMaker"""

    def setUp(self):
        """Configuração inicial"""
        self.maker = ModelMaker()

    def test_initialization(self):
        """Testa inicialização do ModelMaker"""
        self.assertIsNotNone(self.maker.templates)
        self.assertIsInstance(self.maker.created_models, list)

    def test_create_from_template(self):
        """Testa criação de modelo a partir de template"""
        pipeline = self.maker.create_from_template("minimal")

        self.assertIsNotNone(pipeline)
        self.assertEqual(len(self.maker.created_models), 1)

        # Verificar configuração aplicada
        model_info = self.maker.created_models[0]
        self.assertEqual(model_info['type'], 'template')
        self.assertIn('template', model_info['config'])
        self.assertEqual(model_info['config']['template'], 'minimal')

    def test_create_custom(self):
        """Testa criação de modelo customizado"""
        pipeline = self.maker.create_custom(embed_dim=96, vocab_size=768)

        self.assertIsNotNone(pipeline)
        self.assertEqual(len(self.maker.created_models), 1)

        model_info = self.maker.created_models[0]
        self.assertEqual(model_info['type'], 'custom')

    def test_create_quantum_optimized(self):
        """Testa criação de modelo quântico otimizado"""
        pipeline = self.maker.create_quantum_optimized("medium")

        self.assertIsNotNone(pipeline)
        self.assertEqual(len(self.maker.created_models), 1)

    def test_template_validation(self):
        """Testa validação de templates"""
        with self.assertRaises(ValueError):
            self.maker.create_from_template("nonexistent_template")

    def test_config_validation(self):
        """Testa validação de configuração"""
        valid_config = {
            'model': {'embed_dim': 64, 'vocab_size': 512},
            'physics': {'I0': 1.0, 'alpha': 1.0, 'beta': 0.5, 'k': 2.0, 'omega': 1.0}
        }

        self.assertTrue(self.maker.validate_model_config(valid_config))

        invalid_config = {
            'model': {'embed_dim': 5, 'vocab_size': 512},  # embed_dim muito pequeno
            'physics': {'I0': 1.0, 'alpha': 1.0, 'beta': 0.5, 'k': 2.0, 'omega': 1.0}
        }

        self.assertFalse(self.maker.validate_model_config(invalid_config))

    def test_list_created_models(self):
        """Testa listagem de modelos criados"""
        # Criar alguns modelos
        self.maker.create_from_template("minimal")
        self.maker.create_custom(embed_dim=128)

        models = self.maker.list_created_models()
        self.assertEqual(len(models), 2)


class TestVocabularyMaker(unittest.TestCase):
    """Testes para VocabularyMaker"""

    def setUp(self):
        """Configuração inicial"""
        self.maker = VocabularyMaker()

    def test_initialization(self):
        """Testa inicialização do VocabularyMaker"""
        self.assertIsInstance(self.maker.created_vocabularies, list)

    def test_create_semantic_vocab(self):
        """Testa criação de vocabulário semântico"""
        base_words = ["quantum", "consciousness", "fractal", "energy"]
        vocab = self.maker.create_semantic_vocab(base_words, expansion_factor=1.5)

        self.assertIn('tokens', vocab)
        self.assertIn('word_to_idx', vocab)
        self.assertIn('idx_to_word', vocab)
        self.assertGreater(len(vocab['tokens']), len(base_words))

        # Verificar que vocabulário foi registrado
        self.assertEqual(len(self.maker.created_vocabularies), 1)

    def test_create_quantum_vocab(self):
        """Testa criação de vocabulário quântico"""
        # Criar features quânticas simuladas
        quantum_features = torch.randn(50, 32)

        vocab = self.maker.create_quantum_vocab(quantum_features, vocab_size=64)

        self.assertIn('tokens', vocab)
        self.assertIn('quantum_patterns', vocab)
        # Pode ter menos tokens se houver menos features
        self.assertLessEqual(len(vocab['tokens']), 64)
        self.assertGreater(len(vocab['tokens']), 0)

    def test_create_hybrid_vocab(self):
        """Testa criação de vocabulário híbrido"""
        text_sources = ["quantum physics", "consciousness theory", "fractal mathematics"]
        quantum_features = torch.randn(20, 16)

        vocab = self.maker.create_hybrid_vocab(text_sources, quantum_features, target_size=128)

        self.assertIn('tokens', vocab)
        self.assertIn('components', vocab)
        self.assertLessEqual(len(vocab['tokens']), 128)

    def test_vocab_validation(self):
        """Testa validação de vocabulário"""
        valid_vocab = {
            'tokens': ['a', 'b', 'c'],
            'word_to_idx': {'a': 0, 'b': 1, 'c': 2},
            'idx_to_word': {0: 'a', 1: 'b', 2: 'c'}
        }

        self.assertTrue(self.maker.validate_vocabulary(valid_vocab))

        invalid_vocab = {
            'tokens': ['a', 'b', 'c'],
            'word_to_idx': {'a': 0, 'b': 1},  # Mapeamento incompleto
            'idx_to_word': {0: 'a', 1: 'b', 2: 'c'}
        }

        self.assertFalse(self.maker.validate_vocabulary(invalid_vocab))

    def test_list_created_vocabularies(self):
        """Testa listagem de vocabulários criados"""
        # Criar alguns vocabulários
        self.maker.create_semantic_vocab(["test"])
        self.maker.create_quantum_vocab(torch.randn(10, 8))

        vocabs = self.maker.list_created_vocabularies()
        self.assertEqual(len(vocabs), 2)


class TestPipelineMaker(unittest.TestCase):
    """Testes para PipelineMaker"""

    def setUp(self):
        """Configuração inicial"""
        self.maker = PipelineMaker()

    def test_initialization(self):
        """Testa inicialização do PipelineMaker"""
        self.assertIsNotNone(self.maker.model_maker)
        self.assertIsNotNone(self.maker.vocab_maker)
        self.assertIsInstance(self.maker.created_pipelines, list)

    def test_create_physics_pipeline(self):
        """Testa criação de pipeline físico"""
        physics_config = {
            'I0': 1.5,
            'alpha': 2.0,
            'beta': 1.0,
            'k': 3.0,
            'omega': 1.5
        }

        pipeline = self.maker.create_physics_pipeline(physics_config, embed_dim=96)

        self.assertIsNotNone(pipeline)
        self.assertEqual(len(self.maker.created_pipelines), 1)

        pipeline_info = self.maker.created_pipelines[0]
        self.assertEqual(pipeline_info['type'], 'physics')

    def test_create_quantum_pipeline(self):
        """Testa criação de pipeline quântico"""
        quantum_config = {
            'embed_dim': 128,
            'vocab_size': 1024,
            'physics': {
                'I0': 2.0,
                'alpha': 1.8,
                'beta': 0.9,
                'k': 4.0,
                'omega': 2.0
            }
        }

        pipeline = self.maker.create_quantum_pipeline(quantum_config, memory_depth=25)

        self.assertIsNotNone(pipeline)
        self.assertEqual(len(self.maker.created_pipelines), 1)

    def test_create_hybrid_pipeline(self):
        """Testa criação de pipeline híbrido"""
        components = ["quantum_memory", "auto_calibration", "physical_harmonics"]

        pipeline = self.maker.create_hybrid_pipeline(components)

        self.assertIsNotNone(pipeline)
        self.assertEqual(len(self.maker.created_pipelines), 1)

        pipeline_info = self.maker.created_pipelines[0]
        self.assertEqual(pipeline_info['type'], 'hybrid')

    def test_create_research_pipeline(self):
        """Testa criação de pipeline de pesquisa"""
        pipeline = self.maker.create_research_pipeline("quantum")

        self.assertIsNotNone(pipeline)
        self.assertEqual(len(self.maker.created_pipelines), 1)

        pipeline_info = self.maker.created_pipelines[0]
        self.assertEqual(pipeline_info['type'], 'research')
        self.assertEqual(pipeline_info['research_focus'], 'quantum')

    def test_create_production_pipeline(self):
        """Testa criação de pipeline de produção"""
        pipeline = self.maker.create_production_pipeline("speed")

        self.assertIsNotNone(pipeline)
        self.assertEqual(len(self.maker.created_pipelines), 1)

        pipeline_info = self.maker.created_pipelines[0]
        self.assertEqual(pipeline_info['type'], 'production')
        self.assertEqual(pipeline_info['performance_target'], 'speed')

    def test_create_with_vocabulary(self):
        """Testa criação de pipeline com vocabulário customizado"""
        vocab_params = {"base_words": ["quantum", "physics", "consciousness"]}
        pipeline_config = {
            'model': {'embed_dim': 64, 'max_history': 10},
            'physics': {'I0': 1.0, 'alpha': 1.0, 'beta': 0.5, 'k': 2.0, 'omega': 1.0}
        }

        pipeline, vocab = self.maker.create_with_vocabulary(
            "semantic", vocab_params, pipeline_config
        )

        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(vocab)
        self.assertIn('tokens', vocab)

    def test_get_pipeline_templates(self):
        """Testa obtenção de templates de pipeline"""
        templates = self.maker.get_pipeline_templates()

        self.assertIsInstance(templates, dict)
        self.assertGreater(len(templates), 0)

    def test_config_validation(self):
        """Testa validação de configuração de pipeline"""
        valid_config = {
            'model': {'embed_dim': 64, 'vocab_size': 512},
            'physics': {'I0': 1.0, 'alpha': 1.0, 'beta': 0.5, 'k': 2.0, 'omega': 1.0}
        }

        self.assertTrue(self.maker.validate_pipeline_config(valid_config))

    def test_list_created_pipelines(self):
        """Testa listagem de pipelines criados"""
        # Criar alguns pipelines
        self.maker.create_physics_pipeline({
            'I0': 1.0, 'alpha': 1.0, 'beta': 0.5, 'k': 2.0, 'omega': 1.0
        })
        self.maker.create_research_pipeline("fractal")

        pipelines = self.maker.list_created_pipelines()
        self.assertEqual(len(pipelines), 2)


class TestMakerIntegration(unittest.TestCase):
    """Testes de integração entre classes Maker"""

    def test_full_pipeline_creation_workflow(self):
        """Testa workflow completo de criação de pipeline"""
        # 1. Criar vocabulário
        vocab_maker = VocabularyMaker()
        vocab = vocab_maker.create_semantic_vocab(
            ["quantum", "consciousness", "fractal", "energy"],
            expansion_factor=2
        )

        # 2. Criar modelo
        model_maker = ModelMaker()
        pipeline = model_maker.create_custom(embed_dim=96, vocab_size=len(vocab['tokens']))

        # 3. Criar pipeline avançado
        pipeline_maker = PipelineMaker()
        advanced_pipeline = pipeline_maker.create_research_pipeline("quantum")

        # Verificações
        self.assertIsNotNone(vocab)
        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(advanced_pipeline)

        # Verificar que todos os makers registraram suas criações
        self.assertEqual(len(vocab_maker.created_vocabularies), 1)
        self.assertEqual(len(model_maker.created_models), 1)
        self.assertEqual(len(pipeline_maker.created_pipelines), 1)

    def test_cross_maker_functionality(self):
        """Testa funcionalidades que cruzam entre makers"""
        pipeline_maker = PipelineMaker()

        # Criar pipeline com vocabulário integrado
        vocab_params = {"base_words": ["physics", "mathematics", "computation"]}
        pipeline_config = {
            'model': {'embed_dim': 64, 'max_history': 10},
            'physics': {'I0': 1.0, 'alpha': 1.0, 'beta': 0.5, 'k': 2.0, 'omega': 1.0}
        }

        pipeline, vocab = pipeline_maker.create_with_vocabulary(
            "semantic", vocab_params, pipeline_config
        )

        # Verificar integração
        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(vocab)
        self.assertGreater(len(vocab['tokens']), len(vocab_params['base_words']))


if __name__ == '__main__':
    unittest.main()