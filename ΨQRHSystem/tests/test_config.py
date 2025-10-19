"""
Testes para SystemConfig e PhysicsConfig
"""

import unittest
import tempfile
import os
import yaml
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.SystemConfig import SystemConfig, ModelConfig, PhysicsConfig


class TestSystemConfig(unittest.TestCase):
    """Testes para SystemConfig"""

    def test_default_config(self):
        """Testa configuração padrão"""
        config = SystemConfig(ModelConfig(), PhysicsConfig())

        self.assertEqual(config.model.embed_dim, 64)
        self.assertEqual(config.physics.I0, 1.0)
        self.assertEqual(config.device, "auto")
        self.assertIn("quantum_memory", config.enable_components)

    def test_yaml_config_loading(self):
        """Testa carregamento de configuração YAML"""
        config_data = {
            'model': {
                'embed_dim': 128,
                'max_history': 20,
                'vocab_size': 512,
                'num_heads': 16,
                'hidden_dim': 256,
                'num_layers': 6
            },
            'physics': {
                'I0': 2.0,
                'alpha': 2.0,
                'beta': 1.0,
                'k': 3.0,
                'omega': 2.0
            },
            'system': {
                'device': 'cuda',
                'enable_components': ['auto_calibration'],
                'validation': {
                    'energy_conservation': False,
                    'unitarity': True
                }
            }
        }

        # Criar arquivo temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = SystemConfig.from_yaml(temp_path)

            # Verificar valores carregados
            self.assertEqual(config.model.embed_dim, 128)
            self.assertEqual(config.model.num_heads, 16)
            self.assertEqual(config.model.num_layers, 6)
            self.assertEqual(config.physics.I0, 2.0)
            self.assertEqual(config.device, 'cuda')
            self.assertEqual(config.enable_components, ['auto_calibration'])
            self.assertFalse(config.validation['energy_conservation'])
            self.assertTrue(config.validation['unitarity'])

        finally:
            os.unlink(temp_path)

    def test_yaml_config_saving(self):
        """Testa salvamento de configuração YAML"""
        config = SystemConfig(ModelConfig(), PhysicsConfig())
        config.model.embed_dim = 256
        config.physics.alpha = 3.0

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            config.to_yaml(temp_path)

            # Carregar e verificar
            with open(temp_path, 'r') as f:
                saved_data = yaml.safe_load(f)
    
                self.assertEqual(saved_data['model']['embed_dim'], 256)
                self.assertEqual(saved_data['model']['num_heads'], 4)
                self.assertEqual(saved_data['model']['num_layers'], 3)
                self.assertEqual(saved_data['physics']['alpha'], 3.0)

        finally:
            os.unlink(temp_path)

    def test_invalid_yaml_file(self):
        """Testa tratamento de arquivo YAML inválido"""
        with self.assertRaises(FileNotFoundError):
            SystemConfig.from_yaml("nonexistent_file.yaml")


class TestModelConfig(unittest.TestCase):
    """Testes para ModelConfig"""

    def test_model_config_creation(self):
        """Testa criação de ModelConfig"""
        config = ModelConfig(embed_dim=128, max_history=15, vocab_size=1000, num_heads=16, hidden_dim=512, num_layers=6)

        self.assertEqual(config.embed_dim, 128)
        self.assertEqual(config.max_history, 15)
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.num_layers, 6)

    def test_model_config_defaults(self):
        """Testa valores padrão de ModelConfig"""
        config = ModelConfig()

        self.assertEqual(config.embed_dim, 64)
        self.assertEqual(config.max_history, 10)
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.num_layers, 3)


class TestPhysicsConfig(unittest.TestCase):
    """Testes para PhysicsConfig"""

    def test_physics_config_creation(self):
        """Testa criação de PhysicsConfig"""
        config = PhysicsConfig(I0=2.5, alpha=1.5, beta=0.8, k=2.5, omega=1.2)

        self.assertEqual(config.I0, 2.5)
        self.assertEqual(config.alpha, 1.5)
        self.assertEqual(config.beta, 0.8)
        self.assertEqual(config.k, 2.5)
        self.assertEqual(config.omega, 1.2)

    def test_physics_config_defaults(self):
        """Testa valores padrão de PhysicsConfig"""
        config = PhysicsConfig()

        self.assertEqual(config.I0, 1.0)
        self.assertEqual(config.alpha, 1.0)
        self.assertEqual(config.beta, 0.5)
        self.assertEqual(config.k, 2.0)
        self.assertEqual(config.omega, 1.0)


if __name__ == '__main__':
    unittest.main()