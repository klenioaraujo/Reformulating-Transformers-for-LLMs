import torch
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ΨQRHSystem.configs.SystemConfig import SystemConfig, PhysicsConfig, ModelConfig
from ΨQRHSystem.core.PipelineManager import PipelineManager
from ΨQRHSystem.core.PhysicalProcessor import PhysicalProcessor
from ΨQRHSystem.core.QuantumMemory import QuantumMemory
from ΨQRHSystem.core.AutoCalibration import AutoCalibration


class ModelMaker:
    """
    ModelMaker - Cria novos modelos ΨQRH com arquitetura dinâmica

    Permite criação programática de modelos ΨQRH com diferentes configurações,
    templates pré-definidos e personalização completa da arquitetura.
    """

    def __init__(self, base_config_path: Optional[str] = None):
        """
        Inicializa ModelMaker com configuração base

        Args:
            base_config_path: Caminho para configuração base (opcional)
        """
        self.base_config_path = base_config_path or "ΨQRHSystem/config/system_config.yaml"
        self.templates = self._load_templates()
        self.created_models = []

        print("🔧 ModelMaker inicializado - criação dinâmica de modelos ΨQRH")

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Carrega templates pré-definidos de modelos

        Returns:
            Dicionário com templates de modelos
        """
        return {
            "minimal": {
                "model": {"embed_dim": 32, "max_history": 5, "vocab_size": 256},
                "physics": {"I0": 0.5, "alpha": 0.5, "beta": 0.25, "k": 1.0, "omega": 0.5},
                "description": "Modelo minimal para testes rápidos"
            },
            "standard": {
                "model": {"embed_dim": 64, "max_history": 10, "vocab_size": 512},
                "physics": {"I0": 1.0, "alpha": 1.0, "beta": 0.5, "k": 2.0, "omega": 1.0},
                "description": "Modelo padrão balanceado"
            },
            "advanced": {
                "model": {"embed_dim": 128, "max_history": 20, "vocab_size": 1024},
                "physics": {"I0": 2.0, "alpha": 1.5, "beta": 0.75, "k": 3.0, "omega": 1.5},
                "description": "Modelo avançado para aplicações complexas"
            },
            "quantum_focused": {
                "model": {"embed_dim": 96, "max_history": 15, "vocab_size": 768},
                "physics": {"I0": 1.5, "alpha": 2.0, "beta": 1.0, "k": 4.0, "omega": 2.0},
                "description": "Modelo otimizado para processamento quântico"
            }
        }

    def create_from_config(self, config: Dict[str, Any]) -> PipelineManager:
        """
        Cria modelo ΨQRH a partir de configuração completa

        Args:
            config: Configuração completa do modelo

        Returns:
            PipelineManager configurado
        """
        try:
            # Criar configuração do sistema
            system_config = SystemConfig(
                model=ModelConfig(**config.get('model', {})),
                physics=PhysicsConfig(**config.get('physics', {})),
                device=config.get('system', {}).get('device', 'auto'),
                enable_components=config.get('system', {}).get('enable_components', None),
                validation=config.get('system', {}).get('validation', None)
            )

            # Criar pipeline
            pipeline = PipelineManager(system_config)

            # Registrar modelo criado
            model_type = config.get('type', 'from_config')
            model_info = {
                'id': f"model_{len(self.created_models)}",
                'type': model_type,
                'config': config,
                'created_at': datetime.now().isoformat(),
                'pipeline': pipeline
            }
            self.created_models.append(model_info)

            print(f"✅ Modelo criado a partir de configuração: {model_info['id']}")
            return pipeline

        except Exception as e:
            print(f"❌ Erro na criação do modelo: {e}")
            raise

    def create_from_template(self, template_name: str, customizations: Optional[Dict[str, Any]] = None) -> PipelineManager:
        """
        Cria modelo ΨQRH a partir de template pré-definido

        Args:
            template_name: Nome do template
            customizations: Personalizações opcionais

        Returns:
            PipelineManager configurado
        """
        if template_name not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"Template '{template_name}' não encontrado. Disponíveis: {available}")

        # Obter configuração base do template
        config = self.templates[template_name].copy()

        # Aplicar customizações
        if customizations:
            for section, params in customizations.items():
                if section in config:
                    config[section].update(params)
                else:
                    config[section] = params

        # Adicionar informações do template
        config['template'] = template_name
        config['description'] = self.templates[template_name]['description']
        config['type'] = 'template'

        print(f"🔧 Criando modelo do template '{template_name}': {config['description']}")

        return self.create_from_config(config)

    def create_custom(self, embed_dim: int, num_heads: Optional[int] = None,
                     vocab_size: Optional[int] = None, physics_params: Optional[Dict[str, float]] = None) -> PipelineManager:
        """
        Cria modelo ΨQRH customizado com parâmetros específicos

        Args:
            embed_dim: Dimensão do embedding
            num_heads: Número de cabeças de atenção (opcional)
            vocab_size: Tamanho do vocabulário (opcional)
            physics_params: Parâmetros físicos customizados (opcional)

        Returns:
            PipelineManager configurado
        """
        # Configuração customizada
        config = {
            'model': {
                'embed_dim': embed_dim,
                'max_history': 10,
                'vocab_size': vocab_size or 512
            },
            'physics': physics_params or {
                'I0': 1.0,
                'alpha': 1.0,
                'beta': 0.5,
                'k': 2.0,
                'omega': 1.0
            },
            'system': {
                'device': 'auto',
                'enable_components': ["quantum_memory", "auto_calibration", "physical_harmonics"],
                'validation': {
                    'energy_conservation': True,
                    'unitarity': True,
                    'numerical_stability': True
                }
            }
        }

        print(f"🔧 Criando modelo customizado: embed_dim={embed_dim}, vocab_size={vocab_size or 512}")

        config['type'] = 'custom'
        return self.create_from_config(config)

    def create_quantum_optimized(self, complexity_level: str = "medium") -> PipelineManager:
        """
        Cria modelo otimizado para processamento quântico

        Args:
            complexity_level: Nível de complexidade ("low", "medium", "high")

        Returns:
            PipelineManager otimizado para quantum
        """
        complexity_configs = {
            "low": {"embed_dim": 64, "vocab_size": 256, "I0": 0.8, "alpha": 0.8},
            "medium": {"embed_dim": 96, "vocab_size": 512, "I0": 1.2, "alpha": 1.2},
            "high": {"embed_dim": 128, "vocab_size": 1024, "I0": 1.5, "alpha": 1.5}
        }

        if complexity_level not in complexity_configs:
            raise ValueError(f"Nível de complexidade inválido: {complexity_level}")

        params = complexity_configs[complexity_level]

        config = {
            'model': {
                'embed_dim': params['embed_dim'],
                'max_history': 15,
                'vocab_size': params['vocab_size']
            },
            'physics': {
                'I0': params['I0'],
                'alpha': params['alpha'],
                'beta': 0.6,
                'k': 2.5,
                'omega': 1.2
            },
            'system': {
                'device': 'auto',
                'enable_components': ["quantum_memory", "auto_calibration", "physical_harmonics"],
                'validation': {
                    'energy_conservation': True,
                    'unitarity': True,
                    'numerical_stability': True
                }
            }
        }

        print(f"🔧 Criando modelo quântico otimizado (complexidade: {complexity_level})")

        return self.create_from_config(config)

    def save_model(self, pipeline: PipelineManager, model_path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Salva modelo criado em arquivo

        Args:
            pipeline: PipelineManager a salvar
            model_path: Caminho para salvar
            metadata: Metadados adicionais (opcional)
        """
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Preparar dados para salvamento
            model_data = {
                'metadata': {
                    'created_by': 'ModelMaker',
                    'created_at': datetime.now().isoformat(),
                    'version': '2.0.0',
                    **(metadata or {})
                },
                'config': {
                    'model': pipeline.config.model.__dict__,
                    'physics': pipeline.config.physics.__dict__,
                    'system': {
                        'device': pipeline.config.device,
                        'enable_components': pipeline.config.enable_components,
                        'validation': pipeline.config.validation
                    }
                },
                'pipeline_state': pipeline.pipeline_state
            }

            # Salvar em JSON
            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False, default=str)

            print(f"💾 Modelo salvo: {model_path}")

        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {e}")
            raise

    def load_model(self, model_path: str) -> PipelineManager:
        """
        Carrega modelo salvo

        Args:
            model_path: Caminho do modelo salvo

        Returns:
            PipelineManager carregado
        """
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)

            config_data = model_data['config']
            pipeline = self.create_from_config(config_data)

            print(f"📂 Modelo carregado: {model_path}")
            return pipeline

        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise

    def list_created_models(self) -> List[Dict[str, Any]]:
        """
        Lista modelos criados nesta sessão

        Returns:
            Lista de informações dos modelos criados
        """
        return self.created_models.copy()

    def get_template_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtém informações dos templates disponíveis

        Returns:
            Dicionário com informações dos templates
        """
        return self.templates.copy()

    def validate_model_config(self, config: Dict[str, Any]) -> bool:
        """
        Valida configuração de modelo

        Args:
            config: Configuração a validar

        Returns:
            True se válida
        """
        required_sections = ['model', 'physics']
        required_model = ['embed_dim', 'vocab_size']
        required_physics = ['I0', 'alpha', 'beta', 'k', 'omega']

        # Verificar seções obrigatórias
        for section in required_sections:
            if section not in config:
                print(f"❌ Seção obrigatória faltando: {section}")
                return False

        # Verificar parâmetros do modelo
        for param in required_model:
            if param not in config['model']:
                print(f"❌ Parâmetro obrigatório do modelo faltando: {param}")
                return False

        # Verificar parâmetros físicos
        for param in required_physics:
            if param not in config['physics']:
                print(f"❌ Parâmetro físico obrigatório faltando: {param}")
                return False

        # Validações de range
        if config['model']['embed_dim'] < 16 or config['model']['embed_dim'] > 512:
            print("❌ embed_dim deve estar entre 16 e 512")
            return False

        if config['physics']['I0'] <= 0:
            print("❌ I0 deve ser positivo")
            return False

        return True