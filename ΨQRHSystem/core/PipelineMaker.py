import torch
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ΨQRHSystem.configs.SystemConfig import SystemConfig
from ΨQRHSystem.core.PipelineManager import PipelineManager
from ΨQRHSystem.core.PhysicalProcessor import PhysicalProcessor
from ΨQRHSystem.core.QuantumMemory import QuantumMemory
from ΨQRHSystem.core.AutoCalibration import AutoCalibration
from ΨQRHSystem.core.ModelMaker import ModelMaker
from ΨQRHSystem.core.VocabularyMaker import VocabularyMaker


class PipelineMaker:
    """
    PipelineMaker - Cria pipelines completos com componentes configuráveis

    Permite criação programática de pipelines ΨQRH com diferentes
    configurações físicas, quânticas e componentes customizáveis.
    """

    def __init__(self):
        """
        Inicializa PipelineMaker
        """
        self.created_pipelines = []
        self.model_maker = ModelMaker()
        self.vocab_maker = VocabularyMaker()

        print("🔧 PipelineMaker inicializado - criação dinâmica de pipelines ΨQRH")

    def create_physics_pipeline(self, physics_config: Dict[str, Any],
                               embed_dim: int = 64) -> PipelineManager:
        """
        Cria pipeline focado em física quântica

        Args:
            physics_config: Configuração física específica
            embed_dim: Dimensão do embedding

        Returns:
            PipelineManager otimizado para física
        """
        try:
            print("🔧 Criando pipeline físico-quântico...")

            # Configuração otimizada para física
            config = {
                'model': {
                    'embed_dim': embed_dim,
                    'max_history': 15,
                    'vocab_size': 512
                },
                'physics': physics_config,
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

            pipeline = self.model_maker.create_from_config(config)

            # Registrar pipeline criado
            pipeline_info = {
                'id': f"pipeline_{len(self.created_pipelines)}",
                'type': 'physics',
                'physics_config': physics_config,
                'embed_dim': embed_dim,
                'created_at': datetime.now().isoformat(),
                'pipeline': pipeline
            }
            self.created_pipelines.append(pipeline_info)

            print(f"✅ Pipeline físico criado: embed_dim={embed_dim}")
            return pipeline

        except Exception as e:
            print(f"❌ Erro na criação do pipeline físico: {e}")
            raise

    def create_quantum_pipeline(self, quantum_config: Dict[str, Any],
                               memory_depth: int = 20) -> PipelineManager:
        """
        Cria pipeline focado em processamento quântico

        Args:
            quantum_config: Configuração quântica específica
            memory_depth: Profundidade da memória quântica

        Returns:
            PipelineManager otimizado para quantum
        """
        try:
            print("🔧 Criando pipeline quântico...")

            # Configuração otimizada para processamento quântico
            config = {
                'model': {
                    'embed_dim': quantum_config.get('embed_dim', 96),
                    'max_history': memory_depth,
                    'vocab_size': quantum_config.get('vocab_size', 768)
                },
                'physics': quantum_config.get('physics', {
                    'I0': 1.5,
                    'alpha': 2.0,
                    'beta': 1.0,
                    'k': 4.0,
                    'omega': 2.0
                }),
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

            pipeline = self.model_maker.create_from_config(config)

            # Configurar memória quântica profunda
            pipeline.quantum_memory.memory_depth = memory_depth

            # Registrar
            pipeline_info = {
                'id': f"pipeline_{len(self.created_pipelines)}",
                'type': 'quantum',
                'quantum_config': quantum_config,
                'memory_depth': memory_depth,
                'created_at': datetime.now().isoformat(),
                'pipeline': pipeline
            }
            self.created_pipelines.append(pipeline_info)

            print(f"✅ Pipeline quântico criado: memory_depth={memory_depth}")
            return pipeline

        except Exception as e:
            print(f"❌ Erro na criação do pipeline quântico: {e}")
            raise

    def create_hybrid_pipeline(self, components: List[str],
                              base_config: Optional[Dict[str, Any]] = None) -> PipelineManager:
        """
        Cria pipeline híbrido com componentes selecionados

        Args:
            components: Lista de componentes a incluir
            base_config: Configuração base (opcional)

        Returns:
            PipelineManager híbrido
        """
        try:
            print(f"🔧 Criando pipeline híbrido: {components}")

            # Configuração base
            if base_config is None:
                base_config = {
                    'model': {
                        'embed_dim': 64,
                        'max_history': 10,
                        'vocab_size': 512
                    },
                    'physics': {
                        'I0': 1.0,
                        'alpha': 1.0,
                        'beta': 0.5,
                        'k': 2.0,
                        'omega': 1.0
                    }
                }

            # Ajustar componentes habilitados
            valid_components = ["quantum_memory", "auto_calibration", "physical_harmonics"]
            enabled_components = [comp for comp in components if comp in valid_components]

            config = {
                **base_config,
                'system': {
                    'device': 'auto',
                    'enable_components': enabled_components,
                    'validation': {
                        'energy_conservation': 'energy' in components,
                        'unitarity': 'unitary' in components,
                        'numerical_stability': True
                    }
                }
            }

            pipeline = self.model_maker.create_from_config(config)

            # Registrar
            pipeline_info = {
                'id': f"pipeline_{len(self.created_pipelines)}",
                'type': 'hybrid',
                'components': components,
                'enabled_components': enabled_components,
                'created_at': datetime.now().isoformat(),
                'pipeline': pipeline
            }
            self.created_pipelines.append(pipeline_info)

            print(f"✅ Pipeline híbrido criado: {len(enabled_components)} componentes ativos")
            return pipeline

        except Exception as e:
            print(f"❌ Erro na criação do pipeline híbrido: {e}")
            raise

    def create_from_template(self, template_name: str, customizations: Optional[Dict[str, Any]] = None) -> PipelineManager:
        """
        Cria pipeline a partir de template usando ModelMaker

        Args:
            template_name: Nome do template
            customizations: Personalizações (opcional)

        Returns:
            PipelineManager do template
        """
        try:
            print(f"🔧 Criando pipeline do template: {template_name}")

            pipeline = self.model_maker.create_from_template(template_name, customizations)

            # Registrar
            pipeline_info = {
                'id': f"pipeline_{len(self.created_pipelines)}",
                'type': 'template',
                'template': template_name,
                'customizations': customizations,
                'created_at': datetime.now().isoformat(),
                'pipeline': pipeline
            }
            self.created_pipelines.append(pipeline_info)

            print(f"✅ Pipeline do template '{template_name}' criado")
            return pipeline

        except Exception as e:
            print(f"❌ Erro na criação do pipeline do template: {e}")
            raise

    def create_research_pipeline(self, research_focus: str) -> PipelineManager:
        """
        Cria pipeline otimizado para pesquisa específica

        Args:
            research_focus: Foco da pesquisa ("fractal", "quantum", "consciousness", "physics")

        Returns:
            PipelineManager otimizado para pesquisa
        """
        try:
            print(f"🔧 Criando pipeline de pesquisa: {research_focus}")

            # Configurações específicas por foco
            research_configs = {
                "fractal": {
                    'model': {'embed_dim': 128, 'max_history': 25, 'vocab_size': 1024},
                    'physics': {'I0': 2.0, 'alpha': 1.5, 'beta': 0.8, 'k': 3.0, 'omega': 1.5}
                },
                "quantum": {
                    'model': {'embed_dim': 96, 'max_history': 20, 'vocab_size': 768},
                    'physics': {'I0': 1.5, 'alpha': 2.0, 'beta': 1.0, 'k': 4.0, 'omega': 2.0}
                },
                "consciousness": {
                    'model': {'embed_dim': 80, 'max_history': 30, 'vocab_size': 640},
                    'physics': {'I0': 1.2, 'alpha': 1.8, 'beta': 0.9, 'k': 3.5, 'omega': 1.8}
                },
                "physics": {
                    'model': {'embed_dim': 64, 'max_history': 15, 'vocab_size': 512},
                    'physics': {'I0': 1.0, 'alpha': 1.0, 'beta': 0.5, 'k': 2.0, 'omega': 1.0}
                }
            }

            if research_focus not in research_configs:
                raise ValueError(f"Foco de pesquisa inválido: {research_focus}")

            config = {
                **research_configs[research_focus],
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

            pipeline = self.model_maker.create_from_config(config)

            # Registrar
            pipeline_info = {
                'id': f"pipeline_{len(self.created_pipelines)}",
                'type': 'research',
                'research_focus': research_focus,
                'created_at': datetime.now().isoformat(),
                'pipeline': pipeline
            }
            self.created_pipelines.append(pipeline_info)

            print(f"✅ Pipeline de pesquisa '{research_focus}' criado")
            return pipeline

        except Exception as e:
            print(f"❌ Erro na criação do pipeline de pesquisa: {e}")
            raise

    def create_production_pipeline(self, performance_target: str = "balanced") -> PipelineManager:
        """
        Cria pipeline otimizado para produção

        Args:
            performance_target: Alvo de performance ("speed", "accuracy", "balanced")

        Returns:
            PipelineManager otimizado para produção
        """
        try:
            print(f"🔧 Criando pipeline de produção: {performance_target}")

            # Configurações de produção
            production_configs = {
                "speed": {
                    'model': {'embed_dim': 32, 'max_history': 5, 'vocab_size': 256},
                    'physics': {'I0': 0.8, 'alpha': 0.8, 'beta': 0.4, 'k': 1.5, 'omega': 0.8}
                },
                "accuracy": {
                    'model': {'embed_dim': 128, 'max_history': 20, 'vocab_size': 1024},
                    'physics': {'I0': 2.0, 'alpha': 1.5, 'beta': 0.8, 'k': 3.0, 'omega': 1.5}
                },
                "balanced": {
                    'model': {'embed_dim': 64, 'max_history': 10, 'vocab_size': 512},
                    'physics': {'I0': 1.0, 'alpha': 1.0, 'beta': 0.5, 'k': 2.0, 'omega': 1.0}
                }
            }

            config = {
                **production_configs[performance_target],
                'system': {
                    'device': 'auto',
                    'enable_components': ["quantum_memory", "auto_calibration"],  # Sem harmonics para velocidade
                    'validation': {
                        'energy_conservation': performance_target != "speed",
                        'unitarity': True,
                        'numerical_stability': True
                    }
                }
            }

            pipeline = self.model_maker.create_from_config(config)

            # Otimizações de produção
            if performance_target == "speed":
                # Desabilitar validações custosas
                pipeline.config.validation['energy_conservation'] = False

            # Registrar
            pipeline_info = {
                'id': f"pipeline_{len(self.created_pipelines)}",
                'type': 'production',
                'performance_target': performance_target,
                'created_at': datetime.now().isoformat(),
                'pipeline': pipeline
            }
            self.created_pipelines.append(pipeline_info)

            print(f"✅ Pipeline de produção '{performance_target}' criado")
            return pipeline

        except Exception as e:
            print(f"❌ Erro na criação do pipeline de produção: {e}")
            raise

    def create_with_vocabulary(self, vocab_type: str, vocab_params: Dict[str, Any],
                              pipeline_config: Dict[str, Any]) -> Tuple[PipelineManager, Dict[str, Any]]:
        """
        Cria pipeline com vocabulário customizado

        Args:
            vocab_type: Tipo de vocabulário ("semantic", "quantum", "hybrid")
            vocab_params: Parâmetros do vocabulário
            pipeline_config: Configuração do pipeline

        Returns:
            Tupla (PipelineManager, Vocabulário criado)
        """
        try:
            print(f"🔧 Criando pipeline com vocabulário {vocab_type}...")

            # Criar vocabulário
            if vocab_type == "semantic":
                vocab = self.vocab_maker.create_semantic_vocab(**vocab_params)
            elif vocab_type == "quantum":
                vocab = self.vocab_maker.create_quantum_vocab(**vocab_params)
            elif vocab_type == "hybrid":
                vocab = self.vocab_maker.create_hybrid_vocab(**vocab_params)
            else:
                raise ValueError(f"Tipo de vocabulário inválido: {vocab_type}")

            # Ajustar configuração do pipeline para o vocabulário
            pipeline_config['model']['vocab_size'] = len(vocab['tokens'])

            # Criar pipeline
            pipeline = self.model_maker.create_from_config(pipeline_config)

            # Integrar vocabulário no pipeline (se suportado)
            if hasattr(pipeline, 'vocabulary'):
                pipeline.vocabulary = vocab

            # Registrar
            pipeline_info = {
                'id': f"pipeline_{len(self.created_pipelines)}",
                'type': 'with_vocabulary',
                'vocab_type': vocab_type,
                'vocab_size': len(vocab['tokens']),
                'created_at': datetime.now().isoformat(),
                'pipeline': pipeline,
                'vocabulary': vocab
            }
            self.created_pipelines.append(pipeline_info)

            print(f"✅ Pipeline com vocabulário criado: {len(vocab['tokens'])} tokens")
            return pipeline, vocab

        except Exception as e:
            print(f"❌ Erro na criação do pipeline com vocabulário: {e}")
            raise

    def save_pipeline(self, pipeline: PipelineManager, pipeline_path: str,
                     include_vocabulary: bool = False):
        """
        Salva pipeline criado

        Args:
            pipeline: PipelineManager a salvar
            pipeline_path: Caminho para salvar
            include_vocabulary: Incluir vocabulário se disponível
        """
        try:
            os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)

            # Preparar dados para salvamento
            pipeline_data = {
                'metadata': {
                    'created_by': 'PipelineMaker',
                    'created_at': datetime.now().isoformat(),
                    'version': '2.0.0'
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

            # Incluir vocabulário se solicitado e disponível
            if include_vocabulary and hasattr(pipeline, 'vocabulary'):
                pipeline_data['vocabulary'] = pipeline.vocabulary

            # Salvar
            with open(pipeline_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_data, f, indent=2, ensure_ascii=False, default=str)

            print(f"💾 Pipeline salvo: {pipeline_path}")

        except Exception as e:
            print(f"❌ Erro ao salvar pipeline: {e}")
            raise

    def load_pipeline(self, pipeline_path: str) -> PipelineManager:
        """
        Carrega pipeline salvo

        Args:
            pipeline_path: Caminho do pipeline

        Returns:
            PipelineManager carregado
        """
        try:
            with open(pipeline_path, 'r', encoding='utf-8') as f:
                pipeline_data = json.load(f)

            config_data = pipeline_data['config']
            pipeline = self.model_maker.create_from_config(config_data)

            # Restaurar vocabulário se disponível
            if 'vocabulary' in pipeline_data:
                pipeline.vocabulary = pipeline_data['vocabulary']

            print(f"📂 Pipeline carregado: {pipeline_path}")
            return pipeline

        except Exception as e:
            print(f"❌ Erro ao carregar pipeline: {e}")
            raise

    def list_created_pipelines(self) -> List[Dict[str, Any]]:
        """
        Lista pipelines criados nesta sessão

        Returns:
            Lista de informações dos pipelines criados
        """
        return self.created_pipelines.copy()

    def get_pipeline_templates(self) -> Dict[str, str]:
        """
        Obtém templates disponíveis de pipeline

        Returns:
            Dicionário com descrições dos templates
        """
        templates = self.model_maker.get_template_info()
        return {name: info['description'] for name, info in templates.items()}

    def validate_pipeline_config(self, config: Dict[str, Any]) -> bool:
        """
        Valida configuração de pipeline

        Args:
            config: Configuração a validar

        Returns:
            True se válida
        """
        # Usar validação do ModelMaker
        return self.model_maker.validate_model_config(config)