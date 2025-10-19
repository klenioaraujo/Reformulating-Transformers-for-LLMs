"""
ΨQRH System - Complete Modular Quantum-Fractal-Optical Processing Framework

Este pacote implementa o sistema ΨQRH completamente modular com 8 componentes,
baseado na equação de Padilha: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

ARQUITETURA COMPLETA (8 COMPONENTES):

CORE PROCESSING (4 componentes):
- PipelineManager: Orquestração completa do pipeline ΨQRH
- PhysicalProcessor: Processamento físico com equação de Padilha
- QuantumMemory: Memória temporal quântica e processamento de consciência
- AutoCalibration: Calibração emergente de parâmetros físicos

CREATION/MAKER (3 componentes):
- ModelMaker: Criação dinâmica de modelos ΨQRH com templates
- VocabularyMaker: Criação de vocabulários de fontes diversas
- PipelineMaker: Criação avançada de pipelines customizados

COMPATIBILITY (1 componente):
- LegacyAdapter: Compatibilidade total com psiqrh.py original

MÓDULOS FÍSICOS:
- PadilhaEquation: Implementação da equação fundamental
- QuaternionOps: Operações quaterniônicas SO(4)
- SpectralFiltering: Filtragem espectral F(k) com conservação

ZERO FALLBACK POLICY: Sistema falha limpo se componentes obrigatórios ausentes.
VALIDAÇÃO MATEMÁTICA: Conservação de energia, unitariedade, estabilidade numérica.
"""

from .configs.SystemConfig import SystemConfig, ModelConfig, PhysicsConfig
from .core.PipelineManager import PipelineManager
from .core.PhysicalProcessor import PhysicalProcessor
from .core.QuantumMemory import QuantumMemory
from .core.AutoCalibration import AutoCalibration
from .core.LegacyAdapter import LegacyAdapter
from .core.ModelMaker import ModelMaker
from .core.VocabularyMaker import VocabularyMaker
from .core.PipelineMaker import PipelineMaker
from .physics.PadilhaEquation import PadilhaEquation
from .physics.QuaternionOps import QuaternionOps
from .physics.SpectralFiltering import SpectralFiltering

__version__ = "2.0.0"
__author__ = "ΨQRH Development Team"

__all__ = [
    # Configuração
    'SystemConfig',
    'ModelConfig',
    'PhysicsConfig',

    # Core Processing (4)
    'PipelineManager',
    'PhysicalProcessor',
    'QuantumMemory',
    'AutoCalibration',

    # Creation/Maker (3)
    'ModelMaker',
    'VocabularyMaker',
    'PipelineMaker',

    # Compatibility (1)
    'LegacyAdapter',

    # Physics Modules
    'PadilhaEquation',
    'QuaternionOps',
    'SpectralFiltering'
]