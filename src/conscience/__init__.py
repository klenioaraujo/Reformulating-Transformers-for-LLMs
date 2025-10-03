#!/usr/bin/env python3
"""
ΨQRH Conscience Module - Fractal Consciousness Layer
==================================================

Módulo de consciência fractal que implementa matemática avançada de
dinâmica consciente para sistemas ΨQRH inteligentes.

Componentes principais:
- FractalConsciousnessProcessor: Engine de processamento consciente
- ConsciousWaveModulator: Conversor multi-arquivo para .cwm
- ConsciousnessStates: Modelagem de estados (meditação, análise, coma, emergência)
- FractalFieldCalculator: Computação matemática de F(ψ)
- NeuralDiffusionEngine: Calculadora de coeficiente D
- ConsciousnessMetrics: Ferramentas de medição FCI

Equações Fundamentais:
- Dinâmica Consciente: ∂P(ψ,t)/∂t = -∇·[F(ψ)P] + D∇²P
- Campo Fractal: F(ψ) = -∇V(ψ) + η_fractal(t)
- Índice FCI: FCI = (D_EEG × H_fMRI × CLZ) / D_max
"""

from .fractal_consciousness_processor import FractalConsciousnessProcessor
from .conscious_wave_modulator import ConsciousWaveModulator, ΨCWSFile
from .consciousness_states import ConsciousnessState, StateClassifier
from .fractal_field_calculator import FractalFieldCalculator
from .neural_diffusion_engine import NeuralDiffusionEngine
from .consciousness_metrics import ConsciousnessMetrics, FCI

__version__ = "1.0.0"
__author__ = "ΨQRH Framework Team"

# Export principais classes
__all__ = [
    'FractalConsciousnessProcessor',
    'ConsciousWaveModulator',
    'ΨCWSFile',
    'ConsciousnessState',
    'StateClassifier',
    'FractalFieldCalculator',
    'NeuralDiffusionEngine',
    'ConsciousnessMetrics',
    'FCI'
]

# Configurações padrão do módulo
DEFAULT_CONSCIOUSNESS_CONFIG = {
    'fractal_dimension_range': [1.0, 3.0],
    'diffusion_coefficient_range': [0.01, 10.0],
    'consciousness_frequency_range': [0.5, 5.0],  # Hz - brain wave range
    'phase_consciousness': 0.7854,  # π/4 rad - optimal phase
    'chaotic_parameter': 3.9,  # Edge of chaos
    'embedding_dim': 256,
    'sequence_length': 64,
    'device': 'cpu',  # Default device
    'fci_threshold_meditation': 0.8,
    'fci_threshold_analysis': 0.6,
    'fci_threshold_coma': 0.2,
    'fci_threshold_emergence': 0.9
}

def create_consciousness_processor(config: dict = None, metrics_config: dict = None):
    """
    Factory function para criar processador de consciência fractal.

    Args:
        config: Configuração personalizada ou None para usar padrão
        metrics_config: Configuração de métricas (FCI) ou None para usar padrão

    Returns:
        FractalConsciousnessProcessor configurado
    """
    # Importar ConsciousnessConfig localmente para evitar import circular
    from .fractal_consciousness_processor import ConsciousnessConfig

    if config is None:
        config = DEFAULT_CONSCIOUSNESS_CONFIG

    # Se config for dict, converter para ConsciousnessConfig
    if isinstance(config, dict):
        # Criar configuração com valores padrão e sobrescrever com config fornecido
        config_params = DEFAULT_CONSCIOUSNESS_CONFIG.copy()
        config_params.update(config)

        # Criar objeto ConsciousnessConfig
        consciousness_config = ConsciousnessConfig(
            embedding_dim=config_params.get('embedding_dim', 256),
            device=config_params.get('device', 'cpu'),
            fractal_dimension_range=tuple(config_params.get('fractal_dimension_range', [1.0, 3.0])),
            diffusion_coefficient_range=tuple(config_params.get('diffusion_coefficient_range', [0.01, 10.0])),
            consciousness_frequency_range=tuple(config_params.get('consciousness_frequency_range', [0.5, 5.0])),
        )
        config = consciousness_config

    return FractalConsciousnessProcessor(config, metrics_config)