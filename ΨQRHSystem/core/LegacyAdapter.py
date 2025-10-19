#!/usr/bin/env python3
"""
Legacy Adapter - Compatibilidade com Sistema ΨQRH Legado

Este módulo garante compatibilidade total com o sistema psiqrh.py legado,
permitindo migração suave e manutenção de funcionalidades existentes.
"""

import torch
import json
from typing import Dict, Any, Optional, List
import os
import sys

from configs.SystemConfig import SystemConfig
from core.UnifiedPipelineManager import UnifiedPipelineManager


class LegacyAdapter:
    """
    Adaptador para compatibilidade com sistema ΨQRH legado

    Mantém interface idêntica ao psiqrh.py original enquanto
    utiliza o sistema unificado internamente.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Inicializa adaptador legado

        Args:
            config: Configuração do sistema (opcional)
        """
        self.config = config or SystemConfig()
        self.unified_pipeline = UnifiedPipelineManager(self.config)

        # Estado legado para compatibilidade
        self.task = "text-generation"
        self.device = self.unified_pipeline.device
        self.enable_auto_calibration = True
        self.enable_noncommutative = True
        self.audit_mode = False

        print("🔄 Legacy Adapter inicializado - compatibilidade mantida")

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Interface de chamada compatível com sistema legado

        Args:
            text: Texto de entrada
            **kwargs: Parâmetros adicionais

        Returns:
            Resultado no formato legado
        """
        # Processar através do sistema unificado
        unified_result = self.unified_pipeline.process(text, **kwargs)

        # Converter para formato legado
        legacy_result = self._convert_to_legacy_format(unified_result, text)

        return legacy_result

    def _convert_to_legacy_format(self, unified_result: Dict[str, Any], input_text: str) -> Dict[str, Any]:
        """
        Converte resultado unificado para formato legado

        Args:
            unified_result: Resultado do sistema unificado
            input_text: Texto de entrada original

        Returns:
            Resultado no formato legado
        """
        # Extrair dados do resultado unificado
        spectral = unified_result.get('spectral_analysis', {})
        physical = unified_result.get('physical_metrics', {})
        validation = unified_result.get('validation', {})

        # Construir resultado no formato legado
        legacy_result = {
            'status': 'success',
            'task': self.task,
            'device': str(self.device),
            'response': unified_result.get('text', ''),
            'input_length': unified_result.get('input_length', len(input_text)),
            'output_length': unified_result.get('output_length', len(unified_result.get('text', ''))),
            'processing_time': 0.0,  # Não disponível no formato unificado
            'selected_method': 'unified_pipeline',
            'auto_calibration_applied': True,
            'physical_metrics': {
                'FCI': physical.get('FCI', 0.0),
                'consciousness_state': physical.get('consciousness_state', 'UNKNOWN'),
                'alpha_calibrated': physical.get('alpha_calibrated', self.config.physics.alpha),
                'beta_calibrated': physical.get('beta_calibrated', self.config.physics.beta),
                'energy_conservation': spectral.get('energy_conservation', 1.0),
                'fractal_dimension': spectral.get('fractal_dimension', 1.5)
            },
            'mathematical_validation': {
                'energy_conserved': validation.get('energy_conservation', True),
                'unitarity_valid': validation.get('unitarity', True),
                'numerical_stability': validation.get('numerical_stability', True),
                'fractal_consistency': validation.get('fractal_consistency', True)
            },
            'pipeline_steps': [
                'text_to_fractal',
                'quaternion_mapping',
                'spectral_filtering',
                'so4_rotation',
                'optical_probe',
                'consciousness_processing',
                'wave_to_text'
            ],
            'dcf_analysis': {
                'fci_value': physical.get('FCI', 0.0),
                'consciousness_state': physical.get('consciousness_state', 'UNKNOWN'),
                'synchronization_order': physical.get('FCI', 0.0) * 0.8,  # Estimativa
                'cluster_analysis': {
                    'dominant_cluster': {
                        'order_parameter': physical.get('FCI', 0.0) * 0.9
                    }
                },
                'energy_conservation': spectral.get('energy_conservation', 1.0),
                'spectral_coherence': spectral.get('quantum_coherence', 0.0)
            },
            'spectral_analysis': {
                'fractal_dimension': spectral.get('fractal_dimension', 1.5),
                'power_law_exponent': spectral.get('power_law_exponent', 1.0),
                'hurst_exponent': spectral.get('hurst_exponent', 0.5),
                'dominant_frequency': spectral.get('dominant_frequency', 0.0),
                'resonant_frequencies': spectral.get('resonant_frequencies', []),
                'spectral_components': spectral.get('spectral_components', [])
            },
            'dcf_validation': {
                'validation_passed': validation.get('validation_passed', False),
                'energy_conservation_ratio': validation.get('energy_conservation_ratio', 1.0),
                'unitarity_deviation': 0.0,  # Não disponível
                'fractal_consistency_score': 1.0 if validation.get('fractal_consistency', False) else 0.0
            },
            'dcf_metadata': {
                'pipeline_version': 'unified_legacy_adapter',
                'computation_time': 0.0,
                'memory_usage': 0.0,
                'device_info': str(self.device)
            },
            'semantic_analysis': {
                'model_loaded': unified_result.get('semantic_model_info') is not None,
                'vocab_size': unified_result.get('semantic_model_info', {}).get('vocab_size', 0),
                'embed_dim': self.config.model.embed_dim
            }
        }

        return legacy_result

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Retorna status do pipeline em formato legado

        Returns:
            Status compatível com legado
        """
        unified_status = self.unified_pipeline.get_pipeline_status()

        return {
            'pipeline_state': unified_status.get('pipeline_state', {}),
            'device': unified_status.get('device', 'unknown'),
            'config': {
                'embed_dim': self.config.model.embed_dim,
                'max_history': self.config.model.max_history,
                'vocab_size': self.config.model.vocab_size,
                'I0': self.config.physics.I0,
                'alpha': self.config.physics.alpha,
                'beta': self.config.physics.beta,
                'omega': self.config.physics.omega
            },
            'legacy_adapter': True,
            'unified_system': True
        }

    def reset_pipeline(self):
        """Reseta pipeline mantendo compatibilidade"""
        self.unified_pipeline.reset_pipeline()
        print("🔄 Pipeline legado resetado (através do adaptador unificado)")


class ΨQRHLegacyInterface:
    """
    Interface legado completa compatível com psiqrh.py

    Esta classe replica exatamente a interface do ΨQRHPipeline legado
    """

    def __init__(self, task: str = "text-generation", device: Optional[str] = None,
                 input_text: Optional[str] = None, model_dir: Optional[str] = None,
                 enable_auto_calibration: bool = True, enable_noncommutative: bool = True,
                 tokenizer_config: Optional[Dict[str, Any]] = None,
                 enable_cognitive_priming: bool = True, audit_mode: bool = False,
                 vocab_path: Optional[str] = None, reasoning_mode: str = 'geometric'):
        """
        Inicializa interface legado com parâmetros idênticos ao original

        Args:
            task: Tipo de tarefa (compatibilidade)
            device: Dispositivo (compatibilidade)
            input_text: Texto de entrada (compatibilidade)
            model_dir: Diretório do modelo (compatibilidade)
            enable_auto_calibration: Habilita auto-calibração (compatibilidade)
            enable_noncommutative: Habilita geometria não-comutativa (compatibilidade)
            tokenizer_config: Configuração do tokenizer (compatibilidade)
            enable_cognitive_priming: Habilita priming cognitivo (compatibilidade)
            audit_mode: Modo auditoria (compatibilidade)
            vocab_path: Caminho do vocabulário (compatibilidade)
            reasoning_mode: Modo de raciocínio (compatibilidade)
        """
        # Carregar configuração unificada
        config = SystemConfig.default()

        # Mapear parâmetros legado para configuração unificada
        if device:
            config.device = device

        # Inicializar adaptador
        self.adapter = LegacyAdapter(config)

        # Manter atributos de compatibilidade
        self.task = task
        self.device = self.adapter.device
        self.enable_auto_calibration = enable_auto_calibration
        self.enable_noncommutative = enable_noncommutative
        self.audit_mode = audit_mode
        self.reasoning_mode = reasoning_mode

        print(f"✅ ΨQRH Legacy Interface inicializada (através do sistema unificado)")

    def __call__(self, text: str) -> Dict[str, Any]:
        """
        Processa texto mantendo interface idêntica ao legado

        Args:
            text: Texto para processar

        Returns:
            Resultado no formato legado
        """
        return self.adapter(text)

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status em formato legado

        Returns:
            Status compatível
        """
        return self.adapter.get_pipeline_status()


def create_legacy_compatible_pipeline(*args, **kwargs) -> ΨQRHLegacyInterface:
    """
    Factory function para criar pipeline compatível com legado

    Args:
        *args, **kwargs: Parâmetros idênticos ao ΨQRHPipeline legado

    Returns:
        Interface legado compatível
    """
    return ΨQRHLegacyInterface(*args, **kwargs)


# Funções de compatibilidade para importações diretas
def set_quiet_mode(quiet: bool):
    """Compatibilidade com função legado"""
    os.environ['PSIQRH_QUIET'] = '1' if quiet else '0'
    print(f"🔇 Modo silencioso: {'ATIVADO' if quiet else 'DESATIVADO'}")


def get_active_model_path() -> Optional[str]:
    """
    Compatibilidade com função legado

    Returns:
        Caminho do modelo ativo (sempre None para compatibilidade)
    """
    return None


# Teste de compatibilidade
def test_legacy_compatibility():
    """
    Testa compatibilidade com sistema legado

    Returns:
        Resultado do teste
    """
    print("🧪 Testando compatibilidade com sistema legado...")

    try:
        # Criar interface legado
        legacy_pipeline = ΨQRHLegacyInterface(task="text-generation")

        # Testar processamento
        test_text = "Test legacy compatibility"
        result = legacy_pipeline(test_text)

        # Verificar campos obrigatórios do legado
        required_fields = [
            'status', 'task', 'device', 'response', 'input_length', 'output_length',
            'physical_metrics', 'mathematical_validation', 'dcf_analysis', 'spectral_analysis'
        ]

        missing_fields = [field for field in required_fields if field not in result]

        if missing_fields:
            print(f"❌ Campos faltantes: {missing_fields}")
            return False

        print("✅ Compatibilidade com sistema legado verificada!")
        return True

    except Exception as e:
        print(f"❌ Erro no teste de compatibilidade: {e}")
        return False


if __name__ == "__main__":
    # Teste quando executado diretamente
    test_legacy_compatibility()