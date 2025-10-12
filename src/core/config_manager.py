#!/usr/bin/env python3
"""
Centralized Configuration Manager with Pydantic Validation
==========================================================

Sistema centralizado de configuração que valida e gerencia todos os arquivos YAML
do projeto ΨQRH, fornecendo uma interface unificada e validada.

Features:
- Validação automática de tipos e valores
- Carregamento hierárquico de configurações
- Suporte a overrides em runtime
- Validação de dependências entre configurações
- Cache inteligente de configurações carregadas

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import json


class KuramotoConfig(BaseModel):
    """Configuração do sistema Kuramoto"""
    coupling_strength: float = Field(1.0, ge=0.0, le=10.0)
    natural_frequency_mean: float = Field(1.0, ge=0.1, le=5.0)
    natural_frequency_std: float = Field(0.5, ge=0.01, le=2.0)
    num_integration_steps: int = Field(100, ge=10, le=1000)
    time_step: float = Field(0.01, ge=0.001, le=0.1)
    diffusion_coefficient: float = Field(0.1, ge=0.001, le=1.0)
    enable_adaptive_coupling: bool = True
    threshold: float = Field(0.8, ge=0.0, le=1.0)
    adaptive_coupling_rate: float = Field(0.01, ge=0.001, le=0.1)

    class Config:
        validate_assignment = True


class SpatialGridConfig(BaseModel):
    """Configuração do grid espacial"""
    height: int = Field(8, ge=4, le=64)
    width: int = Field(8, ge=4, le=64)
    depth: int = Field(1, ge=1, le=8)
    topology: str = Field("grid_2d", pattern="^(grid_2d|grid_3d|hexagonal|random)$")
    periodic_boundary: bool = True

    class Config:
        validate_assignment = True


class ConnectivityConfig(BaseModel):
    """Configuração de conectividade"""
    sigma_connectivity: Optional[float] = Field(None, ge=0.1, le=10.0)
    decay_function: str = Field("gaussian", pattern="^(gaussian|exponential|linear)$")
    normalize_connections: bool = True

    class Config:
        validate_assignment = True


class ConsciousnessMetricsConfig(BaseModel):
    """Configuração das métricas de consciência"""
    d_eeg_max: float = Field(2.0, ge=0.1, le=10.0)
    h_fmri_max: float = Field(1.0, ge=0.1, le=5.0)
    clz_max: float = Field(5.0, ge=0.1, le=10.0)
    fci_thresholds: Dict[str, float] = Field({
        'coma': 0.15,
        'analysis': 0.3,
        'meditation': 0.6,
        'emergence': 0.8
    })

    @field_validator('fci_thresholds')
    @classmethod
    def validate_thresholds(cls, v):
        required_keys = {'coma', 'analysis', 'meditation', 'emergence'}
        if not all(key in v for key in required_keys):
            raise ValueError(f"FCI thresholds must include: {required_keys}")

        # Validar ordem crescente
        values = [v[k] for k in ['coma', 'analysis', 'meditation', 'emergence']]
        if not all(values[i] < values[i+1] for i in range(len(values)-1)):
            raise ValueError("FCI thresholds must be in ascending order")

        return v

    class Config:
        validate_assignment = True


class NeuralDiffusionConfig(BaseModel):
    """Configuração do motor de difusão neural"""
    diffusion_coefficient_range: List[float] = Field([0.01, 10.0], min_items=2, max_items=2)
    epsilon: float = Field(1e-10, ge=1e-15, le=1e-5)
    field_smoothing_kernel: List[float] = Field([0.25, 0.5, 0.25], min_items=3, max_items=3)

    @field_validator('diffusion_coefficient_range')
    @classmethod
    def validate_diffusion_range(cls, v):
        if v[0] >= v[1]:
            raise ValueError("Diffusion coefficient range must be [min, max] with min < max")
        return v

    class Config:
        validate_assignment = True


class DCFConfig(BaseModel):
    """Configuração do sistema DCF (Dinâmica de Consciência Fractal)"""
    n_candidates: int = Field(50, ge=10, le=200)
    kuramoto_steps: int = Field(100, ge=10, le=500)
    initial_coupling_strength: float = Field(1.0, ge=0.1, le=5.0)
    diffusion_modulation_factor: float = Field(2.0, ge=0.5, le=10.0)
    initial_diffusion_coefficient: float = Field(0.5, ge=0.01, le=2.0)

    class Config:
        validate_assignment = True


class PerformanceConfig(BaseModel):
    """Configuração de performance"""
    device: str = Field("auto", pattern="^(auto|cpu|cuda|mps)$")
    enable_jit: bool = True
    batch_processing: bool = True
    memory_efficient: bool = True

    class Config:
        validate_assignment = True


class QRHIntegrationConfig(BaseModel):
    """Configuração da integração ΨQRH"""
    embed_dim: int = Field(64, ge=16, le=1024)
    quaternion_multiplier: int = Field(4, ge=1, le=8)
    use_layer_norm: bool = True
    residual_weight: float = Field(0.1, ge=0.0, le=1.0)
    use_residual_connection: bool = True

    class Config:
        validate_assignment = True


class PsiQRHConfig(BaseModel):
    """Configuração completa do sistema ΨQRH"""
    # Componentes principais
    kuramoto_spectral_layer: Dict[str, Any] = Field(default_factory=dict)
    consciousness_metrics: Dict[str, Any] = Field(default_factory=dict)
    neural_diffusion_engine: Dict[str, Any] = Field(default_factory=dict)
    dcf_config: Dict[str, Any] = Field(default_factory=dict)

    # Configurações derivadas
    kuramoto: KuramotoConfig = Field(default_factory=KuramotoConfig)
    spatial_grid: SpatialGridConfig = Field(default_factory=SpatialGridConfig)
    connectivity: ConnectivityConfig = Field(default_factory=ConnectivityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    qrh_integration: QRHIntegrationConfig = Field(default_factory=QRHIntegrationConfig)

    @model_validator(mode='before')
    @classmethod
    def build_nested_configs(cls, values):
        """Constrói configurações aninhadas a partir dos dicionários YAML"""
        # Kuramoto config
        kuramoto_data = values.get('kuramoto_spectral_layer', {})
        if kuramoto_data:
            values['kuramoto'] = kuramoto_data.get('oscillator_dynamics', {})
            values['spatial_grid'] = kuramoto_data.get('spatial_grid', {})
            values['connectivity'] = kuramoto_data.get('connectivity', {})
            values['performance'] = kuramoto_data.get('performance', {})
            values['qrh_integration'] = kuramoto_data.get('qrh_integration', {})

        return values

    class Config:
        validate_assignment = True


class ConfigManager:
    """
    Gerenciador centralizado de configurações com validação Pydantic.

    Carrega, valida e fornece acesso unificado a todas as configurações do sistema.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Inicializa o gerenciador de configurações.

        Args:
            config_dir: Diretório base para arquivos de configuração
        """
        self.config_dir = Path(config_dir) if config_dir else self._find_config_dir()
        self._config_cache: Dict[str, Any] = {}
        self._validated_configs: Dict[str, BaseModel] = {}

        print("🔧 Inicializando ConfigManager centralizado...")

    def _find_config_dir(self) -> Path:
        """Encontra o diretório de configurações automaticamente"""
        # Tentar caminhos relativos ao arquivo atual
        current_dir = Path(__file__).parent

        # Caminhos possíveis
        possible_paths = [
            current_dir.parent / "configs",  # src/configs
            current_dir.parent.parent / "configs",  # configs/
            Path("configs"),  # configs/ relativo ao cwd
        ]

        for path in possible_paths:
            if path.exists() and path.is_dir():
                return path

        # Fallback: criar diretório se não existir
        fallback_path = Path("configs")
        fallback_path.mkdir(exist_ok=True)
        print(f"⚠️  Diretório de configs não encontrado, criando: {fallback_path}")
        return fallback_path

    def load_config(self, config_name: str, validate: bool = True) -> Dict[str, Any]:
        """
        Carrega uma configuração específica com validação opcional.

        Args:
            config_name: Nome do arquivo de configuração (sem extensão)
            validate: Se deve validar com Pydantic

        Returns:
            Dicionário de configuração validado
        """
        if config_name in self._config_cache:
            return self._config_cache[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            print(f"⚠️  Arquivo de configuração não encontrado: {config_path}")
            return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}

            if validate:
                validated_config = self._validate_config(config_name, config_data)
                self._validated_configs[config_name] = validated_config
                # Converter de volta para dict para compatibilidade
                config_data = validated_config.model_dump()

            self._config_cache[config_name] = config_data
            print(f"✅ Configuração carregada: {config_name}")
            return config_data

        except Exception as e:
            print(f"❌ Erro carregando configuração {config_name}: {e}")
            return {}

    def _validate_config(self, config_name: str, config_data: Dict[str, Any]) -> BaseModel:
        """Valida configuração com modelo Pydantic apropriado"""
        validators = {
            'kuramoto_config': PsiQRHConfig,
            'consciousness_metrics': ConsciousnessMetricsConfig,
            'neural_diffusion_engine': NeuralDiffusionConfig,
            'dcf_config': DCFConfig,
        }

        validator_class = validators.get(config_name)
        if validator_class:
            try:
                return validator_class(**config_data)
            except Exception as e:
                print(f"⚠️  Validação falhou para {config_name}: {e}")
                # Retornar instância com valores padrão
                return validator_class()
        else:
            # Para configs sem validador específico, usar validação básica
            return self._basic_validation(config_data)

    def _basic_validation(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validação básica para configurações sem modelo específico"""
        # Verificar tipos básicos e ranges razoáveis
        validated = {}

        for key, value in config_data.items():
            if isinstance(value, dict):
                validated[key] = self._basic_validation(value)
            elif isinstance(value, (int, float)):
                # Verificar se é um número razoável
                if abs(value) > 1e10:
                    print(f"⚠️  Valor suspeito para {key}: {value}")
                validated[key] = value
            else:
                validated[key] = value

        return validated

    def get_config(self, config_name: str, key_path: Optional[str] = None) -> Any:
        """
        Obtém configuração específica ou valor aninhado.

        Args:
            config_name: Nome da configuração
            key_path: Caminho para valor aninhado (ex: "kuramoto.coupling_strength")

        Returns:
            Valor de configuração ou dicionário completo
        """
        config = self.load_config(config_name)

        if key_path:
            keys = key_path.split('.')
            value = config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value

        return config

    def set_config(self, config_name: str, key_path: str, value: Any) -> bool:
        """
        Define valor de configuração em runtime.

        Args:
            config_name: Nome da configuração
            key_path: Caminho para o valor (ex: "kuramoto.coupling_strength")
            value: Novo valor

        Returns:
            True se definido com sucesso
        """
        config = self.load_config(config_name, validate=False)

        keys = key_path.split('.')
        target = config

        # Navegar até o penúltimo nível
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]

        # Definir valor
        target[keys[-1]] = value

        # Revalidar se possível
        if config_name in self._validated_configs:
            try:
                validated = self._validate_config(config_name, config)
                self._validated_configs[config_name] = validated
                config = validated.model_dump()
            except Exception as e:
                print(f"⚠️  Revalidação falhou após alteração: {e}")

        # Atualizar cache
        self._config_cache[config_name] = config

        return True

    def save_config(self, config_name: str, file_path: Optional[str] = None) -> bool:
        """
        Salva configuração em arquivo YAML.

        Args:
            config_name: Nome da configuração
            file_path: Caminho do arquivo (opcional)

        Returns:
            True se salvo com sucesso
        """
        if config_name not in self._config_cache:
            print(f"⚠️  Configuração não carregada: {config_name}")
            return False

        save_path = Path(file_path) if file_path else self.config_dir / f"{config_name}.yaml"

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config_cache[config_name], f, default_flow_style=False, indent=2)
            print(f"💾 Configuração salva: {save_path}")
            return True
        except Exception as e:
            print(f"❌ Erro salvando configuração: {e}")
            return False

    def validate_all_configs(self) -> Dict[str, bool]:
        """
        Valida todas as configurações encontradas.

        Returns:
            Dicionário com status de validação por configuração
        """
        validation_status = {}

        # Encontrar todos os arquivos YAML
        yaml_files = list(self.config_dir.glob("*.yaml"))

        for yaml_file in yaml_files:
            config_name = yaml_file.stem
            try:
                config = self.load_config(config_name, validate=True)
                validation_status[config_name] = True
                print(f"✅ {config_name}: válido")
            except Exception as e:
                validation_status[config_name] = False
                print(f"❌ {config_name}: inválido - {e}")

        return validation_status

    def get_system_config(self) -> PsiQRHConfig:
        """
        Obtém configuração completa do sistema ΨQRH.

        Returns:
            Configuração validada do sistema
        """
        # Carregar componentes individuais
        kuramoto_config = self.load_config('kuramoto_config')
        consciousness_config = self.load_config('consciousness_metrics')
        diffusion_config = self.load_config('neural_diffusion_engine')
        dcf_config = self.load_config('dcf_config')

        # Combinar em configuração completa
        full_config_data = {
            'kuramoto_spectral_layer': kuramoto_config,
            'consciousness_metrics': consciousness_config,
            'neural_diffusion_engine': diffusion_config,
            'dcf_config': dcf_config,
        }

        try:
            return PsiQRHConfig(**full_config_data)
        except Exception as e:
            print(f"⚠️  Erro criando configuração completa: {e}")
            return PsiQRHConfig()

    def export_to_json(self, config_name: str, file_path: str) -> bool:
        """
        Exporta configuração para formato JSON.

        Args:
            config_name: Nome da configuração
            file_path: Caminho do arquivo JSON

        Returns:
            True se exportado com sucesso
        """
        config = self.load_config(config_name)
        if not config:
            return False

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"📄 Configuração exportada para JSON: {file_path}")
            return True
        except Exception as e:
            print(f"❌ Erro exportando para JSON: {e}")
            return False


# Instância global do gerenciador
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Obtém instância global do gerenciador de configurações"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def load_psiqrh_config() -> PsiQRHConfig:
    """Carrega configuração completa do sistema ΨQRH"""
    manager = get_config_manager()
    return manager.get_system_config()


if __name__ == "__main__":
    # Teste do sistema de configuração
    print("🧪 Testando ConfigManager...")

    manager = ConfigManager()

    # Testar carregamento
    kuramoto_config = manager.load_config('kuramoto_config')
    print(f"📁 Kuramoto config keys: {list(kuramoto_config.keys()) if kuramoto_config else 'None'}")

    # Testar configuração completa
    system_config = manager.get_system_config()
    print(f"🔧 System config kuramoto coupling: {system_config.kuramoto.coupling_strength}")

    # Testar validação
    validation_results = manager.validate_all_configs()
    print(f"🔍 Validation results: {validation_results}")

    print("✅ ConfigManager test completed!")