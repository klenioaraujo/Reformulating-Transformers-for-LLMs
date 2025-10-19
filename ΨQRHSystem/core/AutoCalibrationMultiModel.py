#!/usr/bin/env python3
"""
Auto-Calibration Multi-Model - Calibração Automática Específica por Modelo

Sistema de auto-calibração inteligente que ajusta parâmetros físicos
(alpha, beta, omega, k) especificamente para cada tipo de modelo carregado.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import yaml

from .AutoCalibration import AutoCalibration

class AutoCalibrationMultiModel:
    """
    Sistema de auto-calibração multi-modelo para ΨQRH

    Detecta automaticamente o tipo de modelo e aplica calibração
    específica otimizada para suas características.
    """

    def __init__(self, config_path: str = "../config/multi_model_config.yaml"):
        """
        Inicializa o sistema de auto-calibração multi-modelo

        Args:
            config_path: Caminho para configuração multi-modelo
        """
        self.config = self._load_config(config_path)
        self.base_calibrator = AutoCalibration()

        # Cache de calibrações por modelo
        self.calibration_cache = {}

        # Parâmetros físicos por tipo de modelo
        self.model_type_params = {
            "gpt2": {
                "alpha_range": (0.8, 1.2),
                "beta_range": (0.3, 0.7),
                "I0_range": (0.8, 1.2),
                "omega_range": (0.8, 1.2),
                "k_range": (1.8, 2.5),
                "description": "GPT-2 Base Parameters"
            },
            "deepseek-coder": {
                "alpha_range": (1.0, 1.4),
                "beta_range": (0.5, 0.9),
                "I0_range": (1.2, 1.8),
                "omega_range": (1.1, 1.5),
                "k_range": (2.2, 3.0),
                "description": "DeepSeek Coder Parameters"
            },
            "transformer": {
                "alpha_range": (0.9, 1.3),
                "beta_range": (0.4, 0.8),
                "I0_range": (0.9, 1.3),
                "omega_range": (0.9, 1.3),
                "k_range": (2.0, 2.8),
                "description": "Generic Transformer Parameters"
            },
            "simulated": {
                "alpha_range": (1.0, 1.0),
                "beta_range": (0.5, 0.5),
                "I0_range": (1.0, 1.0),
                "omega_range": (1.0, 1.0),
                "k_range": (2.0, 2.0),
                "description": "Simulated Model Parameters"
            }
        }

        print("🔧 Auto-Calibration Multi-Model inicializado")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega configuração multi-modelo"""
        config_file = Path(__file__).parent.parent / "config" / "multi_model_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Configuração padrão"""
        return {
            "physical_calibration": {
                "gpt2": {"alpha": 1.0, "beta": 0.5, "I0": 1.0, "omega": 1.0, "k": 2.0},
                "deepseek-coder": {"alpha": 1.2, "beta": 0.7, "I0": 1.5, "omega": 1.3, "k": 2.5},
                "wiki": {"alpha": 0.8, "beta": 0.3, "I0": 0.8, "omega": 0.9, "k": 1.8},
                "gpt2_simulated": {"alpha": 1.0, "beta": 0.5, "I0": 1.0, "omega": 1.0, "k": 2.0}
            }
        }

    def calibrate_for_model(self, model_name: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calibra parâmetros físicos especificamente para um modelo

        Args:
            model_name: Nome do modelo
            model_info: Informações do modelo

        Returns:
            Parâmetros físicos calibrados
        """
        # Verificar cache primeiro
        if model_name in self.calibration_cache:
            print(f"✅ Usando calibração em cache para {model_name}")
            return self.calibration_cache[model_name]

        print(f"🔧 Calibrando parâmetros para modelo: {model_name}")

        # Detectar tipo de modelo
        model_type = self._detect_model_type(model_name, model_info)

        # Obter parâmetros base para o tipo
        base_params = self._get_base_params_for_type(model_type)

        # Aplicar calibração específica
        calibrated_params = self._apply_model_specific_calibration(
            model_name, model_info, base_params
        )

        # Armazenar em cache
        self.calibration_cache[model_name] = calibrated_params

        print(f"✅ Calibração concluída para {model_name}:")
        print(f"   α={calibrated_params['alpha']:.3f}, β={calibrated_params['beta']:.3f}")
        print(f"   I₀={calibrated_params['I0']:.3f}, ω={calibrated_params['omega']:.3f}, k={calibrated_params['k']:.3f}")

        return calibrated_params

    def _detect_model_type(self, model_name: str, model_info: Dict[str, Any]) -> str:
        """
        Detecta o tipo de modelo baseado nas informações

        Args:
            model_name: Nome do modelo
            model_info: Informações do modelo

        Returns:
            Tipo detectado do modelo
        """
        # Verificar nome do modelo
        name_lower = model_name.lower()

        if "deepseek" in name_lower:
            return "deepseek-coder"
        elif "gpt2" in name_lower:
            return "gpt2"
        elif "wiki" in name_lower:
            return "transformer"
        elif "simulated" in name_lower:
            return "simulated"

        # Verificar informações do modelo
        model_type = model_info.get("type", "")
        if "semantic" in model_type:
            # Tentar inferir do nome original
            if hasattr(model_info, 'get') and 'source_model' in model_info:
                source = model_info['source_model'].lower()
                if "deepseek" in source:
                    return "deepseek-coder"
                elif "gpt2" in source:
                    return "gpt2"

        # Verificar parâmetros do modelo
        vocab_size = model_info.get("vocab_size", 0)
        embed_dim = model_info.get("embed_dim", 0)

        if vocab_size == 51200 and embed_dim == 4096:
            return "deepseek-coder"
        elif vocab_size == 50257 and embed_dim == 768:
            return "gpt2"

        # Padrão
        return "transformer"

    def _get_base_params_for_type(self, model_type: str) -> Dict[str, Any]:
        """
        Obtém parâmetros base para um tipo de modelo

        Args:
            model_type: Tipo do modelo

        Returns:
            Parâmetros base
        """
        if model_type in self.config.get("physical_calibration", {}):
            return self.config["physical_calibration"][model_type]

        # Usar parâmetros do tipo se disponíveis
        if model_type in self.model_type_params:
            type_config = self.model_type_params[model_type]
            # Retornar valores médios das faixas
            return {
                "alpha": (type_config["alpha_range"][0] + type_config["alpha_range"][1]) / 2,
                "beta": (type_config["beta_range"][0] + type_config["beta_range"][1]) / 2,
                "I0": (type_config["I0_range"][0] + type_config["I0_range"][1]) / 2,
                "omega": (type_config["omega_range"][0] + type_config["omega_range"][1]) / 2,
                "k": (type_config["k_range"][0] + type_config["k_range"][1]) / 2
            }

        # Padrão seguro
        return {
            "alpha": 1.0,
            "beta": 0.5,
            "I0": 1.0,
            "omega": 1.0,
            "k": 2.0
        }

    def _apply_model_specific_calibration(self, model_name: str, model_info: Dict[str, Any],
                                        base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica calibração específica para o modelo

        Args:
            model_name: Nome do modelo
            model_info: Informações do modelo
            base_params: Parâmetros base

        Returns:
            Parâmetros calibrados
        """
        calibrated = base_params.copy()

        # Ajustes baseados no tamanho do vocabulário
        vocab_size = model_info.get("vocab_size", 50257)
        if vocab_size > 50000:  # Modelos grandes como DeepSeek
            calibrated["alpha"] *= 1.1
            calibrated["I0"] *= 1.2
            calibrated["omega"] *= 1.1
        elif vocab_size < 30000:  # Modelos menores
            calibrated["alpha"] *= 0.9
            calibrated["beta"] *= 0.8

        # Ajustes baseados na dimensão de embedding
        embed_dim = model_info.get("embed_dim", 768)
        if embed_dim > 2000:  # Grandes embeddings
            calibrated["k"] *= 1.2
            calibrated["beta"] *= 1.1
        elif embed_dim < 500:  # Pequenos embeddings
            calibrated["k"] *= 0.9

        # Ajustes baseados no número de camadas
        num_layers = model_info.get("num_layers", 12)
        if num_layers > 20:  # Muitos layers
            calibrated["omega"] *= 1.1
        elif num_layers < 8:  # Poucos layers
            calibrated["omega"] *= 0.9

        # Garantir limites físicos
        calibrated = self._ensure_physical_constraints(calibrated)

        return calibrated

    def _ensure_physical_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Garante que os parâmetros respeitam restrições físicas

        Args:
            params: Parâmetros a validar

        Returns:
            Parâmetros validados
        """
        # Alpha deve ser positivo
        params["alpha"] = max(0.1, params["alpha"])

        # Beta deve estar entre 0 e 1
        params["beta"] = max(0.01, min(0.99, params["beta"]))

        # I0 deve ser positivo
        params["I0"] = max(0.1, params["I0"])

        # Omega deve ser positivo
        params["omega"] = max(0.1, params["omega"])

        # k deve ser positivo
        params["k"] = max(0.5, params["k"])

        return params

    def get_calibration_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtém informações de calibração

        Args:
            model_name: Nome do modelo (None para todos)

        Returns:
            Informações de calibração
        """
        if model_name:
            return {
                "model": model_name,
                "calibrated": model_name in self.calibration_cache,
                "parameters": self.calibration_cache.get(model_name, {})
            }
        else:
            return {
                "total_calibrations": len(self.calibration_cache),
                "calibrated_models": list(self.calibration_cache.keys()),
                "model_types": self.model_type_params
            }

    def clear_calibration_cache(self):
        """Limpa cache de calibrações"""
        self.calibration_cache.clear()
        print("🧹 Cache de calibrações limpo")

    def export_calibration_config(self, output_path: str):
        """
        Exporta configuração de calibração

        Args:
            output_path: Caminho de saída
        """
        config = {
            "model_type_params": self.model_type_params,
            "cached_calibrations": self.calibration_cache,
            "exported_at": str(torch.randint(0, 1000, (1,)).item())  # Placeholder
        }

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def import_calibration_config(self, input_path: str):
        """
        Importa configuração de calibração

        Args:
            input_path: Caminho do arquivo
        """
        with open(input_path, 'r') as f:
            config = yaml.safe_load(f)

        if "cached_calibrations" in config:
            self.calibration_cache.update(config["cached_calibrations"])

        print(f"✅ Configuração de calibração importada: {len(self.calibration_cache)} calibrações")

    def optimize_for_model(self, model_name: str, model_info: Dict[str, Any],
                          performance_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Otimiza calibração baseada em métricas de performance

        Args:
            model_name: Nome do modelo
            model_info: Informações do modelo
            performance_metrics: Métricas de performance (opcional)

        Returns:
            Parâmetros otimizados
        """
        # Obter calibração atual
        current_params = self.calibrate_for_model(model_name, model_info)

        if not performance_metrics:
            return current_params

        # Aplicar otimizações baseadas em métricas
        optimized = current_params.copy()

        # Exemplo: ajustar baseado em erro de conservação de energia
        energy_error = performance_metrics.get("energy_conservation_error", 0.0)
        if energy_error > 0.1:  # Alto erro
            optimized["alpha"] *= 0.95
            optimized["beta"] *= 1.05

        # Ajustar baseado em coerência quântica
        coherence = performance_metrics.get("quantum_coherence", 0.5)
        if coherence < 0.3:  # Baixa coerência
            optimized["omega"] *= 1.1
            optimized["k"] *= 0.95

        # Garantir restrições
        optimized = self._ensure_physical_constraints(optimized)

        # Atualizar cache
        self.calibration_cache[model_name] = optimized

        return optimized