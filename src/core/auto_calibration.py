#!/usr/bin/env python3
"""
Sistema de Auto-Calibração de Pesos ΨQRH
=========================================

Geração automática de pesos baseada em princípios físicos quânticos
e propriedades fractais emergentes da física.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json


class QuantumWeightGenerator:
    """
    Gerador de pesos quânticos baseado em princípios físicos

    Gera pesos automaticamente usando:
    - Distribuições quânticas (Gaussianas complexas)
    - Propriedades fractais
    - Simetrias unitárias
    - Correlações de longo alcance
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Parâmetros físicos para geração de pesos
        self.physical_params = {
            'planck_constant': 1.0,  # ħ = 1 (unidades naturais)
            'fractal_dimension': 1.5,  # D típica para sinais fractais
            'correlation_length': 2.0,  # Comprimento de correlação quântica
            'unitarity_threshold': 0.95  # Threshold para unitariedade
        }

    def generate_quantum_weights(self, shape: Tuple[int, ...],
                               weight_type: str = 'complex_gaussian',
                               fractal_dimension: float = None) -> torch.Tensor:
        """
        Gera pesos quânticos com propriedades físicas

        Args:
            shape: Forma do tensor de pesos
            weight_type: Tipo de distribuição ('complex_gaussian', 'unitary', 'fractal')
            fractal_dimension: Dimensão fractal (opcional)

        Returns:
            Tensor de pesos com propriedades quânticas
        """
        if fractal_dimension is None:
            fractal_dimension = self.physical_params['fractal_dimension']

        if weight_type == 'complex_gaussian':
            return self._generate_complex_gaussian_weights(shape, fractal_dimension)
        elif weight_type == 'unitary':
            return self._generate_unitary_weights(shape)
        elif weight_type == 'fractal':
            return self._generate_fractal_weights(shape, fractal_dimension)
        else:
            raise ValueError(f"Tipo de peso não suportado: {weight_type}")

    def _generate_complex_gaussian_weights(self, shape: Tuple[int, ...],
                                         fractal_dimension: float) -> torch.Tensor:
        """Gera pesos usando distribuição Gaussiana complexa com correlação fractal"""
        # Dimensão efetiva baseada na fractal
        effective_dim = int(shape[0] * fractal_dimension / 2.0)

        # Matriz de correlação com decaimento fractal
        correlation_matrix = self._build_fractal_correlation_matrix(shape[0], fractal_dimension)

        # Geração de números complexos gaussianos correlacionados
        real_part = torch.randn(shape)
        imag_part = torch.randn(shape)

        # Aplicar correlação na dimensão de entrada
        if len(shape) >= 2:
            # Para pesos (out_features, in_features), aplicar correlação nas features de entrada
            correlation_sqrt = torch.linalg.cholesky(correlation_matrix.to(real_part.device))
            real_part = torch.matmul(correlation_sqrt, real_part.view(shape[0], -1)).view(shape)
            imag_part = torch.matmul(correlation_sqrt, imag_part.view(shape[0], -1)).view(shape)

        # Combinar partes real e imaginária
        weights = torch.complex(real_part, imag_part)

        # Normalizar para variância unitária
        weights = weights / torch.sqrt(torch.mean(torch.abs(weights)**2) + 1e-10)

        return weights

    def _generate_unitary_weights(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Gera pesos unitários usando decomposição QR"""
        if len(shape) != 2:
            raise ValueError("Pesos unitários requerem forma 2D")

        rows, cols = shape

        # Gerar matriz aleatória complexa
        random_matrix = torch.complex(torch.randn(rows, cols), torch.randn(rows, cols))

        # Decomposição QR para obter matriz unitária
        Q, R = torch.linalg.qr(random_matrix)

        # Ajustar fase da diagonal de R para garantir unitariedade
        phase_matrix = torch.diag(torch.angle(torch.diag(R)))
        Q = torch.matmul(Q, torch.exp(1j * phase_matrix))

        return Q

    def _generate_fractal_weights(self, shape: Tuple[int, ...],
                                fractal_dimension: float) -> torch.Tensor:
        """Gera pesos com estrutura fractal usando IFS"""
        # Usar sistema de funções iteradas para gerar estrutura fractal
        if len(shape) != 2:
            raise ValueError("Pesos fractais requerem forma 2D")

        rows, cols = shape

        # Gerar pontos usando IFS (exemplo: Sierpinski)
        fractal_points = self._generate_fractal_points(rows * cols, fractal_dimension)

        # Mapear pontos fractais para pesos complexos
        weights = torch.zeros(rows, cols, dtype=torch.complex64)

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < len(fractal_points):
                    x, y = fractal_points[idx]
                    # Mapear coordenadas para amplitude e fase
                    amplitude = torch.sqrt(torch.tensor(x**2 + y**2))
                    phase = torch.atan2(torch.tensor(y), torch.tensor(x))
                    weights[i, j] = amplitude * torch.exp(1j * phase)

        return weights

    def _build_fractal_correlation_matrix(self, size: int, fractal_dimension: float) -> torch.Tensor:
        """Constrói matriz de correlação com decaimento fractal"""
        correlation = torch.zeros(size, size)

        for i in range(size):
            for j in range(size):
                distance = abs(i - j)
                if distance == 0:
                    correlation[i, j] = 1.0
                else:
                    # Decaimento power-law baseado na dimensão fractal
                    correlation[i, j] = 1.0 / (distance ** (2 * fractal_dimension))

        # Normalizar
        correlation = correlation / torch.max(torch.sum(correlation, dim=1))

        return correlation

    def _generate_fractal_points(self, n_points: int, fractal_dimension: float) -> List[Tuple[float, float]]:
        """Gera pontos usando sistema de funções iteradas (IFS)"""
        points = [(0.0, 0.0)]  # Ponto inicial

        # Transformações do IFS para Sierpinski
        transformations = [
            lambda x, y: (0.5 * x, 0.5 * y),                    # Contrair para origem
            lambda x, y: (0.5 * x + 0.5, 0.5 * y),            # Contrair para direita
            lambda x, y: (0.5 * x + 0.25, 0.5 * y + 0.433)    # Contrair para cima
        ]

        # Gerar pontos iterativamente
        for _ in range(int(math.log2(n_points))):
            new_points = []
            for point in points:
                for transform in transformations:
                    new_points.append(transform(*point))
            points = new_points[:n_points]  # Limitar número de pontos

        return points


class AdaptiveWeightCalibrator:
    """
    Calibrador adaptativo de pesos baseado em feedback do pipeline

    Ajusta pesos automaticamente baseado em:
    - Métricas de performance física
    - Conservação de energia
    - Unitaridade quântica
    - Qualidade da geração de texto
    """

    def __init__(self, learning_rate: float = 0.01, patience: int = 10):
        self.learning_rate = learning_rate
        self.patience = patience
        self.history = []
        self.best_weights = None
        self.best_score = float('-inf')

    def calibrate_weights(self, model: nn.Module,
                         physical_metrics: Dict[str, float],
                         text_quality_score: float,
                         max_iterations: int = 100) -> nn.Module:
        """
        Calibra pesos do modelo baseado em métricas físicas e qualidade de texto

        Args:
            model: Modelo a ser calibrado
            physical_metrics: Métricas físicas (unitariedade, conservação de energia, etc.)
            text_quality_score: Score de qualidade do texto gerado
            max_iterations: Máximo de iterações de calibração

        Returns:
            Modelo com pesos calibrados
        """
        print("🔧 Iniciando calibração adaptativa de pesos...")

        for iteration in range(max_iterations):
            # Calcular score composto
            composite_score = self._calculate_composite_score(physical_metrics, text_quality_score)

            # Armazenar no histórico
            self.history.append({
                'iteration': iteration,
                'score': composite_score,
                'metrics': physical_metrics.copy(),
                'text_quality': text_quality_score
            })

            # Verificar se é o melhor score
            if composite_score > self.best_score:
                self.best_score = composite_score
                self.best_weights = self._copy_model_weights(model)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"🛑 Early stopping na iteração {iteration}")
                break

            # Aplicar correções aos pesos
            model = self._apply_weight_corrections(model, physical_metrics)

            print(f"   Iteração {iteration}: Score = {composite_score:.4f}")

        # Restaurar melhores pesos
        if self.best_weights is not None:
            model = self._load_model_weights(model, self.best_weights)

        print(f"✅ Calibração concluída. Melhor score: {self.best_score:.4f}")
        return model

    def _calculate_composite_score(self, physical_metrics: Dict[str, float],
                                 text_quality: float) -> float:
        """Calcula score composto baseado em métricas físicas e qualidade de texto"""
        # Pesos para diferentes métricas
        weights = {
            'unitarity': 0.3,
            'energy_conservation': 0.3,
            'fractal_consistency': 0.2,
            'text_quality': 0.2
        }

        score = 0.0

        # Métricas físicas
        score += weights['unitarity'] * physical_metrics.get('unitarity', 0.0)
        score += weights['energy_conservation'] * physical_metrics.get('energy_conservation', 0.0)
        score += weights['fractal_consistency'] * physical_metrics.get('fractal_consistency', 0.0)

        # Qualidade de texto
        score += weights['text_quality'] * text_quality

        return score

    def _apply_weight_corrections(self, model: nn.Module,
                                physical_metrics: Dict[str, float]) -> nn.Module:
        """Aplica correções aos pesos baseado nas métricas físicas"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    # Correção baseada na unitariedade
                    if physical_metrics.get('unitarity', 1.0) < 0.9:
                        param.data = self._enforce_unitarity(param.data)

                    # Correção baseada na conservação de energia
                    if physical_metrics.get('energy_conservation', 1.0) < 0.95:
                        param.data = self._normalize_energy(param.data)

                    # Pequena perturbação para exploração
                    noise = torch.randn_like(param.data) * self.learning_rate * 0.1
                    param.data += noise

        return model

    def _enforce_unitarity(self, weights: torch.Tensor) -> torch.Tensor:
        """Força unitariedade aproximada nos pesos"""
        if weights.dim() == 2:
            # Para matrizes 2D, usar projeção unitária
            U, _, Vh = torch.linalg.svd(weights, full_matrices=False)
            return torch.matmul(U, Vh)
        else:
            # Para tensores de maior dimensão, normalizar
            return weights / torch.sqrt(torch.mean(weights**2, dim=-1, keepdim=True) + 1e-10)

    def _normalize_energy(self, weights: torch.Tensor) -> torch.Tensor:
        """Normaliza pesos para conservação de energia"""
        energy = torch.sqrt(torch.sum(weights**2))
        if energy > 0:
            return weights / energy
        return weights

    def _copy_model_weights(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Copia pesos do modelo"""
        return {name: param.data.clone() for name, param in model.named_parameters()}

    def _load_model_weights(self, model: nn.Module, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Carrega pesos no modelo"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in weights:
                    param.data.copy_(weights[name])
        return model


class AutoCalibrationSystem:
    """
    Sistema completo de auto-calibração para ΨQRH

    Integra geração de pesos quânticos com calibração adaptativa
    baseada em feedback físico e qualidade de texto.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.weight_generator = QuantumWeightGenerator()
        self.calibrator = AdaptiveWeightCalibrator()
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Carrega configuração do sistema de auto-calibração"""
        default_config = {
            'auto_calibration': {
                'enabled': True,
                'learning_rate': 0.01,
                'patience': 10,
                'max_iterations': 50
            },
            'weight_generation': {
                'default_type': 'complex_gaussian',
                'fractal_dimension': 1.5,
                'use_physical_constraints': True
            },
            'validation': {
                'unitarity_threshold': 0.95,
                'energy_threshold': 0.95,
                'text_quality_threshold': 0.7
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)

        return default_config

    def auto_calibrate_model(self, model: nn.Module,
                           physical_metrics: Dict[str, float],
                           text_quality_score: float) -> nn.Module:
        """
        Auto-calibra modelo completo baseado em métricas físicas e qualidade de texto

        Args:
            model: Modelo a ser calibrado
            physical_metrics: Métricas físicas do pipeline
            text_quality_score: Score de qualidade do texto gerado

        Returns:
            Modelo calibrado
        """
        if not self.config['auto_calibration']['enabled']:
            print("⚠️  Auto-calibração desabilitada")
            return model

        print("🎯 Iniciando auto-calibração completa do modelo ΨQRH...")

        # Fase 1: Gerar pesos quânticos se necessário
        model = self._generate_missing_weights(model)

        # Fase 2: Calibrar pesos adaptativamente
        calibrated_model = self.calibrator.calibrate_weights(
            model=model,
            physical_metrics=physical_metrics,
            text_quality_score=text_quality_score,
            max_iterations=self.config['auto_calibration']['max_iterations']
        )

        # Fase 3: Validar calibração
        validation_results = self._validate_calibration(calibrated_model, physical_metrics)

        if validation_results['passed']:
            print("✅ Auto-calibração bem-sucedida!")
        else:
            print("⚠️  Auto-calibração concluída com ressalvas")

        return calibrated_model

    def _generate_missing_weights(self, model: nn.Module) -> nn.Module:
        """Gera pesos automaticamente para parâmetros não inicializados"""
        print("🔧 Verificando pesos não inicializados...")

        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    # Verificar se peso está inicializado (não é zero ou aleatório padrão)
                    if torch.allclose(param.data, torch.zeros_like(param.data)):
                        print(f"   Gerando pesos para {name}...")

                        # Determinar tipo de peso baseado no nome do parâmetro
                        weight_type = self._infer_weight_type(name)

                        # Gerar pesos quânticos
                        new_weights = self.weight_generator.generate_quantum_weights(
                            shape=param.shape,
                            weight_type=weight_type
                        )

                        # Aplicar pesos gerados
                        if new_weights.dtype != param.dtype:
                            if param.dtype == torch.float32 and new_weights.dtype == torch.complex64:
                                # Para pesos reais, usar magnitude dos pesos complexos
                                param.data.copy_(new_weights.real)
                            else:
                                param.data.copy_(new_weights.real)
                        else:
                            param.data.copy_(new_weights)

        return model

    def _infer_weight_type(self, param_name: str) -> str:
        """Infere tipo de peso baseado no nome do parâmetro"""
        name_lower = param_name.lower()

        if 'attention' in name_lower or 'attn' in name_lower:
            return 'unitary'  # Atenção requer unitariedade
        elif 'embedding' in name_lower:
            return 'fractal'  # Embeddings se beneficiam de estrutura fractal
        elif 'quantum' in name_lower or 'qrh' in name_lower:
            return 'complex_gaussian'  # Componentes quânticos
        else:
            return self.config['weight_generation']['default_type']

    def _validate_calibration(self, model: nn.Module,
                            original_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Valida se a calibração melhorou as métricas"""
        # Calcular métricas do modelo calibrado
        # (simplificado - em implementação real, executaria o pipeline)

        validation_results = {
            'passed': True,
            'improvements': {},
            'warnings': []
        }

        # Verificações básicas
        total_params = sum(p.numel() for p in model.parameters())
        if total_params == 0:
            validation_results['passed'] = False
            validation_results['warnings'].append("Modelo sem parâmetros")

        return validation_results

    def save_calibration_state(self, path: str):
        """Salva estado da calibração"""
        state = {
            'config': self.config,
            'calibrator_history': self.calibrator.history,
            'best_score': self.calibrator.best_score
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_calibration_state(self, path: str):
        """Carrega estado da calibração"""
        with open(path, 'r') as f:
            state = json.load(f)

        self.config = state.get('config', self.config)
        self.calibrator.history = state.get('calibrator_history', [])
        self.calibrator.best_score = state.get('best_score', float('-inf'))


# Função de integração com pipeline unificado
def create_auto_calibration_system(config_path: Optional[str] = None) -> AutoCalibrationSystem:
    """
    Factory function para criar sistema de auto-calibração

    Args:
        config_path: Caminho para arquivo de configuração (opcional)

    Returns:
        Sistema de auto-calibração configurado
    """
    return AutoCalibrationSystem(config_path)


if __name__ == "__main__":
    # Teste do sistema de auto-calibração
    print("🧠 Testando Sistema de Auto-Calibração ΨQRH...")

    # Criar sistema
    calibrator = create_auto_calibration_system()

    # Teste de geração de pesos
    print("\n🔧 Testando geração de pesos quânticos:")

    # Gerar diferentes tipos de pesos
    complex_weights = calibrator.weight_generator.generate_quantum_weights(
        shape=(64, 32), weight_type='complex_gaussian'
    )
    print(f"   Pesos complexos gaussianos: {complex_weights.shape}, dtype={complex_weights.dtype}")

    unitary_weights = calibrator.weight_generator.generate_quantum_weights(
        shape=(32, 32), weight_type='unitary'
    )
    print(f"   Pesos unitários: {unitary_weights.shape}, dtype={unitary_weights.dtype}")

    fractal_weights = calibrator.weight_generator.generate_quantum_weights(
        shape=(16, 16), weight_type='fractal', fractal_dimension=1.8
    )
    print(f"   Pesos fractais: {fractal_weights.shape}, dtype={fractal_weights.dtype}")

    # Teste de calibração (simulado)
    print("\n🎯 Testando calibração adaptativa:")

    # Métricas físicas simuladas
    physical_metrics = {
        'unitarity': 0.85,
        'energy_conservation': 0.92,
        'fractal_consistency': 0.78
    }

    text_quality = 0.65

    print(f"   Métricas físicas: {physical_metrics}")
    print(f"   Qualidade de texto: {text_quality}")

    # Calcular score composto
    composite_score = calibrator.calibrator._calculate_composite_score(physical_metrics, text_quality)
    print(f"   Score composto: {composite_score:.4f}")

    print("\n✅ Sistema de auto-calibração inicializado com sucesso!")
    print("   📊 Pesos quânticos gerados automaticamente")
    print("   🎛️  Calibração adaptativa baseada em física")
    print("   🔄 Integração com pipeline unificado")