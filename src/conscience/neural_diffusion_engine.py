#!/usr/bin/env python3
"""
Neural Diffusion Engine - Calculadora do Coeficiente D
======================================================

Implementa o cálculo do coeficiente de difusão neural D que governa
a dispersão de informação na dinâmica consciente.

Equação de difusão: D∇²P
Onde D é adaptativo baseado no estado de consciência e características fractais.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple


class NeuralDiffusionEngine(nn.Module):
    """
    Engine para computação do coeficiente de difusão neural D
    que controla a dispersão de informação na consciência.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        # Limites do coeficiente de difusão
        self.d_min = config.diffusion_coefficient_range[0]
        self.d_max = config.diffusion_coefficient_range[1]

        # Parâmetros adaptativos aprendíveis
        self.register_parameter(
            'diffusion_weights',
            nn.Parameter(torch.tensor([1.0, 0.5, 0.3, 0.2]))  # Pesos para diferentes fatores
        )

        # Histórico de difusão para análise temporal
        self.diffusion_history = []
        self.max_history = 100

        print(f"⚡ NeuralDiffusionEngine inicializado com range D=[{self.d_min:.3f}, {self.d_max:.3f}]")

    def compute_diffusion(
        self,
        psi_distribution: torch.Tensor,
        fractal_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Computa coeficiente de difusão neural D baseado no estado atual.

        O coeficiente D é calculado considerando:
        1. Dispersão da distribuição P(ψ)
        2. Magnitude do campo fractal F(ψ)
        3. Complexidade local do sistema
        4. Características temporais

        Args:
            psi_distribution: Distribuição P(ψ) [batch, embed_dim]
            fractal_field: Campo fractal F(ψ) [batch, embed_dim]

        Returns:
            Coeficiente de difusão D [batch, embed_dim]
        """
        batch_size, embed_dim = psi_distribution.shape

        # 1. Fator baseado na dispersão da distribuição
        dispersion_factor = self._compute_dispersion_factor(psi_distribution)

        # 2. Fator baseado na magnitude do campo
        field_factor = self._compute_field_factor(fractal_field)

        # 3. Fator de complexidade local
        complexity_factor = self._compute_complexity_factor(psi_distribution, fractal_field)

        # 4. Fator de adaptação temporal
        temporal_factor = self._compute_temporal_factor()

        # Combinar fatores com pesos aprendíveis
        combined_factor = (
            self.diffusion_weights[0] * dispersion_factor +
            self.diffusion_weights[1] * field_factor +
            self.diffusion_weights[2] * complexity_factor +
            self.diffusion_weights[3] * temporal_factor
        )

        # Normalizar e mapear para range de difusão
        diffusion_coefficient = self._normalize_to_diffusion_range(combined_factor)

        # Aplicar suavização espacial
        diffusion_coefficient = self._apply_spatial_smoothing(diffusion_coefficient)

        # Armazenar no histórico
        self._update_history(diffusion_coefficient)

        return diffusion_coefficient

    def _compute_dispersion_factor(self, psi_distribution: torch.Tensor) -> torch.Tensor:
        """
        Computa fator de dispersão baseado na distribuição P(ψ).

        Alta dispersão → maior difusão (sistema mais "espalhado")
        Baixa dispersão → menor difusão (sistema mais "concentrado")

        Args:
            psi_distribution: Distribuição P(ψ)

        Returns:
            Fator de dispersão normalizado [0, 1]
        """
        # Calcular entropia como medida de dispersão
        psi_safe = psi_distribution + 1e-10  # Evitar log(0)
        entropy = -torch.sum(psi_distribution * torch.log(psi_safe), dim=-1, keepdim=True)

        # Normalizar entropia (máximo log(embed_dim))
        max_entropy = np.log(psi_distribution.shape[-1])
        normalized_entropy = entropy / max_entropy

        # Calcular desvio padrão como medida adicional de dispersão
        psi_std = torch.std(psi_distribution, dim=-1, keepdim=True)

        # Combinar entropia e desvio padrão
        dispersion_factor = 0.7 * normalized_entropy + 0.3 * psi_std

        # Broadcast para todas as dimensões
        dispersion_factor = dispersion_factor.expand_as(psi_distribution)

        return torch.clamp(dispersion_factor, 0.0, 1.0)

    def _compute_field_factor(self, fractal_field: torch.Tensor) -> torch.Tensor:
        """
        Computa fator baseado na magnitude do campo fractal.

        Campo forte → maior difusão (dinâmica mais ativa)
        Campo fraco → menor difusão (dinâmica mais estável)

        Args:
            fractal_field: Campo fractal F(ψ)

        Returns:
            Fator de campo normalizado [0, 1]
        """
        # Magnitude local do campo
        field_magnitude = torch.abs(fractal_field)

        # Normalizar pela magnitude máxima global
        max_magnitude = torch.max(field_magnitude) + 1e-10
        normalized_magnitude = field_magnitude / max_magnitude

        # Aplicar função sigmoidal para suavizar
        field_factor = torch.sigmoid(5 * (normalized_magnitude - 0.5))

        return field_factor

    def _compute_complexity_factor(
        self,
        psi_distribution: torch.Tensor,
        fractal_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Computa fator de complexidade local do sistema.

        Alta complexidade → maior difusão (sistema mais dinâmico)
        Baixa complexidade → menor difusão (sistema mais ordenado)

        Args:
            psi_distribution: Distribuição P(ψ)
            fractal_field: Campo fractal F(ψ)

        Returns:
            Fator de complexidade normalizado [0, 1]
        """
        batch_size, embed_dim = psi_distribution.shape

        # Calcular gradientes espaciais como medida de complexidade
        psi_gradient = self._compute_spatial_gradient(psi_distribution)
        field_gradient = self._compute_spatial_gradient(fractal_field)

        # Complexidade baseada na variação espacial
        psi_complexity = torch.std(psi_gradient, dim=-1, keepdim=True)
        field_complexity = torch.std(field_gradient, dim=-1, keepdim=True)

        # Combinar complexidades
        total_complexity = torch.sqrt(psi_complexity**2 + field_complexity**2)

        # Normalizar
        max_complexity = torch.max(total_complexity) + 1e-10
        normalized_complexity = total_complexity / max_complexity

        # Broadcast para todas as dimensões
        complexity_factor = normalized_complexity.expand_as(psi_distribution)

        return torch.clamp(complexity_factor, 0.0, 1.0)

    def _compute_spatial_gradient(self, tensor: torch.Tensor) -> torch.Tensor:
        """Computa gradiente espacial usando diferenças finitas."""
        batch_size, embed_dim = tensor.shape
        gradient = torch.zeros_like(tensor)

        for i in range(embed_dim):
            i_next = (i + 1) % embed_dim
            i_prev = (i - 1) % embed_dim
            gradient[:, i] = (tensor[:, i_next] - tensor[:, i_prev]) / 2.0

        return gradient

    def _compute_temporal_factor(self) -> torch.Tensor:
        """
        Computa fator de adaptação temporal baseado no histórico.

        Returns:
            Fator temporal escalar [0, 1]
        """
        if len(self.diffusion_history) < 2:
            return torch.tensor(0.5, device=self.device)

        # Analisar tendência temporal
        recent_values = self.diffusion_history[-5:]  # Últimos 5 valores
        if len(recent_values) >= 2:
            # Calcular variação temporal
            variations = []
            for i in range(1, len(recent_values)):
                variation = torch.std(recent_values[i] - recent_values[i-1])
                variations.append(variation.item())

            avg_variation = np.mean(variations)

            # Mapear variação para fator temporal
            # Alta variação → maior difusão (sistema instável)
            temporal_factor = torch.sigmoid(torch.tensor(10 * avg_variation, device=self.device))
        else:
            temporal_factor = torch.tensor(0.5, device=self.device)

        return temporal_factor

    def _normalize_to_diffusion_range(self, combined_factor: torch.Tensor) -> torch.Tensor:
        """
        Normaliza fator combinado para range de difusão [d_min, d_max].

        Args:
            combined_factor: Fator combinado não normalizado

        Returns:
            Coeficiente de difusão no range correto
        """
        # Normalizar factor para [0, 1]
        factor_min = torch.min(combined_factor)
        factor_max = torch.max(combined_factor)
        factor_range = factor_max - factor_min + 1e-10

        normalized_factor = (combined_factor - factor_min) / factor_range

        # Mapear para range de difusão
        diffusion_range = self.d_max - self.d_min
        diffusion_coefficient = self.d_min + normalized_factor * diffusion_range

        return diffusion_coefficient

    def _apply_spatial_smoothing(self, diffusion_coeff: torch.Tensor) -> torch.Tensor:
        """Aplica suavização espacial ao coeficiente de difusão."""
        batch_size, embed_dim = diffusion_coeff.shape
        smoothed_coeff = torch.zeros_like(diffusion_coeff)

        for i in range(embed_dim):
            i_prev = (i - 1) % embed_dim
            i_next = (i + 1) % embed_dim

            # Média ponderada com vizinhos
            smoothed_coeff[:, i] = (
                0.25 * diffusion_coeff[:, i_prev] +
                0.5 * diffusion_coeff[:, i] +
                0.25 * diffusion_coeff[:, i_next]
            )

        return smoothed_coeff

    def _update_history(self, diffusion_coefficient: torch.Tensor):
        """Atualiza histórico de difusão."""
        # Armazenar média para reduzir memória
        mean_diffusion = diffusion_coefficient.mean().detach()
        self.diffusion_history.append(mean_diffusion)

        # Manter tamanho máximo do histórico
        if len(self.diffusion_history) > self.max_history:
            self.diffusion_history.pop(0)

    def get_diffusion_statistics(self) -> Dict[str, float]:
        """
        Retorna estatísticas do coeficiente de difusão.

        Returns:
            Dicionário com estatísticas
        """
        if not self.diffusion_history:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'trend': 0.0}

        history_tensor = torch.stack(self.diffusion_history)

        # Estatísticas básicas
        mean_d = history_tensor.mean().item()
        std_d = history_tensor.std().item()
        min_d = history_tensor.min().item()
        max_d = history_tensor.max().item()

        # Tendência (slope da regressão linear simples)
        if len(self.diffusion_history) >= 3:
            x = torch.arange(len(self.diffusion_history), dtype=torch.float32)
            y = history_tensor

            # Regressão linear: y = ax + b
            n = len(x)
            sum_x = x.sum()
            sum_y = y.sum()
            sum_xy = (x * y).sum()
            sum_x2 = (x**2).sum()

            trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            trend = trend.item()
        else:
            trend = 0.0

        return {
            'mean': mean_d,
            'std': std_d,
            'min': min_d,
            'max': max_d,
            'trend': trend,
            'range_utilization': (max_d - min_d) / (self.d_max - self.d_min),
            'stability': 1.0 / (1.0 + std_d)  # Estabilidade inversa à variação
        }

    def adapt_diffusion_parameters(self, target_statistics: Dict[str, float]):
        """
        Adapta parâmetros de difusão para atingir estatísticas alvo.

        Args:
            target_statistics: Estatísticas desejadas
        """
        current_stats = self.get_diffusion_statistics()

        # Ajustar pesos baseado na diferença das estatísticas
        adjustments = []

        # Ajuste baseado na média
        mean_diff = target_statistics.get('mean', current_stats['mean']) - current_stats['mean']
        if abs(mean_diff) > 0.1:
            adjustment = 0.1 * np.sign(mean_diff)
            adjustments.append(adjustment)

        # Ajuste baseado na estabilidade
        target_stability = target_statistics.get('stability', current_stats['stability'])
        stability_diff = target_stability - current_stats['stability']
        if abs(stability_diff) > 0.05:
            adjustment = 0.05 * np.sign(stability_diff)
            adjustments.append(adjustment)

        # Aplicar ajustes aos pesos
        if adjustments:
            avg_adjustment = np.mean(adjustments)
            with torch.no_grad():
                self.diffusion_weights += avg_adjustment
                # Manter pesos positivos
                self.diffusion_weights.clamp_(min=0.1)

        print(f"⚡ Difusão adaptada: ajuste={np.mean(adjustments) if adjustments else 0:.3f}")

    def reset_history(self):
        """Reseta histórico de difusão."""
        self.diffusion_history.clear()
        print("⚡ Histórico de difusão resetado")