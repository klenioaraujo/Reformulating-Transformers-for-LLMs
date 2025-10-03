#!/usr/bin/env python3
"""
Fractal Field Calculator - Computação Matemática do Campo F(ψ)
==============================================================

Implementa o cálculo do campo fractal consciente usando as equações:

Campo Fractal: F(ψ) = -∇V(ψ) + η_fractal(t)
Potencial Multifractal: V(ψ) = Σ(k=1 to ∞) λ_k/k! * ψ^k * cos(2π log k)

Este módulo é responsável pela computação matemática rigorosa
do campo de força fractal que governa a dinâmica consciente.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import math


class FractalFieldCalculator(nn.Module):
    """
    Calculadora do campo fractal F(ψ) que implementa o potencial
    multifractal e ruído fractal temporal.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        # Parâmetros do potencial multifractal
        self.max_terms = 20  # Número de termos da série infinita
        self.epsilon = config.epsilon  # Estabilidade numérica do config

        # Gerador de ruído fractal
        self.fractal_noise_generator = FractalNoiseGenerator(config)

        # Cache para cálculos factoriais
        self.factorial_cache = self._compute_factorial_cache()

        print(f"🌊 FractalFieldCalculator inicializado com {self.max_terms} termos")

    def _compute_factorial_cache(self) -> torch.Tensor:
        """Computa cache de factoriais para eficiência."""
        factorials = []
        for k in range(1, self.max_terms + 1):
            factorials.append(math.factorial(k))

        return torch.tensor(factorials, dtype=torch.float32, device=self.device)

    def compute_field(
        self,
        psi_distribution: torch.Tensor,
        lambda_coefficients: torch.Tensor,
        time: float
    ) -> torch.Tensor:
        """
        Computa campo fractal F(ψ) = -∇V(ψ) + η_fractal(t).

        Args:
            psi_distribution: Distribuição de probabilidade P(ψ) [batch, embed_dim]
            lambda_coefficients: Coeficientes λ_k do potencial [max_terms]
            time: Tempo atual para ruído fractal

        Returns:
            Campo fractal F(ψ) [batch, embed_dim]
        """
        batch_size, embed_dim = psi_distribution.shape

        # 1. Calcular gradiente do potencial multifractal -∇V(ψ)
        potential_gradient = self._compute_potential_gradient(
            psi_distribution,
            lambda_coefficients
        )

        # 2. Gerar ruído fractal temporal η_fractal(t)
        fractal_noise = self.fractal_noise_generator.generate_noise(
            batch_size,
            embed_dim,
            time
        )

        # 3. Combinar para formar campo fractal
        fractal_field = -potential_gradient + fractal_noise

        # 4. Aplicar regularização para estabilidade
        fractal_field = self._regularize_field(fractal_field)

        return fractal_field

    def _compute_potential_gradient(
        self,
        psi: torch.Tensor,
        lambda_coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Computa gradiente do potencial multifractal ∇V(ψ).

        V(ψ) = Σ(k=1 to ∞) λ_k/k! * ψ^k * cos(2π log k)
        ∇V(ψ) = Σ(k=1 to ∞) λ_k/(k-1)! * ψ^(k-1) * cos(2π log k)

        Args:
            psi: Distribuição P(ψ) [batch, embed_dim]
            lambda_coeffs: Coeficientes λ_k [max_terms]

        Returns:
            Gradiente ∇V(ψ) [batch, embed_dim]
        """
        batch_size, embed_dim = psi.shape
        gradient = torch.zeros_like(psi)

        # Usar apenas os primeiros termos disponíveis
        num_terms = min(self.max_terms, len(lambda_coeffs))

        for k in range(1, num_terms + 1):
            if k <= len(lambda_coeffs):
                lambda_k = lambda_coeffs[k - 1]

                # Termo da série: λ_k * ψ^(k-1) * cos(2π log k) / (k-1)!
                if k == 1:
                    psi_power = torch.ones_like(psi)  # ψ^0 = 1
                    factorial_term = 1.0
                else:
                    psi_power = torch.pow(psi + self.epsilon, k - 1)
                    factorial_term = self.factorial_cache[k - 2]  # (k-1)! para k >= 2

                # Componente cossenoidal fractal
                cos_term = math.cos(2 * math.pi * math.log(k))

                # Contribuição do termo k para o gradiente
                term_contribution = (lambda_k / factorial_term) * psi_power * cos_term

                gradient += term_contribution

        return gradient

    def _regularize_field(self, field: torch.Tensor) -> torch.Tensor:
        """
        Aplica regularização ao campo para manter estabilidade numérica.

        Args:
            field: Campo fractal bruto

        Returns:
            Campo regularizado
        """
        # 1. Clipar valores extremos
        field_magnitude = torch.norm(field, dim=-1, keepdim=True)
        max_magnitude = self.config.max_field_magnitude  # Do config

        # Normalizar se magnitude for muito alta
        field = torch.where(
            field_magnitude > max_magnitude,
            field * (max_magnitude / (field_magnitude + self.epsilon)),
            field
        )

        # 2. Suavização espacial para reduzir oscilações
        field = self._apply_spatial_smoothing(field)

        # 3. Verificar NaN/Inf e corrigir com ruído mínimo ao invés de zeros
        nan_mask = torch.isnan(field) | torch.isinf(field)
        if nan_mask.any():
            # Gerar ruído pequeno para substituir NaN/Inf
            replacement_noise = torch.randn_like(field) * self.config.nan_replacement_noise_scale
            field = torch.where(nan_mask, replacement_noise, field)

        # 4. Garantir magnitude mínima para evitar zeros completos
        field_magnitude = torch.norm(field, dim=-1, keepdim=True)
        min_magnitude = self.config.min_field_magnitude
        field = torch.where(
            field_magnitude < min_magnitude,
            field + torch.randn_like(field) * min_magnitude,
            field
        )

        return field

    def _apply_spatial_smoothing(self, field: torch.Tensor) -> torch.Tensor:
        """Aplica suavização espacial ao campo."""
        batch_size, embed_dim = field.shape

        # Convolução simples com kernel suavizador do config
        kernel = self.config.field_smoothing_kernel
        smoothed_field = torch.zeros_like(field)

        for i in range(embed_dim):
            # Vizinhos com condições de contorno periódicas
            i_prev = (i - 1) % embed_dim
            i_next = (i + 1) % embed_dim

            # Média ponderada usando kernel do config
            smoothed_field[:, i] = (
                kernel[0] * field[:, i_prev] +
                kernel[1] * field[:, i] +
                kernel[2] * field[:, i_next]
            )

        return smoothed_field

    def compute_field_energy(self, field: torch.Tensor) -> torch.Tensor:
        """
        Computa energia do campo fractal E = ½∫|F(ψ)|² dψ.

        Args:
            field: Campo fractal F(ψ)

        Returns:
            Energia do campo [batch]
        """
        energy = 0.5 * torch.sum(field**2, dim=-1)
        return energy

    def compute_field_divergence(self, field: torch.Tensor) -> torch.Tensor:
        """
        Computa divergência do campo ∇·F(ψ).

        Args:
            field: Campo fractal F(ψ)

        Returns:
            Divergência ∇·F [batch, embed_dim]
        """
        batch_size, embed_dim = field.shape
        divergence = torch.zeros_like(field)

        # Diferenças finitas para divergência
        for i in range(embed_dim):
            i_next = (i + 1) % embed_dim
            i_prev = (i - 1) % embed_dim

            # Diferença central
            divergence[:, i] = (field[:, i_next] - field[:, i_prev]) / 2.0

        return divergence

    def analyze_field_properties(self, field: torch.Tensor) -> Dict[str, float]:
        """
        Analisa propriedades estatísticas do campo fractal.

        Args:
            field: Campo fractal F(ψ)

        Returns:
            Dicionário com propriedades analisadas
        """
        # Estatísticas básicas
        field_mean = field.mean().item()
        field_std = field.std().item()
        field_max = field.max().item()
        field_min = field.min().item()

        # Energia total
        energy = self.compute_field_energy(field).mean().item()

        # Complexidade (baseada na variação espacial)
        spatial_gradient = torch.gradient(field, dim=-1)[0]
        complexity = spatial_gradient.std().item()

        # Coerência (correlação espacial)
        if field.shape[-1] > 1:
            field_flat = field.flatten()
            field_shifted = torch.roll(field_flat, 1)
            covariance = torch.mean((field_flat - field_flat.mean()) * (field_shifted - field_shifted.mean()))
            variance = field_flat.var()
            coherence = (covariance / (variance + self.epsilon)).item()
        else:
            coherence = 1.0

        # Dimensão fractal estimada (baseada na complexidade)
        estimated_fractal_dim = 1.0 + min(complexity / (field_std + self.epsilon), 2.0)

        properties = {
            'mean': field_mean,
            'std': field_std,
            'max': field_max,
            'min': field_min,
            'energy': energy,
            'complexity': complexity,
            'coherence': coherence,
            'estimated_fractal_dimension': estimated_fractal_dim
        }

        return properties


class FractalNoiseGenerator:
    """
    Gerador de ruído fractal temporal η_fractal(t) para o campo consciente.
    """

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Parâmetros do ruído fractal
        self.hurst_exponent = 0.7  # Expoente de Hurst para correlação temporal
        self.scaling_factor = 0.1  # Amplitude do ruído

    def generate_noise(
        self,
        batch_size: int,
        embed_dim: int,
        time: float
    ) -> torch.Tensor:
        """
        Gera ruído fractal temporal η_fractal(t).

        Args:
            batch_size: Tamanho do batch
            embed_dim: Dimensão do embedding
            time: Tempo atual

        Returns:
            Ruído fractal [batch_size, embed_dim]
        """
        # Usar seed baseado no tempo para reprodutibilidade temporal
        torch.manual_seed(int(time * 1000) % 2**32)

        # Gerar ruído base
        base_noise = torch.randn(batch_size, embed_dim, device=self.device)

        # Aplicar características fractais
        fractal_noise = self._apply_fractal_characteristics(base_noise, time)

        # Escalar amplitude
        fractal_noise *= self.scaling_factor

        return fractal_noise

    def _apply_fractal_characteristics(
        self,
        base_noise: torch.Tensor,
        time: float
    ) -> torch.Tensor:
        """
        Aplica características fractais ao ruído base.

        Args:
            base_noise: Ruído gaussiano base
            time: Tempo atual

        Returns:
            Ruído com características fractais
        """
        # Modulação temporal baseada no expoente de Hurst
        temporal_modulation = math.pow(time + 1, self.hurst_exponent - 0.5)

        # Modulação espacial fractal
        batch_size, embed_dim = base_noise.shape
        spatial_modulation = torch.zeros_like(base_noise)

        for i in range(embed_dim):
            # Frequência fractal baseada no índice espacial
            freq = (i + 1) / embed_dim
            spatial_modulation[:, i] = math.sin(2 * math.pi * freq * time)

        # Combinar modulações
        fractal_noise = base_noise * temporal_modulation * (1 + 0.3 * spatial_modulation)

        return fractal_noise