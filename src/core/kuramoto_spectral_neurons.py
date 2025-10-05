#!/usr/bin/env python3
"""
Kuramoto Spectral Neurons - Sistema de Osciladores Acoplados com Localização Espacial
=====================================================================================

Implementa o modelo de Kuramoto estendido com:
1. Localização espacial dos neurônios espectrais (grid 2D/3D)
2. Equações de reação-difusão para fases: ∂θ/∂t - Δθ = ω + g(θ-φ)
3. Sincronização de osciladores acoplados: dθ/dt = ω₁ + g(θ-φ)
4. Integração com estrutura quaterniônica do ΨQRH

Mathematical Framework:
-----------------------
Reaction-Diffusion Kuramoto:
    ∂θ/∂t - Δθ = ω + K·∑ sin(φⱼ - θᵢ)
    ∂φ/∂t - Δφ = ω - K·∑ sin(φⱼ - θᵢ)

Standard Kuramoto:
    dθᵢ/dt = ωᵢ + (K/N)·∑ⱼ sin(θⱼ - θᵢ)

Spatial Laplacian (2D):
    Δθ = ∂²θ/∂x² + ∂²θ/∂y²

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

# Imports do sistema ΨQRH
from .quaternion_operations import quaternion_multiply, quaternion_normalize


def load_kuramoto_config(config_path: Optional[str] = None) -> Dict:
    """
    Carrega configuração de Kuramoto do arquivo YAML.

    Args:
        config_path: Caminho para arquivo de configuração.
                     Se None, usa configs/kuramoto_config.yaml

    Returns:
        Dicionário com configurações
    """
    if config_path is None:
        # Caminho padrão relativo ao repositório
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / "configs" / "kuramoto_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config['kuramoto_spectral_layer']


class SpatialNeuronGrid(nn.Module):
    """
    Grid espacial de neurônios espectrais com localização física.
    Cada neurônio possui coordenadas (x, y, z) e fase θ.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Dimensões do grid
        self.H = config['spatial_grid']['height']
        self.W = config['spatial_grid']['width']
        self.D = config['spatial_grid']['depth']
        self.n_neurons = self.H * self.W * self.D

        self.device = config['performance']['device']
        self.topology = config['spatial_grid']['topology']

        # Coordenadas espaciais dos neurônios
        self.coordinates = self._initialize_coordinates()

        # Frequências naturais dos osciladores (fixas)
        self.natural_frequencies = self._initialize_natural_frequencies()

        # Estado das fases (variável)
        self.register_buffer('phases_theta', torch.zeros(1, self.n_neurons, device=self.device))
        self.register_buffer('phases_phi', torch.zeros(1, self.n_neurons, device=self.device))

        # Matriz de conectividade (baseada em distância espacial)
        self.connectivity_matrix = self._build_connectivity_matrix()

    def _initialize_coordinates(self) -> torch.Tensor:
        """Inicializa coordenadas espaciais dos neurônios"""
        if self.topology == 'grid_2d':
            # Grid 2D regular
            x = torch.arange(self.W, dtype=torch.float32, device=self.device)
            y = torch.arange(self.H, dtype=torch.float32, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='ij')

            coords = torch.stack([
                xx.flatten(),
                yy.flatten(),
                torch.zeros(self.n_neurons, device=self.device)
            ], dim=1)  # [N, 3]

        elif self.topology == 'grid_3d':
            # Grid 3D regular
            x = torch.arange(self.W, dtype=torch.float32, device=self.device)
            y = torch.arange(self.H, dtype=torch.float32, device=self.device)
            z = torch.arange(self.D, dtype=torch.float32, device=self.device)
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

            coords = torch.stack([
                xx.flatten(),
                yy.flatten(),
                zz.flatten()
            ], dim=1)  # [N, 3]

        elif self.topology == 'hexagonal':
            # Grid hexagonal
            coords = self._create_hexagonal_grid()

        else:  # random
            # Coordenadas aleatórias
            coords = torch.rand(self.n_neurons, 3, device=self.device)
            coords[:, 0] *= self.W
            coords[:, 1] *= self.H
            coords[:, 2] *= self.D

        return coords

    def _create_hexagonal_grid(self) -> torch.Tensor:
        """Cria grid hexagonal"""
        coords = []
        for i in range(self.H):
            for j in range(self.W):
                x = j + (0.5 if i % 2 == 1 else 0.0)
                y = i * np.sqrt(3) / 2
                coords.append([x, y, 0.0])

        coords = torch.tensor(coords[:self.n_neurons], dtype=torch.float32, device=self.device)
        return coords

    def _initialize_natural_frequencies(self) -> torch.Tensor:
        """Inicializa frequências naturais ω dos osciladores"""
        mean_freq = self.config['oscillator_dynamics']['natural_frequency_mean']
        std_freq = self.config['oscillator_dynamics']['natural_frequency_std']

        frequencies = torch.randn(self.n_neurons, device=self.device) * std_freq + mean_freq
        return frequencies

    def _build_connectivity_matrix(self) -> torch.Tensor:
        """Constrói matriz de conectividade baseada em distância espacial"""
        # Calcular distâncias euclidianas
        coords = self.coordinates.unsqueeze(0)  # [1, N, 3]
        coords_t = self.coordinates.unsqueeze(1)  # [N, 1, 3]

        distances = torch.sqrt(torch.sum((coords - coords_t) ** 2, dim=-1))  # [N, N]

        # Raio de influência
        sigma = self.config['connectivity'].get('sigma_connectivity', None)
        if sigma is None:
            sigma = max(self.W, self.H) / 4.0

        # Função de decaimento
        decay_fn = self.config['connectivity']['decay_function']
        if decay_fn == 'gaussian':
            connectivity = torch.exp(-distances ** 2 / (2 * sigma ** 2))
        elif decay_fn == 'exponential':
            connectivity = torch.exp(-distances / sigma)
        else:  # linear
            connectivity = torch.clamp(1.0 - distances / sigma, min=0.0)

        # Remover auto-conexões
        connectivity.fill_diagonal_(0.0)

        # Normalizar
        if self.config['connectivity']['normalize_connections']:
            row_sums = connectivity.sum(dim=1, keepdim=True)
            connectivity = connectivity / (row_sums + 1e-8)

        return connectivity


class KuramotoReactionDiffusion(nn.Module):
    """
    Sistema de Kuramoto com equações de reação-difusão espacial.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = config['performance']['device']

        # Grid espacial de neurônios
        self.neuron_grid = SpatialNeuronGrid(config)

        # Parâmetros
        self.K = config['oscillator_dynamics']['coupling_strength']
        self.D = config['reaction_diffusion']['diffusion_coefficient']
        self.dt = config['reaction_diffusion']['time_step']

        # Sincronização adaptativa
        self.adaptive_coupling = config['synchronization']['enable_adaptive_coupling']
        self.sync_threshold = config['synchronization']['threshold']

    def compute_laplacian(self, phases: torch.Tensor) -> torch.Tensor:
        """Computa Laplaciano espacial Δθ usando diferenças finitas"""
        batch_size = phases.shape[0]
        H, W, D = self.neuron_grid.H, self.neuron_grid.W, self.neuron_grid.D

        # Reshape para grid espacial
        if D == 1:  # 2D
            phases_grid = phases.view(batch_size, H, W)

            # Padding
            pad_mode = 'circular' if self.config['spatial_grid']['periodic_boundary'] else 'replicate'
            phases_padded = F.pad(phases_grid, (1, 1, 1, 1), mode=pad_mode)

            # Diferenças finitas de 2ª ordem
            laplacian = (
                phases_padded[:, 1:-1, 2:] +     # direita
                phases_padded[:, 1:-1, :-2] +    # esquerda
                phases_padded[:, 2:, 1:-1] +     # cima
                phases_padded[:, :-2, 1:-1] -    # baixo
                4 * phases_grid
            )

            laplacian = laplacian.view(batch_size, -1)

        else:  # 3D
            phases_grid = phases.view(batch_size, H, W, D)
            pad_mode = 'circular' if self.config['spatial_grid']['periodic_boundary'] else 'replicate'
            phases_padded = F.pad(phases_grid, (1, 1, 1, 1, 1, 1), mode=pad_mode)

            laplacian = (
                phases_padded[:, 1:-1, 1:-1, 2:] +
                phases_padded[:, 1:-1, 1:-1, :-2] +
                phases_padded[:, 1:-1, 2:, 1:-1] +
                phases_padded[:, 1:-1, :-2, 1:-1] +
                phases_padded[:, 2:, 1:-1, 1:-1] +
                phases_padded[:, :-2, 1:-1, 1:-1] -
                6 * phases_grid
            )

            laplacian = laplacian.view(batch_size, -1)

        return laplacian

    def compute_kuramoto_coupling(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computa termo de acoplamento de Kuramoto"""
        # Expandir para computação matricial
        theta_i = theta.unsqueeze(2)  # [B, N, 1]
        phi_j = phi.unsqueeze(1)      # [B, 1, N]

        # Diferença de fases
        phase_diff = phi_j - theta_i  # [B, N, N]
        sin_diff = torch.sin(phase_diff)  # [B, N, N]

        # Aplicar matriz de conectividade espacial
        connectivity = self.neuron_grid.connectivity_matrix.unsqueeze(0)
        weighted_sin = sin_diff * connectivity

        # Somar contribuições
        g_theta = self.K * weighted_sin.sum(dim=2)  # [B, N]
        g_phi = -g_theta

        return g_theta, g_phi

    def forward(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Integra as equações de reação-difusão de Kuramoto"""
        if num_steps is None:
            num_steps = self.config['reaction_diffusion']['num_integration_steps']

        theta_current = theta.clone()
        phi_current = phi.clone()

        sync_orders = []

        # Integração temporal
        for step in range(num_steps):
            # Laplacianos espaciais
            laplacian_theta = self.compute_laplacian(theta_current)
            laplacian_phi = self.compute_laplacian(phi_current)

            # Acoplamento de Kuramoto
            g_theta, g_phi = self.compute_kuramoto_coupling(theta_current, phi_current)

            # Frequências naturais
            omega = self.neuron_grid.natural_frequencies.unsqueeze(0)

            # Equações de reação-difusão
            dtheta_dt = self.D * laplacian_theta + omega + g_theta
            dphi_dt = self.D * laplacian_phi + omega + g_phi

            # Integração Euler
            theta_current = theta_current + self.dt * dtheta_dt
            phi_current = phi_current + self.dt * dphi_dt

            # Normalizar fases para [-π, π]
            theta_current = torch.atan2(torch.sin(theta_current), torch.cos(theta_current))
            phi_current = torch.atan2(torch.sin(phi_current), torch.cos(phi_current))

            # Ordem de sincronização
            r = self.compute_synchronization_order(theta_current)
            sync_orders.append(r.item())

            # Acoplamento adaptativo
            if self.adaptive_coupling:
                target = self.config['synchronization']['target_synchronization']
                rate = self.config['synchronization']['adaptive_coupling_rate']
                self.K = self.K + rate * (target - r.item())

        results = {
            'theta_final': theta_current,
            'phi_final': phi_current,
            'synchronization_order': torch.tensor(sync_orders),
            'is_synchronized': sync_orders[-1] > self.sync_threshold,
            'final_sync_order': sync_orders[-1]
        }

        return results

    def compute_synchronization_order(self, phases: torch.Tensor) -> torch.Tensor:
        """Calcula parâmetro de ordem de Kuramoto: r = |1/N ∑ⱼ e^(iθⱼ)|"""
        complex_phases = torch.exp(1j * phases)
        mean_complex = complex_phases.mean(dim=1)
        r = torch.abs(mean_complex)
        return r.mean()


class KuramotoSpectralLayer(nn.Module):
    """
    Camada neural baseada em Kuramoto para o transformer ΨQRH.
    Integra neurônios espectrais espacialmente localizados com dinâmica de sincronização.
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()

        # Carregar configuração
        self.config = load_kuramoto_config(config_path)
        self.device = self.config['performance']['device']

        # Sistema de Kuramoto com reação-difusão
        self.kuramoto_system = KuramotoReactionDiffusion(self.config)

        # Dimensões
        embed_dim = self.config['qrh_integration']['embed_dim']
        quat_mult = self.config['qrh_integration']['quaternion_multiplier']
        self.full_dim = embed_dim * quat_mult
        n_neurons = self.kuramoto_system.neuron_grid.n_neurons

        # Projeção de entrada: tokens → fases iniciais
        self.input_projection = nn.Linear(self.full_dim, n_neurons * 2)

        # Projeção de saída: fases sincronizadas → representação quaterniônica
        self.output_projection = nn.Linear(n_neurons * 2, self.full_dim)

        # Normalização
        if self.config['qrh_integration']['use_layer_norm']:
            self.layer_norm = nn.LayerNorm(self.full_dim)
        else:
            self.layer_norm = nn.Identity()

        # Peso residual
        self.residual_weight = self.config['qrh_integration']['residual_weight']

    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Processa input através de dinâmica de Kuramoto.

        Args:
            x: Input tensor [batch, seq_len, embed_dim * 4]
            return_metrics: Se True, retorna métricas de sincronização

        Returns:
            output: Tensor processado [batch, seq_len, embed_dim * 4]
            metrics: Métricas de sincronização (opcional)
        """
        batch_size, seq_len, embed_dim = x.shape
        n_neurons = self.kuramoto_system.neuron_grid.n_neurons

        # Processar cada posição da sequência
        outputs = []
        all_sync_orders = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, embed_dim * 4]

            # Projetar para fases iniciais
            phases_initial = self.input_projection(x_t)  # [batch, 2*n_neurons]

            theta_init = phases_initial[:, :n_neurons]
            phi_init = phases_initial[:, n_neurons:]

            # Normalizar para [-π, π]
            theta_init = torch.tanh(theta_init) * np.pi
            phi_init = torch.tanh(phi_init) * np.pi

            # Executar dinâmica de Kuramoto
            kuramoto_results = self.kuramoto_system(theta_init, phi_init)

            # Concatenar fases finais
            theta_final = kuramoto_results['theta_final']
            phi_final = kuramoto_results['phi_final']
            phases_final = torch.cat([theta_final, phi_final], dim=1)

            # Projetar de volta para espaço quaterniônico
            output_t = self.output_projection(phases_final)

            # Conexão residual
            if self.config['qrh_integration']['use_residual_connection']:
                output_t = output_t + self.residual_weight * x_t

            # Normalização
            output_t = self.layer_norm(output_t)

            outputs.append(output_t)
            all_sync_orders.append(kuramoto_results['final_sync_order'])

        # Reconstruir sequência
        output = torch.stack(outputs, dim=1)  # [batch, seq_len, embed_dim * 4]

        # Métricas agregadas
        if return_metrics:
            metrics = {
                'synchronization_order_mean': np.mean(all_sync_orders),
                'synchronization_order_std': np.std(all_sync_orders),
                'is_synchronized': np.mean(all_sync_orders) > self.kuramoto_system.sync_threshold,
                'sync_order_per_token': all_sync_orders
            }
            return output, metrics
        else:
            return output, None


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def create_kuramoto_layer(
    config_path: Optional[str] = None,
    device: str = "cpu"
) -> KuramotoSpectralLayer:
    """
    Factory function para criar camada de Kuramoto.

    Args:
        config_path: Caminho para configuração YAML
        device: Dispositivo (cpu/cuda/mps)

    Returns:
        KuramotoSpectralLayer configurada
    """
    layer = KuramotoSpectralLayer(config_path)
    layer.to(device)
    return layer
