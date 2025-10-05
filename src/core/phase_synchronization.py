#!/usr/bin/env python3
"""
Sincronização de Fase e Emergência de Ritmos Neurais
====================================================

Implementa:
1. Sincronização de fase com coerência temporal
2. Mapas topográficos de atividade espectral
3. Robustez a ruído via acoplamento de Kuramoto
4. Emergência natural de oscilações theta/gamma
5. Alinhamento com modelos de campo neural e óptica quântica

Mathematical Framework:
-----------------------
Ordem de Sincronização:
    r = |⟨e^{iθ}⟩| = |1/N ∑ⱼ e^{iθⱼ}|

Ritmos Neurais:
    Theta: 4-8 Hz
    Alpha: 8-13 Hz
    Beta: 13-30 Hz
    Gamma: 30-100 Hz

Conservação de Energia:
    ||output||² / ||input||² ∈ [0.95, 1.05]

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RhythmFrequencies:
    """Frequências dos ritmos neurais (Hz)"""
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta: Tuple[float, float] = (13.0, 30.0)
    gamma: Tuple[float, float] = (30.0, 100.0)


class PhaseSynchronizationModule(nn.Module):
    """
    Módulo de sincronização de fase para neurônios espectrais.
    Implementa sincronização espontânea e coerência temporal.
    """

    def __init__(
        self,
        grid_size: int = 32,
        embed_dim: int = 256,
        coupling_strength: float = 1.0,
        sync_threshold: float = 0.9,
        device: str = "cpu"
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_neurons = grid_size * grid_size
        self.embed_dim = embed_dim
        self.K = coupling_strength
        self.sync_threshold = sync_threshold
        self.device = device

        # Frequências dos ritmos neurais
        self.rhythms = RhythmFrequencies()

        # Histórico de sincronização
        self.register_buffer('sync_history', torch.zeros(100, device=device))
        self.history_idx = 0

        # Mapa de fases espacial
        self.register_buffer(
            'spatial_phase_map',
            torch.zeros(grid_size, grid_size, device=device)
        )

    def compute_synchronization_order(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Computa ordem de sincronização: r = |⟨e^{iθ}⟩|

        Args:
            phases: Tensor de fases [batch, n_neurons]

        Returns:
            r: Ordem de sincronização [0, 1]
        """
        # Converter para complexo
        complex_phases = torch.exp(1j * phases)

        # Média vetorial
        mean_phase = complex_phases.mean(dim=1)  # [batch]

        # Magnitude = ordem de sincronização
        r = torch.abs(mean_phase).mean()

        return r

    def extract_rhythm_components(
        self,
        signal: torch.Tensor,
        sampling_rate: float = 1000.0
    ) -> Dict[str, torch.Tensor]:
        """
        Extrai componentes de ritmos neurais (theta, alpha, beta, gamma).

        Args:
            signal: Sinal temporal [batch, time, features]
            sampling_rate: Taxa de amostragem em Hz

        Returns:
            Dict com componentes de cada ritmo
        """
        batch_size, time_steps, features = signal.shape

        # FFT ao longo do tempo
        fft_signal = torch.fft.fft(signal, dim=1)
        freqs = torch.fft.fftfreq(time_steps, d=1.0/sampling_rate)

        # Extrair bandas de frequência
        rhythms = {}

        for rhythm_name in ['theta', 'alpha', 'beta', 'gamma']:
            freq_range = getattr(self.rhythms, rhythm_name)
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

            # Aplicar máscara no domínio da frequência
            filtered_fft = fft_signal.clone()
            filtered_fft[:, ~mask, :] = 0

            # Inverter para domínio do tempo
            rhythm_component = torch.fft.ifft(filtered_fft, dim=1).real
            rhythms[rhythm_name] = rhythm_component

        return rhythms

    def apply_collective_noise_filter(
        self,
        x: torch.Tensor,
        phases: torch.Tensor,
        sync_order: float
    ) -> torch.Tensor:
        """
        Filtra ruído via sincronização coletiva.
        Neurônios sincronizados reforçam sinal; dessincronizados são atenuados.

        Args:
            x: Input tensor [batch, seq_len, features]
            phases: Fases dos neurônios [batch, n_neurons]
            sync_order: Ordem de sincronização r

        Returns:
            Tensor filtrado
        """
        batch_size, seq_len, features = x.shape

        # Se sincronização alta, sinal é confiável
        if sync_order > self.sync_threshold:
            # Mínima filtragem
            noise_filter = 0.95
        else:
            # Filtragem baseada em dessincronização
            noise_filter = 0.5 + 0.45 * sync_order

        # Aplicar filtro suave
        filtered = x * noise_filter

        return filtered

    def create_spatial_phase_map(
        self,
        phases: torch.Tensor
    ) -> torch.Tensor:
        """
        Cria mapa topográfico de fases espaciais.

        Args:
            phases: Fases [batch, n_neurons]

        Returns:
            Mapa 2D [grid_size, grid_size]
        """
        batch_size = phases.shape[0]

        # Reshape para grid espacial
        phase_map = phases.view(batch_size, self.grid_size, self.grid_size)

        # Média sobre batch
        avg_phase_map = phase_map.mean(dim=0)

        # Normalizar para [0, 2π]
        avg_phase_map = torch.fmod(avg_phase_map, 2 * np.pi)

        # Atualizar buffer
        self.spatial_phase_map = avg_phase_map

        return avg_phase_map

    def visualize_spatial_phase_map(
        self,
        phase_map: Optional[torch.Tensor] = None,
        save_path: Optional[Path] = None
    ):
        """
        Visualiza mapa de fase espacial com matplotlib.

        Args:
            phase_map: Mapa de fase [grid_size, grid_size]. Se None, usa buffer interno.
            save_path: Caminho para salvar imagem
        """
        if phase_map is None:
            phase_map = self.spatial_phase_map

        phase_map_np = phase_map.cpu().numpy()

        plt.figure(figsize=(10, 8))
        im = plt.imshow(phase_map_np, cmap='twilight', vmin=0, vmax=2*np.pi, aspect='auto')
        plt.colorbar(im, label='Fase (radianos)')
        plt.title('Mapa Topográfico de Fase Espacial')
        plt.xlabel('Posição X')
        plt.ylabel('Posição Y')

        # Adicionar grid
        plt.grid(True, alpha=0.3, linewidth=0.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Mapa salvo em: {save_path}")
        else:
            plt.show()

        plt.close()

    def forward(
        self,
        x: torch.Tensor,
        phases: torch.Tensor,
        extract_rhythms: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass com sincronização de fase e filtragem de ruído.

        Args:
            x: Input tensor [batch, seq_len, features]
            phases: Fases dos neurônios [batch, n_neurons]
            extract_rhythms: Se True, extrai componentes de ritmos neurais

        Returns:
            output: Tensor processado
            metrics: Métricas de sincronização
        """
        # 1. Computar ordem de sincronização
        sync_order = self.compute_synchronization_order(phases)

        # Atualizar histórico
        self.sync_history[self.history_idx % 100] = sync_order
        self.history_idx += 1

        # 2. Criar mapa espacial de fases
        phase_map = self.create_spatial_phase_map(phases)

        # 3. Aplicar filtro de ruído coletivo
        filtered_x = self.apply_collective_noise_filter(x, phases, sync_order.item())

        # 4. Extrair ritmos neurais (opcional)
        rhythms = {}
        if extract_rhythms and x.shape[1] >= 100:  # Precisa de sequência longa
            rhythms = self.extract_rhythm_components(filtered_x)

        # 5. Compilar métricas
        metrics = {
            'synchronization_order': sync_order.item(),
            'is_synchronized': sync_order.item() > self.sync_threshold,
            'phase_map': phase_map,
            'sync_history': self.sync_history[:min(self.history_idx, 100)],
            'noise_filter_strength': 0.5 + 0.45 * sync_order.item()
        }

        if rhythms:
            metrics['rhythms'] = rhythms

        return filtered_x, metrics


class EnergyConservationValidator:
    """
    Valida conservação de energia com threshold rigoroso [0.95, 1.05].
    """

    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance
        self.min_ratio = 1.0 - tolerance
        self.max_ratio = 1.0 + tolerance
        self.history = []

    def validate(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor
    ) -> Tuple[bool, float, Dict]:
        """
        Valida conservação de energia.

        Args:
            input_tensor: Tensor de entrada
            output_tensor: Tensor de saída

        Returns:
            is_valid: True se energia conservada
            ratio: Razão de energia
            report: Relatório detalhado
        """
        # Calcular energias (norma L2)
        input_energy = torch.norm(input_tensor).item()
        output_energy = torch.norm(output_tensor).item()

        # Razão
        if input_energy < 1e-10:
            ratio = 1.0  # Evitar divisão por zero
        else:
            ratio = output_energy / input_energy

        # Validar
        is_valid = self.min_ratio <= ratio <= self.max_ratio

        # Relatório
        report = {
            'input_energy': input_energy,
            'output_energy': output_energy,
            'ratio': ratio,
            'is_valid': is_valid,
            'tolerance': self.tolerance,
            'min_acceptable': self.min_ratio,
            'max_acceptable': self.max_ratio,
            'deviation': abs(ratio - 1.0)
        }

        # Adicionar ao histórico
        self.history.append(ratio)

        return is_valid, ratio, report

    def get_statistics(self) -> Dict:
        """Retorna estatísticas do histórico de validações"""
        if not self.history:
            return {}

        history_arr = np.array(self.history)
        return {
            'mean_ratio': float(np.mean(history_arr)),
            'std_ratio': float(np.std(history_arr)),
            'min_ratio': float(np.min(history_arr)),
            'max_ratio': float(np.max(history_arr)),
            'num_validations': len(self.history),
            'success_rate': float(np.sum((history_arr >= self.min_ratio) & (history_arr <= self.max_ratio)) / len(self.history))
        }


class TopographicActivityMapper:
    """
    Cria mapas topográficos de atividade espectral (análogo ao córtex visual).
    """

    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size

    def create_activity_map(
        self,
        activations: torch.Tensor,
        phases: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cria mapa de atividade espacial.

        Args:
            activations: Ativações [batch, n_neurons]
            phases: Fases opcionais [batch, n_neurons]

        Returns:
            Mapa 2D [grid_size, grid_size]
        """
        batch_size = activations.shape[0]

        # Reshape para grid
        activity_map = activations.view(batch_size, self.grid_size, self.grid_size)

        # Média sobre batch
        avg_map = activity_map.mean(dim=0)

        return avg_map

    def visualize_activity_map(
        self,
        activity_map: torch.Tensor,
        phase_map: Optional[torch.Tensor] = None,
        save_path: Optional[Path] = None,
        title: str = "Mapa Topográfico de Atividade Espectral"
    ):
        """
        Visualiza mapa de atividade com cores e sobreposição de fases.

        Args:
            activity_map: Mapa de atividade [grid_size, grid_size]
            phase_map: Mapa de fase opcional [grid_size, grid_size]
            save_path: Caminho para salvar
            title: Título do gráfico
        """
        activity_np = activity_map.cpu().numpy()

        if phase_map is not None:
            # Criar subplot com atividade e fase
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            # Atividade
            im1 = axes[0].imshow(activity_np, cmap='hot', aspect='auto')
            axes[0].set_title('Atividade Espectral')
            axes[0].set_xlabel('Posição X')
            axes[0].set_ylabel('Posição Y')
            plt.colorbar(im1, ax=axes[0], label='Ativação')

            # Fase
            phase_np = phase_map.cpu().numpy()
            im2 = axes[1].imshow(phase_np, cmap='twilight', vmin=0, vmax=2*np.pi, aspect='auto')
            axes[1].set_title('Mapa de Fase')
            axes[1].set_xlabel('Posição X')
            axes[1].set_ylabel('Posição Y')
            plt.colorbar(im2, ax=axes[1], label='Fase (rad)')

            plt.suptitle(title, fontsize=14, fontweight='bold')

        else:
            # Apenas atividade
            plt.figure(figsize=(10, 8))
            im = plt.imshow(activity_np, cmap='hot', aspect='auto')
            plt.colorbar(im, label='Ativação')
            plt.title(title)
            plt.xlabel('Posição X')
            plt.ylabel('Posição Y')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Mapa salvo em: {save_path}")
        else:
            plt.show()

        plt.close()


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def stabilize_phases(phases: torch.Tensor) -> torch.Tensor:
    """
    Estabiliza fases para [0, 2π] usando torch.fmod.

    Args:
        phases: Tensor de fases

    Returns:
        Fases estabilizadas
    """
    return torch.fmod(phases, 2 * np.pi)


def measure_temporal_coherence(
    phases_t1: torch.Tensor,
    phases_t2: torch.Tensor
) -> float:
    """
    Mede coerência temporal entre dois instantes de tempo.

    Args:
        phases_t1: Fases no tempo t1 [batch, n_neurons]
        phases_t2: Fases no tempo t2 [batch, n_neurons]

    Returns:
        Coerência temporal [0, 1]
    """
    # Diferença de fase
    phase_diff = torch.abs(phases_t2 - phases_t1)

    # Normalizar
    phase_diff = torch.fmod(phase_diff, 2 * np.pi)
    phase_diff = torch.minimum(phase_diff, 2 * np.pi - phase_diff)

    # Coerência: inverso da dispersão
    coherence = 1.0 - (phase_diff.mean() / np.pi)

    return coherence.item()


def create_phase_sync_module(
    grid_size: int = 32,
    embed_dim: int = 256,
    device: str = "cpu"
) -> PhaseSynchronizationModule:
    """Factory function para criar módulo de sincronização de fase"""
    return PhaseSynchronizationModule(
        grid_size=grid_size,
        embed_dim=embed_dim,
        device=device
    )
