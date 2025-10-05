#!/usr/bin/env python3
"""
Acoplamento Harmônico entre Camadas ΨQRH
========================================

Implementa sincronização harmônica entre:
1. Self-Attention (domínio espectral)
2. Kuramoto Spectral Neurons (osciladores acoplados)
3. Conscious Working Memory (memória persistente)
4. Feed-Forward Network (transformação não-linear)

Mathematical Framework:
----------------------
Acoplamento Harmônico:
    H(x₁, x₂, ..., xₙ) = ∑ᵢ wᵢ·xᵢ + K·∑ᵢⱼ sin(φⱼ - φᵢ)

Sincronização de Fase Global:
    r_global = |1/N ∑ₙ e^{iφₙ}|

Conservação de Energia Coletiva:
    ||H(x)||² ≈ ∑ᵢ ||xᵢ||²

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


def load_harmonic_config(config_path: Optional[str] = None, preset: str = 'standard') -> Dict:
    """
    Carrega configuração de acoplamento harmônico do arquivo YAML.

    Args:
        config_path: Caminho para arquivo de config (opcional)
        preset: Preset a usar ('standard', 'strong', 'weak', 'adaptive')

    Returns:
        Dict com configuração completa
    """
    if config_path is None:
        # Usar caminho padrão
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / 'configs' / 'harmonic_coupling_config.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    # Pegar configuração base
    config = full_config['harmonic_coupling'].copy()

    # Aplicar preset se especificado
    if preset and preset in full_config.get('presets', {}):
        preset_config = full_config['presets'][preset]
        # Merge recursivo
        def merge_dicts(base, override):
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result

        config = merge_dicts(config, preset_config)

    return config


@dataclass
class LayerState:
    """Estado de uma camada no sistema acoplado"""
    output: torch.Tensor
    phase: torch.Tensor
    frequency: float
    energy: float
    name: str


class HarmonicLayerCoupling(nn.Module):
    """
    Módulo de acoplamento harmônico entre camadas ΨQRH.
    Sincroniza fases e frequências para processamento coerente.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_layers: int = 4,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        preset: str = 'standard',
        device: str = 'cpu'
    ):
        super().__init__()

        # Carregar configuração
        if config is None:
            config = load_harmonic_config(config_path, preset)

        self.config = config
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.K = config['coupling_strength']
        self.omega_target = config['target_frequency']
        self.device = device

        # Pesos adaptativos para cada camada
        adaptive_weights = config['layer_weights']['adaptive']
        if adaptive_weights:
            self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)
        else:
            # Usar pesos fixos do config
            initial_weights = list(config['layer_weights']['initial_weights'].values())[:n_layers]
            self.layer_weights = nn.Parameter(torch.tensor(initial_weights), requires_grad=False)

        # Frequências naturais de cada camada
        freq_range = config['natural_frequencies']['frequency_range']
        self.natural_frequencies = nn.Parameter(
            torch.linspace(freq_range[0], freq_range[1], n_layers)
        )

        # Histórico de sincronização
        history_size = config['tracking']['history_size']
        self.register_buffer('sync_history', torch.zeros(history_size, device=device))
        self.history_idx = 0

        # Estado interno para tracking
        self.layer_states: List[LayerState] = []

    def extract_phase(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai fase complexa de um tensor.

        Args:
            x: Tensor [batch, seq, features]

        Returns:
            phases: Fases [batch, features]
        """
        # Colapsar sequência tomando a média
        x_collapsed = x.mean(dim=1)  # [batch, features]

        # Converter para complexo via FFT
        x_fft = torch.fft.fft(x_collapsed, dim=-1)

        # Extrair fase
        phases = torch.angle(x_fft)

        return phases

    def compute_global_synchronization(
        self,
        phases_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Computa ordem de sincronização global entre todas as camadas.

        Args:
            phases_list: Lista de fases [batch, features] para cada camada

        Returns:
            r_global: Ordem de sincronização [0, 1]
        """
        # Stack todas as fases
        all_phases = torch.stack(phases_list, dim=1)  # [batch, n_layers, features]

        # Converter para complexo
        complex_phases = torch.exp(1j * all_phases)

        # Média sobre camadas e features
        mean_phase = complex_phases.mean(dim=(1, 2))  # [batch]

        # Magnitude = ordem de sincronização
        r_global = torch.abs(mean_phase).mean()

        return r_global

    def apply_phase_coupling(
        self,
        outputs: List[torch.Tensor],
        phases: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Aplica acoplamento de fase entre camadas via termo de Kuramoto.

        Args:
            outputs: Lista de outputs de cada camada
            phases: Lista de fases de cada camada

        Returns:
            coupled_outputs: Outputs acoplados harmonicamente
        """
        coupled_outputs = []

        for i, (output_i, phase_i) in enumerate(zip(outputs, phases)):
            # Termo de acoplamento: K·∑ⱼ sin(φⱼ - φᵢ)
            coupling_term = torch.zeros_like(output_i)

            for j, phase_j in enumerate(phases):
                if i != j:
                    # Diferença de fase
                    phase_diff = phase_j - phase_i  # [batch, features]

                    # Termo de sincronização
                    sync_term = torch.sin(phase_diff)  # [batch, features]

                    # Expandir para dimensões completas
                    sync_term_expanded = sync_term.unsqueeze(1).expand_as(output_i)

                    # Acumular acoplamento
                    coupling_term += self.K * sync_term_expanded / (self.n_layers - 1)

            # Output acoplado
            coupled_output = output_i + coupling_term
            coupled_outputs.append(coupled_output)

        return coupled_outputs

    def apply_frequency_alignment(
        self,
        outputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Alinha frequências naturais de cada camada à frequência alvo.

        Args:
            outputs: Lista de outputs de cada camada

        Returns:
            aligned_outputs: Outputs com frequências alinhadas
        """
        aligned_outputs = []

        for i, output in enumerate(outputs):
            # Frequência natural da camada
            omega_i = self.natural_frequencies[i]

            # Fator de ajuste para frequência alvo
            freq_adjustment = self.omega_target / (omega_i + 1e-6)

            # Aplicar modulação de frequência
            # No domínio do tempo, isso equivale a dilatar/contrair temporalmente
            # Aqui, aproximamos multiplicando por fator
            aligned_output = output * freq_adjustment

            aligned_outputs.append(aligned_output)

        return aligned_outputs

    def weighted_combination(
        self,
        outputs: List[torch.Tensor],
        preserve_energy: bool = True
    ) -> torch.Tensor:
        """
        Combina outputs de todas as camadas com pesos adaptativos.

        Args:
            outputs: Lista de outputs de cada camada
            preserve_energy: Se True, normaliza para conservar energia

        Returns:
            combined: Output combinado
        """
        # Normalizar pesos
        weights = torch.softmax(self.layer_weights, dim=0)

        # Combinar com pesos
        combined = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            combined += weights[i] * output

        # Conservar energia se solicitado
        if preserve_energy:
            # Calcular energia média das entradas
            input_energies = [torch.norm(out) for out in outputs]
            mean_input_energy = torch.stack(input_energies).mean()

            # Energia do output combinado
            output_energy = torch.norm(combined)

            # Normalizar para conservar energia
            if output_energy > 1e-8:
                energy_scale = mean_input_energy / output_energy
                combined = combined * energy_scale

        return combined

    def forward(
        self,
        layer_outputs: Dict[str, torch.Tensor],
        layer_names: Optional[List[str]] = None,
        input_reference: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass com acoplamento harmônico.

        Args:
            layer_outputs: Dict com outputs de cada camada
                          {'attention': tensor, 'kuramoto': tensor, ...}
            layer_names: Lista ordenada de nomes das camadas
            input_reference: Tensor de entrada original para cálculo de energia

        Returns:
            harmonized_output: Output harmonizado
            metrics: Métricas de sincronização
        """
        if layer_names is None:
            layer_names = list(layer_outputs.keys())

        # Coletar outputs na ordem especificada
        outputs = [layer_outputs[name] for name in layer_names]

        # Verificar que todos os outputs têm mesma shape
        ref_shape = outputs[0].shape
        for i, output in enumerate(outputs):
            if output.shape != ref_shape:
                # Ajustar shape se necessário
                if output.size(-1) != ref_shape[-1]:
                    # Usar pooling ou padding adaptativo
                    if output.size(-1) > ref_shape[-1]:
                        outputs[i] = output[..., :ref_shape[-1]]
                    else:
                        pad_size = ref_shape[-1] - output.size(-1)
                        outputs[i] = torch.nn.functional.pad(output, (0, pad_size))

        # 1. Extrair fases de cada camada
        phases = [self.extract_phase(output) for output in outputs]

        # 2. Computar sincronização global
        r_global = self.compute_global_synchronization(phases)

        # Atualizar histórico
        self.sync_history[self.history_idx % 100] = r_global
        self.history_idx += 1

        # 3. Aplicar acoplamento de fase
        coupled_outputs = self.apply_phase_coupling(outputs, phases)

        # 4. Alinhar frequências
        aligned_outputs = self.apply_frequency_alignment(coupled_outputs)

        # 5. Combinar com pesos adaptativos
        harmonized_output = self.weighted_combination(aligned_outputs)

        # 6. Computar energias
        energies = [torch.norm(output).item() for output in outputs]
        total_energy_in = sum(energies)
        total_energy_out = torch.norm(harmonized_output).item()
        energy_ratio = total_energy_out / (total_energy_in + 1e-8)

        # 7. Aplicar normalização de energia se necessário
        if energy_ratio < 0.5 or energy_ratio > 2.0:
            scale = torch.sqrt(torch.tensor(total_energy_in / (total_energy_out + 1e-8)))
            harmonized_output = harmonized_output * scale
            total_energy_out = torch.norm(harmonized_output).item()
            energy_ratio = total_energy_out / (total_energy_in + 1e-8)

        # Salvar estados para análise
        self.layer_states = [
            LayerState(
                output=outputs[i],
                phase=phases[i],
                frequency=self.natural_frequencies[i].item(),
                energy=energies[i],
                name=layer_names[i]
            )
            for i in range(len(outputs))
        ]

        # Métricas
        metrics = {
            'global_synchronization': r_global.item(),
            'layer_weights': self.layer_weights.detach().cpu().tolist(),
            'natural_frequencies': self.natural_frequencies.detach().cpu().tolist(),
            'energies': energies,
            'energy_ratio': energy_ratio,
            'is_synchronized': r_global.item() > 0.7,
            'sync_history': self.sync_history[:min(self.history_idx, 100)].cpu().tolist()
        }

        return harmonized_output, metrics


class AdaptiveHarmonicGate(nn.Module):
    """
    Gate adaptativo para controlar contribuição de cada camada
    baseado em sincronização harmônica.
    """

    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.n_layers = n_layers

        # Gates aprendíveis
        self.gates = nn.Parameter(torch.ones(n_layers))

    def forward(
        self,
        outputs: List[torch.Tensor],
        sync_order: float
    ) -> List[torch.Tensor]:
        """
        Aplica gating baseado em sincronização.

        Args:
            outputs: Lista de outputs das camadas
            sync_order: Ordem de sincronização r ∈ [0, 1]

        Returns:
            gated_outputs: Outputs com gating aplicado
        """
        # Gate baseado em sincronização
        # Se sincronização alta → gates abertos
        # Se sincronização baixa → gates mais fechados
        sync_factor = torch.sigmoid(torch.tensor(sync_order * 5 - 2.5))

        # Aplicar gates
        gates_normalized = torch.sigmoid(self.gates)
        gated_outputs = []

        for i, output in enumerate(outputs):
            gate_value = gates_normalized[i] * sync_factor
            gated_output = output * gate_value
            gated_outputs.append(gated_output)

        return gated_outputs


def create_harmonic_coupling(
    embed_dim: int = 256,
    n_layers: int = 4,
    config: Optional[Dict] = None,
    config_path: Optional[str] = None,
    preset: str = 'standard',
    device: str = 'cpu'
) -> HarmonicLayerCoupling:
    """Factory function para criar módulo de acoplamento harmônico"""
    return HarmonicLayerCoupling(
        embed_dim=embed_dim,
        n_layers=n_layers,
        config=config,
        config_path=config_path,
        preset=preset,
        device=device
    )


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def visualize_harmonic_coupling(
    coupling_module: HarmonicLayerCoupling,
    save_path: Optional[str] = None
):
    """
    Visualiza estado do acoplamento harmônico entre camadas.

    Args:
        coupling_module: Módulo de acoplamento
        save_path: Caminho para salvar visualização
    """
    import matplotlib.pyplot as plt

    if not coupling_module.layer_states:
        print("⚠️  Nenhum estado salvo para visualizar")
        return

    n_layers = len(coupling_module.layer_states)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Frequências naturais
    ax = axes[0, 0]
    freqs = [state.frequency for state in coupling_module.layer_states]
    names = [state.name for state in coupling_module.layer_states]
    ax.bar(names, freqs, color='steelblue', alpha=0.7)
    ax.axhline(y=coupling_module.omega_target, color='red', linestyle='--',
               label=f'Target: {coupling_module.omega_target:.2f}')
    ax.set_ylabel('Frequência Natural')
    ax.set_title('Frequências das Camadas')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Energias
    ax = axes[0, 1]
    energies = [state.energy for state in coupling_module.layer_states]
    ax.bar(names, energies, color='coral', alpha=0.7)
    ax.set_ylabel('Energia (norma L2)')
    ax.set_title('Energias das Camadas')
    ax.grid(True, alpha=0.3)

    # 3. Histórico de sincronização
    ax = axes[1, 0]
    history = coupling_module.sync_history[:coupling_module.history_idx].cpu().numpy()
    ax.plot(history, color='green', linewidth=2)
    ax.axhline(y=0.7, color='red', linestyle='--', label='Threshold: 0.7')
    ax.set_xlabel('Iteração')
    ax.set_ylabel('Ordem de Sincronização r')
    ax.set_title('Evolução da Sincronização Global')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Pesos das camadas
    ax = axes[1, 1]
    weights = torch.softmax(coupling_module.layer_weights, dim=0).detach().cpu().numpy()
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    ax.pie(weights, labels=names, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Contribuição Relativa das Camadas')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualização salva em: {save_path}")
    else:
        plt.show()

    plt.close()
