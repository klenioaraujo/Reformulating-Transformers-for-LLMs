#!/usr/bin/env python3
"""
Spectral AlexNet - Rede Neural Espectral de 3 Camadas
======================================================

Arquitetura inspirada no AlexNet, mas usando matemática rigorosa:
- Autoacoplagem logística: x_{n+1} = r·x_n·(1-x_n)
- Sonda óptica: f(λ,t) = A·sin(ωt + φ_0 + θ)
- Pontos fixos: x* = 0 ou x* = 1 - 1/r
- Estabilidade: F'(x*) = r - 2r·x*

Camadas de Aprendizagem:
1. Camada Espectral (Banco) - Conhecimento armazenado
2. Camada de Conversão (Input) - Texto → Espectro
3. Camada de Interpretação (Modelo) - Processamento consciente

Comparação entre as 3 camadas usando ressonância espectral.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.quaternion_math import (
    hamilton_product,
    quaternion_normalize,
    so4_rotation,
    create_unit_quaternion_batch
)


class SpectralConvLayer(nn.Module):
    """
    Camada convolucional espectral com autoacoplagem logística.

    Similar ao AlexNet, mas opera no domínio espectral.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        logistic_r: float = 3.8,
        coupling_iterations: int = 5
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.logistic_r = logistic_r
        self.coupling_iterations = coupling_iterations

        # Kernel espectral (sem bias - física pura)
        self.spectral_kernel = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, dtype=torch.complex64)
        )

        # Parâmetros da sonda óptica
        self.register_buffer('probe_omega', torch.tensor(2 * np.pi))
        self.register_buffer('probe_phi0', torch.tensor(0.0))

        # α(D) adaptativo - será atualizado durante forward
        self.alpha_D = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolução espectral com autoacoplagem.

        Args:
            x: Tensor espectral [batch, in_channels, length]

        Returns:
            Tensor processado [batch, out_channels, length]
        """
        batch_size = x.shape[0]

        # 1. Aplicar autoacoplagem logística ao input
        x_coupled = self._apply_logistic_coupling(x)

        # 2. Convolução no domínio espectral (multiplicação pontual)
        # F{f * g} = F{f} · F{g}
        x_fft = torch.fft.fft(x_coupled, dim=-1)
        kernel_fft = torch.fft.fft(self.spectral_kernel, dim=-1)

        # Broadcasting para convolução
        output_fft = torch.zeros(
            batch_size, self.out_channels, x.shape[-1],
            dtype=torch.complex64, device=x.device
        )

        # Ajustar dimensões para compatibilidade
        # kernel_fft: [out_channels, in_channels, kernel_size]
        # x_fft: [batch, in_channels, seq_len]

        # Para cada canal de saída, fazer a convolução
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                # Interpolar magnitude do kernel para tamanho da sequência
                kernel_magnitude = torch.abs(kernel_fft[i, j, :])
                kernel_phase = torch.angle(kernel_fft[i, j, :])

                kernel_magnitude_interp = torch.nn.functional.interpolate(
                    kernel_magnitude.unsqueeze(0).unsqueeze(0),  # [1, 1, kernel_size]
                    size=x.shape[-1],
                    mode='linear',
                    align_corners=False
                ).squeeze(0).squeeze(0)  # [seq_len]

                kernel_phase_interp = torch.nn.functional.interpolate(
                    kernel_phase.unsqueeze(0).unsqueeze(0),  # [1, 1, kernel_size]
                    size=x.shape[-1],
                    mode='linear',
                    align_corners=False
                ).squeeze(0).squeeze(0)  # [seq_len]

                kernel_interp = kernel_magnitude_interp * torch.exp(1j * kernel_phase_interp)

                output_fft[:, i, :] += x_fft[:, j, :] * kernel_interp

        # 3. Transformada inversa
        output = torch.fft.ifft(output_fft, dim=-1)

        # 4. Aplicar sonda óptica como ativação
        output = self._optical_probe_activation(output)

        return output

    def _apply_logistic_coupling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica autoacoplagem logística: x_{n+1} = r·x_n·(1-x_n)
        """
        # Trabalhar com magnitude
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        # Normalizar para [0, 1]
        mag_min = magnitude.amin(dim=(-1, -2), keepdim=True)
        mag_max = magnitude.amax(dim=(-1, -2), keepdim=True)
        x_n = (magnitude - mag_min) / (mag_max - mag_min + 1e-10)

        # Iterações do mapa logístico
        for _ in range(self.coupling_iterations):
            x_n = self.logistic_r * x_n * (1.0 - x_n)

        # Desnormalizar
        magnitude_coupled = x_n * (mag_max - mag_min) + mag_min

        # Reconstruir complexo
        return magnitude_coupled * torch.exp(1j * phase)

    def _optical_probe_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ativação via sonda óptica: f(λ,t) = A·sin(ωt + φ_0 + θ)
        """
        t = 0.0  # Tempo atual

        # Criar índices λ
        lambda_indices = torch.arange(x.shape[-1], device=x.device, dtype=torch.float32)

        # θ = α(D) · λ
        theta = self.alpha_D * lambda_indices

        # f(λ,t) = sin(ωt + φ_0 + θ)
        phase = self.probe_omega * t + self.probe_phi0 + theta
        activation = torch.sin(phase)

        # Aplicar modulação
        return x * activation.unsqueeze(0).unsqueeze(0)

    def compute_fixed_points(self) -> Tuple[float, float]:
        """
        Calcula pontos fixos do mapa logístico:
        x* = 0 ou x* = 1 - 1/r
        """
        fixed_point_1 = 0.0
        fixed_point_2 = 1.0 - 1.0 / self.logistic_r
        return fixed_point_1, fixed_point_2

    def compute_stability(self, x_star: float) -> float:
        """
        Calcula estabilidade: F'(x*) = r - 2r·x*
        """
        return self.logistic_r - 2 * self.logistic_r * x_star


class SpectralAlexNet(nn.Module):
    """
    Rede Neural Espectral de 3 Camadas (inspirada no AlexNet).

    Arquitetura:
    - Camada 1: Banco Espectral (conhecimento armazenado)
    - Camada 2: Conversão Input (texto → espectro)
    - Camada 3: Interpretação (processamento consciente)

    Comparação espectral entre as 3 camadas para aprendizagem.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = [256, 512, 256],
        spectral_vocab_size: int = 256,
        logistic_r: float = 3.8,
        device: str = 'cpu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.spectral_vocab_size = spectral_vocab_size
        self.logistic_r = logistic_r
        self.device = device

        # ============================================================
        # CAMADA 1: Banco Espectral (Conhecimento Armazenado)
        # ============================================================
        self.spectral_bank = nn.Parameter(
            torch.randn(spectral_vocab_size, input_dim, dtype=torch.complex64)
        )

        # Normalizar banco por energia total
        with torch.no_grad():
            energy = torch.abs(self.spectral_bank).sum(dim=-1, keepdim=True)
            self.spectral_bank.data = self.spectral_bank / (energy + 1e-10)

        # ============================================================
        # CAMADA 2: Conversão Input (Texto → Espectro)
        # ============================================================
        self.input_converter = SpectralConvLayer(
            in_channels=1,
            out_channels=hidden_dims[0],
            kernel_size=32,
            logistic_r=logistic_r,
            coupling_iterations=5
        )

        # Projeção para domínio quaterniônico
        # hidden_dims[0] = 256, mas concatenamos real+imag = 512
        self.to_quaternion = nn.Linear(hidden_dims[0] * 2, 4)  # [w, x, y, z]

        # ============================================================
        # CAMADA 3: Interpretação (Processamento Consciente)
        # ============================================================

        # Subcamada 3.1: Evolução SO(4)
        self.register_buffer(
            'q_left',
            create_unit_quaternion_batch(
                torch.tensor([0.5]), torch.tensor([0.3]), torch.tensor([0.7])
            )[0]
        )
        self.register_buffer(
            'q_right',
            create_unit_quaternion_batch(
                torch.tensor([0.7]), torch.tensor([0.5]), torch.tensor([0.3])
            )[0]
        )

        # Subcamada 3.2: Autoacoplagem quaterniônica
        self.quaternion_coupling_iterations = 3

        # Subcamada 3.3: Projeção espectral final
        self.spectral_projection = SpectralConvLayer(
            in_channels=4,  # Quaternions
            out_channels=1,
            kernel_size=16,
            logistic_r=logistic_r,
            coupling_iterations=3
        )

        # ============================================================
        # Comparador de 3 Camadas
        # ============================================================
        self.layer_comparator = LayerComparator(device=device)

        print("🧠 Spectral AlexNet Inicializado")
        print(f"   • Camada 1 (Banco): {self.spectral_bank.shape}")
        print(f"   • Camada 2 (Input): {hidden_dims[0]} canais")
        print(f"   • Camada 3 (Interpretação): SO(4) + Quaternion Coupling")
        print(f"   • Mapa logístico: r = {logistic_r}")

        # Calcular pontos fixos
        fp1, fp2 = self.input_converter.compute_fixed_points()
        stability = self.input_converter.compute_stability(fp2)
        print(f"   • Pontos fixos: x₁*={fp1:.4f}, x₂*={fp2:.4f}")
        print(f"   • Estabilidade F'(x₂*): {stability:.4f}")

    def forward(
        self,
        x: torch.Tensor,
        return_layer_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass com comparação de 3 camadas.

        Args:
            x: Input espectral [batch, input_dim]
            return_layer_outputs: Se True, retorna saídas de todas as camadas

        Returns:
            Dict com outputs e métricas de comparação
        """
        batch_size = x.shape[0]

        # ============================================================
        # CAMADA 1: Banco Espectral
        # ============================================================
        # Buscar padrões mais similares no banco
        bank_output = self._query_spectral_bank(x)  # [batch, input_dim]

        # ============================================================
        # CAMADA 2: Conversão Input
        # ============================================================
        # Adicionar dimensão de canal
        x_input = x.unsqueeze(1)  # [batch, 1, input_dim]

        # Convolução espectral com autoacoplagem
        conv_output = self.input_converter(x_input)  # [batch, hidden_dim, input_dim]

        # Projetar para quaternions
        # Média ao longo de input_dim para obter features
        conv_features = conv_output.mean(dim=-1)  # [batch, hidden_dim]

        # Converter de complexo para real para a camada linear
        # hidden_dim = 256, então conv_features tem shape [batch, 256]
        # Após concatenar real+imag, fica [batch, 512]
        conv_features_real = torch.cat([conv_features.real, conv_features.imag], dim=-1)

        quaternions = self.to_quaternion(conv_features_real)  # [batch, 4]
        quaternions = quaternion_normalize(quaternions)

        # ============================================================
        # CAMADA 3: Interpretação
        # ============================================================

        # 3.1: Evolução SO(4)
        evolved = so4_rotation(quaternions, self.q_left, self.q_right)

        # 3.2: Autoacoplagem quaterniônica
        coupled = self._quaternion_coupling(evolved)

        # 3.3: Projeção espectral final
        # Expandir quaternions para formato de canal
        coupled_expanded = coupled.unsqueeze(-1).expand(-1, -1, self.input_dim)
        # [batch, 4, input_dim]

        interpretation_output = self.spectral_projection(coupled_expanded)
        # [batch, 1, input_dim]

        interpretation_output = interpretation_output.squeeze(1)  # [batch, input_dim]

        # ============================================================
        # Comparação entre as 3 Camadas
        # ============================================================
        comparison = self.layer_comparator.compare_layers(
            layer1_output=bank_output,
            layer2_output=x,  # Input original
            layer3_output=interpretation_output
        )

        # Saída final: combinação das 3 camadas
        final_output = (
            0.3 * bank_output +
            0.3 * x +
            0.4 * interpretation_output
        )

        result = {
            'output': final_output,
            'comparison': comparison,
            'quaternions': coupled
        }

        if return_layer_outputs:
            result.update({
                'layer1_bank': bank_output,
                'layer2_input': x,
                'layer3_interpretation': interpretation_output
            })

        return result

    def _query_spectral_bank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Busca no banco espectral usando correlação.

        Args:
            x: Query espectral [batch, input_dim]

        Returns:
            Padrões mais similares [batch, input_dim]
        """
        batch_size = x.shape[0]

        # Calcular correlação com cada entrada do banco
        # |⟨x, bank⟩|²
        correlations = torch.abs(
            torch.sum(x.unsqueeze(1) * self.spectral_bank.conj(), dim=-1)
        ) ** 2  # [batch, vocab_size]

        # Normalizar por energia
        x_energy = torch.sum(torch.abs(x) ** 2, dim=-1, keepdim=True)
        correlations = correlations / (x_energy + 1e-10)

        # Pegar padrão de maior correlação
        best_indices = torch.argmax(correlations, dim=1)  # [batch]

        # Recuperar do banco
        bank_output = self.spectral_bank[best_indices]  # [batch, input_dim]

        return bank_output

    def _quaternion_coupling(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Autoacoplagem logística no domínio quaterniônico.

        Args:
            quaternions: [batch, 4]

        Returns:
            Quaternions acoplados [batch, 4]
        """
        # Magnitude quaterniônica
        magnitude = torch.norm(quaternions, dim=-1, keepdim=True)

        # Normalizar para [0, 1]
        mag_min = magnitude.min()
        mag_max = magnitude.max()
        x_n = (magnitude - mag_min) / (mag_max - mag_min + 1e-10)

        # Mapa logístico
        for _ in range(self.quaternion_coupling_iterations):
            x_n = self.logistic_r * x_n * (1.0 - x_n)

        # Desnormalizar
        magnitude_coupled = x_n * (mag_max - mag_min) + mag_min

        # Reescalar quaternions
        quaternions_normalized = quaternions / (magnitude + 1e-10)
        coupled = quaternions_normalized * magnitude_coupled

        return quaternion_normalize(coupled)

    def update_spectral_bank(self, new_patterns: torch.Tensor, learning_rate: float = 0.01):
        """
        Atualiza banco espectral com novos padrões aprendidos.

        Args:
            new_patterns: Novos padrões espectrais [n_patterns, input_dim]
            learning_rate: Taxa de aprendizagem
        """
        with torch.no_grad():
            # Adicionar novos padrões com média móvel
            for i, pattern in enumerate(new_patterns):
                # Encontrar padrão mais similar no banco
                correlations = torch.abs(
                    torch.sum(pattern * self.spectral_bank.conj(), dim=-1)
                )
                best_idx = torch.argmax(correlations)

                # Atualizar com média móvel
                self.spectral_bank[best_idx] = (
                    (1 - learning_rate) * self.spectral_bank[best_idx] +
                    learning_rate * pattern
                )

            # Renormalizar energia
            energy = torch.abs(self.spectral_bank).sum(dim=-1, keepdim=True)
            self.spectral_bank.data = self.spectral_bank / (energy + 1e-10)


class LayerComparator(nn.Module):
    """
    Comparador de 3 Camadas usando ressonância espectral.
    """

    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device

    def compare_layers(
        self,
        layer1_output: torch.Tensor,
        layer2_output: torch.Tensor,
        layer3_output: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compara as 3 camadas usando métricas espectrais.

        Returns:
            Dict com métricas de comparação
        """
        # Converter para espectro se necessário
        spec1 = torch.fft.fft(layer1_output, dim=-1)
        spec2 = torch.fft.fft(layer2_output, dim=-1)
        spec3 = torch.fft.fft(layer3_output, dim=-1)

        # Correlações espectrais
        corr_1_2 = self._spectral_correlation(spec1, spec2)
        corr_1_3 = self._spectral_correlation(spec1, spec3)
        corr_2_3 = self._spectral_correlation(spec2, spec3)

        # Distâncias espectrais
        dist_1_2 = torch.norm(spec1 - spec2, dim=-1).mean()
        dist_1_3 = torch.norm(spec1 - spec3, dim=-1).mean()
        dist_2_3 = torch.norm(spec2 - spec3, dim=-1).mean()

        # Energia de ressonância (quanto mais próximo de 1, melhor)
        resonance = (corr_1_2 + corr_1_3 + corr_2_3) / 3.0

        # Coerência espectral
        coherence = 1.0 / (1.0 + dist_1_2 + dist_1_3 + dist_2_3)

        return {
            'corr_bank_input': corr_1_2,
            'corr_bank_interpretation': corr_1_3,
            'corr_input_interpretation': corr_2_3,
            'dist_bank_input': dist_1_2,
            'dist_bank_interpretation': dist_1_3,
            'dist_input_interpretation': dist_2_3,
            'resonance': resonance,
            'coherence': coherence
        }

    def _spectral_correlation(self, spec1: torch.Tensor, spec2: torch.Tensor) -> torch.Tensor:
        """
        Correlação espectral normalizada: |⟨spec1, spec2⟩|²
        """
        inner_product = torch.sum(spec1 * spec2.conj(), dim=-1)
        correlation = torch.abs(inner_product) ** 2

        # Normalizar por energias
        energy1 = torch.sum(torch.abs(spec1) ** 2, dim=-1)
        energy2 = torch.sum(torch.abs(spec2) ** 2, dim=-1)

        correlation = correlation / (energy1 * energy2 + 1e-10)

        return correlation.mean()
