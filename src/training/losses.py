#!/usr/bin/env python3
"""
Physics-Informed Loss Functions for ΨQRH Training
==================================================

Implementa funções de perda que respeitam os princípios físicos do ΨQRH,
transformando o treinamento de "caixa-preta" para "calibração de universo físico".

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math
try:
    from ..processing.physical_tokenizer import PhysicalTokenizer
except ImportError:
    # Fallback para import absoluto quando executado como script
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'processing'))
    from physical_tokenizer import PhysicalTokenizer


def spectral_distance_loss(generated_text: str, target_text: str,
                          physical_tokenizer: PhysicalTokenizer) -> torch.Tensor:
    """
    Perda por Distância Espectral: Mede a distância quântica entre textos.

    Esta função substitui completamente a CrossEntropyLoss por uma medida física
    da "dissonância" entre estados quânticos no espaço de Hilbert quaterniônico.

    Args:
        generated_text: Texto gerado pelo modelo
        target_text: Texto alvo (ground truth)
        physical_tokenizer: Tokenizer físico para codificação quântica

    Returns:
        Distância L2 entre os estados quânticos dos textos
    """
    # Codificar ambos os textos em estados quânticos
    psi_generated = physical_tokenizer.encode(generated_text)  # [1, seq_len_gen, embed_dim, 4]
    psi_target = physical_tokenizer.encode(target_text)        # [1, seq_len_target, embed_dim, 4]

    # Remover dimensão batch (sempre 1)
    psi_generated = psi_generated.squeeze(0)  # [seq_len_gen, embed_dim, 4]
    psi_target = psi_target.squeeze(0)        # [seq_len_target, embed_dim, 4]

    # Alinhar sequências para comparação (padding/truncation)
    max_len = max(psi_generated.shape[0], psi_target.shape[0])

    if psi_generated.shape[0] < max_len:
        # Pad generated
        pad_len = max_len - psi_generated.shape[0]
        psi_generated = F.pad(psi_generated, (0, 0, 0, 0, 0, pad_len))
    elif psi_target.shape[0] < max_len:
        # Pad target
        pad_len = max_len - psi_target.shape[0]
        psi_target = F.pad(psi_target, (0, 0, 0, 0, 0, pad_len))

    # Calcular distância L2 no espaço quântico
    # Norma euclidiana entre os tensores quaterniônicos
    L_semantic = torch.norm(psi_generated - psi_target, p=2)

    return L_semantic


class SpectralPhysicsInformedLoss(nn.Module):
    """
    Função de Perda Fisicamente Informada com Distância Espectral

    Implementa: L_total = L_semantic + λ₁·L_unitarity + λ₂·L_entropy + λ₃·L_fractal

    Onde L_semantic é a DISTÂNCIA ESPECTRAL entre estados quânticos,
    eliminando qualquer dependência de probabilidades estatísticas.

    Esta perda transforma o ΨQRH em um sistema de calibração física pura,
    onde o aprendizado é a minimização da dissonância quântica entre
    o que o modelo gera e o que deveria gerar.
    """

    def __init__(self,
                 physical_tokenizer: PhysicalTokenizer,
                 lambda_unitarity: float = 0.1,
                 lambda_entropy: float = 0.05,
                 lambda_fractal: float = 0.02,
                 device: str = 'cpu'):
        """
        Args:
            physical_tokenizer: Tokenizer físico para codificação quântica
            lambda_unitarity: Peso para penalização de unitariedade
            lambda_entropy: Peso para penalização de entropia
            lambda_fractal: Peso para penalização fractal
            device: Dispositivo de computação
        """
        super().__init__()
        self.physical_tokenizer = physical_tokenizer
        self.lambda_unitarity = lambda_unitarity
        self.lambda_entropy = lambda_entropy
        self.lambda_fractal = lambda_fractal
        self.device = device

        # Parâmetros físicos aprendíveis
        self.register_buffer('target_fractal_dim', torch.tensor(1.5))  # Dimensão fractal alvo
        self.register_buffer('target_entropy', torch.tensor(1.0))     # Entropia alvo

    def forward(self,
                generated_text: str,
                target_text: str,
                quantum_states: Optional[torch.Tensor] = None,
                attention_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcula perda fisicamente informada baseada em distância espectral.

        Args:
            generated_text: Texto gerado pelo modelo (string)
            target_text: Texto alvo verdadeiro (string)
            quantum_states: Estados quânticos para análise física (opcional)
            attention_weights: Pesos de atenção para análise de unitariedade (opcional)

        Returns:
            Tuple: (perda_total, dicionário_com_componentes)
        """
        # ========== L_SEMANTIC: Distância Espectral ==========
        # Mede a dissonância quântica entre textos no espaço de Hilbert
        L_semantic = spectral_distance_loss(generated_text, target_text, self.physical_tokenizer)

        # ========== L_UNITARITY: Penalização de Unitaridade ==========
        L_unitarity = self._compute_unitarity_penalty(attention_weights)

        # ========== L_ENTROPY: Penalização de Entropia ==========
        # Para entropia, usamos o estado quântico do texto gerado
        if quantum_states is not None:
            L_entropy = self._compute_entropy_penalty_from_states(quantum_states)
        else:
            # Fallback: codificar texto gerado para calcular entropia
            psi_generated = self.physical_tokenizer.encode(generated_text).squeeze(0)
            L_entropy = self._compute_entropy_penalty_from_states(psi_generated)

        # ========== L_FRACTAL: Penalização Fractal ==========
        L_fractal = self._compute_fractal_penalty(quantum_states)

        # ========== L_TOTAL: Combinação Fisicamente Informada ==========
        L_total = L_semantic + \
                 self.lambda_unitarity * L_unitarity + \
                 self.lambda_entropy * L_entropy + \
                 self.lambda_fractal * L_fractal

        # Componentes para logging
        components = {
            'L_semantic': L_semantic.item(),
            'L_unitarity': L_unitarity.item(),
            'L_entropy': L_entropy.item(),
            'L_fractal': L_fractal.item(),
            'L_total': L_total.item()
        }

        return L_total, components

    def _compute_unitarity_penalty(self, attention_weights: Optional[torch.Tensor]) -> torch.Tensor:
        """Computa penalização por violação de unitariedade."""
        if attention_weights is None:
            return torch.tensor(0.0, device=self.device)

        batch_size, n_heads, seq_len, _ = attention_weights.shape
        unitarity_penalty = 0.0

        for head in range(n_heads):
            A = attention_weights[:, head, :, :]
            A_transpose = A.transpose(-2, -1)
            asymmetry = torch.norm(A - A_transpose, dim=(-2, -1))
            unitarity_penalty += asymmetry.mean()

        unitarity_penalty /= max(n_heads, 1)
        return unitarity_penalty

    def _compute_entropy_penalty_from_states(self, quantum_states: torch.Tensor) -> torch.Tensor:
        """Computa penalização de entropia baseada em estados quânticos."""
        if quantum_states.dim() < 3:
            return torch.tensor(0.0, device=self.device)

        # Calcular entropia baseada na magnitude dos componentes quaterniônicos
        magnitudes = torch.norm(quantum_states, dim=-1)  # [batch, seq, embed]

        # Normalizar para distribuição de probabilidade
        probs = F.softmax(magnitudes.view(-1), dim=-1)

        # Calcular entropia
        epsilon = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + epsilon))

        # Penalização baseada na distância do alvo
        entropy_penalty = torch.abs(entropy - self.target_entropy)
        return entropy_penalty

    def _compute_fractal_penalty(self, quantum_states: Optional[torch.Tensor]) -> torch.Tensor:
        """Computa penalização por violação de propriedades fractais."""
        if quantum_states is None:
            return torch.tensor(0.0, device=self.device)

        # Estimativa simplificada de dimensão fractal
        if quantum_states.dim() >= 3:
            signal = quantum_states[..., 0].abs().mean(dim=0)  # Média sobre batch
            spectrum = torch.fft.fft(signal)
            power_spectrum = torch.abs(spectrum) ** 2

            freqs = torch.arange(len(power_spectrum), dtype=torch.float32, device=self.device)
            freqs = freqs + 1e-10

            if len(freqs) > 1:
                log_freqs = torch.log(freqs[:10])
                log_power = torch.log(power_spectrum[:10] + 1e-10)

                if len(log_freqs) > 1:
                    beta = (log_power[-1] - log_power[0]) / (log_freqs[-1] - log_freqs[0])
                    beta = torch.clamp(beta, -3.0, 0.0)
                    D = (3.0 + beta) / 2.0
                    fractal_penalty = torch.abs(D - self.target_fractal_dim)
                    return fractal_penalty

        return torch.tensor(0.0, device=self.device)


# Mantém compatibilidade com a versão anterior
class PhysicsInformedLoss(nn.Module):
    """
    Versão legada da PhysicsInformedLoss (para compatibilidade).

    Use SpectralPhysicsInformedLoss para a implementação física pura.
    """

    def __init__(self,
                 lambda_unitarity: float = 0.1,
                 lambda_entropy: float = 0.05,
                 lambda_fractal: float = 0.02,
                 vocab_size: int = 50257,
                 device: str = 'cpu'):
        super().__init__()
        self.lambda_unitarity = lambda_unitarity
        self.lambda_entropy = lambda_entropy
        self.lambda_fractal = lambda_fractal
        self.vocab_size = vocab_size
        self.device = device

        # Componente semântica padrão (CrossEntropy)
        self.semantic_loss = nn.CrossEntropyLoss()

        # Parâmetros físicos aprendíveis
        self.register_buffer('target_fractal_dim', torch.tensor(1.5))
        self.register_buffer('target_entropy', torch.tensor(1.0))

    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                quantum_states: Optional[torch.Tensor] = None,
                attention_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        # ========== L_SEMANTIC: Perda Semântica (CrossEntropy) ==========
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, self.vocab_size)
        flat_labels = shift_labels.view(-1)

        valid_mask = (flat_labels != -100)
        if valid_mask.any():
            L_semantic = self.semantic_loss(flat_logits[valid_mask], flat_labels[valid_mask])
        else:
            L_semantic = torch.tensor(0.0, device=self.device)

        # ========== L_UNITARITY: Penalização de Unitaridade ==========
        L_unitarity = self._compute_unitarity_penalty(attention_weights)

        # ========== L_ENTROPY: Penalização de Entropia ==========
        L_entropy = self._compute_entropy_penalty(logits)

        # ========== L_FRACTAL: Penalização Fractal ==========
        L_fractal = self._compute_fractal_penalty(quantum_states)

        # ========== L_TOTAL: Combinação Fisicamente Informada ==========
        L_total = L_semantic + \
                 self.lambda_unitarity * L_unitarity + \
                 self.lambda_entropy * L_entropy + \
                 self.lambda_fractal * L_fractal

        components = {
            'L_semantic': L_semantic.item(),
            'L_unitarity': L_unitarity.item(),
            'L_entropy': L_entropy.item(),
            'L_fractal': L_fractal.item(),
            'L_total': L_total.item()
        }

        return L_total, components

    def _compute_unitarity_penalty(self, attention_weights: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Computa penalização por violação de unitariedade.

        A unitariedade garante conservação de energia quântica.
        Penaliza quando ||A·A† - I|| > ε
        """
        if attention_weights is None:
            return torch.tensor(0.0, device=self.device)

        # Para cada head de atenção, verificar unitariedade aproximada
        # A†·A deveria ser próximo de I (matriz identidade)
        batch_size, n_heads, seq_len, _ = attention_weights.shape

        unitarity_penalty = 0.0

        for head in range(n_heads):
            A = attention_weights[:, head, :, :]  # [batch, seq, seq]

            # A†·A (aproximado - atenção é softmax, não unitária)
            # Para penalização, usamos ||A - A†|| como proxy
            A_transpose = A.transpose(-2, -1)
            asymmetry = torch.norm(A - A_transpose, dim=(-2, -1))  # Norma Frobenius da diferença

            # Penalização média sobre batch
            unitarity_penalty += asymmetry.mean()

        unitarity_penalty /= max(n_heads, 1)

        return unitarity_penalty

    def _compute_entropy_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Computa penalização por entropia inadequada.

        A entropia deve estar na faixa ótima para geração criativa vs determinística.
        Penaliza quando H(logits) está muito longe do alvo.
        """
        # Calcular entropia das distribuições de probabilidade
        probs = F.softmax(logits, dim=-1)

        # Entropia: H = -∑p_i·log(p_i)
        epsilon = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)

        # Entropia média sobre batch e sequência
        mean_entropy = entropy.mean()

        # Penalização baseada na distância do alvo
        entropy_penalty = torch.abs(mean_entropy - self.target_entropy)

        return entropy_penalty

    def _compute_fractal_penalty(self, quantum_states: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Computa penalização por violação de propriedades fractais.

        A dimensão fractal deve estar próxima de valores físicos realistas (1.0-2.0).
        Penaliza quando D calculada viola leis físicas.
        """
        if quantum_states is None:
            return torch.tensor(0.0, device=self.device)

        # Estimar dimensão fractal via análise de potência
        # P(k) ~ k^(-β) → D = (3-β)/2
        fractal_penalty = self._estimate_fractal_dimension_penalty(quantum_states)

        return fractal_penalty

    def _estimate_fractal_dimension_penalty(self, quantum_states: torch.Tensor) -> torch.Tensor:
        """
        Estima penalização baseada na dimensão fractal dos estados quânticos.
        """
        # Análise espectral simplificada
        # FFT sobre dimensão temporal (sequência)
        if quantum_states.dim() >= 3:  # [batch, seq, features, ...]
            # Pegar magnitude do sinal
            signal = quantum_states[..., 0].abs()  # Usar primeira componente

            # FFT sobre dimensão da sequência
            spectrum = torch.fft.fft(signal, dim=1)
            power_spectrum = torch.abs(spectrum) ** 2

            # Frequências
            freqs = torch.arange(signal.shape[1], device=self.device, dtype=torch.float32)
            freqs = freqs + 1e-10  # Evitar log(0)

            # Power-law fitting simplificado
            # Usar apenas algumas frequências para estimativa
            log_freqs = torch.log(freqs[:min(10, len(freqs))])
            log_power = torch.log(power_spectrum[:, :min(10, len(freqs))].mean(dim=0) + 1e-10)

            # Coeficiente angular aproximado
            if len(log_freqs) > 1:
                beta = (log_power[-1] - log_power[0]) / (log_freqs[-1] - log_freqs[0])
                beta = torch.clamp(beta, -3.0, 0.0)  # Limitar range físico

                # Dimensão fractal: D = (3 - β) / 2
                D = (3.0 + beta) / 2.0  # Note: β é negativo, então + beta

                # Penalização baseada na distância do alvo
                fractal_penalty = torch.abs(D - self.target_fractal_dim)
            else:
                fractal_penalty = torch.tensor(0.0, device=self.device)
        else:
            fractal_penalty = torch.tensor(0.0, device=self.device)

        return fractal_penalty


def create_physics_informed_loss(lambda_unitarity: float = 0.1,
                                lambda_entropy: float = 0.05,
                                lambda_fractal: float = 0.02,
                                vocab_size: int = 50257,
                                device: str = 'cpu') -> PhysicsInformedLoss:
    """
    Factory function para criar PhysicsInformedLoss (versão legada).

    Args:
        lambda_unitarity: Peso para penalização de unitariedade
        lambda_entropy: Peso para penalização de entropia
        lambda_fractal: Peso para penalização fractal
        vocab_size: Tamanho do vocabulário
        device: Dispositivo

    Returns:
        Instância de PhysicsInformedLoss configurada
    """
    return PhysicsInformedLoss(
        lambda_unitarity=lambda_unitarity,
        lambda_entropy=lambda_entropy,
        lambda_fractal=lambda_fractal,
        vocab_size=vocab_size,
        device=device
    )


def create_spectral_physics_informed_loss(physical_tokenizer: PhysicalTokenizer,
                                         lambda_unitarity: float = 0.1,
                                         lambda_entropy: float = 0.05,
                                         lambda_fractal: float = 0.02,
                                         device: str = 'cpu') -> SpectralPhysicsInformedLoss:
    """
    Factory function para criar SpectralPhysicsInformedLoss (versão física pura).

    Args:
        physical_tokenizer: Tokenizer físico para codificação quântica
        lambda_unitarity: Peso para penalização de unitariedade
        lambda_entropy: Peso para penalização de entropia
        lambda_fractal: Peso para penalização fractal
        device: Dispositivo

    Returns:
        Instância de SpectralPhysicsInformedLoss configurada
    """
    return SpectralPhysicsInformedLoss(
        physical_tokenizer=physical_tokenizer,
        lambda_unitarity=lambda_unitarity,
        lambda_entropy=lambda_entropy,
        lambda_fractal=lambda_fractal,
        device=device
    )