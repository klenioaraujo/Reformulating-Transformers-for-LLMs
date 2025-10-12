#!/usr/bin/env python3
"""
Context Funnel - Mecanismo de Atenção Espectral para Histórico de Conversa
==========================================================================

Implementa o "Funil de Contexto" da arquitetura de aprendizado de ponta a ponta.
Este módulo usa atenção espectral (doe.md) para condensar o histórico de conversa
em um único vetor de contexto focado.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class SpectralAttention(nn.Module):
    """
    Atenção Espectral baseada em doe.md - Mecanismo de Foco Quântico

    Implementa atenção através de transformadas de Fourier e filtragem espectral,
    permitindo que o sistema "preste atenção" a padrões de frequência específicos
    no histórico de conversa.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, alpha: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.alpha = alpha

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Projeções lineares para Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Projeção de saída
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Filtro espectral aprendível
        self.spectral_filter = nn.Parameter(torch.randn(embed_dim))

    def spectral_filter_kernel(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Gera kernel de filtro espectral baseado em doe.md
        F(k) = exp(i α · arctan(ln(|k| + ε)))
        """
        k = torch.arange(seq_len, dtype=torch.float32, device=device)
        k = k + 1e-10  # Evitar log(0)

        # Filtro espectral (doe.md)
        epsilon = 1e-10
        filter_kernel = torch.exp(1j * self.alpha * torch.arctan(torch.log(k + epsilon)))

        return filter_kernel

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Atenção Espectral: Aplica transformada de Fourier antes da atenção

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Attention mask [batch_size, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape

        # ========== TRANSFORMAÇÃO ESPECTRAL ==========
        # Aplicar FFT no domínio da sequência
        x_fft = torch.fft.fft(x, dim=1)  # [batch, seq_len, embed_dim]

        # Aplicar filtro espectral aprendível
        filter_kernel = self.spectral_filter_kernel(seq_len, x.device)
        filter_kernel = filter_kernel.unsqueeze(-1).expand(-1, embed_dim)  # [seq_len, embed_dim]

        x_filtered = x_fft * filter_kernel.unsqueeze(0)  # Broadcasting

        # IFFT de volta ao domínio temporal
        x_spectral = torch.fft.ifft(x_filtered, dim=1).real

        # ========== ATENÇÃO MULTI-HEAD ==========
        # Projeções Q, K, V
        Q = self.q_proj(x_spectral).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_spectral).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_spectral).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Similaridade de atenção
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Aplicar máscara se fornecida
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Aplicar atenção aos valores
        attn_output = torch.matmul(attn_weights, V)

        # Concatenar heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Projeção final
        output = self.out_proj(attn_output)

        return output


class ContextFunnel(nn.Module):
    """
    Funil de Contexto - Condensador de Histórico de Conversa

    Este módulo toma o histórico de conversa (sequência de estados quânticos)
    e produz um único vetor de contexto focado através de atenção espectral.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, max_history: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_history = max_history

        # Camada de atenção espectral
        self.spectral_attention = SpectralAttention(embed_dim, num_heads)

        # Camada de condensação (pooling aprendível)
        self.condensation_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )

        # Normalização
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Mecanismo de gating para controle de foco
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, conversation_history: List[torch.Tensor]) -> torch.Tensor:
        """
        Processa histórico de conversa e gera contexto focado

        Args:
            conversation_history: Lista de tensores quânticos [embed_dim, 4] ou [embed_dim]

        Returns:
            Contexto condensado [embed_dim]
        """
        if not conversation_history:
            # Sem histórico - retornar vetor zero (aprendível)
            return torch.zeros(self.embed_dim, device=next(self.parameters()).device)

        # Limitar histórico ao máximo permitido
        if len(conversation_history) > self.max_history:
            conversation_history = conversation_history[-self.max_history:]

        # Preparar sequência para atenção
        # Cada estado quântico pode ter forma diferente, precisamos padronizar
        processed_states = []
        for state in conversation_history:
            if state.dim() == 1:
                # Estado 1D - expandir para formato quântico
                processed_state = state.unsqueeze(-1).expand(-1, 4)
            elif state.dim() == 2 and state.shape[-1] == 4:
                # Já está no formato quântico
                processed_state = state
            else:
                # Outros formatos - flatten e padronizar
                processed_state = state.flatten()
                if len(processed_state) < self.embed_dim:
                    # Padding
                    padding = torch.zeros(self.embed_dim - len(processed_state), device=state.device)
                    processed_state = torch.cat([processed_state, padding])
                elif len(processed_state) > self.embed_dim:
                    # Truncar
                    processed_state = processed_state[:self.embed_dim]
                processed_state = processed_state.unsqueeze(-1).expand(-1, 4)

            # Projetar para embed_dim (se necessário)
            if processed_state.shape[0] != self.embed_dim:
                # Usar média simples para reduzir dimensão
                processed_state = processed_state.mean(dim=0, keepdim=True).expand(self.embed_dim, -1)

            processed_states.append(processed_state)

        # Stack em sequência [seq_len, embed_dim, 4]
        seq_tensor = torch.stack(processed_states, dim=0)  # [seq_len, embed_dim, 4]

        # Flatten último dimensão para atenção
        seq_flat = seq_tensor.view(seq_tensor.shape[0], -1)  # [seq_len, embed_dim * 4]

        # Se dimensão não corresponde, projetar
        if seq_flat.shape[-1] != self.embed_dim:
            # Projeção linear simples
            proj = nn.Linear(seq_flat.shape[-1], self.embed_dim).to(seq_flat.device)
            seq_flat = proj(seq_flat)

        # Adicionar dimensão batch
        seq_batch = seq_flat.unsqueeze(0)  # [1, seq_len, embed_dim]

        # Aplicar atenção espectral
        attended_seq = self.spectral_attention(seq_batch)  # [1, seq_len, embed_dim]

        # Condensar sequência em vetor único
        # Usar média ponderada aprendível
        seq_squeezed = attended_seq.squeeze(0)  # [seq_len, embed_dim]

        # Pesos de condensação aprendíveis
        condensation_weights = self.condensation_net(seq_squeezed)  # [seq_len, embed_dim]
        condensation_weights = F.softmax(condensation_weights.mean(dim=-1), dim=0)  # [seq_len]

        # Aplicar pesos
        context_vector = torch.sum(seq_squeezed * condensation_weights.unsqueeze(-1), dim=0)  # [embed_dim]

        # Aplicar gating para controle de foco
        gate_value = self.gate_net(context_vector)  # [1]
        context_vector = context_vector * gate_value

        # Normalização final
        context_vector = self.layer_norm(context_vector)

        return context_vector


def create_context_funnel(embed_dim: int = 256, num_heads: int = 8, max_history: int = 10) -> ContextFunnel:
    """
    Factory function para criar Context Funnel.

    Args:
        embed_dim: Dimensão de embedding
        num_heads: Número de heads de atenção
        max_history: Máximo de turnos de histórico

    Returns:
        ContextFunnel configurado
    """
    return ContextFunnel(embed_dim, num_heads, max_history)