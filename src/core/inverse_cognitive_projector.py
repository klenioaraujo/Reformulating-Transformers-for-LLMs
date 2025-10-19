#!/usr/bin/env python3
"""
Inverse Cognitive Projector - A Balança de Calibragem Aprendível
================================================================

Implementa a "Balança de Calibragem" da arquitetura de aprendizado de ponta a ponta.
Esta rede neural aprendível substitui o processo inverso fixo, aprendendo a traduzir
estados de pensamento abstratos de volta para o espaço da linguagem.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class InverseCognitiveProjector(nn.Module):
    """
    Balança de Calibragem - Tradutor Aprendível de Pensamentos Abstratos

    Esta rede neural aprende a mapear estados quânticos abstratos Ψ_final
    de volta para representações quânticas puras do espaço da linguagem.

    Arquitetura:
    1. Processamento quântico não-linear (camadas com ativações complexas)
    2. Projeção para espaço de vocabulário
    3. Similaridade com representações quânticas do dicionário
    """

    def __init__(self, embed_dim: int = 64, vocab_size: int = 50257, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Camadas de processamento quântico (aprendíveis)
        self.quantum_processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_layers)
        ])

        # Camada de projeção para espaço de vocabulário
        self.vocab_projection = nn.Linear(hidden_dim, vocab_size)

        # Camada de refinamento quântico (para ajuste fino)
        self.quantum_refinement = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Camada de expansão quaterniônica
        self.quaternion_expansion = nn.Linear(embed_dim, embed_dim * 4)

        # Mecanismo de confiança (para avaliar qualidade da tradução)
        self.confidence_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Inicialização de pesos
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicialização especializada para processamento quântico"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Inicialização Xavier para estabilidade quântica
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, psi_final: torch.Tensor, quantum_vocab: Optional[torch.Tensor] = None,
                return_confidence: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Traduz estado de pensamento abstrato para representação linguística

        Args:
            psi_final: Estado quântico final abstrato [batch, embed_dim] ou [embed_dim]
            quantum_vocab: Dicionário quântico [vocab_size, embed_dim, 4] (opcional)
            return_confidence: Se deve retornar score de confiança

        Returns:
            Tuple de (psi_reconstructed, confidence_score)
            psi_reconstructed: Estado reconstruído [batch, embed_dim] ou [embed_dim]
        """
        # Garantir formato batch
        if psi_final.dim() == 1:
            psi_final = psi_final.unsqueeze(0)  # [1, embed_dim]
            was_single = True
        else:
            was_single = False

        batch_size = psi_final.shape[0]

        # ========== PROCESSAMENTO QUÂNTICO NÃO-LINEAR ==========
        x = psi_final
        for layer in self.quantum_processor:
            x = layer(x)

        # ========== PROJEÇÃO PARA ESPAÇO DE VOCABULÁRIO ==========
        vocab_logits = self.vocab_projection(x)  # [batch, vocab_size]

        # Aplicar softmax para obter distribuição de probabilidade
        vocab_probs = F.softmax(vocab_logits, dim=-1)  # [batch, vocab_size]

        # ========== RECONSTRUÇÃO QUÂNTICA ==========
        # Se temos dicionário quântico, usar para reconstrução precisa
        if quantum_vocab is not None:
            # quantum_vocab: [vocab_size, embed_dim, 4]
            vocab_size, embed_dim, quat_dim = quantum_vocab.shape

            # Para cada item no batch, encontrar melhor reconstrução
            reconstructed_batch = []
            confidence_batch = []

            for b in range(batch_size):
                probs_b = vocab_probs[b]  # [vocab_size]

                # Calcular reconstrução ponderada baseada nas probabilidades
                # Usar as top-k representações mais prováveis para melhor qualidade
                top_k = min(10, vocab_size)  # Top 10 para balancear qualidade/complexidade
                top_probs, top_indices = torch.topk(probs_b, top_k)

                # Reconstrução ponderada
                weighted_sum = torch.zeros(embed_dim, quat_dim, device=psi_final.device)
                total_weight = 0.0

                for prob, idx in zip(top_probs, top_indices):
                    quantum_state = quantum_vocab[idx]  # [embed_dim, 4]
                    weighted_sum += prob * quantum_state
                    total_weight += prob

                if total_weight > 0:
                    psi_reconstructed = weighted_sum / total_weight  # [embed_dim, 4]
                else:
                    psi_reconstructed = torch.zeros(embed_dim, quat_dim, device=psi_final.device)

                # Calcular confiança baseada na concentração das probabilidades
                entropy = -torch.sum(top_probs * torch.log(top_probs + 1e-10))
                max_prob = torch.max(top_probs)
                confidence = max_prob * (1.0 - entropy / torch.log(torch.tensor(float(top_k))))

                reconstructed_batch.append(psi_reconstructed)
                confidence_batch.append(confidence)

            psi_reconstructed = torch.stack(reconstructed_batch)  # [batch, embed_dim, 4]
            confidence_scores = torch.stack(confidence_batch)  # [batch]
        else:
            # Fallback: usar projeção aprendível sem dicionário quântico
            psi_reconstructed = self.quantum_refinement(vocab_logits)  # [batch, embed_dim]
            confidence_scores = self.confidence_net(psi_reconstructed).squeeze(-1)  # [batch]

        # ========== REFINAMENTO FINAL ==========
        # Aplicar refinamento quântico adicional apenas quando não há dicionário quântico
        if quantum_vocab is None:
            # Para formato quântico, manter estrutura
            psi_reconstructed = self.quaternion_expansion(psi_reconstructed).view(batch_size, self.embed_dim, 4)

        # Remover dimensão batch se entrada era single
        if was_single:
            psi_reconstructed = psi_reconstructed.squeeze(0)
            if return_confidence:
                confidence_scores = confidence_scores.squeeze(0)

        if return_confidence:
            return psi_reconstructed, confidence_scores
        else:
            return psi_reconstructed

    def compute_loss(self, psi_predicted: torch.Tensor, psi_target: torch.Tensor,
                    confidence_weight: float = 0.1) -> torch.Tensor:
        """
        Calcula perda para treinamento da Balança de Calibragem

        Args:
            psi_predicted: Estado previsto pela rede [batch, embed_dim] ou [batch, embed_dim, 4]
            psi_target: Estado alvo (representação quântica pura) [batch, embed_dim] ou [batch, embed_dim, 4]
            confidence_weight: Peso para termo de confiança

        Returns:
            Perda total escalar
        """
        # Garantir formatos compatíveis
        if psi_predicted.shape != psi_target.shape:
            # Se um é quântico e outro não, tentar compatibilizar
            if psi_predicted.dim() == 3 and psi_target.dim() == 2:
                # psi_predicted é [batch, embed_dim, 4], psi_target é [batch, embed_dim]
                # Usar média sobre dimensão quântica
                psi_predicted = psi_predicted.mean(dim=-1)
            elif psi_predicted.dim() == 2 and psi_target.dim() == 3:
                # Inverso
                psi_target = psi_target.mean(dim=-1)

        # Perda principal: MSE entre estados quânticos
        mse_loss = F.mse_loss(psi_predicted, psi_target)

        # Perda de confiança: incentivar alta confiança para previsões corretas
        if hasattr(self, 'confidence_net') and confidence_weight > 0:
            confidence_scores = self.confidence_net(psi_predicted.mean(dim=-1, keepdim=True)).squeeze(-1)
            confidence_loss = -torch.log(confidence_scores + 1e-10).mean()
            total_loss = mse_loss + confidence_weight * confidence_loss
        else:
            total_loss = mse_loss

        return total_loss


def create_inverse_cognitive_projector(embed_dim: int = 256, vocab_size: int = 50257,
                                      hidden_dim: int = 512, num_layers: int = 3,
                                      dropout: float = 0.1) -> InverseCognitiveProjector:
    """
    Factory function para criar Inverse Cognitive Projector.

    Args:
        embed_dim: Dimensão de embedding
        vocab_size: Tamanho do vocabulário
        hidden_dim: Dimensão oculta das camadas
        num_layers: Número de camadas de processamento
        dropout: Taxa de dropout

    Returns:
        InverseCognitiveProjector configurado
    """
    return InverseCognitiveProjector(embed_dim, vocab_size, hidden_dim, num_layers, dropout)