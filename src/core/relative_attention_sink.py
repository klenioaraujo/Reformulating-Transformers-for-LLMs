import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple, List, Dict


class RelativeAttentionSink(nn.Module):
    """
    Identifica e gerencia coletores de atenção baseados na posição relativa
    """
    def __init__(self, hidden_size, sink_strength=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.sink_strength = sink_strength
        self.sink_projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, positions, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            positions: [batch_size, seq_len] - posições absolutas ou relativas
            attention_mask: [batch_size, seq_len] - máscara de atenção
        Returns:
            sink_enhanced_states: Estados com coletor identificado
            sink_indices: Índices dos tokens coletores
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Encontra o token com menor posição relativa em cada sequência
        if attention_mask is not None:
            # Considera apenas tokens não mascarados
            masked_positions = positions.masked_fill(attention_mask == 0, float('inf'))
            sink_indices = torch.argmin(masked_positions, dim=-1)  # [batch_size]
        else:
            sink_indices = torch.argmin(positions, dim=-1)  # [batch_size]

        # Coleta os tokens coletores
        sink_tokens = torch.stack([
            hidden_states[i, sink_indices[i]] for i in range(batch_size)
        ])  # [batch_size, hidden_size]

        # Projeta os coletores para criar um viés de atenção
        enhanced_sink_tokens = self.sink_projection(sink_tokens)  # [batch_size, hidden_size]

        return enhanced_sink_tokens, sink_indices

    def apply_sink_bias(self, attention_scores, sink_indices, sink_strength=None):
        """
        Aplica viés de coletor aos scores de atenção
        """
        if sink_strength is None:
            sink_strength = self.sink_strength

        batch_size, num_heads, seq_len, _ = attention_scores.shape

        # Cria máscara de viés para os tokens coletores
        sink_bias = torch.zeros_like(attention_scores)
        for i in range(batch_size):
            sink_idx = sink_indices[i]
            sink_bias[i, :, :, sink_idx] = sink_strength

        return attention_scores + sink_bias


class SinkAwareAttention(nn.Module):
    """
    Mecanismo de atenção que explicitamente identifica e utiliza coletores relativos
    """
    def __init__(self, d_model, n_heads, sink_strength=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.sink_manager = RelativeAttentionSink(self.d_model, sink_strength)

    def forward(self, query, key, value, positions, attention_mask=None):
        batch_size, seq_len, hidden_dim = query.shape

        # Projeções Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Calcula scores de atenção
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Identifica coletores e aplica viés
        sink_tokens, sink_indices = self.sink_manager(query, positions, attention_mask)
        attention_scores = self.sink_manager.apply_sink_bias(attention_scores, sink_indices)

        # Aplica máscara de atenção
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        # Softmax e atenção
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, v)

        # Reorganiza e projeta saída
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        return self.out_proj(attention_output), sink_indices


class SinkAwarePsiQRHBlock(nn.Module):
    """
    Bloco Psi-QRH com consciência de coletores relativos
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, sink_strength=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.ln1 = nn.LayerNorm(hidden_size)
        self.attention = SinkAwareAttention(hidden_size, num_heads, sink_strength)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, positions, attention_mask=None):
        # Atenção com coletor relativo
        attn_out, sink_indices = self.attention(
            self.ln1(x), self.ln1(x), self.ln1(x), positions, attention_mask
        )
        x = x + self.dropout(attn_out)

        # MLP
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.dropout(mlp_out)

        return x, sink_indices


def monitor_sink_formation(hidden_states, positions, layer_idx):
    """
    Monitora a formação de coletores em diferentes camadas
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape

    # Calcula similaridade entre tokens para identificar coletores
    similarities = torch.matmul(
        hidden_states, hidden_states.transpose(-2, -1)
    ) / math.sqrt(hidden_dim)

    # Encontra tokens que recebem atenção consistente (coletores potenciais)
    attention_pattern = torch.softmax(similarities, dim=-1)
    sink_scores = attention_pattern.mean(dim=1)  # Média de atenção recebida

    # Identifica coletor baseado na posição
    actual_sink_indices = torch.argmin(positions, dim=-1)

    print(f"Camada {layer_idx}:")
    print(f"  - Coletor identificado: posições {actual_sink_indices.cpu().numpy()}")
    print(f"  - Score do coletor: {sink_scores[range(batch_size), actual_sink_indices].cpu().numpy()}")

    return actual_sink_indices, sink_scores


def forward_with_relative_sink(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                                 positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Modificação do forward do transformer principal para usar coletores relativos

    Args:
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        positions: [batch_size, seq_len] - posições para identificação de coletores

    Returns:
        hidden_states: Estados finais do transformer
        all_sink_indices: Lista de índices de coletores por camada
    """
    if positions is None:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

    hidden_states = self.embedding(input_ids)
    all_sink_indices = []

    for layer_idx, layer in enumerate(self.layers):
        hidden_states, sink_indices = layer(hidden_states, positions, attention_mask)
        all_sink_indices.append(sink_indices)

        # Monitora formação do coletor (opcional)
        if layer_idx in [0, 4, 8, 16, 31]:  # Camadas estratégicas
            monitor_sink_formation(hidden_states, positions, layer_idx)

    return hidden_states, all_sink_indices