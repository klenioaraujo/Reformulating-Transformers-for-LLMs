#!/usr/bin/env python3
"""
DynamicQuantumCharacterMatrix - Matriz quântica dinâmica baseada em caracteres

Implementa uma matriz quântica que opera no nível de caracteres em vez de palavras,
permitindo geração mais granular e controle preciso sobre a saída de texto.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, List, Tuple


class DynamicQuantumCharacterMatrix(nn.Module):
    """
    Matriz quântica dinâmica baseada em caracteres.

    Esta matriz opera no nível de caracteres individuais, permitindo:
    - Geração mais granular de texto
    - Melhor controle sobre caracteres especiais
    - Vocabulário menor e mais eficiente
    - Representação quântica de caracteres ASCII/Unicode
    """

    def __init__(self,
                 char_vocab_size: int = 256,  # ASCII básico + caracteres especiais
                 hidden_size: int = 256,      # Deve ser múltiplo de 4 para quaternions
                 device: str = 'cpu'):

        super().__init__()
        self.char_vocab_size = char_vocab_size
        self.hidden_size = hidden_size
        self.device = device

        # Verificar se hidden_size é múltiplo de 4 para quaternions
        if hidden_size % 4 != 0:
            raise ValueError(f"hidden_size deve ser múltiplo de 4 para quaternions, recebido: {hidden_size}")

        # Dimensão quaterniônica
        self.quaternion_dim = hidden_size // 4

        # Embeddings quânticos de caracteres
        self.char_embeddings = nn.Embedding(char_vocab_size, hidden_size)

        # Camada de rotação SO(4) para quaternions
        self.rotation_layer = nn.Linear(hidden_size, hidden_size, bias=False)

        # Inicializar pesos com distribuição normal
        self._initialize_weights()

        print(f"🔬 Dynamic Quantum Character Matrix inicializada")
        print(f"   📊 Vocab: {char_vocab_size} caracteres, Hidden: {hidden_size} (quaternion_dim: {self.quaternion_dim})")
        print(f"   🔄 Camada de rotação SO(4): Implementada com multiplicação quaterniônica")

    def _initialize_weights(self):
        """Inicializa os pesos da matriz com distribuição normal."""
        # Inicializar embeddings de caracteres
        nn.init.normal_(self.char_embeddings.weight, mean=0.0, std=0.02)

        # Inicializar camada de rotação
        nn.init.orthogonal_(self.rotation_layer.weight)

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiplicação quaterniônica para dois tensores.

        Args:
            q1: Primeiro quaternion [batch_size, hidden_size]
            q2: Segundo quaternion [batch_size, hidden_size]

        Returns:
            Produto quaterniônico [batch_size, hidden_size]
        """
        # Reorganizar para dimensão quaterniônica
        q1_reshaped = q1.view(-1, self.quaternion_dim, 4)
        q2_reshaped = q2.view(-1, self.quaternion_dim, 4)

        # Extrair componentes
        a1, b1, c1, d1 = q1_reshaped.unbind(dim=2)
        a2, b2, c2, d2 = q2_reshaped.unbind(dim=2)

        # Multiplicação quaterniônica
        a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

        # Combinar componentes
        result = torch.stack([a, b, c, d], dim=2)
        return result.view(-1, self.hidden_size)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da matriz quântica de caracteres.

        Args:
            char_ids: IDs de caracteres [batch_size, seq_len]

        Returns:
            Representações quânticas [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = char_ids.shape

        # Obter embeddings de caracteres
        char_embeds = self.char_embeddings(char_ids)  # [batch_size, seq_len, hidden_size]

        # Aplicar rotação quaterniônica
        rotated_embeds = self.rotation_layer(char_embeds)

        # Aplicar multiplicação quaterniônica
        char_embeds_flat = char_embeds.view(-1, self.hidden_size)
        rotated_embeds_flat = rotated_embeds.view(-1, self.hidden_size)

        quantum_embeds = self._quaternion_multiply(char_embeds_flat, rotated_embeds_flat)
        quantum_embeds = quantum_embeds.view(batch_size, seq_len, self.hidden_size)

        return quantum_embeds

    def get_character_vocabulary(self) -> Dict[str, int]:
        """
        Retorna o vocabulário de caracteres padrão.

        Returns:
            Dicionário mapeando caracteres para IDs
        """
        # Caracteres ASCII básicos + caracteres especiais
        char_vocab = {}

        # Caracteres ASCII imprimíveis (32-126) - começando do ID 1
        for i in range(32, 127):
            char_vocab[chr(i)] = i - 31  # IDs de 1 a 95

        # Adicionar caracteres de controle importantes com IDs específicos
        control_chars = {
            chr(0): 0,   # Null character
            chr(1): 1,   # Start of header
            chr(10): 96, # New line (\n)
            chr(9): 97,  # Tab (\t)
            chr(32): 98  # Space
        }
        char_vocab.update(control_chars)

        # Caracteres especiais adicionais - continuando dos IDs
        special_chars = {
            'á': 99, 'é': 100, 'í': 101, 'ó': 102, 'ú': 103,  # Acentos
            'à': 104, 'è': 105, 'ì': 106, 'ò': 107, 'ù': 108,
            'â': 109, 'ê': 110, 'î': 111, 'ô': 112, 'û': 113,
            'ã': 114, 'õ': 115, 'ç': 116,
            'Á': 117, 'É': 118, 'Í': 119, 'Ó': 120, 'Ú': 121,
            'À': 122, 'È': 123, 'Ì': 124, 'Ò': 125, 'Ù': 126,
            'Â': 127, 'Ê': 128, 'Î': 129, 'Ô': 130, 'Û': 131,
            'Ã': 132, 'Õ': 133, 'Ç': 134
        }

        char_vocab.update(special_chars)

        # Garantir que não exceda o tamanho do vocabulário
        if len(char_vocab) > self.char_vocab_size:
            # Manter apenas os primeiros char_vocab_size caracteres
            char_vocab = dict(list(char_vocab.items())[:self.char_vocab_size])

        return char_vocab

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Codifica texto em IDs de caracteres.

        Args:
            text: Texto para codificar

        Returns:
            Tensor de IDs de caracteres [1, seq_len]
        """
        char_vocab = self.get_character_vocabulary()

        # Converter texto para IDs
        char_ids = []
        for char in text:
            if char in char_vocab:
                char_ids.append(char_vocab[char])
            else:
                # Usar espaço como fallback para caracteres desconhecidos
                char_ids.append(char_vocab.get(' ', 97))

        return torch.tensor([char_ids], dtype=torch.long, device=self.device)

    def decode_text(self, char_ids: torch.Tensor) -> str:
        """
        Decodifica IDs de caracteres em texto.

        Args:
            char_ids: Tensor de IDs de caracteres [batch_size, seq_len]

        Returns:
            Texto decodificado
        """
        char_vocab = self.get_character_vocabulary()

        # Inverter mapeamento
        id_to_char = {v: k for k, v in char_vocab.items()}

        # Converter IDs para texto
        text_chars = []
        for char_id in char_ids.cpu().numpy().flatten():
            if char_id in id_to_char:
                text_chars.append(id_to_char[char_id])
            else:
                text_chars.append('?')

        return ''.join(text_chars)

    def get_character_embeddings(self) -> torch.Tensor:
        """
        Retorna os embeddings de caracteres.

        Returns:
            Tensor de embeddings [char_vocab_size, hidden_size]
        """
        return self.char_embeddings.weight

    def analyze_character_distribution(self, text: str) -> Dict[str, Any]:
        """
        Analisa a distribuição de caracteres no texto.

        Args:
            text: Texto para analisar

        Returns:
            Dicionário com estatísticas de caracteres
        """
        char_vocab = self.get_character_vocabulary()

        # Contar frequência de caracteres
        char_counts = {}
        for char in text:
            if char in char_vocab:
                char_counts[char] = char_counts.get(char, 0) + 1

        # Calcular estatísticas
        total_chars = len(text)
        unique_chars = len(char_counts)

        return {
            'total_characters': total_chars,
            'unique_characters': unique_chars,
            'character_frequencies': char_counts,
            'vocabulary_coverage': unique_chars / len(char_vocab) if char_vocab else 0
        }