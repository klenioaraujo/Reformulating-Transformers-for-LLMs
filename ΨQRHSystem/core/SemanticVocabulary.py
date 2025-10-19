#!/usr/bin/env python3
"""
Módulo de Vocabulário Semântico para o sistema ΨQRH.
"""

import torch
import torch.nn as nn
import json
from typing import Dict, Optional

class SemanticVocabulary(nn.Module):
    """
    Carrega e gerencia um vocabulário semântico pré-treinado (como o do GPT-2)
    e fornece embeddings para os tokens.

    Esta classe substitui a abordagem algorítmica da QuantumWordMatrix por um
    embedding lookup direto, que é a abordagem correta e comprovada do
    pipeline legado (psiqrh.py).
    """
    def __init__(self, vocab_path: str, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        print(f"📚 [SemanticVocabulary] Carregando vocabulário de: {vocab_path}")
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
        except Exception as e:
            raise IOError(f"Não foi possível ler o arquivo de vocabulário em {vocab_path}: {e}")

        if not isinstance(vocab_data, dict) or 'token_to_id' not in vocab_data:
            raise ValueError("Formato de vocabulário inválido. Esperado um JSON com a chave 'token_to_id'.")

        self.word_to_id: Dict[str, int] = vocab_data['token_to_id']
        self.id_to_word: Dict[int, str] = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)

        # Camada de embedding que armazena os vetores para cada token.
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        print(f"✅ [SemanticVocabulary] Vocabulário carregado com sucesso: {self.vocab_size} tokens.")

    def get_id_to_word_map(self) -> Dict[int, str]:
        """Retorna o mapeamento de ID para palavra."""
        return self.id_to_word

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Retorna os embeddings para uma lista de IDs de token.
        """
        return self.embedding(token_ids)
