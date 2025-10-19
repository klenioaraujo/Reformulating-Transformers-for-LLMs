#!/usr/bin/env python3
"""
Gerenciador de Vocabulário Quântico.

Este módulo é uma refatoração direta da lógica de vocabulário do `psiqrh.py`.
Ele é responsável por carregar o vocabulário semântico e criar uma matriz de 
representações quânticas para permitir a decodificação por similaridade.
"""

import torch
import torch.nn as nn
import json
import math
from typing import Dict, List, Tuple

class QuantumVocabularyManager(nn.Module):
    """
    Carrega e gerencia as representações quânticas pré-computadas para o vocabulário.
    """
    def __init__(self, vocab_path: str, representations_path: str, embed_dim: int, device: str):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim

        # 1. Carregar o vocabulário semântico (JSON)
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            self.id_to_word: Dict[int, str] = {int(v): k for k, v in vocab_data['token_to_id'].items()}
            self.vocab_size = len(self.id_to_word)
            print(f"✅ [VocabManager] Mapeamento de vocabulário carregado: {self.vocab_size} tokens.")
        except Exception as e:
            raise IOError(f"Falha ao carregar o vocabulário de {vocab_path}: {e}")

        # 2. Carregar as representações quânticas pré-computadas (Tensor)
        try:
            print(f"⚡️ [VocabManager] Carregando representações quânticas de: {representations_path}")
            self.quantum_representations = torch.load(representations_path, map_location=self.device)
            if self.quantum_representations.shape[0] != self.vocab_size or self.quantum_representations.shape[1] != self.embed_dim:
                raise ValueError(f"Incompatibilidade de dimensão! Vocab: {self.vocab_size} tokens, Embed dim: {self.embed_dim}. Tensor: {self.quantum_representations.shape}")
            print(f"✅ [VocabManager] Representações carregadas. Shape: {self.quantum_representations.shape}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo de representações quânticas não encontrado em '{representations_path}'. Execute o script 'scripts/build_vocab.py' primeiro.")
        except Exception as e:
            raise IOError(f"Falha ao carregar o tensor de representações de {representations_path}: {e}")

    def decode_state(self, psi_final_abstract: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Decodifica um estado quântico abstrato encontrando o token mais similar na matriz de vocabulário.
        Esta é a implementação correta da decodificação, baseada em similaridade no espaço quântico.
        """
        if psi_final_abstract.dim() != 1 or psi_final_abstract.shape[0] != self.embed_dim:
            raise ValueError(f"Estado quântico final inválido. Esperado shape [{self.embed_dim}], mas recebido {psi_final_abstract.shape}")

        # Normalizar o estado de entrada para a comparação
        psi_norm = psi_final_abstract / (torch.norm(psi_final_abstract) + 1e-9)

        # Calcular similaridade de cosseno com toda a matriz de vocabulário de uma vez
        # Achatamos as representações para o cálculo
        vocab_flat = self.quantum_representations.view(self.vocab_size, -1)
        vocab_norms = vocab_flat / (torch.norm(vocab_flat, dim=1, keepdim=True) + 1e-9)
        psi_flat = psi_norm.view(1, -1)

        # Produto de ponto para obter similaridade de cosseno
        similarities = torch.matmul(vocab_norms, psi_flat.T).squeeze()

        # Obter os top_k resultados
        top_scores, top_indices = torch.topk(similarities, top_k)

        results = []
        for i in range(top_k):
            token_id = top_indices[i].item()
            score = top_scores[i].item()
            word = self.id_to_word.get(token_id, "<UNK>")
            results.append((word, score))
            
        return results
