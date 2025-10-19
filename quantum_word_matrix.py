#!/usr/bin/env python3
"""
Matriz Quântica de Conversão para Palavras (Versão Refatorada)
=========================================================

Esta versão foi corrigida para abandonar a geração de representações algorítmicas
e utilizar uma camada de embedding padrão (torch.nn.Embedding), que é a abordagem
correta para representar vocabulários semânticos. A decodificação é feita via
busca por similaridade de cosseno, alinhando o comportamento com a referência
do pipeline legado, mas dentro da nova arquitetura modular.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class QuantumWordMatrix(nn.Module):
    """
    Gerencia o vocabulário semântico através de uma camada de embedding e fornece
    funcionalidades de codificação e decodificação baseadas em similaridade.
    """

    def __init__(self, 
                 embed_dim: int, 
                 device: str, 
                 word_to_id: Dict[str, int], 
                 id_to_word: Dict[int, str]):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim

        if not word_to_id or not id_to_word:
            raise ValueError("Os dicionários de vocabulário (word_to_id, id_to_word) não podem ser vazios.")

        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.vocab_size = len(word_to_id)

        # A abordagem correta: usar uma camada de embedding para representar as palavras.
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        print(f"✅ [QuantumWordMatrix] Camada de Embedding criada com sucesso para {self.vocab_size} tokens.")

        self.to(device)

    def encode_word(self, word: str) -> torch.Tensor:
        """Codifica uma palavra para seu vetor de embedding."""
        word_id = self.word_to_id.get(word, 0) # Default para token desconhecido
        token_id_tensor = torch.tensor([word_id], dtype=torch.long, device=self.device)
        return self.embedding(token_id_tensor).squeeze(0)

    def decode_quantum_state(self, quantum_state: torch.Tensor, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Decodifica um estado quântico encontrando a(s) palavra(s) mais similar(es)
        no espaço de embedding via similaridade de cosseno.
        """
        if quantum_state.dim() == 0 or quantum_state.numel() != self.embed_dim:
             # Se o estado for inválido, retorna um token de erro
            print(f"⚠️  Estado quântico para decodificação tem dimensão inválida: {quantum_state.shape}")
            return [("<DECODE_ERROR>", 0.0)]

        # Normalizar o estado de entrada
        state_norm = F.normalize(quantum_state.unsqueeze(0), p=2, dim=1)

        # Normalizar todos os vetores do vocabulário
        embedding_matrix = self.embedding.weight
        matrix_norm = F.normalize(embedding_matrix, p=2, dim=1)

        # Calcular similaridade de cosseno entre o estado e todos os vetores
        similarities = torch.matmul(state_norm, matrix_norm.T).squeeze(0)

        # Obter os top_k resultados
        top_scores, top_indices = torch.topk(similarities, top_k)

        # Mapear de volta para palavras
        results = []
        for i in range(top_k):
            # Extrair token_id corretamente baseado no shape dos tensores
            if top_indices.dim() == 1:
                token_id = top_indices[i].item()
            else:
                token_id = top_indices[0][i].item()

            # Extrair score corretamente baseado no shape dos tensores
            if top_scores.dim() == 1:
                score = top_scores[i].item()
            else:
                score = top_scores[0][i].item()

            word = self.id_to_word.get(token_id, "<UNK>")
            results.append((word, score))
            
        return results

    @property
    def quantum_representations(self):
        """
        Propriedade para compatibilidade com CognitiveEngine.
        Retorna a matriz de embeddings como representações quânticas.
        """
        return self.embedding.weight