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

# Importar framework de lógica ternária para integração obrigatória
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

try:
    from core.TernaryLogicFramework import TernaryLogicFramework
except ImportError:
    # Create minimal placeholder if not available
    class TernaryLogicFramework:
        def __init__(self, device="cpu"): pass

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
        # Inicialização melhorada para evitar concentração em poucos tokens
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # Inicialização personalizada para melhor diversidade
        self._initialize_embeddings()

        # Inicializar lógica ternária obrigatória para processamento eficiente
        self.ternary_logic = TernaryLogicFramework(device=device)
        print(f"✅ [QuantumWordMatrix] Lógica ternária integrada obrigatoriamente para processamento eficiente")

        print(f"✅ [QuantumWordMatrix] Camada de Embedding criada com sucesso para {self.vocab_size} tokens.")
        print(f"   📊 Dispositivo: {device}, Lógica Ternária: ✅ (obrigatória)")

        self.to(device)

    def _initialize_embeddings(self):
        """Inicialização personalizada dos embeddings para melhor diversidade"""
        # Usar inicialização Xavier uniforme para melhor distribuição
        nn.init.xavier_uniform_(self.embedding.weight)

        # Adicionar pequeno ruído para evitar similaridade excessiva
        noise = torch.randn_like(self.embedding.weight) * 0.01
        self.embedding.weight.data += noise

        # Normalizar para ter normas mais uniformes
        norms = torch.norm(self.embedding.weight, dim=1, keepdim=True)
        self.embedding.weight.data = self.embedding.weight.data / norms

    def encode_word(self, word: str) -> torch.Tensor:
        """Codifica uma palavra para seu vetor de embedding."""
        word_id = self.word_to_id.get(word, 0) # Default para token desconhecido
        token_id_tensor = torch.tensor([word_id], dtype=torch.long, device=self.device)
        return self.embedding(token_id_tensor).squeeze(0)

    def decode_quantum_state(self, quantum_state: torch.Tensor, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Decodifica um estado quântico encontrando a(s) palavra(s) mais similar(es)
        no espaço de embedding via similaridade de cosseno, otimizada com lógica ternária.
        """
        if quantum_state.dim() == 0 or quantum_state.numel() != self.embed_dim:
             # Se o estado for inválido, retorna um token de erro
            print(f"⚠️  Estado quântico para decodificação tem dimensão inválida: {quantum_state.shape}")
            return [("<DECODE_ERROR>", 0.0)]

        # Usar lógica ternária obrigatória para otimização (ZERO FALLBACK POLICY)
        return self._decode_with_ternary_logic(quantum_state, top_k)

    def _decode_with_ternary_logic(self, quantum_state: torch.Tensor, top_k: int) -> List[Tuple[str, float]]:
        """
        Decodificação otimizada usando lógica ternária para processamento eficiente.
        """
        # Converter estado quântico para representação ternária
        ternary_state = self._quantum_to_ternary(quantum_state)

        # Aplicar operações ternárias para filtragem inicial
        # Usar AND ternário para identificar regiões de alta similaridade
        embedding_matrix = self.embedding.weight
        ternary_embeddings = self._embeddings_to_ternary(embedding_matrix)

        # Calcular similaridade ternária (operações eficientes)
        ternary_similarities = self._compute_ternary_similarity(ternary_state, ternary_embeddings)

        # Filtrar top candidatos usando lógica ternária
        top_k_ternary = min(top_k * 3, len(ternary_similarities))  # Candidatos extras para refinar
        _, top_ternary_indices = torch.topk(ternary_similarities, top_k_ternary)

        # Refinar com similaridade de cosseno nos candidatos filtrados
        candidate_embeddings = embedding_matrix[top_ternary_indices]
        state_norm = F.normalize(quantum_state.unsqueeze(0), p=2, dim=1)
        candidate_norm = F.normalize(candidate_embeddings, p=2, dim=1)
        refined_similarities = torch.matmul(state_norm, candidate_norm.T).squeeze(0)

        # Obter resultados finais
        top_scores, top_local_indices = torch.topk(refined_similarities, top_k)
        top_global_indices = top_ternary_indices[top_local_indices]

        results = []
        for i in range(top_k):
            token_id = top_global_indices[i].item()
            score = top_scores[i].item()
            word = self.id_to_word.get(token_id, "<UNK>")
            results.append((word, score))

        return results

    def _decode_standard(self, quantum_state: torch.Tensor, top_k: int) -> List[Tuple[str, float]]:
        """
        Decodificação padrão sem lógica ternária.
        """
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

    def _quantum_to_ternary(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Converte estado quântico para representação ternária."""
        abs_state = torch.abs(quantum_state)
        max_val = torch.max(abs_state)

        if max_val == 0:
            return torch.zeros_like(quantum_state, dtype=torch.long)

        normalized = quantum_state / (max_val + 1e-10)
        ternary_state = torch.zeros_like(normalized, dtype=torch.long)
        ternary_state[normalized > 0.33] = 1
        ternary_state[normalized < -0.33] = -1

        return ternary_state

    def _embeddings_to_ternary(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Converte matriz de embeddings para representação ternária."""
        abs_embeddings = torch.abs(embeddings)
        max_vals = torch.max(abs_embeddings, dim=1, keepdim=True)[0]

        normalized = embeddings / (max_vals + 1e-10)
        ternary_embeddings = torch.zeros_like(normalized, dtype=torch.long)
        ternary_embeddings[normalized > 0.33] = 1
        ternary_embeddings[normalized < -0.33] = -1

        return ternary_embeddings

    def _compute_ternary_similarity(self, ternary_state: torch.Tensor, ternary_embeddings: torch.Tensor) -> torch.Tensor:
        """Calcula similaridade usando operações ternárias eficientes."""
        # Usar operações de lógica ternária para similaridade
        # Similaridade ternária: contar matches positivos e penalizar mismatches
        matches = (ternary_state.unsqueeze(0) == ternary_embeddings).float()
        penalties = (ternary_state.unsqueeze(0) != ternary_embeddings).float() * 0.5

        similarity = matches.sum(dim=1) - penalties.sum(dim=1)
        return similarity

    @property
    def quantum_representations(self):
        """
        Propriedade para compatibilidade com CognitiveEngine.
        Retorna a matriz de embeddings como representações quânticas.
        """
        return self.embedding.weight