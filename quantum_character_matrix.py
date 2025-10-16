#!/usr/bin/env python3
"""
Matriz Quântica de Conversão Aprimorada para Caracteres
=======================================================

Sistema avançado de mapeamento quântico de caracteres que utiliza um vocabulário flexível
e uma abordagem puramente algorítmica, sem cache.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any

# Importar sistema de similaridade aprimorado
try:
    from quantum_similarity_enhanced import QuantumSimilarityEnhanced
    HAS_ENHANCED_SIMILARITY = True
except ImportError:
    HAS_ENHANCED_SIMILARITY = False
    print("⚠️  Sistema de similaridade aprimorado não disponível, usando implementação básica")

class QuantumCharacterMatrix(nn.Module):
    """
    Matriz Quântica de Conversão Aprimorada para Caracteres com Vocabulário Flexível.
    Implementação puramente algorítmica, sem cache, para aderir aos princípios de design.

    NOTA DE IMPLEMENTAÇÃO:
    Esta versão é puramente algorítmica e não utiliza cache para a decodificação,
    conforme solicitado para garantir a pureza da lógica. Para um ambiente de produção,
    é altamente recomendável reativar uma estratégia de cache (como a pré-computação
    dos estados transformados) para evitar recálculos intensivos e melhorar
    drasticamente a performance da decodificação.
    """

    def __init__(self,
                 embed_dim: int = 64,
                 alpha: float = 1.5,
                 beta: float = 0.8,
                 fractal_dim: float = 1.7,
                 device: str = 'cpu',
                 vocabulary: Optional[List[str]] = None):
        """
        Inicializa a Matriz Quântica de Conversão.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.alpha = alpha
        self.beta = beta
        self.fractal_dim = fractal_dim
        self.device = device

        # ========== VOCABULÁRIO FLEXÍVEL ==========
        if vocabulary is None:
            self.vocabulary = [chr(i) for i in range(32, 127)]
        else:
            self.vocabulary = sorted(list(set(vocabulary)))
        
        if '<UNK>' not in self.vocabulary:
            self.vocabulary.append('<UNK>')

        self.char_to_idx = {char: i for i, char in enumerate(self.vocabulary)}
        self.idx_to_char = {i: char for i, char in enumerate(self.vocabulary)}
        self.unk_idx = self.char_to_idx['<UNK>']

        # ========== PARÂMETROS APRENDÍVEIS ==========
        self.adaptive_transform = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.to(device)

    def _get_transformed_state_for_char(self, char: str, position: int = 0) -> torch.Tensor:
        """
        Aplica a pipeline de encoding completa para um único caractere.
        Helper para ser usado tanto no encoding quanto na decodificação on-the-fly.
        """
        char_idx = self.char_to_idx.get(char, self.unk_idx)

        base_state = self._apply_padilha_wave_equation(char_idx, position)
        filtered_state = self._apply_spectral_filtering(base_state)
        rotated_state = self._apply_so4_rotation(filtered_state)
        state_flat = rotated_state.view(-1).real
        adapted_state = self.adaptive_transform(state_flat)
        normalized_state = self.layer_norm(adapted_state)

        # Adicionar ruído específico por caractere para aumentar distinção
        char_noise = torch.normal(0.0, 0.01, size=normalized_state.shape, device=self.device)
        normalized_state = normalized_state + char_noise

        quaternion_state = torch.zeros(self.embed_dim, 4, dtype=torch.float32, device=self.device)
        quaternion_state[:, 0] = normalized_state

        char_category = self._categorize_character(char)
        semantic_weights = self._get_semantic_weights(char_category)

        for i in range(self.embed_dim):
            phase_shift = 2 * math.pi * i / self.embed_dim
            # Adicionar variação específica por caractere nas fases
            char_phase_mod = char_idx * 0.01
            quaternion_state[i, 1] = normalized_state[i] * semantic_weights[0] * torch.cos(torch.tensor(phase_shift + char_phase_mod))
            quaternion_state[i, 2] = normalized_state[i] * semantic_weights[1] * torch.sin(torch.tensor(phase_shift + char_phase_mod * 1.3))
            quaternion_state[i, 3] = normalized_state[i] * semantic_weights[2] * torch.cos(torch.tensor(2 * phase_shift + char_phase_mod * 0.7))

        return quaternion_state

    def _apply_padilha_wave_equation(self, char_idx: int, position: int = 0) -> torch.Tensor:
        lambda_pos = char_idx / len(self.vocabulary)
        t = position / 100.0  # Usar posição para variação temporal
        I0 = 1.0
        omega = 2 * math.pi * self.alpha
        k = 2 * math.pi / self.fractal_dim

        # Adicionar mais variação usando diferentes componentes
        phase_term1 = omega * t - k * lambda_pos + self.beta * lambda_pos**2
        phase_term2 = omega * t * 1.5 - k * lambda_pos * 0.7 + self.beta * lambda_pos**2 * 1.3

        amplitude_term1 = I0 * math.sin(omega * t + self.alpha * lambda_pos)
        amplitude_term2 = I0 * math.cos(omega * t * 0.8 + self.alpha * lambda_pos * 1.2)

        wave_function1 = amplitude_term1 * torch.exp(torch.tensor(1j * phase_term1))
        wave_function2 = amplitude_term2 * torch.exp(torch.tensor(1j * phase_term2))

        expanded_state = torch.zeros(self.embed_dim, dtype=torch.complex64, device=self.device)
        for i in range(self.embed_dim):
            # Modulação mais complexa usando diferentes frequências
            modulation1 = torch.exp(torch.tensor(1j * 2 * math.pi * i / self.embed_dim * lambda_pos))
            modulation2 = torch.exp(torch.tensor(1j * 4 * math.pi * i / self.embed_dim * lambda_pos * 1.7))

            # Combinação ponderada das duas ondas
            weight1 = 0.6 + 0.4 * math.sin(i * 0.1)
            weight2 = 0.4 + 0.6 * math.cos(i * 0.15)

            expanded_state[i] = (weight1 * wave_function1 * modulation1 +
                                weight2 * wave_function2 * modulation2)
        return expanded_state

    def _apply_spectral_filtering(self, quantum_state: torch.Tensor) -> torch.Tensor:
        k_values = torch.arange(1, self.embed_dim + 1, dtype=torch.float32, device=self.device)

        # Filtro espectral mais complexo com múltiplas componentes
        spectral_filter1 = torch.exp(1j * self.alpha * torch.arctan(torch.log(k_values + 1e-10)))
        spectral_filter2 = torch.exp(1j * self.beta * torch.sin(k_values * 0.1))

        # Combinação de filtros
        combined_filter = 0.7 * spectral_filter1 + 0.3 * spectral_filter2

        filtered_state = quantum_state * combined_filter
        return filtered_state / (torch.norm(filtered_state) + 1e-8)

    def _apply_so4_rotation(self, quantum_state: torch.Tensor) -> torch.Tensor:
        theta = 0.1
        cos_theta = torch.cos(torch.tensor(theta))
        sin_theta = torch.sin(torch.tensor(theta))
        rotated_real = quantum_state.real * cos_theta - quantum_state.imag * sin_theta
        rotated_imag = quantum_state.real * sin_theta + quantum_state.imag * cos_theta
        return torch.complex(rotated_real, rotated_imag)

    def encode_character(self, char: str, position: int = 0) -> torch.Tensor:
        if len(char) != 1:
            char = '<UNK>'
        return self._get_transformed_state_for_char(char, position)

    def decode_quantum_state(self, quantum_state: torch.Tensor, top_k: int = 5, position: int = 0) -> List[Tuple[str, float]]:
        """
        CORREÇÃO: Decodificação com suporte a posição para alinhamento com encoding.
        """
        similarities = []
        with torch.no_grad():
            for i, char in enumerate(self.vocabulary):
                # 🔥 USAR MESMA POSIÇÃO DO ENCODING
                reference_qs = self._get_transformed_state_for_char(char, position=position)
                similarity = self._quaternion_similarity(quantum_state, reference_qs)
                similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        max_similarity = similarities[0][1] if similarities else 1.0
        for idx, similarity in similarities[:top_k]:
            char = self.idx_to_char[idx]
            confidence = similarity / max_similarity if max_similarity > 0 else 0.0
            results.append((char, float(confidence)))

        return results

    def _quaternion_similarity(self, q1: torch.Tensor, q2: torch.Tensor) -> float:
        """
        CORREÇÃO FINAL: Similaridade quântica aprimorada.
        Substitui a implementação anterior para melhor discriminação.
        """
        if HAS_ENHANCED_SIMILARITY:
            return QuantumSimilarityEnhanced.enhanced_quaternion_similarity(q1, q2)
        else:
            # Fallback para implementação básica
            q1_flat = q1.flatten()
            q2_flat = q2.flatten()

            # Normalização robusta
            q1_norm = q1_flat / (torch.norm(q1_flat) + 1e-8)
            q2_norm = q2_flat / (torch.norm(q2_flat) + 1e-8)

            # Similaridade cosseno (mais alta = mais similar)
            cosine_sim = torch.dot(q1_norm, q2_norm)

            # Garantir que a similaridade esteja no intervalo [0, 1]
            similarity = (cosine_sim + 1.0) / 2.0  # Mapear de [-1,1] para [0,1]

            return float(similarity)

    def _categorize_character(self, char: str) -> str:
        if char in 'aeiouAEIOU': return 'vowels'
        if char in 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ': return 'consonants'
        if char in '0123456789': return 'digits'
        if char == ' ': return 'whitespace'
        if char == '<UNK>': return 'unknown'
        return 'punctuation'

    def _get_semantic_weights(self, category: str) -> torch.Tensor:
        semantic_weights = {
            'vowels': [1.2, 0.8, 1.5, 0.9],
            'consonants': [0.9, 1.1, 0.7, 1.3],
            'digits': [1.0, 1.0, 1.0, 1.0],
            'punctuation': [0.5, 0.5, 0.8, 0.8],
            'whitespace': [0.3, 0.3, 0.3, 0.3],
            'unknown': [0.1, 0.1, 0.1, 0.1]
        }
        weights = torch.tensor(semantic_weights[category], dtype=torch.float32)
        return weights / torch.norm(weights)

    def update_spectral_parameters(self, alpha: Optional[float] = None, beta: Optional[float] = None, fractal_dim: Optional[float] = None):
        if alpha is not None: self.alpha = alpha
        if beta is not None: self.beta = beta
        if fractal_dim is not None: self.fractal_dim = fractal_dim
        print(f"✅ Parâmetros espectrais atualizados.")

    def debug_character_similarities(self, chars: List[str]):
        """
        Método de debug para analisar similaridades entre caracteres.
        """
        print(f"\n🔍 DEBUG DE SIMILARIDADES ENTRE CARACTERES")
        print("=" * 50)

        for i, char1 in enumerate(chars):
            for j, char2 in enumerate(chars):
                if i >= j:  # Evitar duplicatas
                    continue

                state1 = self.encode_character(char1, position=0)
                state2 = self.encode_character(char2, position=0)

                similarity = self._quaternion_similarity(state1, state2)

                if HAS_ENHANCED_SIMILARITY:
                    QuantumSimilarityEnhanced.debug_similarity_analysis(
                        state1, state2, char1, char2
                    )
                else:
                    print(f"   {char1}-{char2}: {similarity:.4f}")