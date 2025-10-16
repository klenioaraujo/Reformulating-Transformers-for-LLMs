#!/usr/bin/env python3
"""
ΨQRH Pipeline usando QuantumCharacterMatrix
===========================================

Pipeline alternativo usando apenas a QuantumCharacterMatrix para geração de texto.
"""

import torch
import sys
import os
from typing import List

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix

class ΨQRHPipelineQuantumMatrix:
    """
    Pipeline simples usando apenas QuantumCharacterMatrix para geração.
    """

    def __init__(self, vocabulary: List[str] = None):
        if vocabulary is None:
            vocabulary = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?")

        self.qcm = QuantumCharacterMatrix(vocabulary=vocabulary)
        self.vocabulary = vocabulary

        print("✅ ΨQRH Pipeline QuantumMatrix inicializado com sucesso.")
        print(f"   📚 Vocabulário: {len(self.vocabulary)} caracteres.")

    def process(self, input_text: str, max_length: int = 20) -> str:
        """
        Gera texto usando apenas QuantumCharacterMatrix com preservação de contexto melhorada.
        """
        print(f"\n🔄 Processando: '{input_text}'")

        # Codificar input
        input_states = []
        for i, char in enumerate(input_text):
            if char in self.vocabulary:
                state = self.qcm.encode_character(char, position=i)
                input_states.append(state)

        if not input_states:
            return ""

        # Usar uma combinação dos estados de input como contexto inicial
        # Ponderar mais os últimos caracteres
        weights = torch.linspace(0.5, 1.0, len(input_states))
        weighted_states = [state.flatten() * weight for state, weight in zip(input_states, weights)]
        current_context = torch.stack(weighted_states).mean(dim=0)

        generated_chars = []
        current_position = len(input_text)

        for i in range(max_length):
            with torch.no_grad():
                # Decodificar contexto atual
                context_to_decode = current_context.view(self.qcm.embed_dim, 4)
                decoded_results = self.qcm.decode_quantum_state(
                    context_to_decode, top_k=5, position=current_position
                )

                if not decoded_results:
                    break

                # 🔥 SELEÇÃO INTELIGENTE COM DIVERSIDADE
                next_char = None
                best_score = -1.0

                for char_idx, (char, confidence) in enumerate(decoded_results):
                    # Penalizar caracteres repetidos recentemente
                    repetition_penalty = 0.0
                    if len(generated_chars) > 0:
                        recent_chars = generated_chars[-3:]  # Últimos 3 caracteres
                        if char in recent_chars:
                            repetition_penalty = 0.3

                    # Penalizar posição no ranking (incentivar diversidade)
                    rank_penalty = char_idx * 0.1

                    # Bonus para caracteres alfanuméricos e espaços
                    content_bonus = 0.0
                    if char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?':
                        content_bonus = 0.2

                    # Score final
                    score = confidence - repetition_penalty - rank_penalty + content_bonus

                    if score > best_score:
                        next_char = char
                        best_score = score

                if next_char is None:
                    next_char, _ = decoded_results[0]

                # Critério de parada
                if next_char == '<UNK>' or best_score < 0.05:
                    break

                generated_chars.append(next_char)

                # 🔥 ATUALIZAÇÃO DINÂMICA DO CONTEXTO
                new_char_state = self.qcm.encode_character(next_char, position=current_position)

                # Blend ratio dinâmico: mais conservador no início, mais criativo depois
                if len(generated_chars) < 5:
                    context_blend_ratio = 0.6  # Mais conservador
                else:
                    context_blend_ratio = 0.4  # Mais criativo

                current_context = (
                    context_blend_ratio * current_context +
                    (1 - context_blend_ratio) * new_char_state.flatten()
                )

                # 🔥 ADICIONAR RUÍDO CONTROLADO PARA DIVERSIDADE
                if len(generated_chars) > 3:
                    noise = torch.normal(0.0, 0.01, size=current_context.shape)
                    current_context = current_context + noise

                current_position += 1

        generated_text = "".join(generated_chars)
        print(f"   🔬 Resposta Gerada: '{generated_text}'")
        return generated_text

def main():
    """Teste do pipeline alternativo."""
    pipeline = ΨQRHPipelineQuantumMatrix()

    test_inputs = [
        "hello",
        "what is",
        "the meaning of",
        "life is"
    ]

    for input_text in test_inputs:
        print(f"\n🎯 Input: '{input_text}'")
        result = pipeline.process(input_text)
        print(f"🎯 Output: '{result}'")

if __name__ == "__main__":
    main()