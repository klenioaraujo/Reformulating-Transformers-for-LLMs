#!/usr/bin/env python3
"""
ΨQRH Pipeline usando InverseCognitiveProjector
==============================================

Pipeline alternativo usando InverseCognitiveProjector para geração de texto.
"""

import torch
import sys
import os
from typing import List

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.core.inverse_cognitive_projector import InverseCognitiveProjector
from quantum_word_matrix import QuantumWordMatrix

class ΨQRHPipelineInverseProjector:
    """
    Pipeline usando InverseCognitiveProjector para geração.
    """

    def __init__(self, vocab_size: int = 256):
        self.icp = InverseCognitiveProjector(embed_dim=256, vocab_size=vocab_size)
        self.qcm = QuantumCharacterMatrix()
        self.vocab_size = vocab_size

        print("✅ ΨQRH Pipeline InverseProjector inicializado com sucesso.")
        print(f"   📚 Vocabulário: {vocab_size} caracteres.")

    def process(self, input_text: str, max_length: int = 20) -> str:
        """
        Gera texto usando InverseCognitiveProjector.
        """
        print(f"\n🔄 Processando: '{input_text}'")

        # Codificar input usando QuantumCharacterMatrix
        input_states = []
        for i, char in enumerate(input_text):
            state = self.qcm.encode_character(char, position=i)
            input_states.append(state.flatten())

        if not input_states:
            return ""

        # Combinar estados para criar contexto (usar apenas o último estado)
        context = input_states[-1]  # Último estado como contexto

        generated_chars = []
        current_position = len(input_text)

        for i in range(max_length):
            with torch.no_grad():
                # Usar InverseCognitiveProjector para gerar próximo estado
                # Achatando o contexto para compatibilidade com a rede
                context_flat = context.view(-1)
                psi_reconstructed, confidence = self.icp(
                    context_flat,
                    return_confidence=True
                )

                # Decodificar estado usando QuantumCharacterMatrix
                if psi_reconstructed.dim() == 3:  # Formato quântico [embed_dim, 4]
                    decoded_results = self.qcm.decode_quantum_state(
                        psi_reconstructed, top_k=3, position=current_position
                    )
                else:  # Formato achatado
                    context_to_decode = psi_reconstructed.view(self.qcm.embed_dim, 4)
                    decoded_results = self.qcm.decode_quantum_state(
                        context_to_decode, top_k=3, position=current_position
                    )

                if not decoded_results:
                    break

                # Selecionar próximo caractere
                next_char = None
                best_confidence = 0.0

                for char, conf in decoded_results:
                    if char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?':
                        if conf > best_confidence:
                            next_char = char
                            best_confidence = conf

                if next_char is None:
                    next_char, best_confidence = decoded_results[0]

                # Critério de parada
                if next_char == '<UNK>' or best_confidence < 0.1:
                    break

                generated_chars.append(next_char)

                # Atualizar contexto usando o novo caractere
                new_char_state = self.qcm.encode_character(next_char, position=current_position)
                context_blend_ratio = 0.7
                context = (
                    context_blend_ratio * context +
                    (1 - context_blend_ratio) * new_char_state.flatten()
                )

                current_position += 1

        generated_text = "".join(generated_chars)
        print(f"   🔬 Resposta Gerada: '{generated_text}'")
        return generated_text

def main():
    """Teste do pipeline com InverseCognitiveProjector."""
    pipeline = ΨQRHPipelineInverseProjector()

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