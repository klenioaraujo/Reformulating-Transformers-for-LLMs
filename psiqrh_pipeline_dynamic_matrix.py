#!/usr/bin/env python3
"""
ΨQRH Pipeline usando DynamicQuantumCharacterMatrix
==================================================

Pipeline alternativo usando DynamicQuantumCharacterMatrix para geração de texto.
"""

import torch
import sys
import os
from typing import List

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.core.dynamic_quantum_matrix import DynamicQuantumCharacterMatrix

class ΨQRHPipelineDynamicMatrix:
    """
    Pipeline usando DynamicQuantumCharacterMatrix para geração.
    """

    def __init__(self, vocab_size: int = 256):
        self.dqcm = DynamicQuantumCharacterMatrix(vocab_size=vocab_size, hidden_size=256)
        self.vocab_size = vocab_size

        # Adaptar para modelo padrão
        try:
            self.dqcm.adapt_to_model('gpt2')
        except:
            print("⚠️  Não foi possível adaptar para modelo específico, usando parâmetros padrão")

        print("✅ ΨQRH Pipeline DynamicMatrix inicializado com sucesso.")
        print(f"   📚 Vocabulário: {vocab_size} caracteres.")

    def process(self, input_text: str, max_length: int = 20) -> str:
        """
        Gera texto usando DynamicQuantumCharacterMatrix.
        """
        print(f"\n🔄 Processando: '{input_text}'")

        # Codificar input
        try:
            encoded = self.dqcm.encode_text(input_text)

            # Decodificar para obter índices
            decoded_indices = self.dqcm.decode_text(encoded)

            # Converter índices para caracteres
            generated_chars = []
            for idx in decoded_indices:
                if idx < self.vocab_size:
                    char = chr(idx % 128)  # ASCII básico
                    if 32 <= ord(char) <= 126:  # Caracteres imprimíveis
                        generated_chars.append(char)

            generated_text = "".join(generated_chars)

            # Se não gerou texto suficiente, adicionar geração adicional
            if len(generated_text) < max_length:
                additional_text = self._generate_additional_text(input_text, max_length - len(generated_text))
                generated_text += additional_text

        except Exception as e:
            print(f"   ⚠️  Erro na geração: {e}")
            generated_text = ""

        print(f"   🔬 Resposta Gerada: '{generated_text}'")
        return generated_text

    def _generate_additional_text(self, context: str, max_length: int) -> str:
        """
        Gera texto adicional baseado no contexto usando codificação/decodificação simples.
        """
        additional_chars = []

        # Usar o contexto como base para geração
        for i in range(max_length):
            # Codificar caractere do contexto (cíclico)
            context_char = context[i % len(context)]
            try:
                encoded = self.dqcm.encode_text(context_char)
                decoded_indices = self.dqcm.decode_text(encoded)

                if decoded_indices:
                    idx = decoded_indices[0]
                    if idx < self.vocab_size:
                        char = chr(idx % 128)
                        if 32 <= ord(char) <= 126:
                            additional_chars.append(char)
            except:
                break

        return "".join(additional_chars)

def main():
    """Teste do pipeline com DynamicQuantumCharacterMatrix."""
    pipeline = ΨQRHPipelineDynamicMatrix()

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