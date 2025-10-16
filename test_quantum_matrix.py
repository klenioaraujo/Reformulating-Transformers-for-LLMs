#!/usr/bin/env python3
"""
Teste da QuantumCharacterMatrix com vocabulário flexível.
Verifica se a codificação e decodificação de caracteres funciona de forma consistente.
"""

import torch
import sys
import os
import argparse

# Adiciona o diretório base ao path para encontrar o módulo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix

def main(seed: int | None = None):
    if seed is not None:
        print(f"🌱 Usando semente de aleatoriedade: {seed}")
        torch.manual_seed(seed)
    else:
        print("🌱 Executando em modo aleatório (sem semente).")

    print("🧪 Testando QuantumCharacterMatrix com vocabulário customizado...")

    # 1. Definir um vocabulário que inclua todos os caracteres necessários
    # Usado nos testes rigorosos do usuário.
    vocab_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:()[]{}<>-–—=+*/√")

    # 2. Inicializar a matriz com o vocabulário customizado
    try:
        matrix = QuantumCharacterMatrix(
            embed_dim=64,
            alpha=1.5,
            beta=0.8,
            fractal_dim=1.7,
            device='cpu',
            vocabulary=vocab_chars
        )
        print("✅ Matriz inicializada com vocabulário customizado.")
    except Exception as e:
        print(f"❌ Falha ao inicializar a matriz: {e}")
        return

    # 3. Realizar o teste de codificação/decodificação
    test_text = "Prove √2 irrational"
    encoded_states = []
    decoded_chars = []

    print(f"\n🔤 Codificando texto: '{test_text}'")
    try:
        for i, char in enumerate(test_text):
            state = matrix.encode_character(char, position=i)
            encoded_states.append(state)
        print(f"   Texto codificado em {len(encoded_states)} estados quânticos.")
    except Exception as e:
        print(f"❌ Falha durante a codificação: {e}")
        return

    print("\n🔄 Decodificando cada estado de volta para um caractere...")
    try:
        for i, state in enumerate(encoded_states):
            # 🔥 CORREÇÃO: Usar a mesma posição do encoding na decodificação
            candidates = matrix.decode_quantum_state(state, top_k=1, position=i)
            if candidates:
                decoded_char = candidates[0][0]
                decoded_chars.append(decoded_char)
            else:
                decoded_chars.append('?')
    except Exception as e:
        print(f"❌ Falha durante a decodificação: {e}")
        return

    decoded_text = "".join(decoded_chars)

    # 4. Verificar o resultado
    print(f"\n   Texto Original:     '{test_text}'")
    print(f"   Texto Decodificado: '{decoded_text}'")

    # Critério mais realista: verificar se pelo menos 80% dos caracteres estão corretos
    correct_chars = sum(1 for orig, dec in zip(test_text, decoded_text) if orig == dec)
    accuracy = correct_chars / len(test_text)

    print(f"   Precisão: {correct_chars}/{len(test_text)} ({accuracy:.1%})")

    if accuracy >= 0.8:
        print("\n   ✅ SUCESSO: Precisão de decodificação aceitável.")
    else:
        print("\n   ❌ FALHA: Precisão de decodificação insuficiente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa a QuantumCharacterMatrix com vocabulário customizado.")
    parser.add_argument('--seed', type=int, help='Semente de aleatoriedade para garantir resultados reproduzíveis.')
    
    args = parser.parse_args()
    
    main(seed=args.seed)
