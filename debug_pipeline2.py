#!/usr/bin/env python3
"""
Debug do Pipeline ΨQRH - Parte 2
================================

Script para debug mais detalhado da decodificação.
"""

import torch
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix

def debug_decoding():
    """Debug detalhado da decodificação"""
    print("🔬 DEBUG DETALHADO DA DECODIFICAÇÃO")
    print("=" * 50)

    # Criar QCM
    qcm = QuantumCharacterMatrix(embed_dim=64, device='cpu')

    # Testar decodificação com estado real de um caractere
    print("\n🔍 Testando decodificação com estado real:")

    # Codificar um caractere específico
    test_char = 'h'
    encoded_state = qcm.encode_character(test_char)
    print(f"   Estado codificado de '{test_char}':")
    print(f"      Shape: {encoded_state.shape}")
    print(f"      Norma: {torch.norm(encoded_state.flatten()):.3f}")

    # Decodificar o próprio estado
    decoded = qcm.decode_quantum_state(encoded_state, top_k=5)
    print(f"   Decodificação do próprio estado:")
    for char, confidence in decoded:
        print(f"      '{char}': {confidence:.3f}")

    # Verificar se o caractere original está no topo
    top_char = decoded[0][0]
    if top_char == test_char:
        print(f"   ✅ Caractere original '{test_char}' está no topo!")
    else:
        print(f"   ⚠️  Caractere original '{test_char}' não está no topo (topo: '{top_char}')")

    # Testar com estado de outro caractere
    print("\n🔍 Testando decodificação cruzada:")
    char_a = 'h'
    char_b = 'e'

    state_a = qcm.encode_character(char_a)
    state_b = qcm.encode_character(char_b)

    # Decodificar estado A com referência B
    decoded_a_from_b = qcm.decode_quantum_state(state_a, top_k=3)
    decoded_b_from_a = qcm.decode_quantum_state(state_b, top_k=3)

    print(f"   Estado '{char_a}' decodificado:")
    for char, confidence in decoded_a_from_b:
        print(f"      '{char}': {confidence:.3f}")

    print(f"   Estado '{char_b}' decodificado:")
    for char, confidence in decoded_b_from_a:
        print(f"      '{char}': {confidence:.3f}")

    # Verificar similaridades entre todos os caracteres
    print("\n🔍 Verificando similaridades entre caracteres:")
    test_chars = ['h', 'e', 'l', 'o', ' ', '!']

    for i, char1 in enumerate(test_chars):
        for j, char2 in enumerate(test_chars):
            if i < j:
                state1 = qcm.encode_character(char1)
                state2 = qcm.encode_character(char2)
                similarity = qcm._quaternion_similarity(state1, state2)
                print(f"   '{char1}' vs '{char2}': {similarity:.3f}")

    # Testar com contexto do pipeline (estado médio)
    print("\n🔍 Testando com contexto do pipeline:")
    text = "hello"
    states = [qcm.encode_character(c) for c in text]
    context = torch.mean(torch.stack([s.flatten() for s in states]), dim=0)

    # Remodelar para o formato esperado
    context_reshaped = context.view(qcm.embed_dim, 4)

    decoded_context = qcm.decode_quantum_state(context_reshaped, top_k=5)
    print(f"   Contexto de '{text}' -> top 5 caracteres:")
    for char, confidence in decoded_context:
        print(f"      '{char}': {confidence:.3f}")

if __name__ == "__main__":
    debug_decoding()