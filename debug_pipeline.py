#!/usr/bin/env python3
"""
Debug do Pipeline ΨQRH
=====================

Script para debug do pipeline e identificação de problemas na decodificação.
"""

import torch
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix

def debug_quantum_matrix():
    """Debug da QuantumCharacterMatrix"""
    print("🔬 DEBUG DA QUANTUM CHARACTER MATRIX")
    print("=" * 50)

    # Criar QCM
    qcm = QuantumCharacterMatrix(embed_dim=64, device='cpu')

    print(f"📚 Vocabulário: {len(qcm.vocabulary)} caracteres")
    print(f"📊 Primeiros 10 caracteres: {qcm.vocabulary[:10]}")

    # Testar codificação de caracteres simples
    test_chars = ['h', 'e', 'l', 'l', 'o', ' ']

    print("\n🔍 Testando codificação de caracteres:")
    for char in test_chars:
        encoded = qcm.encode_character(char)
        print(f"   '{char}' -> shape: {encoded.shape}, dtype: {encoded.dtype}")
        print(f"      Valores: min={encoded.min():.3f}, max={encoded.max():.3f}, mean={encoded.mean():.3f}")

    # Testar decodificação com estado aleatório
    print("\n🔍 Testando decodificação com estado aleatório:")
    random_state = torch.randn(64, 4)
    decoded = qcm.decode_quantum_state(random_state, top_k=3)

    print(f"   Estado aleatório -> top 3 caracteres:")
    for char, confidence in decoded:
        print(f"      '{char}': {confidence:.3f}")

    # Testar decodificação com estado zero
    print("\n🔍 Testando decodificação com estado zero:")
    zero_state = torch.zeros(64, 4)
    decoded_zero = qcm.decode_quantum_state(zero_state, top_k=3)

    print(f"   Estado zero -> top 3 caracteres:")
    for char, confidence in decoded_zero:
        print(f"      '{char}': {confidence:.3f}")

    # Testar similaridade entre caracteres
    print("\n🔍 Testando similaridade entre caracteres:")
    char_a = qcm.encode_character('h')
    char_b = qcm.encode_character('e')
    char_c = qcm.encode_character('h')  # Mesmo caractere

    similarity_ab = qcm._quaternion_similarity(char_a, char_b)
    similarity_ac = qcm._quaternion_similarity(char_a, char_c)

    print(f"   Similaridade 'h' vs 'e': {similarity_ab:.3f}")
    print(f"   Similaridade 'h' vs 'h': {similarity_ac:.3f}")

    # Verificar se há problemas com valores NaN ou infinitos
    print("\n🔍 Verificando problemas numéricos:")
    all_finite = True
    for char in qcm.vocabulary[:20]:  # Verificar apenas os primeiros 20
        encoded = qcm.encode_character(char)
        if not torch.isfinite(encoded).all():
            print(f"   ⚠️  Caractere '{char}' tem valores não-finitos!")
            all_finite = False

    if all_finite:
        print("   ✅ Todos os caracteres verificados têm valores finitos")

if __name__ == "__main__":
    debug_quantum_matrix()