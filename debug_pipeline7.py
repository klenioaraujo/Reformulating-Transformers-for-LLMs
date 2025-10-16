#!/usr/bin/env python3
"""
Debug do Pipeline ΨQRH - Parte 7
================================

Script para debug do problema fundamental: similaridade 1.000 no topo.
"""

import torch
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix

def debug_similarity_issue():
    """Debug do problema de similaridade 1.000"""
    print("🔬 DEBUG DO PROBLEMA DE SIMILARIDADE 1.000")
    print("=" * 50)

    # Criar QCM
    qcm = QuantumCharacterMatrix(embed_dim=64, device='cpu')

    # Testar com estado aleatório
    print("\n🔍 Teste com Estado Aleatório:")
    random_state = torch.randn(64, 4)
    decoded_random = qcm.decode_quantum_state(random_state, top_k=5)

    print(f"   Estado aleatório -> top 5 caracteres:")
    for char, confidence in decoded_random:
        print(f"      '{char}': {confidence:.3f}")

    # Testar com estado zero
    print("\n🔍 Teste com Estado Zero:")
    zero_state = torch.zeros(64, 4)
    decoded_zero = qcm.decode_quantum_state(zero_state, top_k=5)

    print(f"   Estado zero -> top 5 caracteres:")
    for char, confidence in decoded_zero:
        print(f"      '{char}': {confidence:.3f}")

    # Testar com estado normalizado
    print("\n🔍 Teste com Estado Normalizado:")
    normalized_state = random_state / torch.norm(random_state)
    decoded_normalized = qcm.decode_quantum_state(normalized_state, top_k=5)

    print(f"   Estado normalizado -> top 5 caracteres:")
    for char, confidence in decoded_normalized:
        print(f"      '{char}': {confidence:.3f}")

    # Verificar se há problema com a função de similaridade
    print("\n🔍 Verificando Função de Similaridade:")
    test_char = 'a'
    test_state = qcm.encode_character(test_char)

    # Testar similaridade com ela mesma
    self_similarity = qcm._quaternion_similarity(test_state, test_state)
    print(f"   Similaridade '{test_char}' com ela mesma: {self_similarity:.6f}")

    # Testar similaridade com estado diferente
    test_char2 = 'b'
    test_state2 = qcm.encode_character(test_char2)
    cross_similarity = qcm._quaternion_similarity(test_state, test_state2)
    print(f"   Similaridade '{test_char}' com '{test_char2}': {cross_similarity:.6f}")

    # Verificar se há estados idênticos
    print("\n🔍 Verificando Estados Idênticos:")
    identical_pairs = []
    for i, char1 in enumerate(qcm.vocabulary[:10]):
        state1 = qcm.encode_character(char1)
        for j, char2 in enumerate(qcm.vocabulary[:10]):
            if i < j:
                state2 = qcm.encode_character(char2)
                similarity = qcm._quaternion_similarity(state1, state2)
                if similarity > 0.999:
                    identical_pairs.append((char1, char2, similarity))

    if identical_pairs:
        print(f"   ⚠️  Encontrados {len(identical_pairs)} pares com similaridade > 0.999:")
        for char1, char2, sim in identical_pairs[:5]:
            print(f"      '{char1}' vs '{char2}': {sim:.6f}")
    else:
        print(f"   ✅ Nenhum par com similaridade > 0.999 encontrado")

    # Verificar se há problema com a normalização
    print("\n🔍 Verificando Normalização:")
    for char in qcm.vocabulary[:5]:
        state = qcm.encode_character(char)
        norm = torch.norm(state.flatten())
        print(f"   '{char}': norma = {norm:.6f}")

    # Testar com ruído adicional
    print("\n🔍 Teste com Ruído Adicional:")
    noisy_state = test_state + torch.normal(0.0, 0.1, size=test_state.shape)
    decoded_noisy = qcm.decode_quantum_state(noisy_state, top_k=5)

    print(f"   Estado com ruído -> top 5 caracteres:")
    for char, confidence in decoded_noisy:
        print(f"      '{char}': {confidence:.3f}")

    # Conclusão
    print(f"\n🎯 DIAGNÓSTICO:")
    if decoded_random[0][1] == 1.0:
        print(f"   ⚠️  PROBLEMA CRÍTICO: Similaridade 1.000 mesmo com estado aleatório")
        print(f"   🔧 SOLUÇÃO: Corrigir a função _quaternion_similarity")
    elif decoded_zero[0][1] == 1.0:
        print(f"   ⚠️  PROBLEMA: Similaridade 1.000 com estado zero")
        print(f"   🔧 SOLUÇÃO: Adicionar verificação para estado zero")
    else:
        print(f"   ✅ Similaridade funcionando corretamente")

if __name__ == "__main__":
    debug_similarity_issue()