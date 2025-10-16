#!/usr/bin/env python3
"""
TESTE ESPECÍFICO PARA CORREÇÃO DA SIMILARIDADE
"""

import torch
import sys
import os

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix

def test_enhanced_similarity():
    """Testa a nova função de similaridade"""
    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDE "))

    print("🚀 TESTE DA SIMILARIDADE APRIMORADA")
    print("=" * 50)

    # Testar similaridades entre caracteres diferentes
    test_pairs = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('A', ' ')]

    for char1, char2 in test_pairs:
        state1 = matrix.encode_character(char1, position=0)
        state2 = matrix.encode_character(char2, position=0)

        similarity = matrix._quaternion_similarity(state1, state2)

        print(f"   {char1}-{char2}: {similarity:.4f}")

        # Critério de sucesso ajustado: similaridade < 0.95 para caracteres diferentes
        if char1 != char2:
            if similarity < 0.95:
                print(f"   ✅ DISCRIMINAÇÃO ADEQUADA")
            else:
                print(f"   ❌ SIMILARIDADE MUITO ALTA")
        else:
            if abs(similarity - 1.0) < 0.01:
                print(f"   ✅ AUTO-SIMILARIDADE CORRETA")
            else:
                print(f"   ❌ AUTO-SIMILARIDADE INCORRETA")

def test_decoding_accuracy():
    """Testa se a decodificação ainda funciona com a nova similaridade"""
    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDE "))

    test_chars = ['A', 'B', 'C', 'D', 'E']

    print(f"\n🔍 TESTE DE PRECISÃO NA DECODIFICAÇÃO")
    print("=" * 50)

    correct = 0
    total = 0

    for char in test_chars:
        state = matrix.encode_character(char, position=0)
        results = matrix.decode_quantum_state(state, top_k=1, position=0)

        if results and results[0][0] == char:
            correct += 1
            print(f"   {char} → {results[0][0]} ✅")
        else:
            decoded = results[0][0] if results else "NONE"
            print(f"   {char} → {decoded} ❌")

        total += 1

    accuracy = correct / total
    print(f"\n📊 Precisão: {correct}/{total} ({accuracy:.1%})")

    if accuracy >= 0.8:
        print("🎯 DECODIFICAÇÃO: ACEITÁVEL")
    else:
        print("⚠️  DECODIFICAÇÃO: PROBLEMÁTICA")

def test_detailed_similarity_analysis():
    """Teste detalhado com análise de componentes da similaridade"""
    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDE "))

    print(f"\n🔍 ANÁLISE DETALHADA DA SIMILARIDADE")
    print("=" * 50)

    # Executar debug detalhado
    matrix.debug_character_similarities(['A', 'B', 'C'])

def test_similarity_range():
    """Testa se a similaridade está no intervalo correto [0, 1]"""
    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDE "))

    print(f"\n🔍 TESTE DO INTERVALO DA SIMILARIDADE")
    print("=" * 50)

    test_chars = ['A', 'B', 'C', 'D', 'E', ' ']
    min_sim = 1.0
    max_sim = 0.0

    for i, char1 in enumerate(test_chars):
        for j, char2 in enumerate(test_chars):
            if i <= j:  # Testar todos os pares
                state1 = matrix.encode_character(char1, position=0)
                state2 = matrix.encode_character(char2, position=0)
                similarity = matrix._quaternion_similarity(state1, state2)

                min_sim = min(min_sim, similarity)
                max_sim = max(max_sim, similarity)

    print(f"   Mínima similaridade: {min_sim:.4f}")
    print(f"   Máxima similaridade: {max_sim:.4f}")

    if 0.0 <= min_sim <= max_sim <= 1.0:
        print("   ✅ INTERVALO CORRETO [0, 1]")
    else:
        print("   ❌ INTERVALO INCORRETO")

if __name__ == "__main__":
    test_enhanced_similarity()
    test_decoding_accuracy()
    test_detailed_similarity_analysis()
    test_similarity_range()