#!/usr/bin/env python3
"""
Debug do Encoding de Caracteres
===============================

Análise detalhada do encoding/decoding de caracteres especiais.
"""

import torch
import sys
import os

# Adiciona o diretório base ao path para encontrar o módulo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix


def debug_character_encoding():
    """
    Debug detalhado do encoding/decoding de caracteres especiais.
    """
    print("🔍 DEBUG DETALHADO - CARACTERES ESPECIAIS")
    print("=" * 60)

    # Usar vocabulário customizado que inclui caracteres especiais
    vocab_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:()[]{}<>-–—=+*/√²∛∞π≠≤≥")

    matrix = QuantumCharacterMatrix(
        embed_dim=64,
        alpha=1.5,
        beta=0.8,
        fractal_dim=1.7,
        device='cpu',
        vocabulary=vocab_chars
    )

    problem_chars = "√²∛∞π≠≤≥"

    for char in problem_chars:
        char_code = ord(char)
        print(f"\n🎯 Caractere: '{char}' (Unicode: {char_code})")

        # Verificar se está no vocabulário
        in_vocab = char in matrix.vocabulary
        print(f"   📍 No vocabulário: {in_vocab}")

        if not in_vocab:
            print(f"   ⚠️  Fora do vocabulário! Será mapeado para '<UNK>'")

        # Codificar
        encoded_state = matrix.encode_character(char)
        print(f"   🔄 Estado codificado: shape {encoded_state.shape}")

        # Decodificar
        candidates = matrix.decode_quantum_state(encoded_state, top_k=5)
        print(f"   🔍 Top 5 candidatos: {candidates}")

        # Verificar similaridades
        if candidates:
            best_char, best_conf = candidates[0]
            print(f"   🎯 Melhor candidato: '{best_char}' (conf: {best_conf:.3f})")

            if best_char == char:
                print("   ✅ RECONHECIDO CORRETAMENTE")
            else:
                print(f"   ❌ FALHA: Esperado '{char}', obtido '{best_char}'")


def test_unicode_support():
    """
    Teste do suporte a Unicode expandido.
    """
    print("\n🔤 TESTE DE SUPORTE UNICODE EXPANDIDO")
    print("=" * 50)

    # Vocabulário com caracteres especiais
    vocab_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:()[]{}<>-–—=+*/√²∛∞π≠≤≥")

    matrix = QuantumCharacterMatrix(
        embed_dim=64,
        alpha=1.5,
        beta=0.8,
        fractal_dim=1.7,
        device='cpu',
        vocabulary=vocab_chars
    )

    test_texts = [
        "ABC",
        "Hello World",
        "12345",
        "How many dwarves?",
        "Prove √2 irrational",  # Agora deve funcionar!
        "Math: π ≠ 3.14, ² + ³",
        "Symbols: ∞ ≤ ≥"
    ]

    total_chars = 0
    correct_chars = 0

    for text in test_texts:
        print(f"\n🔍 Testando: '{text}'")

        # Codificar texto completo
        encoded_states = []
        for char in text:
            state = matrix.encode_character(char)
            encoded_states.append(state)

        # Decodificar
        decoded_text = ""
        for i, state in enumerate(encoded_states):
            candidates = matrix.decode_quantum_state(state, top_k=1)
            decoded_char = candidates[0][0] if candidates else "?"
            decoded_text += decoded_char

            if decoded_char == text[i]:
                correct_chars += 1
                status = "✅"
            else:
                status = "❌"

            print(f"   {status} '{text[i]}' → '{decoded_char}'")
            total_chars += 1

        print(f"   📝 Resultado: '{decoded_text}'")

    accuracy = correct_chars / total_chars if total_chars > 0 else 0
    print(f"\n📊 PRECISÃO FINAL: {correct_chars}/{total_chars} = {accuracy:.1%}")

    return accuracy


def analyze_vocabulary_coverage():
    """
    Analisa a cobertura do vocabulário para caracteres comuns.
    """
    print("\n📊 ANÁLISE DE COBERTURA DO VOCABULÁRIO")
    print("=" * 50)

    # Testar com vocabulário padrão vs expandido
    vocab_default = None  # Usará ASCII 32-126
    vocab_expanded = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:()[]{}<>-–—=+*/√²∛∞π≠≤≥")

    test_chars = "ABCD123√π∞≠"

    print(f"Testando caracteres: {test_chars}")

    # Testar com vocabulário padrão
    print("\n🔬 VOCABULÁRIO PADRÃO (ASCII 32-126):")
    matrix_default = QuantumCharacterMatrix(vocabulary=vocab_default)
    for char in test_chars:
        candidates = matrix_default.decode_quantum_state(matrix_default.encode_character(char), top_k=1)
        result = candidates[0][0] if candidates else "?"
        status = "✅" if result == char else "❌"
        print(f"   {status} '{char}' → '{result}'")

    # Testar com vocabulário expandido
    print("\n🔬 VOCABULÁRIO EXPANDIDO:")
    matrix_expanded = QuantumCharacterMatrix(vocabulary=vocab_expanded)
    for char in test_chars:
        candidates = matrix_expanded.decode_quantum_state(matrix_expanded.encode_character(char), top_k=1)
        result = candidates[0][0] if candidates else "?"
        status = "✅" if result == char else "❌"
        print(f"   {status} '{char}' → '{result}'")


if __name__ == "__main__":
    debug_character_encoding()
    accuracy = test_unicode_support()
    analyze_vocabulary_coverage()

    if accuracy > 0.8:
        print("\n🎉 SISTEMA CORRIGIDO COM SUCESSO!")
    elif accuracy > 0.5:
        print("\n⚠️  Melhoria significativa, mas ainda pode melhorar.")
    else:
        print("\n❌ Problemas persistentes precisam de investigação.")