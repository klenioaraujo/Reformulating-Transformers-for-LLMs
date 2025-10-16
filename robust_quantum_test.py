#!/usr/bin/env python3
"""
Teste Robusto da QuantumCharacterMatrix
=======================================

Teste que verifica a robustez do sistema sem hardcoding e com fallbacks adequados.
"""

import torch
import sys
import os

# Adiciona o diretório base ao path para encontrar o módulo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix


def test_robust_character_handling():
    """
    Testa o tratamento robusto de caracteres, incluindo casos extremos.
    """
    print("🧪 TESTE ROBUSTO - TRATAMENTO DE CARACTERES")
    print("=" * 60)

    # Testar com vocabulário padrão (ASCII 32-126)
    matrix_default = QuantumCharacterMatrix()

    # Testar com vocabulário expandido
    vocab_expanded = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:()[]{}<>-–—=+*/√²∛∞π≠≤≥")
    matrix_expanded = QuantumCharacterMatrix(vocabulary=vocab_expanded)

    test_cases = [
        # (descrição, caractere, esperado_no_default, esperado_no_expandido)
        ("ASCII básico", "A", "A", "A"),
        ("ASCII básico", "1", "1", "1"),
        ("ASCII básico", " ", " ", " "),
        ("Símbolo matemático", "√", "<UNK>", "√"),
        ("Símbolo matemático", "π", "<UNK>", "π"),
        ("Símbolo matemático", "∞", "<UNK>", "∞"),
        ("Caractere inválido", "\x00", "<UNK>", "<UNK>"),
        ("String vazia", "", "<UNK>", "<UNK>"),
        ("String longa", "ABC", "<UNK>", "<UNK>"),
    ]

    print("\n🔬 TESTE DE CASOS INDIVIDUAIS")
    print("-" * 40)

    for desc, char, expected_default, expected_expanded in test_cases:
        print(f"\n📋 {desc}: '{char}'")

        # Testar com vocabulário padrão
        try:
            state_default = matrix_default.encode_character(char)
            candidates_default = matrix_default.decode_quantum_state(state_default, top_k=1)
            result_default = candidates_default[0][0] if candidates_default else "ERROR"
            status_default = "✅" if result_default == expected_default else "❌"
            print(f"   {status_default} Default: '{result_default}' (esperado: '{expected_default}')")
        except Exception as e:
            print(f"   ❌ Default: ERRO - {e}")

        # Testar com vocabulário expandido
        try:
            state_expanded = matrix_expanded.encode_character(char)
            candidates_expanded = matrix_expanded.decode_quantum_state(state_expanded, top_k=1)
            result_expanded = candidates_expanded[0][0] if candidates_expanded else "ERROR"
            status_expanded = "✅" if result_expanded == expected_expanded else "❌"
            print(f"   {status_expanded} Expandido: '{result_expanded}' (esperado: '{expected_expanded}')")
        except Exception as e:
            print(f"   ❌ Expandido: ERRO - {e}")


def test_robust_text_processing():
    """
    Testa o processamento robusto de textos completos.
    """
    print("\n\n🔤 TESTE ROBUSTO - PROCESSAMENTO DE TEXTOS")
    print("=" * 60)

    # Vocabulário robusto que inclui caracteres comuns
    robust_vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:()[]{}<>-–—=+*/√²∛∞π≠≤≥")
    matrix = QuantumCharacterMatrix(vocabulary=robust_vocab)

    test_texts = [
        "Hello World",
        "Prove √2 irrational",
        "Math: π ≠ 3.14",
        "Symbols: ∞ ≤ ≥",
        "Mixed: A1 √ π ∞",
        "With spaces and punctuation: Hello, world!"
    ]

    total_chars = 0
    correct_chars = 0

    for text in test_texts:
        print(f"\n🔍 Processando: '{text}'")

        # Codificar texto completo
        encoded_states = []
        for char in text:
            try:
                state = matrix.encode_character(char)
                encoded_states.append(state)
            except Exception as e:
                print(f"   ⚠️  Erro ao codificar '{char}': {e}")
                encoded_states.append(None)

        # Decodificar
        decoded_text = ""
        for i, state in enumerate(encoded_states):
            if state is not None:
                try:
                    candidates = matrix.decode_quantum_state(state, top_k=1)
                    decoded_char = candidates[0][0] if candidates else "?"
                    decoded_text += decoded_char

                    if decoded_char == text[i]:
                        correct_chars += 1
                        status = "✅"
                    else:
                        status = "❌"

                    print(f"   {status} '{text[i]}' → '{decoded_char}'")
                except Exception as e:
                    print(f"   ❌ Erro ao decodificar '{text[i]}': {e}")
                    decoded_text += "?"
            else:
                decoded_text += "?"
                print(f"   ❌ Estado nulo para '{text[i]}'")

            total_chars += 1

        print(f"   📝 Resultado: '{decoded_text}'")

    accuracy = correct_chars / total_chars if total_chars > 0 else 0
    print(f"\n📊 PRECISÃO ROBUSTA: {correct_chars}/{total_chars} = {accuracy:.1%}")

    return accuracy


def test_error_handling():
    """
    Testa o tratamento de erros e casos extremos.
    """
    print("\n\n🛡️ TESTE ROBUSTO - TRATAMENTO DE ERROS")
    print("=" * 60)

    matrix = QuantumCharacterMatrix()

    error_cases = [
        ("Estado nulo", None),
        ("Estado com shape errado", torch.randn(10)),
        ("Estado com tipo errado", "string_invalida"),
        ("Estado com valores NaN", torch.tensor([float('nan')])),
        ("Estado com valores infinitos", torch.tensor([float('inf')])),
    ]

    for desc, invalid_state in error_cases:
        print(f"\n📋 Testando: {desc}")

        try:
            if invalid_state is not None and isinstance(invalid_state, torch.Tensor):
                # Tentar decodificar estado inválido
                candidates = matrix.decode_quantum_state(invalid_state, top_k=1)
                if candidates:
                    print(f"   ⚠️  Inesperado: Decodificou para '{candidates[0][0]}'")
                else:
                    print(f"   ✅ Comportamento esperado: Nenhum candidato")
            else:
                # Tentar passar objeto inválido
                candidates = matrix.decode_quantum_state(invalid_state, top_k=1)
                print(f"   ❌ ERRO: Deveria ter falhado")
        except Exception as e:
            print(f"   ✅ Comportamento esperado: Erro capturado - {type(e).__name__}")


def test_parameter_robustness():
    """
    Testa a robustez do sistema com diferentes parâmetros.
    """
    print("\n\n⚙️ TESTE ROBUSTO - PARÂMETROS")
    print("=" * 60)

    test_params = [
        ("Parâmetros padrão", {"alpha": 1.5, "beta": 0.8, "fractal_dim": 1.7}),
        ("Parâmetros extremos", {"alpha": 0.1, "beta": 0.1, "fractal_dim": 1.1}),
        ("Parâmetros altos", {"alpha": 3.0, "beta": 2.0, "fractal_dim": 2.5}),
        ("Parâmetros negativos", {"alpha": -1.0, "beta": -0.5, "fractal_dim": 0.5}),
    ]

    test_text = "ABC123"

    for desc, params in test_params:
        print(f"\n🔬 {desc}: α={params['alpha']}, β={params['beta']}, D={params['fractal_dim']}")

        try:
            matrix = QuantumCharacterMatrix(
                alpha=params['alpha'],
                beta=params['beta'],
                fractal_dim=params['fractal_dim']
            )

            # Testar codificação/decodificação
            encoded_states = [matrix.encode_character(c) for c in test_text]
            decoded_text = "".join([matrix.decode_quantum_state(s, top_k=1)[0][0] for s in encoded_states])

            if decoded_text == test_text:
                print(f"   ✅ Sucesso: '{test_text}' → '{decoded_text}'")
            else:
                print(f"   ⚠️  Diferença: '{test_text}' → '{decoded_text}'")

        except Exception as e:
            print(f"   ❌ Falha: {e}")


if __name__ == "__main__":
    test_robust_character_handling()
    accuracy = test_robust_text_processing()
    test_error_handling()
    test_parameter_robustness()

    print(f"\n{'='*60}")
    if accuracy > 0.95:
        print("🎉 SISTEMA ROBUSTO - EXCELENTE DESEMPENHO!")
    elif accuracy > 0.8:
        print("✅ SISTEMA ROBUSTO - BOM DESEMPENHO!")
    else:
        print("⚠️  SISTEMA PRECISA DE MELHORIAS NA ROBUSTEZ")
    print(f"{'='*60}")