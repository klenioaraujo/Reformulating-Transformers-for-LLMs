#!/usr/bin/env python3
"""
Validação das Operações Quânticas
================================

Teste para verificar se as operações quânticas estão realmente transformando os estados
e detectar possíveis falsos positivos.
"""

import torch
from quantum_character_matrix import QuantumCharacterMatrix


def validate_quantum_operations():
    """
    Valida se as operações quânticas estão realmente transformando os estados.
    """
    print("🔍 VALIDAÇÃO DAS OPERAÇÕES QUÂNTICAS")
    print("=" * 50)

    matrix = QuantumCharacterMatrix()
    test_char = 'A'

    print(f"Testando caractere: '{test_char}'")

    # 1. Estado base
    base_state = matrix._apply_padilha_wave_equation(ord(test_char), 0)
    print(f"📊 Norma do estado base: {torch.norm(base_state):.6f}")

    # 2. Após filtragem espectral
    filtered_state = matrix._apply_spectral_filtering(base_state)
    print(f"📊 Norma após filtragem: {torch.norm(filtered_state):.6f}")

    # 3. Após rotação SO(4)
    rotated_state = matrix._apply_so4_rotation(filtered_state)
    print(f"📊 Norma após rotação: {torch.norm(rotated_state):.6f}")

    # 4. Verificar se há mudança real
    base_real = base_state.real.mean().item()
    filtered_real = filtered_state.real.mean().item()
    rotated_real = rotated_state.real.mean().item()

    print(f"📈 Média parte real - Base: {base_real:.6f}, Filtrado: {filtered_real:.6f}, Rotacionado: {rotated_real:.6f}")

    # Se não há mudança significativa, é falso positivo
    changes = abs(filtered_real - base_real) > 1e-6 or abs(rotated_real - filtered_real) > 1e-6
    print(f"🔍 Mudanças significativas: {'✅ SIM' if changes else '❌ NÃO - FALSO POSITIVO'}")

    # 5. Verificar similaridade entre estados
    similarity_base_filtered = torch.abs(torch.dot(base_state, filtered_state.conj())) / (torch.norm(base_state) * torch.norm(filtered_state))
    similarity_filtered_rotated = torch.abs(torch.dot(filtered_state, rotated_state.conj())) / (torch.norm(filtered_state) * torch.norm(rotated_state))

    print(f"🔍 Similaridade Base→Filtrado: {similarity_base_filtered:.6f}")
    print(f"🔍 Similaridade Filtrado→Rotacionado: {similarity_filtered_rotated:.6f}")

    # Estados deveriam ser diferentes após transformações
    significant_differences = similarity_base_filtered < 0.99 and similarity_filtered_rotated < 0.99
    print(f"🔍 Diferenças significativas: {'✅ SIM' if significant_differences else '❌ NÃO - FALSO POSITIVO'}")

    return changes and significant_differences


def test_encoding_decoding_consistency():
    """
    Testa se a codificação e decodificação são consistentes.
    """
    print("\n🔁 TESTE DE CONSISTÊNCIA CODIFICAÇÃO-DECODIFICAÇÃO")
    print("=" * 50)

    matrix = QuantumCharacterMatrix()
    test_chars = ['A', 'B', 'C', '1', '2', '3']

    perfect_reconstruction = 0
    total_chars = 0

    for char in test_chars:
        # Codificar
        encoded_state = matrix.encode_character(char)

        # Decodificar
        candidates = matrix.decode_quantum_state(encoded_state, top_k=1)

        if candidates:
            decoded_char = candidates[0][0]
            confidence = candidates[0][1]

            status = "✅" if decoded_char == char else "❌"
            print(f"   {status} '{char}' → '{decoded_char}' (conf: {confidence:.3f})")

            if decoded_char == char:
                perfect_reconstruction += 1
        else:
            print(f"   ❌ '{char}' → NENHUM CANDIDATO")

        total_chars += 1

    accuracy = perfect_reconstruction / total_chars if total_chars > 0 else 0
    print(f"\n📊 Precisão de reconstrução: {accuracy:.1%} ({perfect_reconstruction}/{total_chars})")

    return accuracy


if __name__ == "__main__":
    has_real_operations = validate_quantum_operations()

    if not has_real_operations:
        print("\n🚨 ALERTA CRÍTICO: Operações quânticas não estão transformando os estados!")
        print("   O sistema está provavelmente retornando estados idênticos ou muito similares.")
        print("   Isso caracteriza um FALSO POSITIVO nos testes.")
    else:
        print("\n✅ Operações quânticas estão funcionando corretamente.")

    # Testar consistência
    accuracy = test_encoding_decoding_consistency()

    if accuracy < 0.5:
        print("\n⚠️  ALERTA: Baixa precisão na reconstrução - possíveis problemas no sistema.")
    elif accuracy == 1.0:
        print("\n⚠️  ALERTA: Precisão perfeita de 100% - possível falso positivo.")
    else:
        print("\n✅ Sistema funcionando com precisão realista.")