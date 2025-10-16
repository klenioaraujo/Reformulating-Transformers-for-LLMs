#!/usr/bin/env python3
"""
Debug do Sistema de Matriz Quântica
===================================

Análise detalhada para identificar problemas no sistema de matriz quântica.
"""

import torch
from quantum_character_matrix import QuantumCharacterMatrix


def debug_cache_issue():
    """
    Analisa o problema do cache de estados transformados.
    """
    print("🔍 DEBUG: ANÁLISE DO CACHE DE ESTADOS")
    print("=" * 50)

    matrix = QuantumCharacterMatrix()

    # Verificar o cache
    print(f"📊 Cache shape: {matrix.transformed_base_states.shape}")
    print(f"📊 Cache dtype: {matrix.transformed_base_states.dtype}")
    print(f"📊 Cache device: {matrix.transformed_base_states.device}")

    # Verificar alguns estados no cache
    print("\n📊 Primeiros 5 estados no cache:")
    for i in range(5):
        state = matrix.transformed_base_states[i]
        print(f"   Estado {i} (char '{chr(i+32)}'): shape {state.shape}, norm {torch.norm(state):.6f}")
        print(f"     Valores médios: {state.mean(dim=0)}")

    # Verificar se todos os estados são muito similares
    first_state = matrix.transformed_base_states[0]
    similarities = []
    for i in range(1, min(10, len(matrix.transformed_base_states))):
        state = matrix.transformed_base_states[i]
        similarity = torch.abs(torch.dot(first_state.flatten(), state.flatten())) / (torch.norm(first_state) * torch.norm(state))
        similarities.append(similarity.item())

    print(f"\n📊 Similaridade média entre primeiros 10 estados: {sum(similarities)/len(similarities):.6f}")

    if all(s > 0.99 for s in similarities):
        print("🚨 PROBLEMA: Todos os estados no cache são quase idênticos!")
        print("   Isso explica por que a decodificação sempre retorna o mesmo caractere.")


def debug_encoding_process():
    """
    Analisa o processo de encoding passo a passo.
    """
    print("\n🔍 DEBUG: PROCESSO DE ENCODING")
    print("=" * 50)

    matrix = QuantumCharacterMatrix()
    test_char = 'A'

    print(f"Testando caractere: '{test_char}' (ASCII: {ord(test_char)})")

    # Passo 1: Estado base
    base_state = matrix._apply_padilha_wave_equation(ord(test_char), 0)
    print(f"\n📊 Passo 1 - Estado base:")
    print(f"   Shape: {base_state.shape}")
    print(f"   Norma: {torch.norm(base_state):.6f}")
    print(f"   Tipo: {base_state.dtype}")
    print(f"   Primeiros 5 valores: {base_state[:5]}")

    # Passo 2: Filtragem espectral
    filtered_state = matrix._apply_spectral_filtering(base_state)
    print(f"\n📊 Passo 2 - Após filtragem:")
    print(f"   Shape: {filtered_state.shape}")
    print(f"   Norma: {torch.norm(filtered_state):.6f}")
    print(f"   Primeiros 5 valores: {filtered_state[:5]}")

    # Passo 3: Rotação SO(4)
    rotated_state = matrix._apply_so4_rotation(filtered_state)
    print(f"\n📊 Passo 3 - Após rotação:")
    print(f"   Shape: {rotated_state.shape}")
    print(f"   Norma: {torch.norm(rotated_state):.6f}")
    print(f"   Primeiros 5 valores: {rotated_state[:5]}")

    # Passo 4: Transformação adaptativa
    state_flat = rotated_state.view(-1).real
    adapted_state = matrix.adaptive_transform(state_flat)
    normalized_state = matrix.layer_norm(adapted_state)
    print(f"\n📊 Passo 4 - Após transformação adaptativa:")
    print(f"   Shape: {normalized_state.shape}")
    print(f"   Norma: {torch.norm(normalized_state):.6f}")
    print(f"   Primeiros 5 valores: {normalized_state[:5]}")

    # Passo 5: Mapeamento quaterniónico
    final_state = matrix.encode_character(test_char)
    print(f"\n📊 Passo 5 - Estado final (quaterniónico):")
    print(f"   Shape: {final_state.shape}")
    print(f"   Norma: {torch.norm(final_state):.6f}")
    print(f"   Primeiros 5 valores: {final_state[:5]}")


def debug_decoding_issue():
    """
    Analisa o problema de decodificação.
    """
    print("\n🔍 DEBUG: PROBLEMA DE DECODIFICAÇÃO")
    print("=" * 50)

    matrix = QuantumCharacterMatrix()
    test_chars = ['A', 'B', 'C']

    for char in test_chars:
        print(f"\nTestando caractere: '{char}'")

        # Codificar
        encoded_state = matrix.encode_character(char)
        print(f"   Estado codificado shape: {encoded_state.shape}")
        print(f"   Estado codificado dtype: {encoded_state.dtype}")

        # Decodificar
        candidates = matrix.decode_quantum_state(encoded_state, top_k=5)
        print(f"   Candidatos decodificados: {candidates}")

        # Verificar similaridade com todos os estados no cache
        similarities = []
        for i, ref_state in enumerate(matrix.transformed_base_states):
            similarity = matrix._quaternion_similarity(encoded_state, ref_state)
            similarities.append((chr(i+32), similarity))

        # Ordenar por similaridade
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"   Top 5 similaridades no cache: {similarities[:5]}")


def debug_quaternion_similarity():
    """
    Analisa a função de similaridade quaterniónica.
    """
    print("\n🔍 DEBUG: FUNÇÃO DE SIMILARIDADE QUATERNIÓNICA")
    print("=" * 50)

    matrix = QuantumCharacterMatrix()

    # Criar dois estados diferentes
    state1 = torch.randn(64, 4, dtype=torch.float32)
    state2 = torch.randn(64, 4, dtype=torch.float32)

    print(f"Estado 1 shape: {state1.shape}")
    print(f"Estado 2 shape: {state2.shape}")

    # Calcular similaridade
    similarity = matrix._quaternion_similarity(state1, state2)
    print(f"Similaridade entre estados aleatórios: {similarity:.6f}")

    # Testar com estados idênticos
    similarity_identical = matrix._quaternion_similarity(state1, state1)
    print(f"Similaridade com ele mesmo: {similarity_identical:.6f}")

    # Testar com estados do cache
    if matrix.transformed_base_states is not None:
        cache_state1 = matrix.transformed_base_states[0]
        cache_state2 = matrix.transformed_base_states[1]
        similarity_cache = matrix._quaternion_similarity(cache_state1, cache_state2)
        print(f"Similaridade entre primeiros dois estados do cache: {similarity_cache:.6f}")


if __name__ == "__main__":
    debug_cache_issue()
    debug_encoding_process()
    debug_decoding_issue()
    debug_quaternion_similarity()