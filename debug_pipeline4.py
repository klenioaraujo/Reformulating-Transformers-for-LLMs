#!/usr/bin/env python3
"""
Debug do Pipeline ΨQRH - Parte 4
================================

Script para debug do problema fundamental: similaridade muito alta entre caracteres.
"""

import torch
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix

def debug_similarity_problem():
    """Debug do problema de similaridade"""
    print("🔬 DEBUG DO PROBLEMA DE SIMILARIDADE")
    print("=" * 50)

    # Criar QCM
    qcm = QuantumCharacterMatrix(embed_dim=64, device='cpu')

    # Analisar similaridades entre todos os caracteres
    print("\n🔍 Analisando similaridades entre caracteres:")

    # Selecionar alguns caracteres representativos
    test_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                  ' ', '!', '.', ',', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Calcular matriz de similaridade
    similarity_matrix = np.zeros((len(test_chars), len(test_chars)))

    for i, char1 in enumerate(test_chars):
        state1 = qcm.encode_character(char1)
        for j, char2 in enumerate(test_chars):
            state2 = qcm.encode_character(char2)
            similarity = qcm._quaternion_similarity(state1, state2)
            similarity_matrix[i, j] = similarity

    # Estatísticas da matriz de similaridade
    print(f"\n📊 Estatísticas da matriz de similaridade ({len(test_chars)}x{len(test_chars)}):")
    print(f"   Média: {similarity_matrix.mean():.3f}")
    print(f"   Desvio padrão: {similarity_matrix.std():.3f}")
    print(f"   Mínimo: {similarity_matrix.min():.3f}")
    print(f"   Máximo: {similarity_matrix.max():.3f}")

    # Contar similaridades altas
    high_sim_count = np.sum(similarity_matrix > 0.95)
    very_high_sim_count = np.sum(similarity_matrix > 0.99)

    print(f"\n🔍 Similaridades altas:")
    print(f"   > 0.95: {high_sim_count} pares ({high_sim_count / len(test_chars)**2 * 100:.1f}%)")
    print(f"   > 0.99: {very_high_sim_count} pares ({very_high_sim_count / len(test_chars)**2 * 100:.1f}%)")

    # Encontrar os pares mais similares
    print(f"\n🔍 Pares mais similares:")
    for _ in range(10):
        max_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        max_sim = similarity_matrix[max_idx]
        if max_sim < 0.9:
            break

        char1 = test_chars[max_idx[0]]
        char2 = test_chars[max_idx[1]]
        print(f"   '{char1}' vs '{char2}': {max_sim:.3f}")

        # Zerar este valor para encontrar o próximo
        similarity_matrix[max_idx] = -1

    # Verificar se há caracteres com representação única
    print(f"\n🔍 Caracteres com representação única:")
    unique_chars = []
    for i, char in enumerate(test_chars):
        # Verificar se este caractere tem similaridade baixa com todos os outros
        min_sim = np.min(similarity_matrix[i, :])
        max_sim = np.max(similarity_matrix[i, :])

        if max_sim < 0.8:
            unique_chars.append((char, max_sim))

    if unique_chars:
        print(f"   Encontrados {len(unique_chars)} caracteres com similaridade máxima < 0.8:")
        for char, max_sim in unique_chars[:10]:
            print(f"      '{char}': {max_sim:.3f}")
    else:
        print(f"   ⚠️  Nenhum caractere tem representação única (todos são muito similares)")

    # Análise da distribuição de similaridade
    print(f"\n📊 Distribuição de similaridade:")
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    hist, _ = np.histogram(similarity_matrix.flatten(), bins=bins)

    for i in range(len(bins)-1):
        count = hist[i]
        percentage = count / len(similarity_matrix.flatten()) * 100
        print(f"   [{bins[i]:.1f}-{bins[i+1]:.1f}]: {count} pares ({percentage:.1f}%)")

    # Conclusão
    print(f"\n🎯 CONCLUSÃO:")
    if similarity_matrix.mean() > 0.8:
        print(f"   ⚠️  PROBLEMA ENCONTRADO: Similaridade média muito alta ({similarity_matrix.mean():.3f})")
        print(f"   🔧 SOLUÇÃO: Aumentar a dimensionalidade ou modificar a função de codificação")
    else:
        print(f"   ✅ Similaridade média aceitável ({similarity_matrix.mean():.3f})")

if __name__ == "__main__":
    debug_similarity_problem()