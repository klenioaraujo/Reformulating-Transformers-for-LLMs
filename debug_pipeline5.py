#!/usr/bin/env python3
"""
Debug do Pipeline ΨQRH - Parte 5
================================

Script para debug da função de codificação e identificação do problema estrutural.
"""

import torch
import numpy as np
import sys
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix

def debug_encoding_function():
    """Debug da função de codificação"""
    print("🔬 DEBUG DA FUNÇÃO DE CODIFICAÇÃO")
    print("=" * 50)

    # Criar QCM
    qcm = QuantumCharacterMatrix(embed_dim=64, device='cpu')

    # Testar diferentes etapas da codificação
    test_char = 'h'
    char_idx = qcm.char_to_idx[test_char]

    print(f"\n🔍 Debug da codificação para '{test_char}' (índice {char_idx}):")

    # Etapa 1: Equação de Padilha
    print("\n📊 Etapa 1: Equação de Padilha")
    base_state = qcm._apply_padilha_wave_equation(char_idx, position=0)
    print(f"   Shape: {base_state.shape}")
    print(f"   Tipo: {base_state.dtype}")
    print(f"   Norma: {torch.norm(base_state):.3f}")
    print(f"   Valores: min={base_state.real.min():.3f}, max={base_state.real.max():.3f}, mean={base_state.real.mean():.3f}")

    # Etapa 2: Filtragem Espectral
    print("\n📊 Etapa 2: Filtragem Espectral")
    filtered_state = qcm._apply_spectral_filtering(base_state)
    print(f"   Shape: {filtered_state.shape}")
    print(f"   Norma: {torch.norm(filtered_state):.3f}")
    print(f"   Valores: min={filtered_state.real.min():.3f}, max={filtered_state.real.max():.3f}, mean={filtered_state.real.mean():.3f}")

    # Etapa 3: Rotação SO(4)
    print("\n📊 Etapa 3: Rotação SO(4)")
    rotated_state = qcm._apply_so4_rotation(filtered_state)
    print(f"   Shape: {rotated_state.shape}")
    print(f"   Norma: {torch.norm(rotated_state):.3f}")
    print(f"   Valores: min={rotated_state.real.min():.3f}, max={rotated_state.real.max():.3f}, mean={rotated_state.real.mean():.3f}")

    # Etapa 4: Transformação adaptativa
    print("\n📊 Etapa 4: Transformação Adaptativa")
    state_flat = rotated_state.view(-1).real
    adapted_state = qcm.adaptive_transform(state_flat)
    print(f"   Shape: {adapted_state.shape}")
    print(f"   Norma: {torch.norm(adapted_state):.3f}")
    print(f"   Valores: min={adapted_state.min():.3f}, max={adapted_state.max():.3f}, mean={adapted_state.mean():.3f}")

    # Etapa 5: Normalização
    print("\n📊 Etapa 5: Normalização")
    normalized_state = qcm.layer_norm(adapted_state)
    print(f"   Shape: {normalized_state.shape}")
    print(f"   Norma: {torch.norm(normalized_state):.3f}")
    print(f"   Valores: min={normalized_state.min():.3f}, max={normalized_state.max():.3f}, mean={normalized_state.mean():.3f}")

    # Etapa 6: Quaternion final
    print("\n📊 Etapa 6: Quaternion Final")
    final_state = qcm.encode_character(test_char)
    print(f"   Shape: {final_state.shape}")
    print(f"   Norma: {torch.norm(final_state.flatten()):.3f}")
    print(f"   Valores: min={final_state.min():.3f}, max={final_state.max():.3f}, mean={final_state.mean():.3f}")

    # Comparar com outro caractere
    print(f"\n🔍 Comparação com outro caractere:")
    test_char2 = 'e'
    final_state2 = qcm.encode_character(test_char2)

    similarity = qcm._quaternion_similarity(final_state, final_state2)
    print(f"   Similaridade '{test_char}' vs '{test_char2}': {similarity:.3f}")

    # Análise da variação entre caracteres
    print(f"\n🔍 Análise da variação entre caracteres:")
    test_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    states = [qcm.encode_character(c) for c in test_chars]

    # Calcular matriz de covariância
    states_tensor = torch.stack([s.flatten() for s in states])
    cov_matrix = torch.cov(states_tensor.T)

    print(f"   Dimensão do espaço: {states_tensor.shape[1]}")
    print(f"   Rank da matriz de covariância: {torch.linalg.matrix_rank(cov_matrix)}")
    print(f"   Determinante da matriz de covariância: {torch.det(cov_matrix):.6e}")

    # Autovalores
    eigenvalues = torch.linalg.eigvals(cov_matrix).real
    print(f"   Autovalores (top 5): {eigenvalues[:5]}")
    print(f"   Número de autovalores > 1e-6: {torch.sum(eigenvalues > 1e-6).item()}")

    # Verificar se há colinearidade
    print(f"\n🔍 Verificando colinearidade:")
    correlations = []
    for i in range(len(test_chars)):
        for j in range(i+1, len(test_chars)):
            corr = torch.corrcoef(torch.stack([states_tensor[i], states_tensor[j]]))[0, 1]
            correlations.append(corr.item())

    print(f"   Correlação média: {np.mean(correlations):.3f}")
    print(f"   Correlação máxima: {np.max(correlations):.3f}")
    print(f"   Correlação mínima: {np.min(correlations):.3f}")

    # Conclusão
    print(f"\n🎯 CONCLUSÃO:")
    if np.mean(correlations) > 0.8:
        print(f"   ⚠️  PROBLEMA: Alta colinearidade entre caracteres")
        print(f"   🔧 SOLUÇÃO: Modificar a função de codificação para gerar representações mais distintas")
    else:
        print(f"   ✅ Colinearidade aceitável")

if __name__ == "__main__":
    debug_encoding_function()