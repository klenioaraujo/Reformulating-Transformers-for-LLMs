#!/usr/bin/env python3
"""
Debug do Pipeline ΨQRH - Parte 3
================================

Script para debug do pipeline completo e identificação do problema de geração.
"""

import torch
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix
from src.core.context_funnel import ContextFunnel

def debug_pipeline():
    """Debug do pipeline completo"""
    print("🔬 DEBUG DO PIPELINE COMPLETO")
    print("=" * 50)

    # Criar componentes
    qcm = QuantumCharacterMatrix(embed_dim=64, device='cpu')
    context_funnel = ContextFunnel(
        embed_dim=qcm.embed_dim * 4,
        num_heads=8,
        max_history=50
    ).to('cpu')

    # Texto de teste
    input_text = "hello world"
    print(f"\n📝 Processando: '{input_text}'")

    # Etapa 1: Codificação
    print("\n🔍 Etapa 1: Codificação")
    input_states = [qcm.encode_character(char, position=i) for i, char in enumerate(input_text)]
    flattened_input_states = [s.flatten() for s in input_states]

    print(f"   Número de estados: {len(input_states)}")
    print(f"   Shape de cada estado: {input_states[0].shape}")
    print(f"   Shape achatado: {flattened_input_states[0].shape}")

    # Etapa 2: ContextFunnel
    print("\n🔍 Etapa 2: ContextFunnel")
    with torch.no_grad():
        current_context = context_funnel(flattened_input_states)

    print(f"   Contexto gerado: shape={current_context.shape}")
    print(f"   Norma do contexto: {torch.norm(current_context):.3f}")

    # Etapa 3: Loop de geração
    print("\n🔍 Etapa 3: Loop de Geração")
    max_length = 5  # Testar apenas 5 iterações
    generated_chars = []

    for i in range(max_length):
        with torch.no_grad():
            # Remodelar contexto para decodificação
            context_to_decode = current_context.view(qcm.embed_dim, 4)
            print(f"\n   Iteração {i+1}:")
            print(f"      Contexto remodelado: shape={context_to_decode.shape}")
            print(f"      Norma do contexto remodelado: {torch.norm(context_to_decode.flatten()):.3f}")

            # Decodificar
            decoded_results = qcm.decode_quantum_state(context_to_decode, top_k=3)
            print(f"      Top 3 caracteres decodificados:")
            for char, confidence in decoded_results:
                print(f"         '{char}': {confidence:.3f}")

            next_char = decoded_results[0][0] if decoded_results else '<UNK>'
            generated_chars.append(next_char)

            # Codificar novo caractere
            new_char_state = qcm.encode_character(next_char, position=i)
            print(f"      Novo caractere '{next_char}' codificado: shape={new_char_state.shape}")

            # Atualizar contexto
            old_context = current_context.clone()
            current_context = (current_context + new_char_state.flatten()) / 2.0

            print(f"      Mudança no contexto: {torch.norm(current_context - old_context):.3f}")

    generated_text = "".join(generated_chars)
    print(f"\n🎯 Texto gerado: '{generated_text}'")

    # Análise adicional do problema
    print("\n🔍 Análise do Problema:")

    # Verificar se há caracteres dominantes
    char_counts = {}
    for char in qcm.vocabulary:
        state = qcm.encode_character(char)
        similarity = qcm._quaternion_similarity(context_to_decode, state)
        char_counts[char] = similarity

    # Ordenar por similaridade
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"   Top 5 caracteres mais similares ao contexto:")
    for char, sim in sorted_chars[:5]:
        print(f"      '{char}': {sim:.3f}")

    # Verificar se há caracteres com similaridade muito alta
    high_sim_chars = [(char, sim) for char, sim in sorted_chars if sim > 0.99]
    print(f"   Caracteres com similaridade > 0.99: {len(high_sim_chars)}")
    if len(high_sim_chars) > 10:
        print(f"   ⚠️  Muitos caracteres têm similaridade muito alta - pode ser um problema de resolução!")

if __name__ == "__main__":
    debug_pipeline()