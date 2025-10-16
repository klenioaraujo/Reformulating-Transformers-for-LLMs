#!/usr/bin/env python3
"""
Debug do Pipeline ΨQRH - Parte 6
================================

Script para debug final do problema de geração no pipeline.
"""

import torch
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix
from src.core.context_funnel import ContextFunnel

def debug_pipeline_final():
    """Debug final do pipeline"""
    print("🔬 DEBUG FINAL DO PIPELINE")
    print("=" * 50)

    # Criar componentes
    qcm = QuantumCharacterMatrix(embed_dim=64, device='cpu')
    context_funnel = ContextFunnel(
        embed_dim=qcm.embed_dim * 4,
        num_heads=8,
        max_history=50
    ).to('cpu')

    # Texto de teste simples
    input_text = "test"
    print(f"\n📝 Processando: '{input_text}'")

    # Etapa 1: Codificação
    print("\n🔍 Etapa 1: Codificação")
    input_states = [qcm.encode_character(char, position=i) for i, char in enumerate(input_text)]
    flattened_input_states = [s.flatten() for s in input_states]

    print(f"   Estados codificados: {len(input_states)}")
    for i, (char, state) in enumerate(zip(input_text, input_states)):
        print(f"      '{char}': shape={state.shape}, norm={torch.norm(state.flatten()):.3f}")

    # Etapa 2: ContextFunnel
    print("\n🔍 Etapa 2: ContextFunnel")
    with torch.no_grad():
        current_context = context_funnel(flattened_input_states)

    print(f"   Contexto gerado: shape={current_context.shape}")
    print(f"   Norma do contexto: {torch.norm(current_context):.3f}")

    # Etapa 3: Loop de geração (apenas primeira iteração)
    print("\n🔍 Etapa 3: Primeira Iteração de Geração")
    with torch.no_grad():
        # Remodelar contexto para decodificação
        context_to_decode = current_context.view(qcm.embed_dim, 4)
        print(f"   Contexto remodelado: shape={context_to_decode.shape}")
        print(f"   Norma do contexto remodelado: {torch.norm(context_to_decode.flatten()):.3f}")

        # Decodificar
        decoded_results = qcm.decode_quantum_state(context_to_decode, top_k=10)
        print(f"   Top 10 caracteres decodificados:")
        for char, confidence in decoded_results:
            print(f"      '{char}': {confidence:.3f}")

        next_char = decoded_results[0][0] if decoded_results else '<UNK>'
        print(f"   Próximo caractere selecionado: '{next_char}'")

        # Analisar similaridade do contexto com todos os caracteres
        print(f"\n🔍 Análise de Similaridade do Contexto:")
        similarities = []
        for char in qcm.vocabulary[:20]:  # Apenas primeiros 20
            ref_state = qcm.encode_character(char)
            similarity = qcm._quaternion_similarity(context_to_decode, ref_state)
            similarities.append((char, similarity))

        # Ordenar por similaridade
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"   Top 5 caracteres mais similares ao contexto:")
        for char, sim in similarities[:5]:
            print(f"      '{char}': {sim:.3f}")

        print(f"   Caracteres menos similares ao contexto:")
        for char, sim in similarities[-5:]:
            print(f"      '{char}': {sim:.3f}")

    # Verificar se há problema com o ContextFunnel
    print(f"\n🔍 Verificando ContextFunnel:")
    print(f"   Input shapes: {[s.shape for s in flattened_input_states]}")
    print(f"   ContextFunnel output shape: {current_context.shape}")

    # Verificar se o contexto está normalizado
    context_norm = torch.norm(current_context)
    print(f"   Norma do contexto: {context_norm:.3f}")
    if context_norm > 20:
        print(f"   ⚠️  Contexto pode estar com norma muito alta")

    # Testar com contexto simples (média dos estados)
    print(f"\n🔍 Teste com Contexto Simples (Média):")
    simple_context = torch.mean(torch.stack(flattened_input_states), dim=0)
    simple_context_reshaped = simple_context.view(qcm.embed_dim, 4)

    decoded_simple = qcm.decode_quantum_state(simple_context_reshaped, top_k=5)
    print(f"   Contexto simples -> top 5 caracteres:")
    for char, confidence in decoded_simple:
        print(f"      '{char}': {confidence:.3f}")

    # Conclusão
    print(f"\n🎯 CONCLUSÃO:")
    if decoded_results[0][1] > 0.99:
        print(f"   ⚠️  PROBLEMA: Similaridade muito alta no topo ({decoded_results[0][1]:.3f})")
        print(f"   🔧 SOLUÇÃO: Ajustar a função de codificação ou o ContextFunnel")
    else:
        print(f"   ✅ Similaridade no topo aceitável ({decoded_results[0][1]:.3f})")

if __name__ == "__main__":
    debug_pipeline_final()