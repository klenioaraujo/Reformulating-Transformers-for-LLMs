#!/usr/bin/env python3
"""
DEBUG DA SIMILARIDADE QUÂNTICA
=============================

Script para investigar por que a similaridade não está discriminando corretamente.
"""

import torch
import sys
import os

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix

def debug_similarity():
    """Debug detalhado da similaridade quântica"""
    print("🔍 DEBUG DA SIMILARIDADE QUÂNTICA")
    print("="*60)

    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDE "))

    # Codificar caracteres
    char_a = matrix.encode_character('A', position=0)
    char_b = matrix.encode_character('B', position=0)
    char_c = matrix.encode_character('C', position=0)

    print(f"\n📊 DIMENSÕES DOS ESTADOS:")
    print(f"   Estado A: {char_a.shape}")
    print(f"   Estado B: {char_b.shape}")
    print(f"   Estado C: {char_c.shape}")

    print(f"\n📊 VALORES DOS ESTADOS (primeiros 5 elementos):")
    print(f"   A: {char_a.flatten()[:5]}")
    print(f"   B: {char_b.flatten()[:5]}")
    print(f"   C: {char_c.flatten()[:5]}")

    # Calcular similaridades
    sim_ab = matrix._quaternion_similarity(char_a, char_b)
    sim_ac = matrix._quaternion_similarity(char_a, char_c)
    sim_aa = matrix._quaternion_similarity(char_a, char_a)

    print(f"\n📊 SIMILARIDADES:")
    print(f"   A-B: {sim_ab:.4f}")
    print(f"   A-C: {sim_ac:.4f}")
    print(f"   A-A: {sim_aa:.4f}")

    # Verificar se os estados são diferentes
    print(f"\n📊 DIFERENÇAS ENTRE ESTADOS:")
    diff_ab = torch.norm(char_a - char_b).item()
    diff_ac = torch.norm(char_a - char_c).item()
    print(f"   Norma da diferença A-B: {diff_ab:.4f}")
    print(f"   Norma da diferença A-C: {diff_ac:.4f}")

    # Testar com diferentes posições
    print(f"\n📊 TESTE COM DIFERENTES POSIÇÕES:")
    for pos in [0, 1, 5, 10]:
        char_a_pos = matrix.encode_character('A', position=pos)
        char_b_pos = matrix.encode_character('B', position=pos)
        sim_pos = matrix._quaternion_similarity(char_a_pos, char_b_pos)
        print(f"   Posição {pos}: A-B = {sim_pos:.4f}")

def debug_decoding():
    """Debug da decodificação"""
    print(f"\n\n🔍 DEBUG DA DECODIFICAÇÃO")
    print("="*60)

    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDE "))

    # Testar decodificação de caracteres individuais
    for char in "ABCDE":
        encoded = matrix.encode_character(char, position=0)
        decoded = matrix.decode_quantum_state(encoded, top_k=3, position=0)
        print(f"\n🔍 Decodificando '{char}':")
        for result_char, confidence in decoded:
            print(f"   {result_char}: {confidence:.4f}")

def debug_vocabulary():
    """Debug do vocabulário"""
    print(f"\n\n🔍 DEBUG DO VOCABULÁRIO")
    print("="*60)

    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDE "))

    print(f"Vocabulário: {matrix.vocabulary}")
    print(f"Tamanho do vocabulário: {len(matrix.vocabulary)}")
    print(f"Mapeamento char->idx: {matrix.char_to_idx}")

def main():
    """Executa todos os debug"""
    debug_similarity()
    debug_decoding()
    debug_vocabulary()

if __name__ == "__main__":
    main()