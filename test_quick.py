#!/usr/bin/env python3
"""
TESTE RÁPIDO DO SISTEMA ΨQRH CORRIGIDO
=====================================
"""

import torch
import sys
import os

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix

def test_similarity_quick():
    """Teste rápido da similaridade"""
    print("🔍 TESTE RÁPIDO DA SIMILARIDADE")
    print("="*60)

    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDE "))

    # Testar similaridade entre caracteres diferentes
    char_a = matrix.encode_character('A', position=0)
    char_b = matrix.encode_character('B', position=0)
    char_c = matrix.encode_character('C', position=0)

    # Similaridade entre A e B
    sim_ab = matrix._quaternion_similarity(char_a, char_b)
    # Similaridade entre A e C
    sim_ac = matrix._quaternion_similarity(char_a, char_c)
    # Similaridade entre A e A (deve ser muito alta)
    sim_aa = matrix._quaternion_similarity(char_a, char_a)

    print(f"   Similaridade A-B: {sim_ab:.4f}")
    print(f"   Similaridade A-C: {sim_ac:.4f}")
    print(f"   Similaridade A-A: {sim_aa:.4f}")

    # Verificar se a similaridade é discriminativa
    # A-A deve ser muito maior que A-B e A-C
    if sim_aa > 0.99 and sim_ab < 0.95 and sim_ac < 0.95:
        print("   ✅ DISCRIMINAÇÃO DE SIMILARIDADE: OK")
        return True
    else:
        print("   ❌ FALHA NA DISCRIMINAÇÃO DE SIMILARIDADE")
        return False

def test_decoding_quick():
    """Teste rápido da decodificação"""
    print("\n🔍 TESTE RÁPIDO DA DECODIFICAÇÃO")
    print("="*60)

    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDE "))

    # Testar se podemos decodificar corretamente
    char_a = matrix.encode_character('A', position=0)
    decoded = matrix.decode_quantum_state(char_a, top_k=1, position=0)

    if decoded and decoded[0][0] == 'A':
        print(f"   ✅ DECODIFICAÇÃO: OK (decodificado '{decoded[0][0]}')")
        return True
    else:
        print(f"   ❌ FALHA NA DECODIFICAÇÃO: esperado 'A', obtido '{decoded[0][0] if decoded else 'N/A'}'")
        return False

def main():
    """Executa testes rápidos"""
    print("🚀 TESTES RÁPIDOS DO ΨQRH CORRIGIDO")

    results = []
    results.append(("Similaridade", test_similarity_quick()))
    results.append(("Decodificação", test_decoding_quick()))

    # Relatório final
    print("\n" + "="*60)
    print("📊 RELATÓRIO FINAL")
    print("="*60)

    passed = sum(1 for name, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {name}")

    print(f"\n🎯 Resultado: {passed}/{total} testes passados ({passed/total:.1%})")

    if passed == total:
        print("\n🎉 CORREÇÕES BÁSICAS FUNCIONANDO!")
        return True
    else:
        print(f"\n⚠️  AINDA HÁ PROBLEMAS: {total-passed} teste(s) falhou/falharam")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)