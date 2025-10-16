#!/usr/bin/env python3
"""
TESTE COMPREENSIVO DO SISTEMA ΨQRH CORRIGIDO
============================================

Teste que verifica as correções implementadas:
- Consistência encoding/decoding com posições
- Pipeline com preservação de contexto
- Similaridade quântica discriminativa
"""

import torch
import sys
import os

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_character_matrix import QuantumCharacterMatrix
from psiqrh_pipeline import ΨQRHPipeline

def test_consistency_with_position():
    """Teste que verifica encoding/decoding com posições variáveis"""
    print("\n" + "="*60)
    print("🔍 TESTE DE CONSISTÊNCIA POSICIONAL")
    print("="*60)

    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ "))

    test_cases = [
        ("A", "A"),
        ("HELLO", "WORLD"),
        ("TEST", "CASE"),
        ("QUANTUM", "COMPUTING")
    ]

    all_passed = True

    for input_text, expected_prefix in test_cases:
        print(f"\n🔍 Testando: '{input_text}' -> esperado '{expected_prefix}'")

        # Codificar
        encoded_states = [matrix.encode_character(char, position=i) for i, char in enumerate(input_text)]

        # Decodificar COM POSIÇÃO CORRETA
        decoded_chars = []
        for i, state in enumerate(encoded_states):
            candidates = matrix.decode_quantum_state(state, top_k=1, position=i)
            if candidates:
                decoded_chars.append(candidates[0][0])

        decoded_text = "".join(decoded_chars)
        print(f"   Resultado: '{decoded_text}'")

        # Verificar consistência
        if input_text == decoded_text:
            print("   ✅ CONSISTÊNCIA POSICIONAL: OK")
        else:
            print(f"   ❌ FALHA NA CONSISTÊNCIA POSICIONAL: esperado '{input_text}', obtido '{decoded_text}'")
            all_passed = False

    return all_passed

def test_similarity_discrimination():
    """Teste robusto que verifica se a similaridade quântica é discriminativa"""
    print("\n" + "="*60)
    print("🔍 TESTE DE DISCRIMINAÇÃO DE SIMILARIDADE")
    print("="*60)

    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDE "))

    # Testar todos os pares possíveis para verificação robusta
    chars = ['A', 'B', 'C', 'D', 'E', ' ']

    auto_similarities = []
    cross_similarities = []

    for i, char1 in enumerate(chars):
        for j, char2 in enumerate(chars):
            if i <= j:  # Evitar duplicatas
                state1 = matrix.encode_character(char1, position=0)
                state2 = matrix.encode_character(char2, position=0)
                similarity = matrix._quaternion_similarity(state1, state2)

                if char1 == char2:
                    auto_similarities.append(similarity)
                else:
                    cross_similarities.append(similarity)

    # Análise estatística
    min_auto = min(auto_similarities)
    max_cross = max(cross_similarities)
    margin = min_auto - max_cross

    print(f"   Auto-similaridade mínima: {min_auto:.4f}")
    print(f"   Cross-similaridade máxima: {max_cross:.4f}")
    print(f"   Margem de discriminação: {margin:.4f}")

    # Critério robusto: margem mínima de 0.05
    if margin > 0.05:
        print("   ✅ DISCRIMINAÇÃO DE SIMILARIDADE: OK (margem positiva)")
        return True
    else:
        print("   ❌ FALHA NA DISCRIMINAÇÃO DE SIMILARIDADE (margem insuficiente)")
        return False

def test_pipeline_generation():
    """Teste do pipeline completo com casos realistas"""
    print("\n" + "="*60)
    print("🔍 TESTE DO PIPELINE COMPLETO")
    print("="*60)

    pipeline = ΨQRHPipeline()

    test_inputs = [
        "hello",
        "what is",
        "the meaning of",
        "life is"
    ]

    valid_outputs = 0

    for input_text in test_inputs:
        print(f"\n🎯 Input: '{input_text}'")
        result = pipeline.process(input_text)
        print(f"🎯 Output: '{result}'")

        # Avaliação básica de qualidade
        if len(result) > 0 and all(ord(c) < 127 and c.isprintable() for c in result):
            print("   ✅ SAÍDA TEXTUAL VÁLIDA")
            valid_outputs += 1
        else:
            print("   ⚠️  SAÍDA POTENCIALMENTE PROBLEMÁTICA")

    success_rate = valid_outputs / len(test_inputs)
    print(f"\n📊 Taxa de sucesso: {valid_outputs}/{len(test_inputs)} ({success_rate:.1%})")

    return success_rate >= 0.5  # Pelo menos 50% de sucesso

def test_context_preservation():
    """Teste que verifica a preservação de contexto no pipeline"""
    print("\n" + "="*60)
    print("🔍 TESTE DE PRESERVAÇÃO DE CONTEXTO")
    print("="*60)

    matrix = QuantumCharacterMatrix(vocabulary=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ "))

    # Testar se o contexto se mantém estável durante a geração
    input_text = "HELLO"

    # Codificar input
    input_states = [matrix.encode_character(char, position=i) for i, char in enumerate(input_text)]

    # Simular algumas iterações de geração
    # Usar apenas o primeiro estado como contexto inicial
    current_context = input_states[0].flatten()

    context_variations = []
    for i in range(5):
        # Decodificar com posição correta
        context_to_decode = current_context.view(matrix.embed_dim, 4)
        candidates = matrix.decode_quantum_state(context_to_decode, top_k=1, position=len(input_text) + i)

        if candidates:
            next_char = candidates[0][0]
            new_char_state = matrix.encode_character(next_char, position=len(input_text) + i)

            # Atualização ponderada do contexto (como no pipeline corrigido)
            context_blend_ratio = 0.7
            current_context = (
                context_blend_ratio * current_context +
                (1 - context_blend_ratio) * new_char_state.flatten()
            )

            context_variations.append(torch.norm(current_context).item())

    # Verificar se o contexto não diverge muito
    max_variation = max(context_variations) - min(context_variations)
    print(f"   Variação máxima do contexto: {max_variation:.4f}")

    if max_variation < 1.0:  # Variação aceitável
        print("   ✅ PRESERVAÇÃO DE CONTEXTO: OK")
        return True
    else:
        print("   ❌ FALHA NA PRESERVAÇÃO DE CONTEXTO")
        return False

def main():
    """Executa todos os testes e relata resultados"""
    print("🚀 INICIANDO TESTES COMPREENSIVOS DO ΨQRH CORRIGIDO")

    results = []

    # Executar testes
    results.append(("Consistência Posicional", test_consistency_with_position()))
    results.append(("Discriminação de Similaridade", test_similarity_discrimination()))
    results.append(("Pipeline Completo", test_pipeline_generation()))
    results.append(("Preservação de Contexto", test_context_preservation()))

    # Relatório final
    print("\n" + "="*60)
    print("📊 RELATÓRIO FINAL DE TESTES")
    print("="*60)

    passed = sum(1 for name, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {name}")

    print(f"\n🎯 Resultado: {passed}/{total} testes passados ({passed/total:.1%})")

    if passed == total:
        print("\n🎉 SISTEMA ΨQRH CORRIGIDO: TODOS OS TESTES PASSARAM!")
        return True
    else:
        print(f"\n⚠️  SISTEMA ΨQRH CORRIGIDO: {total-passed} teste(s) falhou/falharam")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)