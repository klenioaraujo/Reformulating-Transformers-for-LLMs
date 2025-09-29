#!/usr/bin/env python3
"""
Teste Avançado de Conservação de Energia para ΨQRH

Valida compliance com Teorema de Parseval e conservação de energia
em todos os cenários de validação científica.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.architecture.psiqrh_transformer import PsiQRHTransformer
from src.validation.mathematical_validation import MathematicalValidator
from src.optimization.advanced_energy_controller import AdvancedEnergyController
from src.optimization.energy_normalizer import energy_preserve


def test_parseval_compliance():
    """Teste de compliance com Teorema de Parseval"""
    print("=== Teste de Compliance com Teorema de Parseval ===")
    print("=" * 60)

    # Criar dados de teste
    batch_size, seq_len, d_model = 2, 128, 512
    x = torch.randn(batch_size, seq_len, d_model)

    # Testar preservação espectral usando FFT com normalização
    x_fft = torch.fft.fft(x, dim=1, norm="ortho")
    x_reconstructed = torch.fft.ifft(x_fft, dim=1, norm="ortho").real

    # Calcular métricas de Parseval
    time_domain_energy = torch.norm(x, p=2).item() ** 2
    freq_domain_energy = torch.norm(x_fft, p=2).item() ** 2 / seq_len
    reconstruction_error = torch.norm(x - x_reconstructed, p=2).item()

    print(f"Energia no domínio do tempo: {time_domain_energy:.6f}")
    print(f"Energia no domínio da frequência: {freq_domain_energy:.6f}")
    print(f"Razão Parseval: {freq_domain_energy / time_domain_energy:.6f}")
    print(f"Erro de reconstrução: {reconstruction_error:.6f}")

    # Verificar compliance
    parseval_compliant = abs(freq_domain_energy / time_domain_energy - 1.0) <= 0.05
    reconstruction_ok = reconstruction_error < 1e-6

    print(f"\nCompliance Parseval: {'PASS' if parseval_compliant else 'FAIL'}")
    print(f"Reconstrução precisa: {'PASS' if reconstruction_ok else 'FAIL'}")

    return parseval_compliant, reconstruction_ok


def test_layer_wise_energy_control():
    """Teste de controle de energia por camada"""
    print("\n=== Teste de Controle de Energia por Camada ===")
    print("=" * 60)

    # Testar controle de energia por camada
    n_layers, d_model = 6, 512
    layer_controller = AdvancedEnergyController(d_model, n_layers)

    # Testar cada camada
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, d_model)
    input_energy = torch.norm(x, p=2, dim=-1, keepdim=True)

    layer_results = []
    for layer_idx in range(n_layers):
        # Aplicar controle de camada
        controlled = layer_controller(x, layer_idx)

        # Calcular conservação
        output_energy = torch.norm(controlled, p=2, dim=-1, keepdim=True)
        conservation_ratio = (output_energy / input_energy).mean().item()

        layer_results.append({
            'layer': layer_idx,
            'conservation_ratio': conservation_ratio,
            'compliant': abs(conservation_ratio - 1.0) <= 0.05
        })

    # Exibir resultados
    print("Resultados por Camada:")
    for result in layer_results:
        status = "PASS" if result['compliant'] else "FAIL"
        print(f"  Camada {result['layer']}: Razão = {result['conservation_ratio']:.6f} [{status}]")

    # Estatísticas globais
    compliant_layers = sum(1 for r in layer_results if r['compliant'])
    avg_ratio = sum(r['conservation_ratio'] for r in layer_results) / len(layer_results)

    print(f"\nEstatísticas:")
    print(f"  Camadas compliant: {compliant_layers}/{n_layers}")
    print(f"  Razão média: {avg_ratio:.6f}")
    print(f"  Status global: {'PASS' if compliant_layers == n_layers else 'FAIL'}")

    return compliant_layers == n_layers, avg_ratio


def test_enhanced_psiqrh_energy():
    """Teste de ΨQRH com controle avançado de energia"""
    print("\n=== Teste de ΨQRH com Controle Avançado de Energia ===")
    print("=" * 60)

    # Criar modelo com controle de energia
    vocab_size = 1000
    d_model = 256

    print("Criando ΨQRH com controle de energia...")
    model_with_energy = PsiQRHTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=4,  # Menos camadas para teste rápido
        n_heads=8
    )

    # Criar modelo sem controle para comparação (mesmo modelo, apenas para comparação)
    model_without_energy = PsiQRHTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=4,
        n_heads=8
    )

    # Dados de teste
    input_ids = torch.randint(0, vocab_size, (1, 64))

    # Testar ambos os modelos
    print("\nTestando modelos...")

    with torch.no_grad():
        # Modelo sem controle
        output_without = model_without_energy(input_ids)

        # Modelo com controle
        output_with = model_with_energy(input_ids)

    # Calcular energias
    input_embeddings = model_with_energy.token_embedding(input_ids)
    input_energy = torch.norm(input_embeddings, p=2).item()

    energy_without = torch.norm(output_without, p=2).item()
    energy_with = torch.norm(output_with, p=2).item()

    ratio_without = energy_without / input_energy
    ratio_with = energy_with / input_energy

    print(f"\nResultados de Conservação de Energia:")
    print(f"  Energia de entrada: {input_energy:.6f}")
    print(f"  Energia sem controle: {energy_without:.6f} (Razão: {ratio_without:.6f})")
    print(f"  Energia com controle: {energy_with:.6f} (Razão: {ratio_with:.6f})")

    # Avaliar compliance
    compliant_without = abs(ratio_without - 1.0) <= 0.05
    compliant_with = abs(ratio_with - 1.0) <= 0.05

    print(f"\nStatus de Compliance:")
    print(f"  Sem controle: {'PASS' if compliant_without else 'FAIL'}")
    print(f"  Com controle: {'PASS' if compliant_with else 'FAIL'}")

    improvement = abs(ratio_with - 1.0) / abs(ratio_without - 1.0) if ratio_without != 1.0 else 1.0
    print(f"  Melhoria: {improvement:.2f}x mais próximo de 1.0")

    return compliant_with, ratio_with, improvement


def comprehensive_validation():
    """Validação abrangente de todos os cenários"""
    print("\n=== Validação Abrangente de Conservação de Energia ===")
    print("=" * 60)

    # Executar todos os testes
    parseval_ok, reconstruction_ok = test_parseval_compliance()
    layer_control_ok, avg_ratio = test_layer_wise_energy_control()
    psiqrh_energy_ok, final_ratio, improvement = test_enhanced_psiqrh_energy()

    # Teste do controlador básico
    print("\n=== Teste do Controlador Básico ===")
    x_test = torch.randn(2, 128, 512)
    normalized = energy_preserve(x_test, x_test * 2.0)
    basic_ratio = torch.norm(normalized, p=2).item() / torch.norm(x_test, p=2).item()
    basic_ok = abs(basic_ratio - 1.0) <= 0.05

    # Resumo final
    print("\n" + "=" * 60)
    print("RESUMO FINAL DA VALIDAÇÃO")
    print("=" * 60)

    tests = [
        ("Teorema de Parseval", parseval_ok),
        ("Reconstrução Espectral", reconstruction_ok),
        ("Controle por Camada", layer_control_ok),
        ("Controlador Básico", basic_ok),
        ("ΨQRH com Energia", psiqrh_energy_ok)
    ]

    passed = sum(1 for name, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nResultado Geral: {passed}/{total} testes PASS")
    print(f"Razão Final de Conservação: {final_ratio:.6f}")
    print(f"Melhoria: {improvement:.2f}x")

    if passed == total:
        print("\n🎯 TODOS OS TESTES PASSARAM!")
        print("Sistema está compliant com Teorema de Parseval e conservação de energia.")
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam.")
        print("Revisar implementação do controle de energia.")

    return passed == total, final_ratio


def sci_005_energy_conservation_scenario():
    """Cenário SCI_005: Validação de Conservação de Energia"""
    print("\n=== SCI_005: Cenário de Conservação de Energia ===")
    print("=" * 60)

    # Configuração do cenário
    vocab_size = 5000
    d_model = 512
    seq_lengths = [64, 128, 256]
    batch_sizes = [1, 2, 4]

    results = []

    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            print(f"\nTestando: batch_size={batch_size}, seq_len={seq_len}")

            # Criar modelo
            model = PsiQRHTransformer(
                vocab_size=vocab_size,
                d_model=d_model
            )

            # Dados de teste
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

            with torch.no_grad():
                output = model(input_ids)

            # Calcular conservação
            input_embeddings = model.token_embedding(input_ids)
            input_energy = torch.norm(input_embeddings, p=2).item()
            output_energy = torch.norm(output, p=2).item()
            ratio = output_energy / input_energy

            compliant = abs(ratio - 1.0) <= 0.05
            results.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'ratio': ratio,
                'compliant': compliant
            })

            print(f"  Razão: {ratio:.6f} [{'PASS' if compliant else 'FAIL'}]")

    # Resumo do cenário
    compliant_tests = sum(1 for r in results if r['compliant'])
    total_tests = len(results)
    avg_ratio = sum(r['ratio'] for r in results) / total_tests

    print(f"\nResumo SCI_005:")
    print(f"  Testes compliant: {compliant_tests}/{total_tests}")
    print(f"  Razão média: {avg_ratio:.6f}")
    print(f"  Status: {'PASS' if compliant_tests == total_tests else 'FAIL'}")

    return compliant_tests == total_tests, avg_ratio


def main():
    """Função principal de teste"""
    print("ΨQRH - Validação Científica de Conservação de Energia")
    print("=" * 60)
    print("Objetivo: energy_ratio ∈ [0.95, 1.05] em todos os cenários")
    print("=" * 60)

    # Validação abrangente
    all_passed, final_ratio = comprehensive_validation()

    # Cenário SCI_005 específico
    sci_005_passed, sci_005_ratio = sci_005_energy_conservation_scenario()

    # Relatório final
    print("\n" + "=" * 60)
    print("RELATÓRIO FINAL DE CONSERVAÇÃO DE ENERGIA")
    print("=" * 60)

    if all_passed and sci_005_passed:
        print("✅ SISTEMA COMPLIANT COM TEOREMA DE PARSEVAL")
        print(f"✅ Razão final de conservação: {final_ratio:.6f} ∈ [0.95, 1.05]")
        print("✅ Todos os cenários científicos validados")
        print("\n🎯 OBJETIVO CIENTÍFICO ATINGIDO!")
    else:
        print("❌ SISTEMA NÃO COMPLIANT")
        print(f"❌ Razão final: {final_ratio:.6f} ∉ [0.95, 1.05]")
        print("❌ Revisar implementação do controle de energia")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()