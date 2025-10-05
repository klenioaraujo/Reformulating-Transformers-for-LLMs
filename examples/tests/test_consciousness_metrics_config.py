#!/usr/bin/env python3
"""
Test Consciousness Metrics Configuration
=========================================

Demonstra o uso das configurações de métricas de consciência
e valida o mapeamento correto Dimensão Fractal → FCI.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
from dataclasses import dataclass


@dataclass
class MockConfig:
    """Mock config para teste."""
    device: str = 'cpu'
    diffusion_coefficient_range: tuple = (0.0, 1.0)
    epsilon: float = 1e-10


def load_consciousness_metrics_config():
    """Carrega configuração de métricas."""
    config_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'configs',
        'consciousness_metrics.yaml'
    )

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_fractal_dimension_to_fci_mapping():
    """Testa mapeamento D → FCI."""
    print("=" * 60)
    print("🧪 TEST: Fractal Dimension → FCI Mapping")
    print("=" * 60)

    # Carregar config
    metrics_config = load_consciousness_metrics_config()

    # Importar ConsciousnessMetrics
    from src.conscience.consciousness_metrics import ConsciousnessMetrics

    # Criar instância
    mock_config = MockConfig()
    metrics = ConsciousnessMetrics(mock_config, metrics_config)

    # Casos de teste
    test_cases = [
        (1.0, 0.0, "Linha suave (mínima complexidade)"),
        (1.25, 0.125, "Linha costeira típica"),
        (1.5, 0.25, "Ruído 1/f"),
        (1.7, 0.35, "Movimento browniano fracionário"),
        (2.0, 0.5, "Browniano padrão"),
        (2.2, 0.6, "Alta atividade neural"),
        (2.5, 0.75, "Dinâmica complexa"),
        (2.8, 0.9, "Pico de consciência"),
        (3.0, 1.0, "Preenchimento total do espaço"),
    ]

    print("\n📊 Mapeamento Dimensão Fractal → FCI:\n")
    print(f"{'D (Fractal)':<15} {'FCI (Calculado)':<18} {'FCI (Esperado)':<18} {'Estado':<15} {'Descrição'}")
    print("-" * 100)

    all_passed = True
    for dimension, expected_fci, description in test_cases:
        calculated_fci = metrics.compute_fci_from_fractal_dimension(dimension)
        state = metrics._classify_fci_state(calculated_fci)

        # Verificar se está dentro da tolerância
        tolerance = 0.01
        passed = abs(calculated_fci - expected_fci) < tolerance
        symbol = "✓" if passed else "✗"

        print(f"{dimension:<15.2f} {calculated_fci:<18.3f} {expected_fci:<18.3f} {state:<15} {symbol} {description}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ TODOS OS TESTES PASSARAM!")
    else:
        print("❌ ALGUNS TESTES FALHARAM!")
    print("=" * 60)

    return all_passed


def test_state_thresholds():
    """Testa classificação de estados com thresholds configuráveis."""
    print("\n" + "=" * 60)
    print("🧪 TEST: State Classification Thresholds")
    print("=" * 60)

    metrics_config = load_consciousness_metrics_config()

    from src.conscience.consciousness_metrics import ConsciousnessMetrics

    mock_config = MockConfig()
    metrics = ConsciousnessMetrics(mock_config, metrics_config)

    # Pegar thresholds da configuração
    thresholds = metrics_config['state_thresholds']

    print("\n🎯 Thresholds Configurados:\n")
    print(f"EMERGENCE: FCI ≥ {thresholds['emergence']['min_fci']:.2f} (D ≥ {thresholds['emergence']['fractal_dimension_min']:.2f})")
    print(f"MEDITATION: FCI ≥ {thresholds['meditation']['min_fci']:.2f} (D ≥ {thresholds['meditation']['fractal_dimension_min']:.2f})")
    print(f"ANALYSIS:   FCI ≥ {thresholds['analysis']['min_fci']:.2f} (D ≥ {thresholds['analysis']['fractal_dimension_min']:.2f})")
    print(f"COMA:       FCI < {thresholds['analysis']['min_fci']:.2f} (D < {thresholds['analysis']['fractal_dimension_min']:.2f})")

    # Testar casos limítrofes
    test_cases = [
        (0.85, "EMERGENCE"),
        (0.75, "MEDITATION"),
        (0.45, "ANALYSIS"),
        (0.20, "COMA"),
    ]

    print("\n📊 Classificação de Estados:\n")
    print(f"{'FCI':<10} {'Estado Esperado':<20} {'Estado Calculado':<20} {'Status'}")
    print("-" * 70)

    all_passed = True
    for fci_value, expected_state in test_cases:
        calculated_state = metrics._classify_fci_state(fci_value)
        passed = calculated_state == expected_state
        symbol = "✓" if passed else "✗"

        print(f"{fci_value:<10.2f} {expected_state:<20} {calculated_state:<20} {symbol}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ CLASSIFICAÇÃO CORRETA!")
    else:
        print("❌ CLASSIFICAÇÃO INCORRETA!")
    print("=" * 60)

    return all_passed


def test_real_world_examples():
    """Testa exemplos do mundo real da configuração."""
    print("\n" + "=" * 60)
    print("🧪 TEST: Real-World Examples from Config")
    print("=" * 60)

    metrics_config = load_consciousness_metrics_config()

    from src.conscience.consciousness_metrics import ConsciousnessMetrics

    mock_config = MockConfig()
    metrics = ConsciousnessMetrics(mock_config, metrics_config)

    examples = metrics_config['real_world_examples']

    print("\n🌍 Exemplos do Mundo Real:\n")
    print(f"{'Exemplo':<25} {'D':<10} {'FCI (Calc)':<15} {'FCI (Esp)':<15} {'Estado':<15} {'Status'}")
    print("-" * 95)

    all_passed = True
    for name, data in examples.items():
        dimension = data['fractal_dimension']
        expected_fci = data['expected_fci']
        expected_state = data['state']

        calculated_fci = metrics.compute_fci_from_fractal_dimension(dimension)
        calculated_state = metrics._classify_fci_state(calculated_fci)

        # Tolerância
        fci_passed = abs(calculated_fci - expected_fci) < 0.01
        state_passed = calculated_state == expected_state
        passed = fci_passed and state_passed

        symbol = "✓" if passed else "✗"

        print(f"{name:<25} {dimension:<10.2f} {calculated_fci:<15.3f} {expected_fci:<15.3f} {calculated_state:<15} {symbol}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ TODOS OS EXEMPLOS CORRETOS!")
    else:
        print("❌ ALGUNS EXEMPLOS FALHARAM!")
    print("=" * 60)

    return all_passed


def main():
    """Executa todos os testes."""
    print("\n" + "=" * 60)
    print("🧠 CONSCIOUSNESS METRICS CONFIGURATION TEST SUITE")
    print("=" * 60)

    results = []

    # Teste 1: Mapeamento D → FCI
    results.append(("Fractal D → FCI Mapping", test_fractal_dimension_to_fci_mapping()))

    # Teste 2: Thresholds de estados
    results.append(("State Thresholds", test_state_thresholds()))

    # Teste 3: Exemplos do mundo real
    results.append(("Real-World Examples", test_real_world_examples()))

    # Resumo final
    print("\n" + "=" * 60)
    print("📋 RESUMO DOS TESTES")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{name:<30} {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("\n✅ Fórmula corrigida: FCI = (D - 1.0) / 2.0")
        print("✅ Thresholds configuráveis funcionando")
        print("✅ Mapeamento D → FCI → Estado correto")
    else:
        print("⚠️  ALGUNS TESTES FALHARAM - Verifique a configuração")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)