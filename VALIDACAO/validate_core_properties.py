#!/usr/bin/env python3
"""
ΨQRH Core Properties Validation Script
======================================

Unified validation script that executes the four critical validation tests:
1. Energy Conservation Test
2. Parseval Theorem Validation
3. Memory Benchmark Test
4. Rotational Quaternion Properties Test

This script ensures that the core mathematical properties of ΨQRH are maintained
and that the system operates with energy stability and numerical correctness.

Author: ΨQRH Validation Framework
Date: 2025-10-02
Version: 1.0.0
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_module_exists(module_name: str) -> bool:
    """Check if a Python module exists and can be imported."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ImportError:
        return False


def run_energy_conservation_test() -> bool:
    """Run energy conservation test."""
    print("\n🔋 Teste 1: Validação de Conservação de Energia")
    print("=" * 50)

    if not check_module_exists('examples.energy_conservation_test'):
        print("❌ Módulo energy_conservation_test não encontrado")
        return False

    try:
        # Import and run the test module
        from examples.energy_conservation_test import main as energy_test_main

        # Execute the test
        result = energy_test_main()

        # Handle tuple return (energy_ratio, conservation_ratio)
        if isinstance(result, tuple):
            energy_ratio, conservation_ratio = result
            success = abs(energy_ratio - 1.0) <= 0.05 and abs(conservation_ratio - 1.0) <= 0.05
        else:
            success = result is None or result == True

        if success:
            print("✅ Teste de conservação de energia PASSOU")
            return True
        else:
            print("❌ Teste de conservação de energia FALHOU")
            return False

    except Exception as e:
        print(f"❌ Erro no teste de conservação de energia: {e}")
        return False


def run_parseval_validation_test() -> bool:
    """Run Parseval theorem validation test."""
    print("\n📊 Teste 2: Validação do Teorema de Parseval")
    print("=" * 50)

    if not check_module_exists('examples.parseval_validation_test'):
        print("❌ Módulo parseval_validation_test não encontrado")
        return False

    try:
        # Import and run the test module
        from examples.parseval_validation_test import main as parseval_test_main

        # Execute the test
        result = parseval_test_main()

        # Handle boolean or None return
        success = result is None or result == True

        if success:
            print("✅ Teste do teorema de Parseval PASSOU")
            return True
        else:
            print("❌ Teste do teorema de Parseval FALHOU")
            return False

    except Exception as e:
        print(f"❌ Erro no teste do teorema de Parseval: {e}")
        return False


def run_memory_benchmark_test() -> bool:
    """Run memory benchmark test."""
    print("\n💾 Teste 3: Benchmark de Memória")
    print("=" * 50)

    if not check_module_exists('examples.memory_benchmark_test'):
        print("❌ Módulo memory_benchmark_test não encontrado")
        return False

    try:
        # Import and run the test module
        from examples.memory_benchmark_test import main as memory_test_main

        # Execute the test
        result = memory_test_main()

        if result:
            print("✅ Teste de benchmark de memória PASSOU")
            return True
        else:
            print("❌ Teste de benchmark de memória FALHOU")
            return False

    except Exception as e:
        print(f"❌ Erro no teste de benchmark de memória: {e}")
        return False


def run_rotational_quaternion_test() -> bool:
    """Run rotational quaternion properties test."""
    print("\n🔄 Teste 4: Propriedades do Quaternion Rotacional")
    print("=" * 50)

    if not check_module_exists('examples.test_rotational_quaternion'):
        print("❌ Módulo test_rotational_quaternion não encontrado")
        return False

    try:
        # Import and run the test module
        from examples.test_rotational_quaternion import test_rotational_quaternion_efficiency as quaternion_test_main

        # Execute the test
        result = quaternion_test_main()

        # Handle None return (function doesn't return explicit success)
        success = result is None or result == True

        if success:
            print("✅ Teste de propriedades do quaternion rotacional PASSOU")
            return True
        else:
            print("❌ Teste de propriedades do quaternion rotacional FALHOU")
            return False

    except Exception as e:
        print(f"❌ Erro no teste de propriedades do quaternion rotacional: {e}")
        return False


def main() -> bool:
    """Main validation function that runs all core property tests."""
    print("🔬 VALIDAÇÃO DAS PROPRIEDADES DO NÚCLEO ΨQRH")
    print("=" * 60)
    print("Este script valida as propriedades matemáticas fundamentais do sistema ΨQRH")
    print("para garantir estabilidade de energia e corretude numérica.")
    print()

    # Run all tests
    test_results = []

    test_results.append(("Conservação de Energia", run_energy_conservation_test()))
    test_results.append(("Teorema de Parseval", run_parseval_validation_test()))
    test_results.append(("Benchmark de Memória", run_memory_benchmark_test()))
    test_results.append(("Quaternion Rotacional", run_rotational_quaternion_test()))

    # Print summary
    print("\n" + "=" * 60)
    print("📋 RESUMO DA VALIDAÇÃO DO NÚCLEO")
    print("=" * 60)

    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {test_name}: {status}")

    print(f"\n📊 Resultado: {passed_tests}/{total_tests} testes passaram")

    if passed_tests == total_tests:
        print("\n🎉 ✅ VALIDAÇÃO DAS PROPRIEDADES DO NÚCLEO CONCLUÍDA COM SUCESSO!")
        print("O sistema ΨQRH mantém todas as propriedades matemáticas fundamentais.")
        return True
    else:
        print(f"\n⚠️  ❌ VALIDAÇÃO FALHOU: {total_tests - passed_tests} teste(s) falharam")
        print("Revise as configurações ou o código modificado.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Validação interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erro crítico durante a validação: {e}")
        sys.exit(1)