#!/usr/bin/env python3
"""
Script para executar todos os testes do sistema ΨQRH
"""

import unittest
import sys
import os
from pathlib import Path

# Adicionar diretório raiz do projeto ao path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Adicionar diretório ΨQRHSystem ao path
psiqrh_root = Path(__file__).parent.parent
sys.path.insert(0, str(psiqrh_root))


def run_all_tests():
    """Executa todos os testes"""
    print("🚀 Executando testes do sistema ΨQRH")
    print("=" * 50)

    # Descobrir todos os testes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Diretório de testes
    test_dir = Path(__file__).parent

    # Carregar testes
    test_files = [
        'test_config.py',
        'test_physics.py',
        'test_core.py',
        'test_integration.py'
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"\n📋 Carregando testes: {test_file}")
            try:
                # Importar módulo diretamente
                module_name = test_file[:-3]  # Remove .py
                if module_name == 'test_config':
                    import test_config as module
                elif module_name == 'test_physics':
                    import test_physics as module
                elif module_name == 'test_core':
                    import test_core as module
                elif module_name == 'test_integration':
                    import test_integration as module
                else:
                    raise ImportError(f"Módulo {module_name} não suportado")

                tests = loader.loadTestsFromModule(module)
                suite.addTests(tests)

                # Contar testes
                test_count = tests.countTestCases()
                total_tests += test_count
                print(f"   ✅ {test_count} testes carregados")

            except Exception as e:
                print(f"   ❌ Erro ao carregar {test_file}: {e}")
        else:
            print(f"   ⚠️  Arquivo de teste não encontrado: {test_file}")

    print(f"\n🎯 Total de testes a executar: {total_tests}")
    print("=" * 50)

    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    print("📊 RESULTADO DOS TESTES")
    print("=" * 50)

    print(f"Total de testes: {result.testsRun}")
    print(f"Aprovados: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Falhas: {len(result.failures)}")
    print(f"Erros: {len(result.errors)}")

    if result.failures:
        print(f"\n❌ FALHAS:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback[:100]}...")

    if result.errors:
        print(f"\n💥 ERROS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback[:100]}...")

    # Resumo final
    if result.wasSuccessful():
        print(f"\n🎉 TODOS OS TESTES APROVADOS! ✅")
        return 0
    else:
        print(f"\n⚠️  Alguns testes falharam. Verifique os detalhes acima.")
        return 1


def run_specific_test(test_name):
    """Executa um teste específico"""
    print(f"🎯 Executando teste específico: {test_name}")

    try:
        # Importar módulo de teste diretamente
        if test_name == 'test_config':
            import test_config as module
        elif test_name == 'test_physics':
            import test_physics as module
        elif test_name == 'test_core':
            import test_core as module
        elif test_name == 'test_integration':
            import test_integration as module
        else:
            raise ImportError(f"Teste '{test_name}' não encontrado")

        # Carregar e executar
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return 0 if result.wasSuccessful() else 1

    except ImportError:
        print(f"❌ Teste '{test_name}' não encontrado")
        return 1


def main():
    """Função principal"""
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        exit_code = run_specific_test(test_name)
    else:
        exit_code = run_all_tests()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()