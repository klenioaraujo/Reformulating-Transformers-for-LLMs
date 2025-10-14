# teste_completo_automatico.py
import subprocess
import sys
import os

def run_test(test_file):
    """Executa um teste e retorna se foi bem-sucedido"""
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos de timeout
        )

        if result.returncode == 0:
            print(f"✅ {test_file} - SUCESSO")
            return True
        else:
            print(f"❌ {test_file} - FALHA")
            print(f"   Erro: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {test_file} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {test_file} - ERRO: {e}")
        return False

def main():
    print("🎯 EXECUTANDO SUITE DE TESTES ΨQRH")
    print("=" * 50)

    tests = [
        "teste_basico.py",
        "teste_performance.py",
        "teste_integracao.py",
        "teste_carga_pesada.py"
    ]

    results = []
    for test in tests:
        if os.path.exists(test):
            success = run_test(test)
            results.append((test, success))
        else:
            print(f"⚠️  {test} - ARQUIVO NÃO ENCONTRADO")
            results.append((test, False))

    print("\n📋 RESUMO DOS TESTES:")
    print("=" * 30)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test}")

    print(f"\n📊 RESULTADO: {passed}/{total} testes passaram")

    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("🚀 SISTEMA ΨQRH VALIDADO COM SUCESSO!")
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam")
        sys.exit(1)

if __name__ == "__main__":
    main()