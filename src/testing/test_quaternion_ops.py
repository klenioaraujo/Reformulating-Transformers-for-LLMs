"""
Testes Mínimos para Operações Quaternion

Testes básicos para validar as operações quaternion fundamentais.
"""

import torch
import sys
import os

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_quaternion_multiply():
    """Teste da multiplicação quaternion (Hamilton product)"""
    from src.core.quaternion_operations import quaternion_multiply

    # Quaternions simples
    q1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    q2 = torch.tensor([5.0, 6.0, 7.0, 8.0])

    result = quaternion_multiply(q1, q2)

    # Verificar resultado esperado
    expected = torch.tensor([-60.0, 12.0, 30.0, 24.0])

    assert torch.allclose(result, expected, rtol=1e-5), f"Expected {expected}, got {result}"
    print("✅ Teste de multiplicação quaternion passou")

def test_quaternion_conjugate():
    """Teste do conjugado quaternion"""
    from src.core.quaternion_operations import quaternion_conjugate

    q = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = quaternion_conjugate(q)

    expected = torch.tensor([1.0, -2.0, -3.0, -4.0])

    assert torch.allclose(result, expected, rtol=1e-5), f"Expected {expected}, got {result}"
    print("✅ Teste de conjugado quaternion passou")

def test_quaternion_norm():
    """Teste da norma quaternion"""
    from src.core.quaternion_operations import quaternion_norm

    q = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = quaternion_norm(q)

    expected = torch.sqrt(torch.tensor(30.0))  # sqrt(1² + 2² + 3² + 4²) = sqrt(30)

    assert torch.allclose(result, expected, rtol=1e-5), f"Expected {expected}, got {result}"
    print("✅ Teste de norma quaternion passou")

def run_all_tests():
    """Executar todos os testes"""
    print("🧪 Executando Testes Mínimos de Operações Quaternion...")

    try:
        test_quaternion_multiply()
        test_quaternion_conjugate()
        test_quaternion_norm()
        print("\n🎉 Todos os testes mínimos passaram!")
        return True
    except Exception as e:
        print(f"\n❌ Teste falhou: {e}")
        return False

if __name__ == "__main__":
    run_all_tests()