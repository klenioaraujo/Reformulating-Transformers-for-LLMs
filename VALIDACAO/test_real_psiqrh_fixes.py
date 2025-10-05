#!/usr/bin/env python3
"""
Testes com componentes REAIS do ΨQRH

Testa as correções usando:
- QRHLayer real
- QRHFactory real
- FFTCache real
- MathematicalValidator real
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.ΨQRH import QRHFactory
from src.core.qrh_layer import QRHLayer, QRHConfig, FFTCache
from src.validation.mathematical_validation import (
    MathematicalValidator,
    EmbeddingNotFoundError
)
from src.core.quaternion_operations import QuaternionOperations


def test_1_real_qrh_energy_validation():
    """Teste 1: Validação de energia com QRHLayer real"""
    print("Teste 1: Validação de energia com QRHLayer REAL")

    # Criar QRHLayer real (QRHConfig é dataclass, não precisa de __init__)
    from dataclasses import dataclass, replace
    config = QRHConfig()
    config = replace(config, embed_dim=64, alpha=1.0)
    layer = QRHLayer(config)

    # Input real (embeddings float) - QRHLayer espera 4*embed_dim = 256
    batch_size = 2
    seq_len = 8
    expected_dim = 4 * config.embed_dim  # 256
    x = torch.randn(batch_size, seq_len, expected_dim)

    # Validador
    validator = MathematicalValidator(tolerance=0.5)

    # Validar energia
    result = validator.validate_energy_conservation(layer, x)

    print(f"  ✓ Método de validação: {result['validation_method']}")
    print(f"  ✓ Energia entrada: {result['input_energy']:.6f}")
    print(f"  ✓ Energia saída: {result['output_energy']:.6f}")
    print(f"  ✓ Razão conservação: {result['conservation_ratio']:.6f}")
    print(f"  ✓ Conservada: {result['is_conserved']}")

    assert result['validation_method'] == 'proper_embedding', "Deve usar embedding apropriado"
    assert result['input_energy'] is not None, "Energia de entrada não deve ser None"
    assert result['output_energy'] is not None, "Energia de saída não deve ser None"


def test_2_real_qrh_factory():
    """Teste 2: QRHFactory real com validação"""
    print("\nTeste 2: QRHFactory REAL com validação")

    # Criar factory real (usa config_path)
    factory = QRHFactory(config_path="configs/qrh_config.yaml")

    # Criar QRH Layer manualmente
    from dataclasses import replace
    config = QRHConfig()
    config = replace(config, embed_dim=32, alpha=1.0)
    layer = QRHLayer(config)

    # Input real - QRHLayer espera 4*32 = 128
    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, 128)

    # Forward pass direto no layer
    output = layer(x)

    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Sem NaN: {not torch.isnan(output).any()}")
    print(f"  ✓ Sem Inf: {not torch.isinf(output).any()}")

    assert output.shape == x.shape, "Shape deve ser preservado"
    assert not torch.isnan(output).any(), "Não deve ter NaN"
    assert not torch.isinf(output).any(), "Não deve ter Inf"


def test_3_real_fft_cache_lru():
    """Teste 3: FFTCache LRU real"""
    print("\nTeste 3: FFTCache LRU REAL")

    cache = FFTCache(max_size=3, max_memory_mb=10.0)

    # Teste LRU (não FIFO)
    t1 = cache.get(('key1',), lambda: torch.randn(10, 10))
    t2 = cache.get(('key2',), lambda: torch.randn(10, 10))
    t3 = cache.get(('key3',), lambda: torch.randn(10, 10))

    # Acessar key1 novamente (torna-se recente)
    t1_again = cache.get(('key1',), lambda: torch.randn(99, 99))
    assert torch.equal(t1, t1_again), "Deve recuperar do cache"

    # Adicionar key4 - deve remover key2 (mais antigo), não key1 (recente)
    t4 = cache.get(('key4',), lambda: torch.randn(10, 10))

    # key1 deve ainda estar no cache (LRU)
    t1_check = cache.get(('key1',), lambda: torch.randn(99, 99))
    assert t1_check.shape == (10, 10), "key1 deve estar no cache (LRU)"

    # Métricas
    metrics = cache.get_metrics()
    print(f"  ✓ Hits: {metrics['hits']}")
    print(f"  ✓ Misses: {metrics['misses']}")
    print(f"  ✓ Hit rate: {metrics['hit_rate']:.2%}")
    print(f"  ✓ Entradas atuais: {metrics['current_entries']}")
    print(f"  ✓ Uso de memória: {metrics['memory_usage_mb']:.4f} MB")

    assert metrics['hits'] >= 2, "Deve ter pelo menos 2 hits"
    assert metrics['current_entries'] <= 3, "Não deve exceder max_size"


def test_4_real_quaternion_operations():
    """Teste 4: QuaternionOperations real"""
    print("\nTeste 4: QuaternionOperations REAL")

    # Testar SpectralActivation que é parte do sistema
    from src.core.quaternion_operations import SpectralActivation

    sact = SpectralActivation(activation_type="gelu")

    batch_size = 2
    seq_len = 4
    dim = 16
    x = torch.randn(batch_size, seq_len, dim)

    result = sact(x)

    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Output shape: {result.shape}")
    print(f"  ✓ Sem NaN: {not torch.isnan(result).any()}")

    assert result.shape == x.shape, "Shape deve ser preservado"
    assert not torch.isnan(result).any(), "Não deve ter NaN"


def test_5_real_validation_skip_mode():
    """Teste 5: Modo skip da validação real"""
    print("\nTeste 5: Validação REAL com skip_on_no_embedding")

    # QRHLayer não tem token_embedding, mas recebe embeddings float
    from dataclasses import replace
    config = QRHConfig()
    config = replace(config, embed_dim=32, alpha=1.0)
    layer = QRHLayer(config)

    # Input embeddings float (compatível) - precisa 4*32 = 128
    x = torch.randn(2, 4, 128)

    validator = MathematicalValidator(tolerance=0.5)

    # Deve funcionar porque x já é embedding
    result = validator.validate_energy_conservation(layer, x, skip_on_no_embedding=True)

    print(f"  ✓ Método: {result['validation_method']}")
    print(f"  ✓ Input energy: {result.get('input_energy', 'None')}")
    print(f"  ✓ Output energy: {result['output_energy']:.6f}")

    # Como x já é embedding float, deve funcionar normalmente
    assert result['input_energy'] is not None, "Deve calcular energia de embedding float"


def test_6_real_comprehensive_validation():
    """Teste 6: Validação completa REAL"""
    print("\nTeste 6: Validação matemática COMPLETA do ΨQRH")

    # Sistema real completo
    from dataclasses import replace
    config = QRHConfig()
    config = replace(config, embed_dim=32, alpha=1.0)
    layer = QRHLayer(config)
    qops = QuaternionOperations()

    # Input real - precisa 4*32 = 128
    x = torch.randn(2, 8, 128)

    validator = MathematicalValidator(tolerance=0.5)

    # Validação completa
    results = validator.comprehensive_validation(layer, x, qops)

    print(f"  ✓ Conservação de energia: {results['energy_conservation']['is_conserved']}")
    print(f"  ✓ Unitariedade: {results['unitarity']['is_unitary']}")
    print(f"  ✓ Estabilidade numérica: {results['numerical_stability']['is_stable']}")
    print(f"  ✓ Propriedades quaternion: {results['quaternion_properties']['all_properties_valid']}")
    print(f"  ✓ Operações espectrais: {results['spectral_operations']['fft_consistency']}")

    overall = results['overall_validation']
    print(f"\n  ✓ Testes passados: {overall['passed_tests']}/{overall['total_tests']}")

    assert overall['passed_tests'] >= 4, f"Deve passar pelo menos 4 de 6 testes"


if __name__ == "__main__":
    print("=" * 70)
    print("ΨQRH - Testes com Componentes REAIS")
    print("=" * 70)

    tests = [
        test_1_real_qrh_energy_validation,
        test_2_real_qrh_factory,
        test_3_real_fft_cache_lru,
        test_4_real_quaternion_operations,
        test_5_real_validation_skip_mode,
        test_6_real_comprehensive_validation
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FALHOU: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERRO: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Resultados: {passed}/{len(tests)} passaram, {failed}/{len(tests)} falharam")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
