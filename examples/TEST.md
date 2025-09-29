$ python3 basic_usage.py
python3 parseval_validation_test.py
python3 energy_conservation_test.py
python3 advanced_energy_test.py
ΨQRH Transformer Demonstration
==================================================
ΨQRH Transformer - Basic Usage Example
==================================================
Creating ΨQRH Transformer...

Model Architecture:
  vocab_size: 10000
  d_model: 512
  n_layers: 6
  n_heads: 8
  dim_feedforward: 2048
  total_parameters: 192115600
  architecture: ΨQRH Transformer

Input shape: torch.Size([2, 128])

Running forward pass...
Output shape: torch.Size([2, 128, 10000])
Output range: [-2.2599, 2.3605]

==================================================
Mathematical Validation
==================================================
Running comprehensive mathematical validation...

ΨQRH Mathematical Validation Report
==================================================
Energy Conservation: PASS
  Input Norm: 256.103119
  Output Norm: 256.103119
  Ratio: 1.000000 (target: 1.0 ± 0.05)
Unitarity: FAIL
  Mean Magnitude: 0.841400 (target: 1.0 ± 0.05)
  Std Magnitude: 0.562917
Numerical Stability: PASS
  Passes: 1000
  NaN Count: 0
  Inf Count: 0
Quaternion Properties: PASS
  Identity: PASS
  Inverse: PASS
Spectral Operations: FAIL
  FFT Consistency: PASS
  Parseval Theorem: FAIL
  Parseval Ratio: 0.015625
--------------------------------------------------
Overall Validation: FAIL
  Tests Passed: 4/6

==================================================
Basic Performance Comparison
==================================================
/home/padilha/.local/lib/python3.12/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Measuring ΨQRH inference time...
ΨQRH Inference Time: 2.3683 seconds
ΨQRH Output Shape: torch.Size([4, 256, 5000])

Note: Full performance comparison requires proper standard transformer setup
This example demonstrates the basic ΨQRH usage pattern.

==================================================
Demonstration Complete!
==================================================

Next steps:
1. Run mathematical validation on your specific use case
2. Compare performance with standard transformers
3. Explore different model configurations
4. Check out the implementation roadmap for upcoming features
ΨQRH - Complete Parseval Theorem Validation
============================================================
Objective: Verify compliance with ||x||² = ||F{x}||²
============================================================

=== COMPREHENSIVE PARSEVAL VALIDATION ===
==================================================

Running: FFT Compliance
=== FFT/Parseval Compliance Test ===
==================================================
Signal shape: torch.Size([1, 128, 1])

1. FFT without normalization:
   Time energy: 78.127945
   Frequency energy: 10000.376953
   Ratio: 0.007812
   Parseval valid: ❌ NO

2. FFT with norm='ortho':
   Time energy: 78.127945
   Frequency energy: 78.127937
   Ratio: 1.000000
   Parseval valid: ✅ YES

3. IFFT Reconstruction:
   Reconstruction error: 0.000001
   Perfect reconstruction: ❌ NO
   Result: ❌ FAIL

Running: Energy Preservation

=== Energy Preservation Function Test ===
==================================================
Input shape: torch.Size([2, 128, 512])
Output shape: torch.Size([2, 128, 512])

Energies before:
   Input: 510.763306
   Output: 4601.382812
   Ratio: 9.008836

Energies after:
   Normalized: 510.763306
   Ratio: 1.000000
   Preservation: ✅ PERFECT
   Result: ✅ PASS

Running: ΨQRH Parseval

=== ΨQRH with Parseval Validation Test ===
==================================================
Creating ΨQRH transformer...
Input shape: torch.Size([1, 64])

Validating Parseval during forward pass...

Results:
   Input energy: 1023.397095
   Output energy: 1023.397095
   Conservation ratio: 1.000000
   Initial Parseval: ✅ OK
   Final Parseval: ✅ OK
   Energy conservation: ✅ OK
   Result: ✅ PASS

Running: Spectral Operation

=== Spectral Operation Wrapper Test ===
==================================================
Input shape: torch.Size([1, 128, 512])
Result shape: torch.Size([1, 128, 512])
Parseval preserved: ❌ NO
   Result: ❌ FAIL

==================================================
PARSEVAL VALIDATION SUMMARY
==================================================
  FFT Compliance: ❌ FAIL
  Energy Preservation: ✅ PASS
  ΨQRH Parseval: ✅ PASS
  Spectral Operation: ❌ FAIL

Total: 2/4 tests passed

⚠️  2 test(s) failed
❌ Review spectral implementations

============================================================
❌ PARSEVAL VALIDATION FAILED
❌ Review FFT implementations
============================================================
ΨQRH Energy Conservation Test Suite
==================================================
Testing Energy Normalizer
==================================================
Basic Energy Normalizer:
  Input Energy: 362.788788
  Output Energy: 362.788788
  Conservation Ratio: 1.000000
  Target: 1.000000 ± 0.05
  Status: PASS

Advanced Energy Controller:
  Controlled Energy: 362.788788
  Controlled Ratio: 1.000000
  Status: PASS

==================================================
Testing Enhanced ΨQRH with Energy Conservation
==================================================
Running enhanced ΨQRH forward pass...
Output shape: torch.Size([1, 64, 1000])
Output range: [-4.5010, 4.8504]

Enhanced ΨQRH Energy Conservation:
  Input Energy: 256.260742
  Output Energy: 256.260773
  Conservation Ratio: 1.000000
  Target: 1.000000 ± 0.05
  Status: PASS

==================================================
Comparing Original vs Enhanced ΨQRH
==================================================
Testing Original ΨQRH...
Testing Enhanced ΨQRH...

Comparison Results:
  Input Energy: 256.582550
  Original Output Energy: 256.582458
  Original Ratio: 1.000000
  Enhanced Output Energy: 256.582550
  Enhanced Ratio: 1.000000
  Improvement: 0.00x closer to 1.0

==================================================
Summary of Energy Conservation Improvements
==================================================
Original ΨQRH Conservation Ratio: 1.000000
Enhanced ΨQRH Conservation Ratio: 1.000000
Target Range: 0.95 - 1.05

Original Deviation from 1.0: 0.000000
Enhanced Deviation from 1.0: 0.000000
Improvement: 100.0% closer to target

==================================================
Energy Conservation Test Complete!
==================================================
ΨQRH - Validação Científica de Conservação de Energia
============================================================
Objetivo: energy_ratio ∈ [0.95, 1.05] em todos os cenários
============================================================

=== Validação Abrangente de Conservação de Energia ===
============================================================
=== Teste de Compliance com Teorema de Parseval ===
============================================================
Energia no domínio do tempo: 130441.221500
Energia no domínio da frequência: 1019.072904
Razão Parseval: 0.007813
Erro de reconstrução: 0.000050

Compliance Parseval: FAIL
Reconstrução precisa: FAIL

=== Teste de Controle de Energia por Camada ===
============================================================
Resultados por Camada:
  Camada 0: Razão = 1.000000 [PASS]
  Camada 1: Razão = 1.000000 [PASS]
  Camada 2: Razão = 1.000000 [PASS]
  Camada 3: Razão = 1.000000 [PASS]
  Camada 4: Razão = 1.000000 [PASS]
  Camada 5: Razão = 1.000000 [PASS]

Estatísticas:
  Camadas compliant: 6/6
  Razão média: 1.000000
  Status global: PASS

=== Teste de ΨQRH com Controle Avançado de Energia ===
============================================================
Criando ΨQRH com controle de energia...

Testando modelos...

Resultados de Conservação de Energia:
  Energia de entrada: 256.766266
  Energia sem controle: 255.316772 (Razão: 0.994355)
  Energia com controle: 256.766083 (Razão: 0.999999)

Status de Compliance:
  Sem controle: PASS
  Com controle: PASS
  Melhoria: 0.00x mais próximo de 1.0

=== Teste do Controlador Básico ===

============================================================
RESUMO FINAL DA VALIDAÇÃO
============================================================
  Teorema de Parseval: FAIL
  Reconstrução Espectral: FAIL
  Controle por Camada: PASS
  Controlador Básico: PASS
  ΨQRH com Energia: PASS

Resultado Geral: 3/5 testes PASS
Razão Final de Conservação: 0.999999
Melhoria: 0.00x

⚠️  2 teste(s) falharam.
Revisar implementação do controle de energia.

=== SCI_005: Cenário de Conservação de Energia ===
============================================================

Testando: batch_size=1, seq_len=64
  Razão: 0.999998 [PASS]

Testando: batch_size=2, seq_len=64
  Razão: 0.999996 [PASS]

Testando: batch_size=4, seq_len=64
  Razão: 0.999989 [PASS]

Testando: batch_size=1, seq_len=128
  Razão: 0.999997 [PASS]

Testando: batch_size=2, seq_len=128
  Razão: 0.999989 [PASS]

Testando: batch_size=4, seq_len=128
  Razão: 0.999970 [PASS]

Testando: batch_size=1, seq_len=256
  Razão: 0.999988 [PASS]

Testando: batch_size=2, seq_len=256
  Razão: 0.999972 [PASS]

Testando: batch_size=4, seq_len=256
  Razão: 0.999920 [PASS]

Resumo SCI_005:
  Testes compliant: 9/9
  Razão média: 0.999980
  Status: PASS

============================================================
RELATÓRIO FINAL DE CONSERVAÇÃO DE ENERGIA
============================================================
❌ SISTEMA NÃO COMPLIANT
❌ Razão final: 0.999999 ∉ [0.95, 1.05]
❌ Revisar implementação do controle de energia

============================================================
