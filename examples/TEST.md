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
Output range: [-2.1624, 2.2475]

==================================================
Mathematical Validation
==================================================
Running comprehensive mathematical validation...

ΨQRH Mathematical Validation Report
==================================================
Energy Conservation: PASS
  Input Energy: 65718.656250
  Output Energy: 65718.664062
  Ratio: 1.000000 (target: 1.0 ± 0.05)
Unitarity: FAIL
  Mean Magnitude: 0.856090 (target: 1.0 ± 0.05)
  Std Magnitude: 0.542188
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
  Parseval Ratio: 0.160304+0.000000j
--------------------------------------------------
Overall Validation: FAIL
  Tests Passed: 4/6

==================================================
Basic Performance Comparison
==================================================
/home/padilha/.local/lib/python3.12/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Measuring ΨQRH inference time...
ΨQRH Inference Time: 2.3847 seconds
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
   Input: 513.604553
   Output: 4620.888184
   Ratio: 8.996977

Energies after:
   Normalized: 513.604553
   Ratio: 1.000000
   Preservation: ✅ PERFECT
   Result: ✅ PASS

Running: ΨQRH Parseval

=== ΨQRH with Parseval Validation Test ===
==================================================
Creating ΨQRH transformer...
Input shape: torch.Size([1, 64])

Validating Parseval during forward pass...
⚠️  Parseval violation: ratio=117.936951+0.000006j, tolerance=1e-05
   Time domain energy: 65228.250000
   Freq domain energy: 553.077271-0.000030j
⚠️  Parseval violation in input_embeddings
   Time domain energy: 65228.250000
   Freq domain energy: 553.077271-0.000030j
   Ratio: 117.936957+0.000006j
⚠️  Parseval violation: ratio=255.437439-0.000017j, tolerance=1e-05
   Time domain energy: 65228.250000
   Freq domain energy: 255.359009+0.000017j
⚠️  Parseval violation in final_output
   Time domain energy: 65228.250000
   Freq domain energy: 255.359009+0.000017j
   Ratio: 255.437434-0.000017j

Results:
   Input energy: 1019.191406
   Output energy: 1019.191406
   Conservation ratio: 1.000000
   Initial Parseval: ❌ FAILED
   Final Parseval: ❌ FAILED
   Energy conservation: ✅ OK
   Result: ❌ FAIL

Running: Spectral Operation

=== Spectral Operation Wrapper Test ===
==================================================
Input shape: torch.Size([1, 128, 512])
⚠️  Parseval violation: ratio=94.296898-0.000003j, tolerance=1e-05
   Time domain energy: 65851.656250
   Freq domain energy: 698.343811+0.000025j
⚠️  Parseval violation in test_spectral_op_input
   Time domain energy: 65851.656250
   Freq domain energy: 698.343811+0.000025j
   Ratio: 94.296900-0.000003j
⚠️  Parseval violation: ratio=85.527771+0.000002j, tolerance=1e-05
   Time domain energy: 65535.996094
   Freq domain energy: 766.253967-0.000014j
⚠️  Parseval violation in test_spectral_op_output
   Time domain energy: 65535.996094
   Freq domain energy: 766.253967-0.000014j
   Ratio: 85.527774+0.000002j
Result shape: torch.Size([1, 128, 512])
⚠️  Parseval violation: ratio=1.004817, tolerance=1e-05
   Time domain energy: 65851.656250
   Freq domain energy: 65535.996094
Parseval preserved: ❌ NO
   Result: ❌ FAIL

==================================================
PARSEVAL VALIDATION SUMMARY
==================================================
  FFT Compliance: ❌ FAIL
  Energy Preservation: ✅ PASS
  ΨQRH Parseval: ❌ FAIL
  Spectral Operation: ❌ FAIL

Total: 1/4 tests passed

⚠️  3 test(s) failed
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
  Input Energy: 362.055267
  Output Energy: 362.055267
  Conservation Ratio: 1.000000
  Target: 1.000000 ± 0.05
  Status: PASS

Advanced Energy Controller:
  Controlled Energy: 362.055267
  Controlled Ratio: 1.000000
  Status: PASS

==================================================
Testing Enhanced ΨQRH with Energy Conservation
==================================================
Running enhanced ΨQRH forward pass...
Output shape: torch.Size([1, 64, 1000])
Output range: [-3.8060, 4.5551]

Enhanced ΨQRH Energy Conservation:
  Input Energy: 254.983063
  Output Energy: 254.983078
  Conservation Ratio: 1.000000
  Target: 1.000000 ± 0.05
  Status: PASS

==================================================
Comparing Original vs Enhanced ΨQRH
==================================================
Testing Original ΨQRH...
Testing Enhanced ΨQRH...

Comparison Results:
  Input Energy: 255.532730
  Original Output Energy: 255.532623
  Original Ratio: 1.000000
  Enhanced Output Energy: 255.532700
  Enhanced Ratio: 1.000000
  Improvement: 0.29x closer to 1.0

==================================================
Summary of Energy Conservation Improvements
==================================================
Original ΨQRH Conservation Ratio: 1.000000
Enhanced ΨQRH Conservation Ratio: 1.000000
Target Range: 0.95 - 1.05

Original Deviation from 1.0: 0.000000
Enhanced Deviation from 1.0: 0.000000
Improvement: 71.4% closer to target

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
Energia no domínio do tempo: 130715.017404
Energia no domínio da frequência: 1021.209005
Razão Parseval: 0.007812
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
  Energia de entrada: 255.054855
  Energia sem controle: 256.394012 (Razão: 1.005250)
  Energia com controle: 255.054810 (Razão: 1.000000)

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
Razão Final de Conservação: 1.000000
Melhoria: 0.00x

⚠️  2 teste(s) falharam.
Revisar implementação do controle de energia.

=== SCI_005: Cenário de Conservação de Energia ===
============================================================

Testando: batch_size=1, seq_len=64
  Razão: 0.999999 [PASS]

Testando: batch_size=2, seq_len=64
  Razão: 0.999997 [PASS]

Testando: batch_size=4, seq_len=64
  Razão: 0.999992 [PASS]

Testando: batch_size=1, seq_len=128
  Razão: 0.999997 [PASS]

Testando: batch_size=2, seq_len=128
  Razão: 0.999989 [PASS]

Testando: batch_size=4, seq_len=128
  Razão: 0.999970 [PASS]

Testando: batch_size=1, seq_len=256
  Razão: 0.999990 [PASS]

Testando: batch_size=2, seq_len=256
  Razão: 0.999974 [PASS]

Testando: batch_size=4, seq_len=256
  Razão: 0.999915 [PASS]

Resumo SCI_005:
  Testes compliant: 9/9
  Razão média: 0.999980
  Status: PASS

============================================================
RELATÓRIO FINAL DE CONSERVAÇÃO DE ENERGIA
============================================================
❌ SISTEMA NÃO COMPLIANT
❌ Razão final: 1.000000 ∉ [0.95, 1.05]
❌ Revisar implementação do controle de energia

============================================================
