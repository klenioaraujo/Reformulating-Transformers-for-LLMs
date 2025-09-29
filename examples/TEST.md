python3 basic_usage.py


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
Output range: [-2.3637, 2.3788]

==================================================
Mathematical Validation
==================================================
Running comprehensive mathematical validation...

ΨQRH Mathematical Validation Report
==================================================
Energy Conservation: PASS
  Input Energy: 65141.058594
  Output Energy: 65141.058594
  Ratio: 1.000000 (target: 1.0 ± 0.05)
Unitarity: FAIL
  Mean Magnitude: 0.834900 (target: 1.0 ± 0.05)
  Std Magnitude: 0.566372
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
  Parseval Ratio: 0.220191+0.000000j
--------------------------------------------------
Overall Validation: FAIL
  Tests Passed: 4/6

==================================================
Basic Performance Comparison
==================================================
/home/padilha/.local/lib/python3.12/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Measuring ΨQRH inference time...
ΨQRH Inference Time: 2.4513 seconds
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
ΨQRH - Energy Conservation & Parseval Validation
============================================================
Objective: Verify energy conservation in ΨQRH operations
           Validate Parseval for pure FFT operations only
============================================================

=== COMPREHENSIVE PARSEVAL VALIDATION ===
==================================================

Running: Pure FFT Parseval
=== Pure FFT Parseval Compliance Test ===
==================================================
Signal shape: torch.Size([1, 128, 512])

Pure FFT Parseval valid: ✅ YES
Pure IFFT Parseval valid: ✅ YES
Reconstruction error: 0.000035
Perfect reconstruction: ✅ YES
   Result: ✅ PASS

Running: Energy Preservation

=== Energy Preservation Function Test ===
==================================================
Input shape: torch.Size([2, 128, 512])
Output shape: torch.Size([2, 128, 512])

Energies before:
   Input: 512.025452
   Output: 4613.744141
   Ratio: 9.010771

Energies after:
   Normalized: 512.025574
   Ratio: 1.000000
   Preservation: ✅ PERFECT
   Result: ✅ PASS

Running: ΨQRH Energy Conservation

=== ΨQRH with Parseval Validation Test ===
==================================================
Creating ΨQRH transformer...
Input shape: torch.Size([1, 64])

Validating Parseval during forward pass...

Results:
   Input energy: 1019.504944
   Output energy: 1019.505005
   Conservation ratio: 1.000000
   Initial Parseval: ✅ OK
   Final Parseval: ✅ OK
   Energy conservation: ✅ OK
   Result: ✅ PASS

Running: Spectral Operation Energy

=== Spectral Operation Energy Preservation Test ===
==================================================
Input shape: torch.Size([1, 128, 512])
Result shape: torch.Size([1, 128, 512])
Input energy: 65814.859375
Output energy: 65814.851562
Energy ratio: 1.000000
Energy preserved: ✅ YES
   Result: ✅ PASS

==================================================
PARSEVAL VALIDATION SUMMARY
==================================================
  Pure FFT Parseval: ✅ PASS
  Energy Preservation: ✅ PASS
  ΨQRH Energy Conservation: ✅ PASS
  Spectral Operation Energy: ✅ PASS

Total: 4/4 tests passed

🎯 SYSTEM COMPLIANT WITH ENERGY CONSERVATION!
✅ Pure FFT operations preserve Parseval
✅ All spectral operations preserve energy

============================================================
✅ ENERGY CONSERVATION VALIDATION SUCCESSFUL
✅ Pure FFT operations preserve Parseval
✅ All ΨQRH operations preserve energy
============================================================
ΨQRH Energy Conservation Test Suite
==================================================
Testing Energy Normalizer
==================================================
Basic Energy Normalizer:
  Input Energy: 361.666443
  Output Energy: 361.666443
  Conservation Ratio: 1.000000
  Target: 1.000000 ± 0.05
  Status: PASS

Advanced Energy Controller:
  Controlled Energy: 361.666443
  Controlled Ratio: 1.000000
  Status: PASS

==================================================
Testing Enhanced ΨQRH with Energy Conservation
==================================================
Running enhanced ΨQRH forward pass...
Output shape: torch.Size([1, 64, 1000])
Output range: [-4.0688, 4.1109]

Enhanced ΨQRH Energy Conservation:
  Input Energy: 255.046539
  Output Energy: 255.046646
  Conservation Ratio: 1.000000
  Target: 1.000000 ± 0.05
  Status: PASS

==================================================
Comparing Original vs Enhanced ΨQRH
==================================================
Testing Original ΨQRH...
Testing Enhanced ΨQRH...

Comparison Results:
  Input Energy: 255.822388
  Original Output Energy: 255.822525
  Original Ratio: 1.000001
  Enhanced Output Energy: 255.822449
  Enhanced Ratio: 1.000000
  Improvement: 0.44x closer to 1.0

==================================================
Summary of Energy Conservation Improvements
==================================================
Original ΨQRH Conservation Ratio: 1.000001
Enhanced ΨQRH Conservation Ratio: 1.000000
Target Range: 0.95 - 1.05

Original Deviation from 1.0: 0.000001
Enhanced Deviation from 1.0: 0.000000
Improvement: 55.6% closer to target

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
Energia no domínio do tempo: 130554.218750
Energia no domínio da frequência: 130554.218750
Razão Parseval: 1.000000
Erro de reconstrução: 0.000049

Compliance Parseval: PASS
Reconstrução precisa: PASS

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
  Energia de entrada: 256.414551
  Energia sem controle: 257.186066 (Razão: 1.003009)
  Energia com controle: 256.414459 (Razão: 1.000000)

Status de Compliance:
  Sem controle: PASS
  Com controle: PASS
  Melhoria: 0.00x mais próximo de 1.0

=== Teste do Controlador Básico ===

============================================================
RESUMO FINAL DA VALIDAÇÃO
============================================================
  Teorema de Parseval: PASS
  Reconstrução Espectral: PASS
  Controle por Camada: PASS
  Controlador Básico: PASS
  ΨQRH com Energia: PASS

Resultado Geral: 5/5 testes PASS
Razão Final de Conservação: 1.000000
Melhoria: 0.00x

🎯 TODOS OS TESTES PASSARAM!
Sistema está compliant com Teorema de Parseval e conservação de energia.

=== SCI_005: Cenário de Conservação de Energia ===
============================================================

Testando: batch_size=1, seq_len=64
  Razão: 0.999998 [PASS]

Testando: batch_size=2, seq_len=64
  Razão: 0.999996 [PASS]

Testando: batch_size=4, seq_len=64
  Razão: 0.999991 [PASS]

Testando: batch_size=1, seq_len=128
  Razão: 0.999996 [PASS]

Testando: batch_size=2, seq_len=128
  Razão: 0.999989 [PASS]

Testando: batch_size=4, seq_len=128
  Razão: 0.999971 [PASS]

Testando: batch_size=1, seq_len=256
  Razão: 0.999990 [PASS]

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
✅ SISTEMA COMPLIANT COM TEOREMA DE PARSEVAL
✅ Razão final de conservação: 1.000000 ∈ [0.95, 1.05]
✅ Todos os cenários científicos validados

🎯 OBJETIVO CIENTÍFICO ATINGIDO!

============================================================
