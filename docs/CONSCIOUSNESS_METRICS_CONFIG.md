# Consciousness Metrics Configuration Guide

## Visão Geral

Este guia documenta o sistema de configuração das métricas de consciência fractal (FCI - Fractal Consciousness Index) e como personalizar os thresholds de estados de consciência.

## Problema Corrigido

### ❌ Problema Original
A fórmula de mapeamento estava incorreta:
```python
# Incorreto (divisão por 9.0)
fci = (dimension - 1.0) / 9.0
```

Para um movimento browniano com dimensão fractal D = 1.7:
```
fci = (1.7 - 1.0) / 9.0 = 0.078  # Valor muito baixo!
```

### ✅ Solução Implementada
Fórmula corrigida alinhada com a escala real de dimensões fractais:
```python
# Correto (divisão por 2.0)
fci = (dimension - 1.0) / 2.0
```

Para o mesmo movimento browniano:
```
fci = (1.7 - 1.0) / 2.0 = 0.35  # Valor correto!
```

## Arquivo de Configuração

**Localização**: `configs/consciousness_metrics.yaml`

### Estrutura Principal

```yaml
# Dimensão fractal de referência
fractal_dimension:
  min: 1.0          # Linha suave (sem complexidade)
  max: 3.0          # Preenchimento total do espaço
  normalizer: 2.0   # max - min = 3.0 - 1.0 = 2.0

# Thresholds para estados de consciência
state_thresholds:
  emergence:
    min_fci: 0.8
    fractal_dimension_min: 2.6

  meditation:
    min_fci: 0.6
    fractal_dimension_min: 2.2

  analysis:
    min_fci: 0.3
    fractal_dimension_min: 1.6

  coma:
    max_fci: 0.3
    fractal_dimension_max: 1.6
```

## Mapeamento Dimensão Fractal → FCI

### Fórmula

```
FCI = (D - D_min) / (D_max - D_min)
FCI = (D - 1.0) / 2.0
```

Onde:
- `D` = Dimensão fractal medida
- `D_min` = 1.0 (linha suave, complexidade mínima)
- `D_max` = 3.0 (preenchimento completo do espaço)
- `FCI` ∈ [0, 1]

### Tabela de Referência

| D (Dimensão Fractal) | FCI | Estado | Descrição |
|----------------------|-----|--------|-----------|
| 1.00 | 0.00 | COMA | Linha suave (sem complexidade) |
| 1.25 | 0.125 | COMA | Linha costeira típica |
| 1.50 | 0.25 | COMA | Ruído 1/f |
| **1.70** | **0.35** | **ANALYSIS** | **Movimento browniano fracionário** |
| 2.00 | 0.50 | ANALYSIS | Browniano padrão |
| 2.20 | 0.60 | MEDITATION | Alta atividade neural |
| 2.40 | 0.70 | MEDITATION | Dinâmica complexa |
| 2.80 | 0.90 | EMERGENCE | Pico de consciência |
| 3.00 | 1.00 | EMERGENCE | Preenchimento total do espaço |

## Estados de Consciência

### 1. COMA (FCI < 0.3)
- **Dimensão Fractal**: D < 1.6
- **Características**: Atividade consciente mínima
- **Exemplos**: Linhas costeiras, ruído simples

### 2. ANALYSIS (0.3 ≤ FCI < 0.6)
- **Dimensão Fractal**: 1.6 ≤ D < 2.2
- **Características**: Processamento lógico e analítico
- **Exemplos**: Movimento browniano, processos estocásticos

### 3. MEDITATION (0.6 ≤ FCI < 0.8)
- **Dimensão Fractal**: 2.2 ≤ D < 2.6
- **Características**: Análise introspectiva profunda
- **Exemplos**: Alta atividade neural organizada

### 4. EMERGENCE (FCI ≥ 0.8)
- **Dimensão Fractal**: D ≥ 2.6
- **Características**: Máxima criatividade e insight
- **Exemplos**: Picos de atividade consciente

## Como Usar

### 1. Carregar Configuração

```python
import yaml

# Carregar configuração
with open('configs/consciousness_metrics.yaml', 'r') as f:
    metrics_config = yaml.safe_load(f)

# Criar ConsciousnessMetrics com configuração
from src.conscience.consciousness_metrics import ConsciousnessMetrics

metrics = ConsciousnessMetrics(config, metrics_config)
```

### 2. Calcular FCI de Dimensão Fractal

```python
# Movimento browniano fracionário
fractal_dimension = 1.7
fci = metrics.compute_fci_from_fractal_dimension(fractal_dimension)

print(f"D = {fractal_dimension} → FCI = {fci:.3f}")
# Output: D = 1.7 → FCI = 0.350

# Classificar estado
state = metrics._classify_fci_state(fci)
print(f"Estado: {state}")
# Output: Estado: ANALYSIS
```

### 3. Personalizar Thresholds

Edite `configs/consciousness_metrics.yaml`:

```yaml
state_thresholds:
  emergence:
    min_fci: 0.85  # Aumentar threshold (mais restritivo)

  meditation:
    min_fci: 0.65  # Ajustar conforme necessário

  analysis:
    min_fci: 0.35  # Modificar limite inferior
```

## Validação

Execute o script de teste para validar a configuração:

```bash
python3 examples/test_consciousness_metrics_config.py
```

### Saída Esperada

```
🧠 CONSCIOUSNESS METRICS CONFIGURATION TEST SUITE
============================================================

✅ Fractal D → FCI Mapping: PASSOU
✅ State Thresholds: PASSOU
✅ Real-World Examples: PASSOU

🎉 TODOS OS TESTES PASSARAM!

✅ Fórmula corrigida: FCI = (D - 1.0) / 2.0
✅ Thresholds configuráveis funcionando
✅ Mapeamento D → FCI → Estado correto
```

## Exemplos do Mundo Real

O arquivo de configuração inclui exemplos calibrados:

```yaml
real_world_examples:
  brownian_motion:
    fractal_dimension: 1.7
    expected_fci: 0.35
    state: "ANALYSIS"

  coastline:
    fractal_dimension: 1.25
    expected_fci: 0.125
    state: "COMA"

  neural_activity_high:
    fractal_dimension: 2.4
    expected_fci: 0.7
    state: "MEDITATION"

  neural_activity_peak:
    fractal_dimension: 2.8
    expected_fci: 0.9
    state: "EMERGENCE"
```

## Logs e Debug

Ativar logs detalhados na configuração:

```yaml
debug:
  log_fci_calculations: true  # Mostrar cálculos FCI
  log_component_details: true  # Detalhes dos componentes
  warn_on_threshold_violations: true
  verbose_state_transitions: true
```

Saída de exemplo:
```
🔬 FCI Calculation: D=1.700 → FCI=0.350
```

## Componentes do FCI

Além do mapeamento direto D → FCI, o sistema calcula FCI baseado em três componentes:

```yaml
component_max_values:
  d_eeg_max: 10.0   # Dimensão EEG máxima
  h_fmri_max: 5.0   # Hemodinâmica máxima
  clz_max: 3.0      # Complexidade Lempel-Ziv máxima

fci_weights:
  d_eeg: 0.4   # 40% peso para EEG
  h_fmri: 0.3  # 30% peso para fMRI
  clz: 0.3     # 30% peso para complexidade
```

## Referências Matemáticas

### Dimensão Fractal
- **Definição**: Medida de complexidade que indica como um objeto preenche o espaço
- **Escala**: D ∈ [1, 3]
  - D = 1: Linha suave (Hausdorff dimension)
  - D = 2: Superfície plana
  - D = 3: Volume completo

### Movimento Browniano Fracionário
- **Dimensão**: D = 2 - H
- **Expoente de Hurst**: H ∈ [0, 1]
- **Exemplo**: H = 0.3 → D = 1.7

### Teorema da Box-Counting
```
D = lim(ε→0) [log N(ε) / log(1/ε)]
```
Onde N(ε) é o número de caixas de tamanho ε necessárias para cobrir o objeto.

## Migração de Código Legado

Se você estava usando a fórmula antiga:

```python
# ❌ Antiga (incorreta)
fci = (dimension - 1.0) / 9.0

# ✅ Nova (correta)
from src.conscience.consciousness_metrics import ConsciousnessMetrics
fci = metrics.compute_fci_from_fractal_dimension(dimension)
```

## Troubleshooting

### FCI sempre retorna valores baixos
✅ **Solução**: Verifique se está usando a fórmula corrigida `(D - 1.0) / 2.0`

### Estados não estão sendo classificados corretamente
✅ **Solução**: Ajuste os thresholds em `configs/consciousness_metrics.yaml`

### Testes falhando
✅ **Solução**: Execute `python3 examples/test_consciousness_metrics_config.py` para diagnóstico

## Contato e Contribuições

Para dúvidas ou contribuições relacionadas às métricas de consciência:
- Abra uma issue no repositório
- Consulte a documentação em `docs/`
- Execute os testes de validação

---

**Última atualização**: 2025-09-30
**Versão da configuração**: 1.0
**Status**: ✅ Validado