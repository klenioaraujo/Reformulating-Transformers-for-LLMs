# 📁 Configurações ΨQRH

Este diretório contém todos os arquivos de configuração YAML para o sistema ΨQRH.

## 📋 Índice de Arquivos

### 1. `qrh_config.yaml` ⚙️
**Configuração principal do QRHLayer**

Controla:
- Parâmetros quaterniônicos (rotações, dimensões)
- Processamento espectral (FFT, janelamento)
- Normalização e regularização
- Configuração de dispositivo (CPU/GPU)

**Parâmetros principais:**
```yaml
qrh_layer:
  embed_dim: 64                  # Dimensão base do embedding
  alpha: 1.0                     # Coeficiente do filtro espectral
  use_learned_rotation: true     # Rotações aprendidas vs fixas
  spectral_dropout_rate: 0.0     # Regularização espectral
```

**Usado por:**
- `src/core/qrh_layer.py`
- `src/core/enhanced_qrh_processor.py`

---

### 2. `cognitive_filters_config.yaml` 🧠 ✨ NOVO
**Configuração dos filtros cognitivos adaptativos**

Controla os 3 filtros semânticos:

#### Contradiction Detector (Detecção de Contradições)
```yaml
contradiction_detector:
  contradiction_threshold: 0.3        # Threshold de detecção (0-1)
  contradiction_sensitivity: 2.0      # Sensibilidade (amplificação)
  phase_rotation_strength: 0.5        # Força da atenuação
```

#### Irrelevance Filter (Filtro de Irrelevâncias)
```yaml
irrelevance_filter:
  irrelevance_threshold: 0.4          # Threshold de relevância
  enable_fft_filtering: true          # Filtragem FFT
```

#### Bias Filter (Filtro de Vieses)
```yaml
bias_filter:
  bias_threshold: 0.6                 # Threshold de detecção
  num_bias_patterns: 10               # Número de padrões
```

**Usado por:**
- `src/core/enhanced_qrh_processor.py`
- `src/cognitive/semantic_adaptive_filters.py`

**Documentação:** Ver `QUICKSTART_COGNITIVE.md`

---

### 3. `consciousness_metrics.yaml` 🌊
**Configuração do Fractal Consciousness Index (FCI)**

Controla:
- Thresholds de estados de consciência (EMERGENCE, MEDITATION, ANALYSIS, COMA)
- Cálculo do FCI baseado em dimensão fractal
- Normalização de componentes (D_EEG, H_fMRI, CLZ)
- Método de correlação espacial

**Mapeamento FCI:**
```yaml
state_thresholds:
  emergence:
    min_fci: 0.8              # FCI ≥ 0.8 → Estado emergente
  meditation:
    min_fci: 0.6              # FCI ≥ 0.6 → Estado meditativo
  analysis:
    min_fci: 0.3              # FCI ≥ 0.3 → Estado analítico
  coma:
    max_fci: 0.3              # FCI < 0.3 → Estado coma
```

**Usado por:**
- `src/conscience/consciousness_metrics.py`
- `src/conscience/fractal_consciousness_processor.py`

---

### 4. `fractal_config.yaml` 🌀
**Configuração do processamento fractal**

Controla:
- Cálculo de campos fractais
- Difusão neural
- Parâmetros caóticos

**Usado por:**
- `src/conscience/fractal_field_calculator.py`
- `src/conscience/neural_diffusion_engine.py`

---

### 5. `example_configs.yaml` 📚
**Exemplos de configurações para diferentes cenários**

Contém templates para:
- Configuração mínima
- Configuração de alta performance
- Configuração para pesquisa
- Configuração para produção

**Útil para:** Referência rápida e casos de uso específicos

---

## 🔧 Como Usar

### Carregar configuração padrão (automático):
```python
from src.core.enhanced_qrh_processor import create_enhanced_processor

# Carrega automaticamente de configs/
processor = create_enhanced_processor()
```

### Carregar configuração customizada:
```python
processor = create_enhanced_processor(
    cognitive_config_path="path/to/custom_cognitive_config.yaml"
)
```

### Modificar configuração via código:
```python
import yaml

# Carregar
with open("configs/cognitive_filters_config.yaml") as f:
    config = yaml.safe_load(f)

# Modificar
config['contradiction_detector']['contradiction_threshold'] = 0.2

# Salvar
with open("configs/my_custom_config.yaml", "w") as f:
    yaml.dump(config, f)
```

---

## 📊 Hierarquia de Configurações

```
configs/
├── qrh_config.yaml                    # QRHLayer base
│   └── Usado por: QRHLayer
│
├── cognitive_filters_config.yaml      # Filtros cognitivos ✨ NOVO
│   └── Usado por: SemanticAdaptiveFilter
│
├── consciousness_metrics.yaml         # Métricas FCI
│   └── Usado por: ConsciousnessMetrics
│
├── fractal_config.yaml                # Processamento fractal
│   └── Usado por: FractalFieldCalculator
│
└── example_configs.yaml               # Templates e exemplos
    └── Referência
```

---

## 🎯 Casos de Uso

### Aumentar sensibilidade à contradição:
```yaml
# Em cognitive_filters_config.yaml
contradiction_detector:
  contradiction_threshold: 0.2        # Reduzir (padrão: 0.3)
  contradiction_sensitivity: 3.0      # Aumentar (padrão: 2.0)
```

### Filtrar mais irrelevâncias:
```yaml
# Em cognitive_filters_config.yaml
irrelevance_filter:
  irrelevance_threshold: 0.6          # Aumentar (padrão: 0.4)
```

### Detectar mais vieses:
```yaml
# Em cognitive_filters_config.yaml
bias_filter:
  bias_threshold: 0.4                 # Reduzir (padrão: 0.6)
  num_bias_patterns: 15               # Aumentar (padrão: 10)
```

### Alterar thresholds de consciência:
```yaml
# Em consciousness_metrics.yaml
state_thresholds:
  emergence:
    min_fci: 0.85                     # Mais restritivo (padrão: 0.8)
```

---

## ⚠️ Notas Importantes

1. **Backup antes de modificar**: Sempre faça backup dos configs originais
2. **Validação**: Após modificar, execute testes para validar
3. **Ranges válidos**:
   - Thresholds: 0.0 - 1.0
   - FCI: 0.0 - 1.0
   - Dimensões: > 0
4. **Compatibilidade**: Mantenha estrutura YAML válida

---

## 🧪 Testar Configurações

### Teste básico:
```bash
python3 test_cognitive_integration.py
```

### Demo interativa:
```bash
python3 demo_cognitive_filters.py
```

### Validar config:
```python
import yaml

with open("configs/cognitive_filters_config.yaml") as f:
    config = yaml.safe_load(f)
    print("✅ Config válido!")
```

---

## 📚 Documentação Relacionada

- `../COGNITIVE_INTEGRATION_SUMMARY.md` - Integração completa
- `../QUICKSTART_COGNITIVE.md` - Quick start
- `../docs/CONSCIOUSNESS_METRICS_CONFIG.md` - Detalhes do FCI

---

## 🔄 Histórico de Versões

### v1.1.0 (2025-09-30) - Filtros Cognitivos ✨
- ✅ Adicionado `cognitive_filters_config.yaml`
- ✅ Integração com `enhanced_qrh_processor.py`
- ✅ Suporte a métricas cognitivas

### v1.0.0 (2025-09-29) - Release Inicial
- ✅ `qrh_config.yaml`
- ✅ `consciousness_metrics.yaml`
- ✅ `fractal_config.yaml`

---

**Mantido por:** ΨQRH Project Team
**Última atualização:** 2025-09-30