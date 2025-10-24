# ΨQRH System - Quantum-Physical Consciousness Processing Framework

## 📚 Overview

The **ΨQRH (Psi Quantum Relativity Harmonics)** is an advanced language processing system based on quantum, fractal, and optical physical principles. It implements the **Padilha Wave Equation** for text processing through rigorous physical transformations, now enhanced with **ternary logic** for more sophisticated quantum-like processing.

### 🎯 What is ΨQRH?

ΨQRH transforms text into quantum physical representations through:
- **Padilha Wave Equation**: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
- **Quaternion Mapping**: Ψ(x) → 4D SO(4) space
- **Spectral Filtering**: F(k) = exp(i α · arctan(ln|k| + ε))
- **Consciousness Processing**: FCI (Fractal Consciousness Index)
- **Ternary Logic Framework**: Enhanced processing with -1, 0, 1 states

### 🔺 Ternary Logic Advantages

The ΨQRH system now operates with **ternary logic** instead of traditional binary logic, providing several key advantages:

#### **1. Quantum-Like Superposition**
- **Ternary States**: -1 (False/Inactive), 0 (Neutral/Undefined), 1 (True/Active)
- **Superposition Representation**: Allows quantum-like uncertainty and intermediate states
- **Consensus Mechanisms**: Ternary majority voting for robust decision-making

#### **2. Enhanced Consciousness Modeling**
- **Intermediate States**: Better representation of consciousness levels between discrete states
- **Consensus-Based Classification**: More nuanced state transitions (COMA ↔ ANALYSIS ↔ EMERGENCE)
- **Uncertainty Handling**: Neutral states for ambiguous or transitional conditions

#### **3. Improved Stability**
- **Ternary Validation**: Combined binary+ternary validation for comprehensive consistency checks
- **State Distribution Analysis**: Ensures balanced ternary state distributions across processing
- **Consensus Thresholds**: Configurable thresholds for ternary consensus operations

#### **4. Physical Consistency**
- **Ternary Physics Validation**: Enhanced validation of energy conservation and unitarity
- **State Stabilization**: Ternary-based stabilization of quantum operations
- **Distribution Consistency**: Maintains ternary state balance throughout processing pipeline

### 🏗️ Class-Based Organizational Structure

The ΨQRH system is organized into distinct classes, each handling specific physical and computational responsibilities:

#### **Core Classes (8 Main Components)**
```
ΨQRHSystem/
├── core/                      # 8 primary components
│   ├── PipelineManager.py     # Complete orchestration with ternary validation
│   ├── PhysicalProcessor.py   # Padilha equation with ternary physics validation
│   ├── QuantumMemory.py       # Quantum temporal memory
│   ├── AutoCalibration.py     # Emergent parameter calibration
│   ├── ModelMaker.py          # Dynamic model creation
│   ├── VocabularyMaker.py     # Dynamic vocabulary creation
│   ├── PipelineMaker.py       # Advanced pipeline construction
│   └── LegacyAdapter.py       # Compatibility with legacy psiqrh.py
```

#### **Physics Classes**
```
├── physics/                   # Physical computation modules
│   ├── PadilhaEquation.py     # Padilha wave equation implementation
│   ├── QuaternionOps.py       # Quaternion operations with ternary stabilization
│   └── SpectralFiltering.py   # Spectral filtering with ternary modulation
```

#### **Consciousness Classes**
```
├── consciousness/             # Consciousness processing
│   ├── consciousness_metrics.py    # FCI calculation with ternary classification
│   ├── consciousness_states.py     # State definitions
│   └── fractal_consciousness_processor.py
```

#### **Configuration Classes**
```
├── config/                    # Configuration system
│   ├── SystemConfig.py        # Unified configuration management
│   └── SystemConfig.py        # Configuration classes
```

## 🚀 Instalação e Configuração

### Pré-requisitos
```bash
pip install torch numpy scipy pyyaml
```

### Basic Configuration
```yaml
# config/system_config.yaml
model:
  embed_dim: 64
  max_history: 10
  vocab_size: 256

physics:
  I0: 1.0      # Base amplitude
  alpha: 1.0   # Linear dispersion parameter
  beta: 0.5    # Quadratic dispersion parameter
  k: 2.0       # Wave number
  omega: 1.0   # Angular frequency

ternary_logic:
  enable_consensus: true
  consensus_threshold: 0.6
  stabilization_enabled: true

system:
  device: auto
  enable_components: ["quantum_memory", "auto_calibration", "physical_harmonics"]
  validation:
    energy_conservation: true
    unitarity: true
    numerical_stability: true
    ternary_consistency: true
```

## 🎮 Principais Comandos

### 1. Interface de Linha de Comando (CLI)

#### Processamento Básico de Texto
```bash
cd ΨQRHSystem

# Processar texto simples
python3 -c "from interfaces.CLI import ΨQRHCLI; cli = ΨQRHCLI(); cli.process_text('Olá mundo quântico')"

# Modo interativo
python3 -c "from interfaces.CLI import main; main()" --interactive

# Com arquivo de configuração customizado
python3 -c "from interfaces.CLI import ΨQRHCLI; cli = ΨQRHCLI(); cli.load_config('config/custom_config.yaml'); cli.process_text('Texto de teste')"
```

#### Exemplos de Uso CLI
```bash
# Processamento de texto
python3 -c "
from ΨQRHSystem.interfaces.CLI import ΨQRHCLI
cli = ΨQRHCLI()
result = cli.process_text('Explique a teoria quântica')
print('Resultado:', result['text'])
print('FCI:', result['physical_metrics']['FCI'])
"

# Análise física
python3 -c "
from ΨQRHSystem.interfaces.CLI import ΨQRHCLI
cli = ΨQRHCLI()
analysis = cli.analyze_text('Texto para análise espectral')
print('Dimensão fractal:', analysis['fractal_dimension'])
"
```

### 2. API REST

#### Iniciar Servidor
```bash
cd ΨQRHSystem

# Servidor básico
python3 -c "from interfaces.API import main; main()" --host 0.0.0.0 --port 5000

# Com configuração customizada
python3 -c "from interfaces.API import main; main()" --config config/production.yaml --port 8080
```

#### Endpoints da API

##### POST /process
Processa texto através do pipeline ΨQRH completo.

```bash
curl -X POST http://localhost:5000/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Texto para processar", "task": "text-generation"}'
```

**Response:**
```json
{
  "status": "success",
  "text": "Processed text",
  "physical_metrics": {
    "FCI": 0.85,
    "fractal_dimension": 1.67,
    "alpha_calibrated": 1.2
  },
  "pipeline_state": {
    "ternary_consistency": 1
  },
  "dcf_analysis": {
    "fci_value": 0.85,
    "consciousness_state": "EMERGENCE"
  }
}
```

##### GET /health
Verifica status do sistema.

```bash
curl http://localhost:5000/health
```

##### POST /analyze
Análise espectral e física do texto.

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Texto para análise"}'
```

### 3. Uso Programático

#### PipelineManager (Recomendado)
```python
from ΨQRHSystem.config.SystemConfig import SystemConfig
from ΨQRHSystem.core.PipelineManager import PipelineManager

# Carregar configuração
config = SystemConfig.from_yaml('config/system_config.yaml')

# Criar pipeline
pipeline = PipelineManager(config)

# Processar texto
result = pipeline.process("Texto de entrada")

print("Generated text:", result['text'])
print("FCI:", result['physical_metrics']['FCI'])
print("Ternary consistency:", result['pipeline_state']['ternary_consistency'])
```

#### ModelMaker (Criação Dinâmica)
```python
from ΨQRHSystem.core.ModelMaker import ModelMaker

# Criar maker
maker = ModelMaker()

# Modelo customizado
pipeline = maker.create_custom(embed_dim=128, vocab_size=1024)

# Modelo a partir de template
pipeline = maker.create_from_template("quantum_focused")

# Modelo quântico otimizado
pipeline = maker.create_quantum_optimized("high")
```

#### VocabularyMaker (Vocabulários Dinâmicos)
```python
from ΨQRHSystem.core.VocabularyMaker import VocabularyMaker

maker = VocabularyMaker()

# Vocabulário semântico
vocab = maker.create_semantic_vocab(
    ["quantum", "consciousness", "fractal", "energy"],
    expansion_factor=2
)

# Vocabulário quântico
vocab = maker.create_quantum_vocab(quantum_features, vocab_size=512)

# Vocabulário híbrido
vocab = maker.create_hybrid_vocab(text_sources, quantum_features)
```

#### PipelineMaker (Pipelines Avançados)
```python
from ΨQRHSystem.core.PipelineMaker import PipelineMaker

maker = PipelineMaker()

# Pipeline físico-quântico
pipeline = maker.create_physics_pipeline({
    'I0': 1.5, 'alpha': 2.0, 'beta': 1.0,
    'k': 3.0, 'omega': 1.5
})

# Pipeline de pesquisa
pipeline = maker.create_research_pipeline("quantum")

# Pipeline de produção
pipeline = maker.create_production_pipeline("speed")
```

### 4. LegacyAdapter (Compatibilidade)

#### Substituição Direta do psiqrh.py Original
```python
# Antes (arquivo original)
from psiqrh import ΨQRHPipeline
pipeline = ΨQRHPipeline()
result = pipeline("Texto de entrada")

# Agora (novo sistema)
from ΨQRHSystem.core.LegacyAdapter import LegacyAdapter
pipeline = LegacyAdapter()  # Interface idêntica
result = pipeline("Texto de entrada")  # Mesmo resultado
```

## 🔬 System Functionality

### Processing Pipeline with Ternary Logic

1. **Text → Fractal Embedding**
   - Sequential conversion to fractal representation
   - Fractal dimension analysis D via power-law fitting

2. **Ψ(x) Quaternion Mapping with Ternary Logic**
   - Mapping to 4D quaternionic space with ternary stabilization
   - w (real), x,y,z (imaginary) components with ternary state validation

3. **Spectral Filtering with Ternary Modulation**
   - Filtering F(k) = exp(i α · arctan(ln|k| + ε)) with ternary frequency modulation
   - Energy conservation guaranteed with ternary consistency checks

4. **SO(4) Rotation with Ternary Validation**
   - Unitary rotations: Ψ' = q_left ⊗ Ψ ⊗ q_right† with ternary unitarity validation
   - Quantum norm preservation with ternary state distribution analysis

5. **Optical Probe with Ternary Enhancement**
   - Waveform generation via Padilha equation with ternary physics validation
   - Physical conversion to optical representation

6. **Consciousness Processing with Ternary Classification**
   - FCI (Fractal Consciousness Index) calculation with ternary state consensus
   - Consciousness states: COMA, ANALYSIS, MEDITATION, EMERGENCE with ternary transitions

7. **Wave-to-Text with Ternary Decoding**
   - Optical to text conversion with ternary pattern recognition
   - Resonance-based decoding with ternary consensus validation

### Key Equations (Based on DOE.md)

**Padilha Wave Equation:**
```
f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
```
Where:
- **λ**: Wavelength (dispersion parameter)
- **t**: Time
- **I₀**: Base amplitude
- **α**: Linear dispersion parameter
- **β**: Quadratic dispersion parameter
- **k**: Wave number
- **ω**: Angular frequency

**Fractal Dimension Mapping:**
```
α(D) = α₀(1 + λ(D - D_euclidean)/D_euclidean)
```
Where D is calculated via power-law fitting: P(k) ~ k^(-β) → D = (3 - β) / 2

**Quaternion Operations:**
```
Hamilton Product: (w1 + x1i + y1j + z1k) * (w2 + x2i + y2j + z2k)
SO(4) Rotations: Ψ' = q_left ⊗ Ψ ⊗ q_right†
```

**Spectral Filtering:**
```
F(k) = exp(i α · arctan(ln(|k| + ε)))
```

**Fractal Consciousness Index:**
```
FCI = (D_EEG × H_fMRI × CLZ) / D_max
```
Where components are calculated with ternary consensus validation.

### Mathematical Validations

- ✅ **Energy Conservation**: ||output|| ≈ ||input|| (within 5%) with ternary consistency
- ✅ **Unitarity**: SO(4) rotations preserve quantum states with ternary validation
- ✅ **Numerical Stability**: Double precision quaternion arithmetic
- ✅ **Fractal Consistency**: D ∈ [1.0, 2.0] with ternary consensus
- ✅ **Ternary Balance**: Balanced ternary state distributions

## 🧪 Testes e Validação

### Executar Todos os Testes
```bash
cd ΨQRHSystem
python3 -m pytest tests/ -v
```

### Testes Específicos
```bash
# Testes dos makers
python3 -m pytest tests/test_makers.py -v

# Testes de configuração
python3 -m pytest tests/test_config.py -v

# Testes físicos
python3 -m pytest tests/test_physics.py -v
```

#### Ternary Logic Validation
```python
from ΨQRHSystem.core.TernaryLogicFramework import TernaryLogicFramework, TernaryValidationFramework

# Test ternary operations
ternary_logic = TernaryLogicFramework(device='cpu')

# Test superposition
superposition = ternary_logic.create_superposition()
print(f"Superposition: value={superposition.value}, confidence={superposition.confidence:.3f}")

# Validate operations
validator = TernaryValidationFramework(ternary_logic)
validation_results = validator.validate_ternary_operations()
print(f"Ternary validation: {validation_results}")
```

### Physical Validation with Ternary Logic
```python
from ΨQRHSystem.physics.PadilhaEquation import PadilhaEquation
from ΨQRHSystem.physics.SpectralFiltering import SpectralFiltering

# Validate Padilha equation with ternary consistency
equation = PadilhaEquation()
result = equation.validate_energy_conservation(input_energy, output_energy)

# Validate spectral filtering with ternary unitarity
filtering = SpectralFiltering()
is_unitary = filtering.validate_filter_unitarity(embed_dim=64)
```

## ⚙️ Configuração Avançada

#### Ternary Logic Parameters
```yaml
ternary_logic:
  enable_consensus: true          # Enable consensus operations
  consensus_threshold: 0.6        # Threshold for consensus decisions
  stabilization_enabled: true     # Enable ternary stabilization
  state_distribution_check: true  # Validate ternary state distributions
```

#### Physical Parameters with Ternary Enhancement
```yaml
physics:
  I0: 1.0           # Base amplitude (0.1 - 5.0)
  alpha: 1.0        # Linear dispersion (0.1 - 3.0)
  beta: 0.5         # Quadratic dispersion (0.01 - 1.0)
  k: 2.0            # Wave number (0.5 - 10.0)
  omega: 1.0        # Angular frequency (0.1 - 5.0)

ternary_physics:
  validation_enabled: true
  distribution_tolerance: 0.35
  consensus_validation: true
```

#### System Components
```yaml
system:
  device: auto      # auto, cpu, cuda, mps
  enable_components:
    - quantum_memory
    - auto_calibration
    - physical_harmonics
  validation:
    energy_conservation: true
    unitarity: true
    numerical_stability: true
    ternary_consistency: true
```

### Templates de Modelo
```python
# Templates disponíveis
templates = maker.get_template_info()
print(templates.keys())
# dict_keys(['minimal', 'standard', 'advanced', 'quantum_focused'])
```

## 🔧 Desenvolvimento e Extensão

### Adicionar Novo Componente
```python
# 1. Criar classe em core/
class NewComponent:
    def __init__(self, config):
        self.config = config

    def process(self, data):
        # Implementação
        return processed_data

# 2. Integrar no PipelineManager
class PipelineManager:
    def __init__(self, config):
        self.new_component = NewComponent(config)
        # ... outros componentes

    def process(self, text):
        # Usar new_component no pipeline
        result = self.new_component.process(data)
        return result
```

### Criar Novo Template
```python
# Adicionar em ModelMaker._load_templates()
"custom_template": {
    "model": {"embed_dim": 96, "max_history": 15, "vocab_size": 512},
    "physics": {"I0": 1.2, "alpha": 1.5, "beta": 0.8, "k": 2.5, "omega": 1.2},
    "description": "Template customizado"
}
```

## 📊 Monitoramento e Debug

### Logs de Auditoria
```python
pipeline = PipelineManager(config)
result = pipeline.process("Texto de teste", audit_mode=True)

print("Logs de auditoria:", result['audit_log_count'])
print("ID da sessão:", result['audit_session_id'])
```

### Métricas Físicas
```python
result = pipeline.process("Texto")

print("Enhanced metrics:")
print(f"  FCI: {result['physical_metrics']['FCI']:.3f}")
print(f"  Fractal dimension: {result['physical_metrics']['fractal_dimension']:.3f}")
print(f"  Ternary consistency: {result['pipeline_state']['ternary_consistency']}")
print(f"  Energy conserved: {result['mathematical_validation']['energy_conservation']}")
```

### Validação de Estado
```python
# Verificar saúde do sistema
from ΨQRHSystem.core.PipelineManager import PipelineManager

pipeline = PipelineManager(config)
health = pipeline.health_check()

print("Status do sistema:")
for component, status in health.items():
    print(f"  {component}: {'✅' if status else '❌'}")
```

## 🚨 Troubleshooting

### Erro: "Componente não disponível"
```
Causa: Componente não instalado ou configuração incorreta
Solução: Verificar config/system_config.yaml e instalar dependências
```

### Erro: "Falha na conservação de energia"
```
Causa: Parâmetros físicos fora do range válido
Solução: Ajustar parâmetros em config/system_config.yaml
```

### Erro: "Memória insuficiente"
```
Causa: embed_dim ou vocab_size muito grandes
Solução: Reduzir dimensões no arquivo de configuração
```

## 📈 Performance e Otimização

#### Use Case Configurations

##### Development (Fast)
```yaml
model:
  embed_dim: 32
  vocab_size: 128
physics:
  I0: 0.5
  alpha: 0.5
ternary_logic:
  consensus_threshold: 0.5  # Lower threshold for speed
```

##### Production (Balanced)
```yaml
model:
  embed_dim: 128
  vocab_size: 1024
physics:
  I0: 2.0
  alpha: 1.5
system:
  device: cuda
ternary_logic:
  consensus_threshold: 0.7  # Higher threshold for accuracy
```

##### Research (Comprehensive)
```yaml
model:
  embed_dim: 256
  vocab_size: 4096
physics:
  I0: 3.0
  alpha: 2.0
  beta: 1.0
ternary_logic:
  enable_consensus: true
  consensus_threshold: 0.8
  stabilization_enabled: true
```

### Benchmarking
```python
import time
from ΨQRHSystem.core.PipelineManager import PipelineManager

pipeline = PipelineManager(config)

# Benchmark
start_time = time.time()
for _ in range(100):
    result = pipeline.process("Texto de teste")
end_time = time.time()

print(f"Tempo médio: {(end_time - start_time) / 100:.3f}s por processamento")
```

## 🤝 Contribuição

### Estrutura de Commits
```
feat: adicionar novo componente físico
fix: corrigir validação de energia
docs: atualizar documentação da API
test: adicionar testes para PipelineMaker
refactor: otimizar operações quaterniônicas
```

#### Code Standards
- Type hints required
- Complete docstrings
- Tests for new functionality
- Mathematical validations
- Ternary logic consistency checks

## 📄 Licença

Este sistema implementa princípios físicos avançados baseados na equação de Padilha e teoria quântica de campos. Uso acadêmico e de pesquisa.

---

**ΨQRH System** - Transforming language through quantum physics, fractal mathematics, and optical principles with enhanced ternary logic processing.