# ΨQRH System - Sistema Físico Quântico-Fractal-Óptico

## 📚 Visão Geral

O **ΨQRH (Psi Quantum Relativity Harmonics)** é um sistema avançado de processamento de linguagem baseado em princípios físicos quânticos, fractais e ópticos. Implementa a **Equação de Padilha** para processamento de texto através de transformações físicas rigorosas.

### 🎯 O que é o ΨQRH?

O ΨQRH transforma texto em representações físicas quânticas através de:
- **Equação de Padilha**: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
- **Mapeamento Quaterniônico**: Ψ(x) → espaço 4D SO(4)
- **Filtragem Espectral**: F(k) = exp(i α · arctan(ln|k| + ε))
- **Processamento de Consciência**: FCI (Fractal Consciousness Index)

### 🏗️ Arquitetura Modular

```
ΨQRHSystem/
├── core/           # 8 componentes principais
│   ├── PipelineManager.py     # Orquestração completa
│   ├── PhysicalProcessor.py   # Equação de Padilha
│   ├── QuantumMemory.py       # Memória temporal quântica
│   ├── AutoCalibration.py     # Calibração emergente
│   ├── ModelMaker.py          # Criação dinâmica de modelos
│   ├── VocabularyMaker.py     # Criação dinâmica de vocabulários
│   ├── PipelineMaker.py       # Pipelines avançados
│   └── LegacyAdapter.py       # Compatibilidade com psiqrh.py
├── physics/        # Módulos físicos
│   ├── PadilhaEquation.py     # Equação de Padilha
│   ├── QuaternionOps.py       # Operações quaterniônicas
│   └── SpectralFiltering.py   # Filtragem espectral
├── config/         # Sistema de configuração
│   └── SystemConfig.py        # Configuração unificada
├── interfaces/     # Interfaces de usuário
│   ├── CLI.py                 # Interface de linha de comando
│   └── API.py                 # API REST
└── tests/          # Testes abrangentes
    └── test_makers.py         # 25+ casos de teste
```

## 🚀 Instalação e Configuração

### Pré-requisitos
```bash
pip install torch numpy scipy pyyaml
```

### Configuração Básica
```yaml
# config/system_config.yaml
model:
  embed_dim: 64
  max_history: 10
  vocab_size: 256

physics:
  I0: 1.0      # Amplitude base
  alpha: 1.0   # Parâmetro de dispersão linear
  beta: 0.5    # Parâmetro de dispersão quadrática
  k: 2.0       # Número de onda
  omega: 1.0   # Frequência angular

system:
  device: auto
  enable_components: ["quantum_memory", "auto_calibration", "physical_harmonics"]
  validation:
    energy_conservation: true
    unitarity: true
    numerical_stability: true
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

**Resposta:**
```json
{
  "status": "success",
  "text": "Texto processado",
  "physical_metrics": {
    "FCI": 0.85,
    "fractal_dimension": 1.67,
    "alpha_calibrated": 1.2
  },
  "dcf_analysis": {
    "fci_value": 0.85,
    "consciousness_state": "ACTIVE"
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

print("Texto gerado:", result['text'])
print("FCI:", result['physical_metrics']['FCI'])
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

## 🔬 Como Funciona o ΨQRH

### Pipeline de Processamento

1. **Texto → Fractal Embedding**
   - Conversão sequencial para representação fractal
   - Análise de dimensão fractal D via power-law fitting

2. **Ψ(x) Quaternion Mapping**
   - Mapeamento para espaço quaterniônico 4D
   - w (real), x,y,z (imaginários) componentes

3. **Spectral Filtering**
   - Filtragem F(k) = exp(i α · arctan(ln|k| + ε))
   - Conservação de energia garantida

4. **SO(4) Rotation**
   - Rotações unitárias: Ψ' = q_left ⊗ Ψ ⊗ q_right†
   - Preservação de norma quântica

5. **Optical Probe**
   - Geração de forma de onda via equação de Padilha
   - Conversão física para representação óptica

6. **Consciousness Processing**
   - Cálculo FCI (Fractal Consciousness Index)
   - Estados de consciência: COMA, DREAM, ACTIVE

7. **Wave-to-Text**
   - Conversão óptica para texto de saída
   - Decodificação baseada em padrões ressonantes

### Equação de Padilha

**f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))**

Onde:
- **λ**: Comprimento de onda (dispersão)
- **t**: Tempo
- **I₀**: Amplitude base
- **α**: Parâmetro de dispersão linear
- **β**: Parâmetro de dispersão quadrática
- **k**: Número de onda
- **ω**: Frequência angular

### Validações Matemáticas

- ✅ **Conservação de Energia**: ||output|| ≈ ||input|| (dentro de 5%)
- ✅ **Unitariedade**: Rotações SO(4) preservam estados quânticos
- ✅ **Estabilidade Numérica**: Aritmética quaterniônica double precision
- ✅ **Consistência Fractal**: D ∈ [1.0, 2.0]

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

### Validação Física
```python
from ΨQRHSystem.physics.PadilhaEquation import PadilhaEquation
from ΨQRHSystem.physics.SpectralFiltering import SpectralFiltering

# Validar equação de Padilha
equation = PadilhaEquation()
result = equation.validate_energy_conservation(input_energy, output_energy)

# Validar filtragem espectral
filtering = SpectralFiltering()
is_unitary = filtering.validate_unitarity(transformation_matrix)
```

## ⚙️ Configuração Avançada

### Parâmetros Físicos
```yaml
physics:
  I0: 1.0           # Amplitude base (0.1 - 5.0)
  alpha: 1.0        # Dispersão linear (0.1 - 3.0)
  beta: 0.5         # Dispersão quadrática (0.01 - 1.0)
  k: 2.0            # Número de onda (0.5 - 10.0)
  omega: 1.0        # Frequência angular (0.1 - 5.0)
```

### Componentes do Sistema
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

print("Métricas físicas:")
print(f"  FCI: {result['physical_metrics']['FCI']:.3f}")
print(f"  Dimensão fractal: {result['physical_metrics']['fractal_dimension']:.3f}")
print(f"  Energia conservada: {result['mathematical_validation']['energy_conserved']}")
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

### Configurações por Caso de Uso

#### Desenvolvimento Rápido
```yaml
model:
  embed_dim: 32
  vocab_size: 128
physics:
  I0: 0.5
  alpha: 0.5
```

#### Produção
```yaml
model:
  embed_dim: 128
  vocab_size: 1024
physics:
  I0: 2.0
  alpha: 1.5
system:
  device: cuda
```

#### Pesquisa
```yaml
model:
  embed_dim: 256
  vocab_size: 4096
physics:
  I0: 3.0
  alpha: 2.0
  beta: 1.0
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

### Padrões de Código
- Type hints obrigatórios
- Docstrings completas
- Testes para novas funcionalidades
- Validações matemáticas

## 📄 Licença

Este sistema implementa princípios físicos avançados baseados na equação de Padilha e teoria quântica de campos. Uso acadêmico e de pesquisa.

---

**ΨQRH System** - Transformando linguagem através da física quântica, fractal e óptica.