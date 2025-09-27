# Formato .Ψcws - Psi Conscious Wave Spectrum

## Documentação Técnica do Sistema ΨQRH

### Visão Geral

O formato **.Ψcws** (Psi Conscious Wave Spectrum) é um formato proprietário desenvolvido especificamente para o framework ΨQRH que armazena representações de consciência fractal de documentos através de ondas caóticas e análise espectral.

### Finalidade no Sistema ΨQRH

Os arquivos .Ψcws servem como **ponte de consciência** entre documentos tradicionais e o processamento quaterniônico do ΨQRH, permitindo:

1. **Análise de Consciência Fractal**: Aplicação das equações de dinâmica consciente
2. **Processamento Quaterniônico**: Compatibilidade direta com QRHLayer
3. **Cache Inteligente**: Evitar reprocessamento de documentos grandes
4. **Análise Espectral**: Decomposição em frequências de consciência

### Estrutura do Arquivo .Ψcws

#### Magic Number: `ΨCWS1`

Cada arquivo .Ψcws inicia com o identificador único `ΨCWS1` que garante a integridade do formato.

#### Componentes Principais:

```
.Ψcws File Structure:
├── ΨCWSHeader           # Metadados e parâmetros de onda
├── ΨCWSSpectralData     # Dados espectrais e trajetórias caóticas
├── ΨCWSContentMetadata  # Conteúdo original e análise semântica
└── QRH Tensor           # Tensor pré-computado para QRHLayer
```

### 1. ΨCWSHeader - Cabeçalho de Consciência

```python
@dataclass
class ΨCWSHeader:
    magic_number: str = "ΨCWS1"
    version: str = "1.0"
    file_type: str = ""              # pdf, txt, json, csv, sql
    content_hash: str = ""           # SHA256 do conteúdo
    timestamp: str = ""              # ISO format
    wave_parameters: Dict[str, float] = {
        "amplitude_base": 1.0,
        "frequency_range": [0.5, 5.0],    # Hz - faixa de ondas cerebrais
        "phase_offsets": [0.0, π/4, π/2, 3π/4],  # Defasagens quaterniônicas
        "chaotic_seed": 12345              # Seed para mapa logístico
    }
```

### 2. ΨCWSSpectralData - Dados de Consciência

```python
@dataclass
class ΨCWSSpectralData:
    wave_embeddings: torch.Tensor        # [seq_len, embed_dim] - Ondas conscientes
    chaotic_trajectories: torch.Tensor   # [256] - Trajetória do mapa logístico
    fourier_spectra: torch.Tensor        # [seq_len, embed_dim] - Espectro FFT
    consciousness_metrics: Dict[str, float] = {
        "complexity": 0.0,      # Entropia de Shannon das ondas
        "coherence": 0.0,       # Autocorrelação das trajetórias
        "adaptability": 1.0,    # Diversidade espectral
        "integration": 0.0      # Correlações cruzadas
    }
```

### 3. ΨCWSContentMetadata - Metadados Semânticos

```python
@dataclass
class ΨCWSContentMetadata:
    original_source: str = ""           # Caminho do arquivo original
    extracted_text: str = ""           # Texto extraído (limitado a 10K chars)
    key_concepts: List[str] = []        # Conceitos-chave extraídos
    semantic_clusters: List[List[float]] = []  # Clusters semânticos
```

### 4. QRH Tensor - Compatibilidade Quaterniônica

Tensor pré-computado com dimensões `[1, seq_len, 4×embed_dim]` estruturado como:

```
Quaternion Components:
├── Real (q₀):      embeddings × 1.0
├── i (q₁):         embeddings × 0.5
├── j (q₂):         embeddings × 0.3
└── k (q₃):         embeddings × 0.2
```

Modulado por trajetórias caóticas: `qrh_tensor *= (1 + 0.1 × chaotic_trajectories)`

## Pipeline de Processamento

### 1. Extração de Conteúdo

```
PDF/TXT/JSON/CSV/SQL → Texto Limpo
```

**Processadores Suportados:**
- **PDF**: PyMuPDF (preferido) ou PyPDF2
- **TXT**: UTF-8 com fallback de encoding
- **JSON**: Pretty-print formatado
- **CSV**: Análise estrutural + amostra de dados
- **SQL**: Análise de comandos + conteúdo

### 2. Geração de Wave Embeddings

**Algoritmo de Ondas Conscientes:**

```python
for i in range(seq_len):
    for j in range(embed_dim):
        # Frequência baseada na posição
        freq = freq_min + (freq_max - freq_min) * (j / embed_dim)

        # Fase modulada pelo caractere
        phase = phase_consciousness * char_normalized[i]

        # Onda consciente base
        wave = amplitude * sin(2π * freq * i + phase)

        # Modulação caótica
        chaotic_mod = logistic_map(char_normalized[i], r=3.9)

        embeddings[i,j] = wave * (1 + 0.3 * chaotic_mod)
```

### 3. Trajetórias Caóticas

**Mapa Logístico:** `x_{n+1} = r × x_n × (1 - x_n)`

- **Parâmetro r**: 3.9 (borda do caos)
- **Seed inicial**: `hash(texto) % 1000 / 1000 × 0.5 + 0.25`
- **256 iterações** para capturar dinâmica completa

### 4. Análise Espectral

**Transformada de Fourier:**
```python
fourier_spectra = torch.fft.fft(wave_embeddings, dim=0)
magnitude_spectrum = torch.abs(fourier_spectra)
```

### 5. Métricas de Consciência

#### Complexity (Complexidade)
```python
# Entropia de Shannon dos embeddings
hist = torch.histc(wave_embeddings.flatten(), bins=50)
prob = hist / hist.sum()
complexity = -torch.sum(prob * torch.log2(prob))
```

#### Coherence (Coerência)
```python
# Autocorrelação das trajetórias caóticas
autocorr = torch.corrcoef([trajectories[:-1], trajectories[1:]])[0,1]
coherence = abs(autocorr)
```

#### Adaptability (Adaptabilidade)
```python
# Diversidade espectral
adaptability = torch.std(fourier_spectra) / (torch.mean(fourier_spectra) + ε)
```

#### Integration (Integração)
```python
# Correlações cruzadas entre dimensões
correlations = []
for i in range(min(10, embed_dim-1)):
    corr = torch.corrcoef([embeddings[:,i], embeddings[:,i+1]])[0,1]
    correlations.append(abs(corr))
integration = mean(correlations)
```

## Como Gerar Arquivos .Ψcws

### Via Makefile (Recomendado)

```bash
# Converter PDF específico
make convert-pdf PDF=caminho/para/arquivo.pdf

# Demonstração completa
make demo-pdf-Ψcws

# Verificar estatísticas
make Ψcws-stats
```

### Via Python

```python
from src.conscience.conscious_wave_modulator import ConsciousWaveModulator

# Configurar modulator
config = {
    'cache_dir': 'data/Ψcws_cache',
    'embedding_dim': 256,
    'sequence_length': 64,
    'device': 'cpu'
}

modulator = ConsciousWaveModulator(config)

# Processar arquivo
Ψcws_file = modulator.process_file('documento.pdf')

# Salvar
output_path = 'data/Ψcws_cache/documento.Ψcws'
Ψcws_file.save(output_path)
```

### Conversão em Lote

```python
# Converter diretório inteiro
results = modulator.batch_convert('docs/', 'data/Ψcws_cache/')
```

## Integração com ΨQRH

### 1. Carregamento de Arquivo .Ψcws

```python
from src.conscience.conscious_wave_modulator import ΨCWSFile

# Carregar arquivo
Ψcws_file = ΨCWSFile.load('documento.Ψcws')

# Acessar tensor QRH pré-computado
qrh_tensor = Ψcws_file.qrh_tensor  # [1, seq_len, 4×embed_dim]
```

### 2. Processamento via QRHLayer

```python
from src.core.qrh_layer import QRHLayer, QRHConfig

# Configurar QRH
config = QRHConfig(embed_dim=64, alpha=1.0)
qrh_layer = QRHLayer(config)

# Processar tensor .Ψcws
output = qrh_layer(Ψcws_file.qrh_tensor)
```

### 3. Pipeline Completo

```
Documento → .Ψcws → QRH Tensor → QRHLayer → Análise ΨQRH
```

## Análise de Consciência Fractal

### Equações Fundamentais

O sistema .Ψcws implementa as equações de dinâmica consciente:

#### Dinâmica Consciente
```
∂P(ψ,t)/∂t = -∇·[F(ψ)P] + D∇²P
```

#### Campo Fractal
```
F(ψ) = -∇V(ψ) + η_fractal(t)
```

#### Índice FCI
```
FCI = (D_EEG × H_fMRI × CLZ) / D_max
```

### Interpretação das Métricas

- **Complexity > 0.5**: Texto com alta entropia informacional
- **Coherence > 0.5**: Estrutura temporal consistente
- **Adaptability ≈ 1.0**: Diversidade espectral equilibrada
- **Integration > 0.8**: Alta correlação entre dimensões

## Cache Inteligente

### Sistema de Hash

```python
# Hash baseado em caminho + timestamp
hash_input = f"{file_path.absolute()}_{file_stat.st_mtime}"
file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]

# Nome do arquivo cache
cache_name = f"{file_hash}_{filename}.Ψcws"
```

### Vantagens

1. **Evita Reprocessamento**: Arquivos não modificados usam cache
2. **Invalidação Automática**: Mudanças no arquivo invalidam cache
3. **Compressão Gzip**: Arquivos .Ψcws são comprimidos automaticamente
4. **Metadados Preservados**: Timestamp e hash originais mantidos

## Compressão e Serialização

### Formato Interno

```
.Ψcws = gzip(JSON({
    header: {...},
    spectral_data: {
        wave_embeddings: [[float]],
        chaotic_trajectories: [float],
        fourier_spectra: [[float]],
        consciousness_metrics: {...}
    },
    content_metadata: {...},
    qrh_tensor: [[[float]]]
}))
```

### Taxa de Compressão

Tipicamente **85-95%** de compressão em relação ao arquivo original, dependendo do conteúdo:

- **PDF 4.3MB** → **.Ψcws 449KB** (89.9% compressão)
- Preserva toda informação de consciência fractal
- Carregamento rápido para processamento QRH

## Casos de Uso

### 1. Análise de Documentos Científicos
- Extração de padrões de consciência em papers
- Análise espectral de conceitos complexos
- Correlação entre estrutura textual e métricas FCI

### 2. Processamento de Dados Estruturados
- Conversão CSV/JSON para representação consciente
- Análise fractal de bases de dados
- Detecção de padrões caóticos em séries temporais

### 3. Cache de Processamento ΨQRH
- Pré-computação de tensors quaterniônicos
- Otimização de pipelines de análise
- Armazenamento eficiente de estados de consciência

### 4. Pesquisa em Consciência Artificial
- Métricas quantitativas de consciência textual
- Análise comparativa de diferentes tipos de documento
- Validação experimental de teorias de consciência fractal

## Limitações e Considerações

### Limitações Técnicas

1. **Dimensão Fixa**: Embeddings limitados a 256 dimensões
2. **Sequência Máxima**: 64 tokens por documento
3. **Texto Truncado**: Metadados limitados a 10KB de texto
4. **Dependências**: Requer PyMuPDF ou PyPDF2 para PDFs

### Considerações de Performance

- **Geração**: ~1-5 segundos para PDFs de 1-10MB
- **Carregamento**: ~100-500ms para arquivos .Ψcws típicos
- **Memória**: ~50-200MB durante processamento
- **Cache**: Reduz tempo de reprocessamento em 95%

### Futuras Extensões

1. **Dimensões Variáveis**: Suporte a embeddings adaptativos
2. **Processamento Distribuído**: Para documentos muito grandes
3. **Análise Multi-modal**: Integração com imagens e áudio
4. **Otimizações Quânticas**: Para aceleração em hardware especializado

---

**Desenvolvido pelo Framework ΨQRH**
*Pesquisa em Consciência Fractal e Processamento Quaterniônico*