# ΨCWS Training System - Sistema de Treinamento ΨCWS

## 📋 Visão Geral

O sistema ΨCWS implementa um pipeline completo de treinamento que converte:
```
TEXT → ESPECTRO → ESPECTRO SAÍDA → ESPECTRO ENTRADA → CONVERSÃO TEXT
```

**Arquitetura:**
- **Base:** Modelos open-source
- **Segurança:** 7 camadas de criptografia
- **Padrão:** Máscara científica para garantir padrão
- **Processamento:** Conversão espectral

## 🚀 Como Usar

### 1. Configuração de Parâmetros

```python
from Ψcws_training_parameters import ΨCWSTrainingParameters

# Configuração padrão
params = ΨCWSTrainingParameters()

# Configuração predefinida
params = get_preset_config("large")  # small, medium, large, spectral_focus

# Otimizar para hardware
params.optimize_for_hardware("gpu")  # gpu, cpu, tpu

# Validar parâmetros
is_valid, errors = params.validate_parameters()
```

### 2. Parâmetros Principais

#### Treinamento
```python
{
    "batch_size": 32,
    "learning_rate": 1e-4,
    "max_epochs": 100,
    "gradient_clip": 1.0,
    "optimizer": "AdamW",
    "scheduler": "cosine"
}
```

#### Modelo
```python
{
    "vocab_size": 50000,
    "embedding_dim": 512,
    "hidden_dim": 1024,
    "num_layers": 6,
    "num_heads": 8,
    "spectral_dim": 256
}
```

#### Espectral
```python
{
    "fft_bins": 128,
    "window_size": 64,
    "hop_length": 32,
    "n_mels": 80,
    "compression_method": "log"
}
```

#### Criptografia
```python
{
    "encryption_layers": 7,
    "encryption_key_size": 32,
    "scientific_mask_enabled": True,
    "mask_pattern": "fractal_gaussian"
}
```

## 🔧 Pipeline de Processamento

### 1. Conversão Text → Espectro
```python
from src.conscience.conscious_wave_modulator import ConsciousWaveModulator

# Configurar modulador
config = {
    'embedding_dim': 256,
    'sequence_length': 64,
    'device': 'cpu'
}
modulator = ConsciousWaveModulator(config)

# Converter arquivo
Ψcws_file = modulator.process_file("documento.pdf")
Ψcws_file.save("output.Ψcws")
```

### 2. Proteção com Criptografia
```python
from src.conscience.secure_Ψcws_protector import create_secure_Ψcws_protector

# Criar protetor
protector = create_secure_Ψcws_protector()

# Proteger arquivo
protected_parts = protector.protect_file("output.Ψcws", parts=4)
```

### 3. Processamento Espectral
```python
# Parâmetros espectrais otimizados
spectral_config = {
    'use_stft': True,
    'n_fft': 1024,
    'n_mels': 80,
    'compression_method': 'log'
}
```

## 🎯 Configurações Predefinidas

### `small` - Teste Rápido
- Batch size: 8
- Embedding: 256
- Layers: 4
- Épocas: 10

### `medium` - Desenvolvimento
- Batch size: 16
- Embedding: 384
- Layers: 6
- Épocas: 50

### `large` - Produção
- Batch size: 32
- Embedding: 512
- Layers: 8
- Épocas: 100

### `spectral_focus` - Foco Espectral
- Spectral dim: 512
- FFT bins: 256
- Mel bands: 128
- MFCC habilitado

## 🔒 Sistema de Segurança

### 7 Camadas de Criptografia
1. **AES-256-GCM** - Criptografia simétrica
2. **ChaCha20-Poly1305** - Criptografia de fluxo
3. **Fernet** - Criptografia autenticada
4. **XOR-Custom** - Obfuscação customizada
5. **Transposition** - Transposição de dados
6. **HMAC-AES** - Autenticação + criptografia
7. **Obfuscation** - Obfuscação final

### Máscara Científica
- Padrão: `fractal_gaussian`
- Threshold de entropia: 0.8
- Garante padrão matemático consistente

## 📊 Métricas de Treinamento

### Consciência
- **Complexidade**: Entropia dos embeddings
- **Coerência**: Autocorrelação de trajetórias
- **Adaptabilidade**: Diversidade espectral
- **Integração**: Correlação cruzada

### Performance
- **Loss**: Cross-entropy
- **Accuracy**: Precisão de conversão
- **Spectral Fidelity**: Fidelidade espectral
- **Encryption Security**: Segurança da criptografia

## 🛠️ Comandos Makefile

### Conversão de Arquivos
```bash
# Converter PDF para ΨCWS
make convert-pdf PDF=documento.pdf

# Estatísticas ΨCWS
make Ψcws-stats

# Listar arquivos ΨCWS
make list-Ψcws
```

### Treinamento
```bash
# Teste rápido
python3 train_Ψcws.py --preset small

# Treinamento completo
python3 train_Ψcws.py --preset large --device gpu

# Treinamento espectral
python3 train_Ψcws.py --preset spectral_focus
```

## 📁 Estrutura de Arquivos

```
Ψcws_training_parameters.py    # Parâmetros de treinamento
src/conscience/
├── conscious_wave_modulator.py    # Conversão text→espectro
├── secure_Ψcws_protector.py       # Sistema de segurança
└── ...
data/Ψcws_cache/               # Cache de arquivos ΨCWS
secure_parts/                  # Partes criptografadas
```

## 🎯 Exemplo Completo

```python
import torch
from Ψcws_training_parameters import ΨCWSTrainingParameters
from src.conscience.conscious_wave_modulator import ConsciousWaveModulator

# 1. Configurar parâmetros
params = ΨCWSTrainingParameters()
params.optimize_for_hardware("gpu")

# 2. Converter texto para espectro
modulator = ConsciousWaveModulator({
    'embedding_dim': params.training_config.embedding_dim,
    'sequence_length': params.training_config.max_sequence_length
})

Ψcws_file = modulator.process_file("input.txt")

# 3. Proteger com criptografia
from src.conscience.secure_Ψcws_protector import create_secure_Ψcws_protector
protector = create_secure_Ψcws_protector()
protected_parts = protector.protect_file("input.Ψcws")

print("✅ Pipeline ΨCWS configurado com sucesso!")
```

## 🔍 Validação

```python
# Validar parâmetros
is_valid, errors = params.validate_parameters()
if is_valid:
    print("✅ Parâmetros válidos")
else:
    print(f"❌ Erros: {errors}")

# Verificar compatibilidade hardware
print(f"Dispositivo: {params.training_config.device}")
print(f"Batch size otimizado: {params.training_config.batch_size}")
```

## 📈 Otimizações

### Para GPU
- Batch size aumentado
- Precisão mista habilitada
- Acumulação de gradiente reduzida

### Para CPU
- Batch size reduzido
- Precisão mista desabilitada
- Acumulação de gradiente aumentada

### Para TPU
- Batch size máximo
- Precisão mista habilitada
- Acumulação mínima

## 🐛 Solução de Problemas

### Erro: "embedding_dim não divisível por num_heads"
```python
# Solução: Ajustar embedding_dim
params.training_config.embedding_dim = 512  # Divisível por 8
```

### Erro: "Nenhuma GPU disponível"
```python
# Solução: Usar CPU
params.training_config.device = "cpu"
params.optimize_for_hardware("cpu")
```

### Erro: "Arquivo ΨCWS corrompido"
```python
# Solução: Verificar criptografia
from src.conscience.secure_Ψcws_protector import create_secure_Ψcws_protector
protector = create_secure_Ψcws_protector()
success = protector.read_protected_file(protected_parts)
```

## 📞 Suporte

Para problemas ou dúvidas:
- Verificar logs de validação
- Consultar parâmetros predefinidos
- Validar compatibilidade hardware
- Verificar integridade dos arquivos ΨCWS