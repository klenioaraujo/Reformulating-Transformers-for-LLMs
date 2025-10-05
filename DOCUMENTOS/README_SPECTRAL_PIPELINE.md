# Pipeline Espectral ΨQRH - Guia de Uso

Este documento explica como usar os pipelines espectrais corrigidos que utilizam modelos convertidos reais ao invés de dados simulados.

## 📋 Visão Geral

Os pipelines espectrais foram atualizados para:

1. **Carregar modelos convertidos espectralmente** - Usa `make convert-model` para converter modelos com análise espectral física
2. **Usar embeddings reais do modelo** - Aplica FFT nos embeddings do modelo ao invés de dados simulados
3. **Gerar texto com o modelo convertido** - Usa o modelo para geração de texto real
4. **Incluir metadados espectrais** - Exibe dimensão fractal (D) e expoente de lei de potência (α)
5. **Processar com componentes ΨQRH** - Aplica fractal field calculator, difusão neural e métricas de consciência

## 🚀 Passo a Passo

### 1. Converter um Modelo para Espectral

Primeiro, você precisa converter um modelo usando análise espectral física:

```bash
# Converter GPT-2 base
make convert-model SOURCE=gpt2 OUTPUT=./models/gpt2_psiqrh

# Converter GPT-2 médio
make convert-model SOURCE=gpt2-medium OUTPUT=./models/gpt2_medium_psiqrh

# Converter BERT
make convert-model SOURCE=bert-base-uncased OUTPUT=./models/bert_psiqrh

# Converter modelo local
make convert-model SOURCE=./path/to/model OUTPUT=./models/converted
```

Este processo aplica:
- **FFT** (Fast Fourier Transform)
- **Análise de Espectro de Potência**
- **Lei de Potência** para determinar expoente α
- **Dimensão Fractal** usando box-counting
- **Correção Leech (Λ₂₄)** para quaternions
- **Validação Energética** (Parseval)

### 2. Executar o Pipeline Completo

#### Opção A: Usar o script shell

```bash
# Executar com modelo padrão (psiqrh_gpt2_MEDIO)
bash examples/run_spectral_pipeline.sh

# Executar com modelo específico
bash examples/run_spectral_pipeline.sh models/gpt2_psiqrh
```

#### Opção B: Executar manualmente

```bash
# Pipeline completo
python3 examples/complete_spectral_pipeline.py models/gpt2_psiqrh

# Pipeline legível
python3 examples/human_readable_spectral_pipeline.py models/gpt2_psiqrh
```

### 3. Pipeline Ponta a Ponta (Automatizado)

Use o comando `new-model` para todo o processo automatizado:

```bash
# Pipeline completo: adquire, converte, treina, certifica e ativa
make new-model SOURCE=gpt2-medium NAME=gpt2_qa

# Com opções específicas
make new-model SOURCE=gpt2-medium NAME=gpt2_qa USE_SPECTRAL=true USE_COMPLETE=true
```

Este comando executa automaticamente:
1. Download/carregamento do modelo fonte
2. Conversão espectral (FFT + Lei de Potência + Leech)
3. Treinamento com PsiQRHTransformerComplete (física rigorosa)
4. Certificação do modelo
5. Ativação como modelo padrão
6. Calibração física por gradiente
7. Teste de eco pós-calibração

## 📊 Arquitetura dos Pipelines

### Complete Spectral Pipeline (`complete_spectral_pipeline.py`)

```python
Pipeline Completo de Treinamento Espectral ΨQRH
├── Carregar modelo convertido espectralmente
│   ├── AutoTokenizer + AutoModelForCausalLM
│   ├── Metadados espectrais (D_fractal, α)
│   └── Dispositivo (CUDA/MPS/CPU)
├── Componentes espectrais ΨQRH
│   ├── FractalFieldCalculator
│   ├── NeuralDiffusionEngine
│   └── ConsciousnessMetrics
├── Processamento de texto
│   ├── Spectral Embedding (FFT nos embeddings do modelo)
│   ├── Treinamento espectral (fractal + difusão)
│   ├── Geração com modelo real
│   └── Enriquecimento com métricas de consciência
└── Validação e saída formatada
```

### Human Readable Spectral Pipeline (`human_readable_spectral_pipeline.py`)

Versão simplificada focada em saída legível para humanos, com mesma estrutura mas menos verbose.

## 📈 Métricas e Validação

Os pipelines calculam e reportam:

### Métricas de Consciência (ΨQRH)
- **FCI** (Fractal Consciousness Index): 0.0 - 1.0
  - FCI ≥ 0.45: Estado de Emergência (alta criatividade)
  - FCI ≥ 0.30: Estado Meditativo (processamento profundo)
  - FCI ≥ 0.15: Estado Analítico (estruturado)
  - FCI < 0.15: Estado Basal (fundamental)

### Métricas Espectrais
- **Dimensão Fractal (D)**: Medida de complexidade estrutural
- **Expoente Lei de Potência (α)**: Característica espectral
- **Entropia Ψ**: Medida de desordem/informação
- **Coerência**: Razão energia baixas frequências / total
- **Magnitude do Campo**: Norma do campo difundido

### Validação Física
- Conservação de energia (Parseval)
- Correção Leech para quaternions (Λ₂₄)
- Validação de transformada espectral

## 🔬 Exemplo de Saída

```
🎯 RESULTADO DO PIPELINE COMPLETO ESPECTRAL ΨQRH
================================================

📤 ENTRADA ORIGINAL:
   "O futuro da inteligência artificial é promissor"

📥 SAÍDA PROCESSADA (LEGÍVEL):
   "O futuro da inteligência artificial é promissor para o desenvolvimento
   de sistemas cada vez mais sofisticados e capazes de auxiliar em diversas
   áreas do conhecimento humano."

   [Métricas ΨQRH: processamento meditativo (FCI ≥ 0.30) (D_fractal=1.847) α=1.234 |
   Entropia=3.456 | Coerência=0.678 | processamento profundo e coerente]

🔍 VALIDAÇÃO:
   • Status: ✅ VÁLIDO
   • Confiança: 0.62
   • Estado Consciente: MEDITATION
   • FCI: 0.3156
   • Dimensão Fractal: 1.847

📊 MÉTRICAS ESPECTRAIS:
   • Entropia Ψ: 3.456
   • Coerência: 0.678
   • Magnitude Média: 12.345

💡 OBSERVAÇÕES:
   ✓ Saída validada com sucesso
   ✓ Estado meditativo - processamento profundo
```

## 🛠️ Troubleshooting

### Erro: "Modelo não encontrado"

**Solução**: Converta um modelo primeiro usando `make convert-model`

```bash
make convert-model SOURCE=gpt2 OUTPUT=./models/gpt2_psiqrh
```

### Erro: "Erro ao carregar modelo"

**Solução**: O pipeline usará fallback (FFT direto) automaticamente. Para usar o modelo:

1. Verifique se o diretório contém os arquivos necessários:
   - `config.json`
   - `pytorch_model.bin` ou `model.safetensors`
   - `tokenizer.json`

2. Reconverta o modelo se necessário:
   ```bash
   make convert-model SOURCE=gpt2 OUTPUT=./models/gpt2_psiqrh
   ```

### Performance lenta

**Solução**: Use GPU se disponível

```bash
# Pipeline detecta automaticamente CUDA/MPS
python3 examples/complete_spectral_pipeline.py models/gpt2_psiqrh
```

## 📚 Referências

- **Makefile Commands**: `make help` para lista completa de comandos
- **Conversão Espectral**: `make convert-model` usa FFT + Lei de Potência + Leech
- **Pipeline Completo**: `make new-model` para processo ponta a ponta
- **Física Rigorosa**: PsiQRHTransformerComplete com validação energética

## 🔗 Comandos Relacionados

```bash
# Ver todos os comandos disponíveis
make help

# Listar modelos certificados
make model-list

# Certificar um modelo
make model-certify MODEL=gpt2_psiqrh

# Chat com modelo ativo
make chat-model

# Validar propriedades do núcleo
make validate-core

# Testes de física
make test-physics
```
