# Plano de Validação Robusta End-to-End - Modelo ΨQRH

## 📋 Visão Geral

Plano sistemático de validação para verificar integridade, qualidade e funcionalidade do modelo ΨQRH treinado pelo script `train_model.py`.

**Objetivo**: Garantir que o modelo treinado:
1. ✅ Foi salvo corretamente e pode ser carregado
2. ✅ Supera o modelo baseline (não treinado)
3. ✅ Produz resultados qualitativamente coerentes

---

## 🔍 Fase 1: Verificação do Artefato de Treinamento

### Objetivo
Garantir que o resultado do `train_model.py` é um artefato válido e funcional.

### 1.1: Validação de Arquivos

**Script**: `validate_training_output.py --model_dir <path>`

**Arquivos Verificados**:
```
✅ pytorch_model.bin      - Pesos do modelo treinado
✅ config.json            - Configuração do modelo
✅ model_info.json        - Metadados ΨQRH
✅ tokenizer_config.json  - Configuração do tokenizer
✅ vocab.json             - Vocabulário
✅ merges.txt             - BPE merges
```

**Comando**:
```bash
python3 validate_training_output.py --model_dir ./models/psiqrh_wikitext_v2
```

**Critério de Sucesso**: Todos os arquivos existem no diretório.

---

### 1.2: Validação de Carregamento

**Script**: `validate_training_output.py` (mesma execução)

**Testes Executados**:
1. Inicializar `QRHFactory` com `model_path`
2. Verificar `pretrained_model` não é `None`
3. Verificar `tokenizer` não é `None`
4. Contar parâmetros do modelo

**Código de Validação**:
```python
try:
    qrh_factory = QRHFactory(model_path="./models/psiqrh_wikitext_v2")

    assert qrh_factory.pretrained_model is not None, "Modelo não carregado"
    assert qrh_factory.tokenizer is not None, "Tokenizer não carregado"

    num_params = sum(p.numel() for p in qrh_factory.pretrained_model.parameters())
    print(f"✅ Modelo carregado: {num_params:,} parâmetros")

except Exception as e:
    print(f"❌ Falha: {e}")
    sys.exit(1)
```

**Critério de Sucesso**: Carregamento sem erros, `num_params > 0`.

---

## 📊 Fase 2: Validação Quantitativa de Qualidade

### Objetivo
Provar que o treinamento melhorou o modelo.

### 2.1: Benchmark de Perplexidade Comparativo

**Script**: `validate_training_output.py --model_dir <path>`

**Modelos Comparados**:
1. **ΨQRH Não Treinado**: Pesos aleatórios iniciais
2. **ΨQRH Treinado**: Pesos carregados do checkpoint

**Dataset**: WikiText-103 validation set (50 amostras)

**Métricas Calculadas**:
- Perplexity (PPL)
- Cross-Entropy Loss
- Tempo de inferência

**Resultado Esperado**:
```
Perplexity(Treinado) < Perplexity(Não Treinado)
```

**Exemplo de Output**:
```
📊 FASE 2.1: Benchmark de Perplexidade Comparativo

  ΨQRH Não Treinado:
    Perplexity: 15234.56
    Loss: 9.6321
    Tempo: 45.2s

  ΨQRH Treinado:
    Perplexity: 487.23
    Loss: 6.1890
    Tempo: 46.1s

  📈 Melhoria: 96.8%
  ✅ Modelo treinado é melhor que não treinado
```

**Comando**:
```bash
# Validação completa (com benchmark)
python3 validate_training_output.py --model_dir ./models/psiqrh_wikitext_v2

# Validação rápida (sem benchmark)
python3 validate_training_output.py --model_dir ./models/psiqrh_wikitext_v2 --skip_benchmark
```

**Critério de Sucesso**: `Perplexity(Treinado) < Perplexity(Não Treinado)`

---

### 2.2: Teste de Métricas Dinâmicas

**Script**: `test_deep_dive_metrics.py`

**Pré-requisito**: `app.py` rodando com modelo treinado carregado

**Atualizar app.py**:
```python
# Em app.py, linha ~29
qrh_factory = QRHFactory(
    config_path="configs/qrh_config.yaml",
    model_path="./models/psiqrh_wikitext_v2"  # ← Adicionar esta linha
)
```

**Testes Executados**:
1. POST `/api/v1/analyze/deep_dive` com texto "ola"
2. POST `/api/v1/analyze/deep_dive` com texto "ola mundo"
3. POST `/api/v1/analyze/deep_dive` com texto "ola mundo como vai voce hoje"

**Comando**:
```bash
# Terminal 1: Iniciar servidor
python3 app.py

# Terminal 2: Executar testes
python3 test_deep_dive_metrics.py
```

**Critério de Sucesso**:
- Métricas (β, D_EEG, H_fMRI, CLZ) são diferentes para textos diferentes
- Valores não são defaults (0.025, 2.0, 0.75)
- Requests retornam HTTP 200

---

## 💬 Fase 3: Validação Qualitativa e de Conversação

### Objetivo
Testar comportamento real do modelo em cenários de uso.

### 3.1: Teste Automático de Cenários

**Script**: `chat_with_model.py --test_mode`

**Cenários de Teste**:

#### 1. Conhecimento Factual
```
Prompt: "Qual é a capital da França?"
Keywords esperadas: ['paris', 'frança']
```

#### 2. Criatividade
```
Prompt: "Conte-me uma pequena história sobre um robô que sonhava em ser um pássaro."
Keywords esperadas: ['robô', 'pássaro', 'sonho']
```

#### 3. Manutenção de Contexto
```
Prompt: "Eu gosto de física quântica. Qual tópico você acha mais interessante?"
Keywords esperadas: ['física', 'quântica']
```

#### 4. Robustez a Ruído
```
Prompt: "rererer rere re"
Keywords esperadas: [] (apenas verificar que não crashou)
```

#### 5. Raciocínio Simples
```
Prompt: "Se eu tenho 5 maçãs e como 2, quantas sobram?"
Keywords esperadas: ['3', 'três', 'sobra']
```

**Comando**:
```bash
python3 chat_with_model.py \
    --model_dir ./models/psiqrh_wikitext_v2 \
    --test_mode \
    --save_results test_results.json
```

**Output Esperado**:
```
🧪 MODO TESTE AUTOMÁTICO - Cenários Qualitativos

[1/5] Teste: Conhecimento Factual
  Prompt: "Qual é a capital da França?"
  Resposta: "Paris é a capital da França."
  Tempo: 1.23s | Tokens: 15
  ✅ PASSOU (keywords encontradas: ['paris', 'frança'])

[2/5] Teste: Criatividade
  ...

📊 RESUMO DOS TESTES
Total de testes: 5
✅ Passou: 4
❌ Falhou: 1
Taxa de sucesso: 80.0%
```

**Critério de Sucesso**: Taxa de sucesso ≥ 60%

---

### 3.2: Chat Interativo Manual

**Script**: `chat_with_model.py` (modo interativo)

**Comando**:
```bash
python3 chat_with_model.py --model_dir ./models/psiqrh_wikitext_v2
```

**Interface**:
```
💬 MODO CHAT INTERATIVO - ΨQRH
======================================================================
Digite 'sair' para encerrar
Digite 'reset' para limpar histórico
Digite 'historico' para ver conversas anteriores
======================================================================

👤 Você: Olá, como você está?
🤖 ΨQRH: Estou funcionando perfeitamente! Como posso ajudá-lo hoje?
   ⏱️  0.85s | 12 tokens | 14.1 tok/s

👤 Você: _
```

**Comandos Disponíveis**:
- `sair` - Encerra o chat
- `reset` - Limpa histórico de conversação
- `historico` - Mostra conversas anteriores

**Testes Sugeridos**:
1. Cumprimento inicial
2. Pergunta factual simples
3. Pergunta complexa multi-turn
4. Teste de criatividade
5. Teste de raciocínio lógico

**Critério de Sucesso**: Respostas coerentes e relevantes aos prompts.

---

## 📁 Estrutura de Arquivos

### Scripts Criados

```
validate_training_output.py    # Fase 1 + Fase 2.1
test_deep_dive_metrics.py      # Fase 2.2
chat_with_model.py             # Fase 3.1 + 3.2
```

### Fluxo de Execução

```
┌─────────────────────────────────────────────────────────┐
│  1. Treinar modelo                                       │
│     python3 train_model.py --epochs 3                   │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  2. Validar artefatos e perplexidade (Fase 1 + 2.1)     │
│     python3 validate_training_output.py \               │
│         --model_dir ./models/psiqrh_wikitext_v2         │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  3. Atualizar app.py com modelo treinado                │
│     qrh_factory = QRHFactory(model_path="...")          │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  4. Testar métricas dinâmicas (Fase 2.2)                │
│     python3 app.py  # Terminal 1                        │
│     python3 test_deep_dive_metrics.py  # Terminal 2     │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  5. Teste automático qualitativo (Fase 3.1)             │
│     python3 chat_with_model.py \                        │
│         --model_dir ./models/psiqrh_wikitext_v2 \       │
│         --test_mode                                      │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  6. Chat interativo manual (Fase 3.2)                   │
│     python3 chat_with_model.py \                        │
│         --model_dir ./models/psiqrh_wikitext_v2         │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Checklist de Validação

### Fase 1: Verificação de Artefatos ✅
- [ ] Todos os arquivos existem no diretório do modelo
- [ ] `pytorch_model.bin` tem tamanho > 0
- [ ] `config.json` é JSON válido
- [ ] Modelo carrega sem erros via `QRHFactory`
- [ ] Tokenizer carrega sem erros
- [ ] Número de parâmetros > 0

### Fase 2: Validação Quantitativa ✅
- [ ] Perplexity do modelo treinado < Perplexity do não treinado
- [ ] Melhoria de perplexity ≥ 10%
- [ ] Métricas dinâmicas funcionam com modelo treinado
- [ ] Endpoint `/deep_dive` retorna valores diferentes para textos diferentes
- [ ] Valores de β, D_EEG, H_fMRI, CLZ não são defaults

### Fase 3: Validação Qualitativa ✅
- [ ] Testes automáticos passam com ≥ 60% de taxa de sucesso
- [ ] Modelo responde a prompts simples
- [ ] Respostas são coerentes (não gibberish)
- [ ] Modelo não crashou em nenhum teste
- [ ] Chat interativo funciona sem travamentos

---

## 📊 Relatório de Validação

### Estrutura do Relatório

```json
{
  "model_info": {
    "path": "./models/psiqrh_wikitext_v2",
    "num_parameters": 15234567,
    "vocab_size": 50257,
    "spectral_dim": 256,
    "n_layers": 6
  },
  "phase1_artifacts": {
    "all_files_present": true,
    "loading_successful": true
  },
  "phase2_quantitative": {
    "untrained_perplexity": 15234.56,
    "trained_perplexity": 487.23,
    "improvement_percent": 96.8,
    "metrics_dynamic": true
  },
  "phase3_qualitative": {
    "test_mode_results": {
      "total_tests": 5,
      "passed": 4,
      "failed": 1,
      "success_rate": 80.0
    },
    "interactive_chat": "Testado manualmente - OK"
  },
  "overall_status": "PASS",
  "validation_date": "2025-10-02T12:34:56Z"
}
```

---

## 🚀 Comandos Rápidos

### Validação Completa (todas as fases)

```bash
# 1. Treinar (se ainda não treinou)
python3 train_model.py \
    --output_dir ./models/psiqrh_wikitext_v2 \
    --epochs 3 \
    --batch_size 8

# 2. Validar artefatos e perplexidade
python3 validate_training_output.py \
    --model_dir ./models/psiqrh_wikitext_v2

# 3. Atualizar app.py (manual)
# Adicionar: qrh_factory = QRHFactory(model_path="./models/psiqrh_wikitext_v2")

# 4. Testar métricas dinâmicas
python3 app.py &  # Em background
sleep 5  # Aguardar inicialização
python3 test_deep_dive_metrics.py

# 5. Teste qualitativo automático
python3 chat_with_model.py \
    --model_dir ./models/psiqrh_wikitext_v2 \
    --test_mode \
    --save_results validation_results.json

# 6. Chat interativo
python3 chat_with_model.py \
    --model_dir ./models/psiqrh_wikitext_v2
```

### Validação Rápida (sem benchmark de perplexidade)

```bash
# Apenas fase 1
python3 validate_training_output.py \
    --model_dir ./models/psiqrh_wikitext_v2 \
    --skip_benchmark

# Teste qualitativo rápido
python3 chat_with_model.py \
    --model_dir ./models/psiqrh_wikitext_v2 \
    --test_mode
```

---

## 🔧 Troubleshooting

### Erro: "Modelo não carregado"

**Causa**: Arquivos de modelo corrompidos ou faltando

**Solução**:
```bash
# Verificar arquivos
ls -lh ./models/psiqrh_wikitext_v2/

# Re-treinar se necessário
python3 train_model.py --output_dir ./models/psiqrh_wikitext_v2 --epochs 3
```

---

### Erro: "Perplexity do treinado pior que não treinado"

**Causa**: Treinamento não convergiu ou overfitting

**Soluções**:
1. Treinar por mais épocas
2. Ajustar learning rate
3. Verificar logs de treinamento

```bash
# Re-treinar com mais épocas
python3 train_model.py --epochs 10 --learning_rate 5e-5
```

---

### Erro: "CUDA out of memory"

**Causa**: Modelo muito grande para GPU disponível

**Solução**:
```bash
# Usar CPU
python3 validate_training_output.py --device cpu
python3 chat_with_model.py --device cpu

# Ou reduzir batch size no treinamento
python3 train_model.py --batch_size 4
```

---

## 📝 Notas Finais

### Observações Importantes

1. **Fase 1** é obrigatória - sem ela, as outras fases falham
2. **Fase 2.1** pode ser demorada (~5-10min) - use `--skip_benchmark` se necessário
3. **Fase 3.2** requer interação manual - reserve tempo para testes exploratórios

### Próximos Passos Após Validação

Se todas as fases passarem:

1. ✅ Modelo está pronto para uso em produção
2. 📝 Documentar resultados no README
3. 🚀 Fazer deploy no servidor
4. 📊 Monitorar métricas de uso real
5. 🔄 Iterar: treinar com mais dados/épocas se necessário

---

**Data de Criação**: 2025-10-02
**Autor**: Claude Code
**Versão**: 1.0.0
