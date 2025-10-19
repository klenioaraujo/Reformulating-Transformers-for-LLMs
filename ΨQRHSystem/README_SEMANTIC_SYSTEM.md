# Configuração do Sistema ΨQRH Semântico

Este documento descreve como configurar o ΨQRHSystem para usar vocabulário semântico e modelo semântico, apresentando informações detalhadas durante a execução seguindo o formato do sistema legado.

## Visão Geral

O sistema semântico ΨQRH permite:
- **Vocabulário Semântico**: Carregamento de vocabulário específico para processamento quântico
- **Modelo Semântico**: Configuração de modelo com informações detalhadas
- **Exibição de Informações**: Apresentação do modelo setado durante execução
- **Contagem de Tokens**: Informação sobre quantidade de tokens no vocabulário

## Uso Básico

### 1. Usando o Script de Configuração

```bash
# Executar com texto específico
python configure_semantic_system.py "Olá mundo quântico"

# Com configuração customizada
python configure_semantic_system.py --config ../config.yaml "Processamento semântico"

# Com vocabulário específico
python configure_semantic_system.py --vocab data/native_vocab.json "Teste"

# Apenas informações do sistema
python configure_semantic_system.py --info
```

### 2. Usando a CLI Original com Configuração Semântica

```bash
# A CLI original agora exibirá informações do modelo semântico
python -c "from interfaces.CLI import ΨQRHCLI; cli = ΨQRHCLI(); cli.process_text('Olá mundo quântico')"
```

## Saída Esperada

Quando executado, o sistema exibirá:

```
📁 Carregando configuração: ../config.yaml
🚀 Inicializando pipeline ΨQRH...
🔬 Physical Processor inicializado com equação de Padilha
   f(λ,t) = 1.0 sin(1.0t + 1.0λ) e^(i(1.0t - 2.0λ + 0.5λ²))
🧠 Quantum Memory inicializada com profundidade temporal: 10
🔧 Auto-Calibration inicializado com parâmetros físicos emergentes
✅ Pipeline Manager inicializado no dispositivo: cpu
✅ Pipeline ΨQRH pronto!

============================================================
🔬 SISTEMA ΨQRH CONFIGURADO
============================================================
🧠 Modelo: ΨQRH Semantic Model
📊 Tipo: semantic_quantum
🔢 Vocabulário: semantic
📈 Tokens: 23
📐 Dimensão: 64
🏗️  Camadas: 3
🎯 Cabeças: 8
💾 Dispositivo: cpu
============================================================

🧠 Processando: 'Olá mundo quântico...'

🔬 EXECUTANDO PIPELINE ΨQRH PARA: 'Olá mundo quântico...'
🔬 Dimensão fractal calculada: D = 1.390
✅ Pipeline concluído com sucesso

============================================================
🎯 RESULTADO ΨQRH
============================================================
📝 Texto: Quantum fractal quantum processing completed.
🔬 Dimensão Fractal: 0.500
⚡ Energia: ✅ CONSERVADA
✅ Validações: APROVADAS
🔧 Pipeline: ATIVO
============================================================

📊 Informações do Modelo Semântico:
   🔢 Tokens no vocabulário: 23
   🧠 Tipo de modelo: semantic_quantum
```

## Configuração de Vocabulário

### Vocabulário Padrão
O sistema inclui um vocabulário semântico padrão com 23 tokens relacionados a conceitos quânticos:
- `quantum`, `consciousness`, `fractal`, `energy`, `harmonic`
- `resonance`, `coherence`, `entanglement`, `dimension`, `field`
- `wave`, `particle`, `probability`, `state`, `transformation`
- `optical`, `spectral`, `temporal`, `spatial`, `geometric`
- `processing`, `completed`, `result`

### Vocabulário Customizado
Para usar vocabulário customizado, crie um arquivo JSON no formato:

```json
{
  "tokens": {
    "palavra1": 0,
    "palavra2": 1,
    "palavra3": 2
  },
  "metadata": {
    "type": "semantic",
    "size": 3,
    "description": "Descrição do vocabulário"
  }
}
```

## Configuração do Modelo

### Parâmetros do Modelo Semântico
- **Nome**: `ΨQRH Semantic Model`
- **Tipo**: `semantic_quantum`
- **Dimensão de Embedding**: 64
- **Número de Camadas**: 3
- **Número de Cabeças**: 8
- **Dimensão Oculta**: 128
- **Histórico Máximo**: 10
- **Dispositivo**: CPU (ou GPU se disponível)

### Configuração via YAML
O sistema carrega automaticamente configurações do arquivo `config.yaml`:

```yaml
model:
  embed_dim: 64
  max_history: 10
  num_heads: 8
  num_layers: 3
  vocab_size: 256
physics:
  I0: 1.0
  alpha: 1.0
  beta: 0.5
  k: 2.0
  omega: 1.0
system:
  device: auto
  enable_auto_calibration: true
  enable_cognitive_priming: true
  enable_noncommutative: true
  name: "ΨQRH Pipeline"
  version: 2.0.0
```

## Integração com o Sistema Legado

O sistema semântico mantém compatibilidade total com o sistema legado:

1. **Mesma CLI**: Interface `ΨQRHCLI` mantida
2. **Mesmo Pipeline**: Processamento físico quântico preservado
3. **Mesmas Validações**: Validações matemáticas rigorosas mantidas
4. **Informações Adicionais**: Exibição de informações do modelo semântico

## Exemplos de Uso

### Exemplo 1: Processamento Simples
```python
from configure_semantic_system import SemanticSystemConfigurator

configurator = SemanticSystemConfigurator()
result = configurator.process_text_semantic("Explique entrelaçamento quântico")
print(result['text'])
```

### Exemplo 2: Configuração Customizada
```python
configurator = SemanticSystemConfigurator("my_config.yaml")
configurator.load_semantic_vocabulary("custom_vocab.json")
result = configurator.process_text_semantic("Análise fractal")
```

### Exemplo 3: Informações do Sistema
```python
configurator = SemanticSystemConfigurator()
configurator.load_semantic_vocabulary()
configurator.configure_semantic_model()
configurator.display_system_info()
```

## Arquitetura

O sistema semântico estende a arquitetura existente:

1. **SemanticSystemConfigurator**: Classe principal de configuração
2. **Vocabulário Semântico**: Carregamento e gerenciamento de tokens
3. **Modelo Semântico**: Configuração e informações do modelo
4. **Integração com Pipeline**: Conexão com o pipeline físico existente

## Compatibilidade

- ✅ Python 3.8+
- ✅ PyTorch 1.9+
- ✅ Sistema ΨQRH legado
- ✅ Configurações YAML existentes
- ✅ Vocabulários JSON customizados

## Troubleshooting

### Erro de Importação
```bash
# Se houver erro de importação, verifique o path
python -c "import sys; print(sys.path)"
```

### Vocabulário Não Encontrado
```bash
# Verificar se o arquivo existe
ls -la data/native_vocab.json
```

### Configuração Não Carregada
```bash
# Verificar caminho do arquivo de configuração
python configure_semantic_system.py --config ../config.yaml --info
```

## Próximos Passos

1. **Expansão de Vocabulário**: Adicionar mais tokens semânticos
2. **Modelos Especializados**: Configurações para diferentes domínios
3. **Otimização**: Melhorias de performance para vocabulários grandes
4. **Integração**: Conectores com outros sistemas de NLP

---

**Desenvolvido para o ΨQRHSystem** - Sistema Físico Quântico-Fractal-Óptico