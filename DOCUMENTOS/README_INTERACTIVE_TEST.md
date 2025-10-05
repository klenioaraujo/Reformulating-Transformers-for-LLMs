# Interactive Pipeline Test - Teste Interativo do Pipeline ΨQRH

## 📋 Sobre

Este script implementa um teste interativo completo do pipeline ΨQRH com logging detalhado de cada processo, gerando um arquivo para cada interação com análise completa.

## 🚀 Como Usar

### Execução Básica
```bash
python3 interactive_pipeline_test.py
```

### Comandos Disponíveis
- `quit`/`exit`/`sair` - Sair do teste
- `help`/`ajuda` - Mostrar ajuda
- `status` - Status do sistema

### Exemplos de Entradas para Teste

1. **Texto Simples (SIMULADO)**
   ```
   Explique o que são quaternions
   ```

2. **Dados Numéricos (REAL)**
   ```
   Processe o sinal [1.0, -2.5, 3.7, 0.8]
   ```

3. **Análise Matemática**
   ```
   Analise matematicamente esta frase
   ```

4. **Teste de Memória**
   ```
   O sistema precisa lembrar desta informação
   ```

5. **Teste Kuramoto**
   ```
   Simule osciladores acoplados com fase
   ```

## 📊 Saída Gerada

### Para Cada Interação
- Arquivo `interaction_XXX.md` com análise completa
- Log detalhado de todas as etapas do pipeline
- Métricas de performance
- Análise de componentes ativos
- Classificação REAL/SIMULADO

### Relatório Final
- Arquivo `FINAL_REPORT.md` consolidado
- Estatísticas gerais
- Distribuição por tipo de tarefa
- Análise de performance

## 🎯 Funcionalidades

### ✅ Detecção Automática de Tarefa
- `text-generation` - Geração de texto
- `signal-processing` - Processamento de sinais numéricos
- `analysis` - Análise matemática

### ✅ Classificação REAL/SIMULADO
- **REAL**: Entradas com dados numéricos explícitos
- **SIMULADO**: Entradas textuais conceituais

### ✅ Análise de Componentes
- Memória de Trabalho
- Sistema Kuramoto
- Métricas de Consciência
- Processador Numérico
- Framework ΨQRH

### ✅ Métricas de Performance
- Tempo de execução
- Comprimento entrada/saída
- Status do processamento
- Componentes ativos

## 📁 Estrutura de Arquivos

```
pipeline_test_logs/
├── interaction_001.md
├── interaction_002.md
├── ...
└── FINAL_REPORT.md
```

## 🔧 Configuração

### Pré-requisitos
- Python 3.7+
- Dependências do projeto ΨQRH
- Acesso aos módulos `src/`

### Personalização
- Modificar `output_dir` no construtor
- Ajustar análise de componentes
- Customizar relatórios

## 🧪 Exemplo de Uso

```bash
# Executar teste
python3 interactive_pipeline_test.py

# Entrar comandos de teste
🤔 Você: Processe o sinal [1.0, -2.5, 3.7]
🤔 Você: Explique transformada de Fourier
🤔 Você: status
🤔 Você: quit

# Verificar arquivos gerados
ls -la pipeline_test_logs/
```

## 📈 Análise de Resultados

Cada arquivo de interação inclui:
- Entrada original do usuário
- Tarefa detectada
- Tipo de processamento
- Tempo de execução
- Resposta do sistema
- Componentes ativos
- Fluxo de dados detalhado
- Métricas de consciência

## 🐛 Solução de Problemas

### Erro de Importação
- Verificar se `sys.path` inclui diretório base
- Confirmar que módulos `src/` estão disponíveis

### Pipeline Não Inicializa
- Verificar dependências do PyTorch
- Confirmar configurações YAML

### Sem Arquivos Gerados
- Verificar permissões de escrita
- Confirmar diretório de saída

## 📞 Suporte

Para problemas ou sugestões:
- Verificar logs de erro
- Consultar documentação do ΨQRH
- Revisar configurações YAML