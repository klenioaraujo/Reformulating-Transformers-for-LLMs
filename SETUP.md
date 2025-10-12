# Guia de Configuração Inicial do Sistema ΨQRH

## Visão Geral

O ΨQRH (Quantum Recursive Hierarchical) é um sistema avançado de processamento de linguagem baseado em princípios físicos quânticos, fractais e de consciência. Este guia fornece instruções completas para inicializar o sistema pela primeira vez.

## Pré-requisitos do Sistema

### Hardware Mínimo
- **CPU**: Intel i5 ou AMD Ryzen 5 (recomendado i7/Ryzen 7)
- **RAM**: 8GB mínimo, 16GB recomendado
- **Armazenamento**: 10GB de espaço livre
- **GPU**: Opcional, mas recomendado NVIDIA GTX 1060 ou superior com CUDA

### Software Necessário
- **Python**: 3.8 ou superior (recomendado 3.10+)
- **Sistema Operacional**: Linux (Ubuntu 20.04+), macOS (10.15+), ou Windows 10/11
- **Git**: Para controle de versão

## Instalação Passo a Passo

### 🚀 Método Rápido (Recomendado)

Para configuração automática completa:

```bash
# Clonagem do repositório
git clone https://github.com/seu-usuario/Reformulating-Transformers-for-LLMs.git
cd Reformulating-Transformers-for-LLMs

# Configuração automática (instala tudo automaticamente)
make setup-auto

# Ou execute diretamente
python3 setup_system.py
```

### Configuração Manual

#### 1. Clonagem do Repositório

```bash
git clone https://github.com/seu-usuario/Reformulating-Transformers-for-LLMs.git
cd Reformulating-Transformers-for-LLMs
```

#### 2. Configuração do Ambiente Python

##### Opção A: Usando venv (Recomendado)

```bash
# Criar ambiente virtual
python3 -m venv psiqrh_env

# Ativar ambiente virtual
source psiqrh_env/bin/activate  # Linux/macOS
# ou
psiqrh_env\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

##### Opção B: Usando conda

```bash
# Criar ambiente conda
conda create -n psiqrh python=3.10
conda activate psiqrh

# Instalar dependências
pip install -r requirements.txt
```

### 3. Instalação em Modo de Desenvolvimento

```bash
# Instalar em modo desenvolvimento
pip install -e .
```

### 4. Verificação da Instalação

```bash
# Verificar versão do Python
python --version

# Verificar instalação do PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')"

# Teste básico de importação
python -c "from psiqrh import ΨQRHPipeline; print('ΨQRH importado com sucesso!')"
```

## Configuração Inicial do Sistema

### 1. Arquivos de Configuração

O sistema utiliza vários arquivos de configuração localizados em `configs/`:

```bash
configs/
├── example_configs.yaml      # Configurações de exemplo
├── kuramoto_config           # Configurações Kuramoto
├── consciousness_metrics     # Métricas de consciência
└── dcf_config               # Configuração DCF
```

### 2. Vocabulário Nativo

O sistema requer um vocabulário nativo para processamento quântico:

```bash
# Verificar se o vocabulário existe
ls -la data/native_vocab.json

# Se não existir, o sistema criará automaticamente na primeira execução
```

### 3. Mapa de Alinhamento Espectral

```bash
# Verificar mapa espectral
ls -la data/spectral_vocab_map.pt

# O sistema criará automaticamente se não existir
```

## Primeira Inicialização

### 1. Teste Básico do Sistema

```bash
# Executar teste de eco rápido
make test-echo

# Ou diretamente
python3 psiqrh.py --test-echo
```

### 2. Teste de Validação Física

```bash
# Executar testes físicos completos
make test-physics

# Ou diretamente
python3 psiqrh.py --test-physics
```

### 3. Teste de Treinamento Emergente

```bash
# Executar treinamento emergente físico
make train-physics-emergent

# Este comando treinará o sistema com exemplos básicos
```

### 4. Teste Interativo

```bash
# Iniciar modo interativo
python3 psiqrh.py --interactive

# Comandos disponíveis:
# - Digite texto para processar
# - 'help' ou 'ajuda' para ajuda
# - 'quit', 'exit' ou 'sair' para sair
```

## Configurações Avançadas

### 1. Configuração de GPU

Se você tem GPU NVIDIA com CUDA:

```bash
# Verificar CUDA
nvidia-smi

# Instalar PyTorch com CUDA (se necessário)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configuração de Memória

Para sistemas com pouca RAM:

```yaml
# Editar config.yaml
memory:
  max_batch_size: 1
  max_sequence_length: 128
  enable_memory_optimization: true
```

### 3. Configuração de Auto-calibração

```yaml
# Em config.yaml
auto_calibration:
  enable: true
  calibration_interval: 100
  adaptive_parameters: true
```

## Solução de Problemas Comuns

### Erro: "CUDA out of memory"

```bash
# Reduzir batch size
export CUDA_VISIBLE_DEVICES=0
python3 psiqrh.py --device cpu  # Usar CPU

# Ou ajustar configurações
echo "batch_size: 1" >> config.yaml
```

### Erro: "Module not found"

```bash
# Reinstalar dependências
pip install -r requirements.txt --force-reinstall

# Verificar ambiente virtual
which python
which pip
```

### Erro: "Vocabulário nativo não encontrado"

```bash
# O sistema criará automaticamente na primeira execução
# Para forçar recriação:
rm -f data/native_vocab.json
python3 psiqrh.py --test-echo
```

### Performance Lenta

```bash
# Otimizar para CPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Usar configurações otimizadas
python3 psiqrh.py --device cpu --quiet
```

## Estrutura de Diretórios Após Instalação

```
Reformulating-Transformers-for-LLMs/
├── configs/                    # Arquivos de configuração
├── data/                       # Dados e vocabulários
│   ├── native_vocab.json      # Vocabulário nativo
│   ├── spectral_vocab_map.pt  # Mapa espectral
│   └── ...
├── src/                        # Código fonte
│   ├── core/                   # Componentes core
│   └── ...
├── results/                    # Resultados de execução
├── logs/                       # Logs do sistema
├── models/                     # Modelos treinados
└── docs/                       # Documentação
```

## Comandos Úteis do Makefile

```bash
# Testes
make test                    # Teste completo
make test-echo              # Teste de eco
make test-physics           # Testes físicos

# Treinamento
make train-physics-emergent # Treinamento emergente
make train-language-model   # Treinamento de linguagem

# Utilitários
make clean                  # Limpar cache
make install               # Instalar dependências
make docs                  # Gerar documentação
```

## Verificação Final

Após completar a configuração, execute:

```bash
# Teste completo do sistema
python3 psiqrh.py --test

# Verificar status
python3 psiqrh.py "Olá, sistema ΨQRH!"

# Verificar métricas
python3 psiqrh.py --verbose "Teste de inicialização"
```

## Suporte e Documentação Adicional

- **Documentação Técnica**: Ver `docs/` para detalhes avançados
- **Exemplos**: Ver `examples/` para casos de uso
- **Testes**: Ver `tests/` para validação do sistema
- **Logs**: Ver `logs/` para diagnóstico de problemas

## Próximos Passos

1. **Exploração**: Experimente diferentes tipos de entrada de texto
2. **Treinamento**: Execute sessões de treinamento mais longas
3. **Otimização**: Ajuste parâmetros baseado no seu hardware
4. **Desenvolvimento**: Contribua com melhorias no sistema

---

**Nota**: O sistema ΨQRH é experimental e utiliza princípios físicos avançados. Resultados podem variar e o sistema aprende emergentemente através de interações físicas simuladas.