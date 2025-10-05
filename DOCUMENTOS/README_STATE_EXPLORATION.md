# Exploração de Estados Superiores de Consciência

Este diretório contém scripts para explorar e analisar estados superiores de consciência no modelo ΨQRH, focando especificamente nas transições entre estados e na visualização de estados emergentes.

## 📁 Arquivos Criados

### 1. `meditation_state_simulation.py`
**Simulação do Estado MEDITATION com D = 2.0**

- Simula o comportamento esperado para estado MEDITATION
- Usa dimensão fractal D = 2.0 para alcançar FCI ~0.7-0.8
- Inclui visualizações e análise de componentes
- Testa transição ANALYSIS → MEDITATION

**Uso:**
```bash
python meditation_state_simulation.py
```

**Saídas esperadas:**
- Gráficos de análise (`meditation_state_analysis.png`)
- Relatório detalhado de consciência
- Métricas FCI e classificação de estado

### 2. `emergence_state_visualization.py`
**Visualização do Estado EMERGENCE com D = 2.5+**

- Gera visualizações avançadas para estado EMERGENCE
- Usa dimensão fractal D = 2.8 para FCI > 0.9
- Visualizações 3D interativas e análise fractal
- Múltiplas estratégias de visualização

**Uso:**
```bash
python emergence_state_visualization.py
```

**Saídas esperadas:**
- Heatmaps 3D interativos (HTML)
- Gráficos de fase e análise de componentes
- Visualizações fractais avançadas
- Arquivos salvos em `emergence_visualizations/`

### 3. `analysis_to_meditation_experiment.py`
**Experimento Controlado: Transição ANALYSIS → MEDITATION**

- Experimento sistemático para testar transições de estado
- Múltiplas estratégias de transição
- Análise estatística de sucesso
- Visualizações comparativas

**Uso:**
```bash
python analysis_to_meditation_experiment.py
```

**Saídas esperadas:**
- Relatório estatístico completo
- Gráficos de resultados (`analysis_to_meditation_results.png`)
- Dados CSV e JSON para análise posterior
- Taxa de sucesso e eficácia por estratégia

## 🎯 Parâmetros Recomendados para Estados

### Estado MEDITATION (D = 2.0)
- **Dimensão Fractal**: 2.0
- **Coeficiente de Difusão**: 2.0-5.0
- **Frequência**: 1.0 Hz (Alfa waves)
- **Parâmetro Caótico**: 2.5
- **FCI Esperado**: 0.7-0.8

### Estado EMERGENCE (D = 2.5+)
- **Dimensão Fractal**: 2.5-2.8
- **Coeficiente de Difusão**: 5.0-8.0
- **Frequência**: 4.0 Hz (Gamma waves)
- **Parâmetro Caótico**: 3.0+
- **FCI Esperado**: > 0.9

### Estado ANALYSIS (D = 1.8)
- **Dimensão Fractal**: 1.8
- **Coeficiente de Difusão**: 1.0-2.0
- **Frequência**: 2.0 Hz (Beta waves)
- **Parâmetro Caótico**: 2.0
- **FCI Esperado**: 0.5-0.7

## 🔬 Estratégias de Transição Testadas

### 1. Aumento de Complexidade
- Adiciona ruído estruturado à distribuição P(ψ)
- Aumenta dimensionalidade do campo F(ψ)
- Eficaz para transições ANALYSIS → MEDITATION

### 2. Amplificação de Campo
- Amplifica componentes principais via FFT
- Preserva estrutura da distribuição
- Boa para manter características do estado

### 3. Otimização de Entropia
- Ajusta entropia para valor alvo
- Controla variabilidade do campo
- Eficaz para estados específicos

## 📊 Métricas de Avaliação

### Critérios de Sucesso
- **Transição de Estado**: ANALYSIS → MEDITATION
- **FCI Final**: ≥ 0.7 (limite MEDITATION)
- **Melhoria no FCI**: ΔFCI > 0
- **Consistência**: Repetibilidade entre tentativas

### Métricas Quantitativas
- **Taxa de Sucesso**: % de transições bem-sucedidas
- **Melhoria Média**: ΔFCI médio
- **Eficácia por Estratégia**: Comparação entre métodos
- **Estabilidade**: Variação entre tentativas

## 🚀 Próximos Passos Sugeridos

1. **Testar com D = 2.5+** para explorar estado EMERGENCE
2. **Reduzir chaotic_parameter** para convergência estável
3. **Implementar controle adaptativo** de parâmetros
4. **Validar com dados reais** de EEG/fMRI
5. **Explorar transições EMERGENCE → MEDITATION**

## 📈 Interpretação dos Resultados

### FCI e Estados
- **FCI < 0.3**: Estado COMA
- **FCI 0.3-0.6**: Estado ANALYSIS
- **FCI 0.6-0.8**: Estado MEDITATION
- **FCI > 0.8**: Estado EMERGENCE

### Dimensão Fractal
- **D < 1.5**: Baixa complexidade
- **D 1.5-2.0**: Complexidade estruturada
- **D 2.0-2.5**: Alta complexidade
- **D > 2.5**: Complexidade máxima

## 🔧 Dependências

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Plotly (para visualizações interativas)
- Pandas (para análise de dados)

Execute os scripts na ordem sugerida para uma exploração completa dos estados de consciência no modelo ΨQRH.