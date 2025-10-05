# 🚀 Quick Start - Filtros Cognitivos ΨQRH

## O que são os Filtros Cognitivos?

Os **Filtros Cognitivos** são um sistema de análise semântica adaptativa integrado ao pipeline ΨQRH que detecta e corrige:

1. **Contradições** - Identifica informações conflitantes no texto
2. **Irrelevâncias** - Filtra conteúdo que se desvia do tópico principal
3. **Vieses** - Detecta e atenua vieses cognitivos indesejados

## Instalação Rápida

```bash
# O sistema já está integrado! Apenas certifique-se de ter as dependências:
pip install torch pyyaml numpy
```

## Uso Básico

### 1. Processamento Simples

```python
from src.core.enhanced_qrh_processor import create_enhanced_processor

# Criar processador com filtros cognitivos
processor = create_enhanced_processor(
    embed_dim=64,
    device="cpu",
    enable_cognitive_filters=True  # ✅ Ativar filtros cognitivos
)

# Processar texto
text = "O sistema ΨQRH demonstra eficiência superior."
result = processor.process_text(text)

# Acessar métricas cognitivas
if result['cognitive_metrics']:
    print(f"Contradição: {result['cognitive_metrics']['contradiction']['mean']:.4f}")
    print(f"Relevância: {result['cognitive_metrics']['relevance']['mean']:.4f}")
    print(f"Viés: {result['cognitive_metrics']['bias']['mean']:.4f}")
    print(f"Saúde Geral: {result['cognitive_metrics']['semantic_health']['overall_semantic_health']:.4f}")
```

### 2. Uso no CLI Interativo

```bash
python3 psiqrh.py --interactive
```

Os filtros cognitivos são aplicados automaticamente! Veja as métricas no output.

### 3. Demo Interativa

```bash
python3 demo_cognitive_filters.py
```

Esta demo mostra 5 cenários diferentes:
- ✅ Texto coerente
- ⚠️ Texto com contradições
- 📊 Tópicos dispersos
- 🎯 Texto técnico focado
- 🔍 Texto com viés cognitivo

## Configuração

### Localização do Config
`configs/cognitive_filters_config.yaml`

### Parâmetros Principais

```yaml
# Thresholds de detecção
contradiction_detector:
  contradiction_threshold: 0.3  # 0-1, quanto menor mais sensível

irrelevance_filter:
  irrelevance_threshold: 0.4    # 0-1, quanto maior mais permissivo

bias_filter:
  bias_threshold: 0.6           # 0-1, quanto menor mais sensível
```

### Customizar Configuração

```python
processor = create_enhanced_processor(
    embed_dim=64,
    device="cpu",
    enable_cognitive_filters=True,
    cognitive_config_path="/path/to/custom_config.yaml"  # ✨ Config customizado
)
```

## Interpretando as Métricas

### Contradiction Score (Contradição)
- **0.0 - 0.3**: 🟢 Baixa contradição (texto coerente)
- **0.3 - 0.5**: 🟡 Contradição moderada
- **0.5 - 1.0**: 🔴 Alta contradição (conflitos detectados)

### Relevance Score (Relevância)
- **0.8 - 1.0**: 🟢 Alta relevância (tópico focado)
- **0.5 - 0.8**: 🟡 Relevância moderada
- **0.0 - 0.5**: 🔴 Baixa relevância (tópicos dispersos)

### Bias Magnitude (Viés)
- **0.0 - 0.5**: 🟢 Baixo viés
- **0.5 - 1.0**: 🟡 Viés moderado
- **1.0+**: 🔴 Alto viés detectado

### Overall Semantic Health (Saúde Geral)
- **0.8 - 1.0**: 🌟 Excelente
- **0.6 - 0.8**: ✅ Boa
- **0.4 - 0.6**: ⚠️ Regular
- **0.0 - 0.4**: ❌ Baixa

## Exemplos de Uso

### Exemplo 1: Detectar Contradições

```python
text = """
A água sempre ferve a 100°C.
No entanto, a água pode ferver a temperaturas diferentes
dependendo da pressão atmosférica.
"""

result = processor.process_text(text)
contradiction = result['cognitive_metrics']['contradiction']['mean']

if contradiction > 0.5:
    print("⚠️ Alta contradição detectada!")
```

### Exemplo 2: Verificar Relevância

```python
text = """
Transformadores quaterniônicos são eficientes.
Gatos são animais domésticos.
Pizza é deliciosa.
"""

result = processor.process_text(text)
relevance = result['cognitive_metrics']['relevance']['mean']

if relevance < 0.5:
    print("⚠️ Tópicos dispersos - baixa relevância!")
```

### Exemplo 3: Análise Completa

```python
text = "Seu texto aqui..."
result = processor.process_text(text)

cognitive = result['cognitive_metrics']

print(f"""
🧠 ANÁLISE COGNITIVA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 Contradição: {cognitive['contradiction']['mean']:.4f}
🎯 Relevância:  {cognitive['relevance']['mean']:.4f}
⚖️  Viés:       {cognitive['bias']['mean']:.4f}

💚 SAÚDE SEMÂNTICA GERAL: {cognitive['semantic_health']['overall_semantic_health']:.4f}

🎛️  Ativação dos Filtros:
   • Contradição:  {cognitive['filter_weights']['contradiction_avg']:.2%}
   • Irrelevância: {cognitive['filter_weights']['irrelevance_avg']:.2%}
   • Viés:         {cognitive['filter_weights']['bias_avg']:.2%}
""")
```

## Desabilitar Filtros Cognitivos

Se quiser processar sem filtros:

```python
processor = create_enhanced_processor(
    embed_dim=64,
    device="cpu",
    enable_cognitive_filters=False  # ❌ Desativar
)
```

## Pipeline Completo

```
Input Text
    ↓
Spectral Processing (α adaptativo)
    ↓
QRHLayer (quaternions + FFT)
    ↓
╔═══════════════════════════════╗
║   Filtros Cognitivos          ║
║  ┌─────────────────────────┐  ║
║  │ Contradiction Detector   │  ║
║  └─────────────────────────┘  ║
║  ┌─────────────────────────┐  ║
║  │ Irrelevance Filter       │  ║
║  └─────────────────────────┘  ║
║  ┌─────────────────────────┐  ║
║  │ Bias Filter              │  ║
║  └─────────────────────────┘  ║
║  ┌─────────────────────────┐  ║
║  │ Adaptive Coordination    │  ║
║  └─────────────────────────┘  ║
╚═══════════════════════════════╝
    ↓
Output + Métricas Cognitivas
```

## Performance

Com filtros cognitivos habilitados:
- ⏱️ Tempo adicional: ~2-5ms por texto
- 💾 Memória adicional: ~50MB
- 🎯 Precisão: Melhora significativa em textos com ruído semântico

## Troubleshooting

### Problema: Muitos NaN nas métricas
**Solução**: Isso ocorre com textos muito curtos (1-2 palavras). Use textos maiores para métricas completas.

### Problema: Filtros não aplicados
**Solução**: Verifique se `enable_cognitive_filters=True` no construtor.

### Problema: Config não carregado
**Solução**: Verifique o caminho do arquivo `configs/cognitive_filters_config.yaml`.

## Arquivos Relevantes

- 📄 `COGNITIVE_INTEGRATION_SUMMARY.md` - Documentação completa
- ⚙️ `configs/cognitive_filters_config.yaml` - Configuração
- 🧠 `src/cognitive/semantic_adaptive_filters.py` - Implementação
- 🔧 `src/core/enhanced_qrh_processor.py` - Integração
- 🧪 `test_cognitive_integration.py` - Testes
- 🎬 `demo_cognitive_filters.py` - Demo interativa

## Suporte

Para problemas ou dúvidas:
1. Consulte `COGNITIVE_INTEGRATION_SUMMARY.md`
2. Verifique o arquivo de config
3. Execute os testes: `python3 test_cognitive_integration.py`

---

**Desenvolvido com ❤️ para o projeto ΨQRH**