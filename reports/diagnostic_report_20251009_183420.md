# Relatório de Diagnóstico do Pipeline ΨQRH

**Timestamp:** 20251009_183420
**Log File:** audit_logs/audit_20251009_170352_log.json
**Input Text:** "Test numerical stability"
**Parameters:** {
  "test": "stability"
}

## Análise de Fidelidade da Reconstrução

- **Erro Quadrático Médio (Input vs. Inverted):** 0.298963
- **Similaridade de Cosseno (Input vs. Inverted):** 0.851251
- **Preservação de Energia:** 1.000000
- **Norma Input:** 128.314941
- **Norma Inverted:** 128.314926

### Diagnóstico de Reconstrução
**❌ CRÍTICO:** Perda significativa de informação. Problemas graves de estabilidade numérica.


## Análise do Espaço de Embedding (dim=64)

- **Distância Média Mínima:** 3.814708
- **Desvio Padrão das Distâncias:** 1.110042

### Pares de Caracteres Mais Problemáticos
- **('o', 'p')**: Similaridade = 0.000004
- **('*', 'T')**: Similaridade = 0.000003
- **('S', 'T')**: Similaridade = 0.000003
- **('S', 'x')**: Similaridade = 0.000003
- **('e', '{')**: Similaridade = 0.000003

### Diagnóstico de Embedding
**✅ BOM:** Boa separabilidade entre caracteres.


## Análise de Interferência Contextual

- **Autocorrelação Média (Absoluta):** 0.104839
- **Desvio Padrão da Autocorrelação:** 0.131048
- **Autocorrelação Máxima (Absoluta):** 0.333141
- **Razão de Alta Correlação (>0.5):** 0.000000
- **Assunção de Independência Válida:** True

### Diagnóstico Contextual
**✅ BOM:** Baixa interferência contextual. Assunção de independência é válida.


## Conclusão e Recomendações

### Problemas Identificados
- ❌ Perda significativa de informação na reconstrução

### Recomendações
- 🔧 Investigar acumulação de erros numéricos em operações FFT/filtro
