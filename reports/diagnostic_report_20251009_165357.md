# Relatório de Diagnóstico do Pipeline ΨQRH

**Timestamp:** 20251009_165357
**Log File:** results/audit_logs/audit_20251009_165031_log.json
**Input Text:** "Hello world"
**Parameters:** {
  "stage": "emergent_generation_start"
}

## Análise de Fidelidade da Reconstrução

- **Erro Quadrático Médio (Input vs. Inverted):** 18.285658
- **Similaridade de Cosseno (Input vs. Inverted):** 0.614350
- **Preservação de Energia:** 0.014537
- **Norma Input:** 69.030685
- **Norma Inverted:** 1.003498

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

- **Autocorrelação Média (Absoluta):** nan
- **Desvio Padrão da Autocorrelação:** nan
- **Autocorrelação Máxima (Absoluta):** nan
- **Razão de Alta Correlação (>0.5):** 0.095238
- **Assunção de Independência Válida:** False

### Diagnóstico Contextual
**❌ CRÍTICO:** Alta interferência contextual. Assunção de independência é **inválida**. Estados quânticos contêm fortes 'ecos' de vizinhos.


## Conclusão e Recomendações

### Problemas Identificados
- ❌ Perda significativa de informação na reconstrução
- ❌ Interferência contextual viola assunção de independência

### Recomendações
- 🔧 Investigar acumulação de erros numéricos em operações FFT/filtro
- 🔧 Implementar probing contextual que considere dependências sequenciais
