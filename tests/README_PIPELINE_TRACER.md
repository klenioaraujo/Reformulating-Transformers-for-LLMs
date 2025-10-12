# Pipeline Tracer - Debug Tool

## Objetivo

Este tool foi criado para **mapear e identificar erros** no pipeline ΨQRH de processamento de texto. Ele traça **passo a passo** como a informação se comporta em cada etapa, criando logs detalhados que servem como mapa de diagnóstico.

## Como Usar

### Execução Básica
```bash
python debug_pipeline_tracer.py "Qual a cor do céu?"
```

### Com Arquivo de Entrada
```bash
python debug_pipeline_tracer.py --file input.txt
```

### Diretório de Saída Customizado
```bash
python debug_pipeline_tracer.py "teste" --output-dir meus_logs
```

## Etapas Rastreadas

O tracer monitora **6 etapas principais** do pipeline:

### 1. 🔮 Text → Fractal Signal
- **Entrada**: Texto bruto
- **Saída**: Sinal fractal complexo
- **Logs**: Forma do tensor, estatísticas, tipo de sinal

### 2. 🌀 Fractal Signal → Quaternions
- **Entrada**: Sinal fractal
- **Saída**: Estado quântico 4D
- **Logs**: Dimensões quaterniônicas, norma, compatibilidade

### 3. 🌊 Spectral Filtering
- **Entrada**: Estado quântico
- **Saída**: Estado filtrado
- **Logs**: Parâmetro alpha, mudança de energia, estatísticas

### 4. 🔄 SO(4) Rotation
- **Entrada**: Estado filtrado
- **Saída**: Estado rotacionado
- **Logs**: Validação de unitariedade, rotações aplicadas

### 5. 🔬 Optical Probe Decoding
- **Entrada**: Estado quântico final
- **Saída**: Token ID
- **Logs**: Status do mapa vocabular, similaridades, dimensões

### 6. 📝 Token → Text
- **Entrada**: Token ID
- **Saída**: Texto final
- **Logs**: Mapeamento vocabular, texto gerado

## Arquivos de Saída

### Log Principal (`debug_logs/pipeline_trace_YYYYMMDD_HHMMSS.jsonl`)
- Formato JSONL (uma linha por etapa)
- Timestamp preciso
- Metadados completos
- Estatísticas de tensores
- Informações de erro

### Estrutura do Log:
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "step": "fractal_conversion",
  "session_id": "20241011_120000",
  "data": {
    "input_text": "Hello",
    "tensor_shape": [1, 64],
    "tensor_stats": {
      "min": -0.5,
      "max": 1.2,
      "mean": 0.3,
      "norm": 8.7
    }
  }
}
```

## Identificação de Erros

### Erros Comuns Detectáveis:

1. **Incompatibilidade Dimensional**
   - Tensores com formas inesperadas
   - Mismatch entre `embed_dim` esperado e real

2. **Perda de Informação**
   - Norma do tensor reduzindo drasticamente
   - Valores NaN ou infinitos

3. **Falha no Optical Probe**
   - Mapa vocabular não carregado
   - Similaridades de cosseno baixas
   - Token ID inválido

4. **Problemas de Unitariedade**
   - Energia não conservada nas rotações
   - Norma alterada após filtragem

### Exemplo de Diagnóstico:
```bash
# Executar tracer
python debug_pipeline_tracer.py "test"

# Analisar logs
cat debug_logs/pipeline_trace_*.jsonl | jq '.step, .data.error // .data.output'
```

## Integração com Pipeline Principal

### Para Debug em Produção:
```python
from debug_pipeline_tracer import PipelineTracer

# Substituir chamada direta do pipeline
tracer = PipelineTracer()
result = tracer.trace_complete_pipeline("input text")
```

### Para Testes Automatizados:
```python
# Criar suite de testes com tracer
import unittest
from debug_pipeline_tracer import PipelineTracer

class TestPipeline(unittest.TestCase):
    def test_pipeline_trace(self):
        tracer = PipelineTracer(output_dir="test_logs")
        result = tracer.trace_complete_pipeline("test input")
        self.assertIsNotNone(result)
```

## Dicas de Uso

1. **Comece com textos simples**: "a", "test", "hello"
2. **Compare múltiplas execuções**: Use diferentes textos de entrada
3. **Verifique os logs após cada erro**: O último log antes da falha contém a causa
4. **Use o session_id**: Para correlacionar múltiplos traces
5. **Monitore estatísticas**: Mudanças abruptas indicam problemas

## Próximos Passos

1. Execute o tracer com texto simples
2. Identifique em qual etapa o erro ocorre
3. Use os logs para diagnosticar a causa raiz
4. Corrija o problema no pipeline principal
5. Re-execute o tracer para validar a correção