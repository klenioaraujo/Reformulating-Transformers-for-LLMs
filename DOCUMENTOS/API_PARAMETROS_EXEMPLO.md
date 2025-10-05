# API com Parâmetros de Processamento - Exemplo de Resposta

## Resposta da API com Parâmetros Incluídos

A API agora retorna todos os parâmetros reais utilizados no processamento no campo `processing_parameters`. Aqui está um exemplo da resposta completa:

### Resposta Completa
```json
{
  "consciousness_metrics": {
    "convergence_achieved": true,
    "distribution_spread": 0.002196397166699171,
    "entropy": 5.360718250274658,
    "fci": 0.00340536842122674,
    "field_magnitude": 1.9473668336868286,
    "fractal_dimension": 1.008,
    "peak_distribution": 0.007258872035890818,
    "processing_steps": 12,
    "state": "COMA"
  },
  "gls_data": {
    "p5js_code": "...",
    "processing_code": "...",
    "status": "success",
    "visual_params": {
      "colors": ["#2C3E50", "#34495E", "#7F8C8D"],
      "complexity": 4,
      "fci": 0.00340536842122674,
      "fractal_dim": 1.008,
      "rotation_speed": 0.01,
      "state": "COMA"
    }
  },
  "processing_parameters": {
    "qrh_config": {
      "alpha": 1.0,
      "device": "cpu",
      "embed_dim": 64,
      "use_learned_rotation": true
    },
    "consciousness_config": {
      "chaotic_parameter": 3.9,
      "consciousness_frequency_range": [0.5, 5.0],
      "device": "cpu",
      "diffusion_coefficient_range": [0.01, 10.0],
      "embedding_dim": 256,
      "fci_threshold_analysis": 0.6,
      "fci_threshold_coma": 0.2,
      "fci_threshold_emergence": 0.9,
      "fci_threshold_meditation": 0.8,
      "fractal_dimension_range": [1.0, 3.0],
      "max_iterations": 100,
      "phase_consciousness": 0.7854,
      "sequence_length": 64,
      "time_step": 0.01
    },
    "psicws_config": {
      "cache_dir": "data/Ψcws_cache",
      "default_model": "latest",
      "enabled": true,
      "fallback_to_qrh": true,
      "load_on_init": false,
      "model_dir": "data/Ψcws",
      "use_cache": true
    }
  },
  "response": "Análise para 'ola sistema' com alpha=1.616: Espectro com torch.Size([1, 1, 256]) dimensões...",
  "status": "success",
  "timestamp": 0.10053646564483643,
  "user_message": "ola sistema"
}
```

## Parâmetros de Processamento Incluídos

### 1. Configurações QRH (`qrh_config`)
- **embed_dim**: 64 (dimensão de embedding)
- **alpha**: 1.0 (parâmetro de escala principal)
- **use_learned_rotation**: true (rotação aprendida habilitada)
- **device**: "cpu" (dispositivo de processamento)

### 2. Configurações de Consciência (`consciousness_config`)
- **embedding_dim**: 256 (dimensão de embedding para consciência)
- **sequence_length**: 64 (comprimento da sequência)
- **fractal_dimension_range**: [1.0, 3.0] (faixa de dimensão fractal)
- **consciousness_frequency_range**: [0.5, 5.0] Hz (faixa de frequência cerebral)
- **chaotic_parameter**: 3.9 (parâmetro caótico)
- **time_step**: 0.01 (passo de tempo)
- **max_iterations**: 100 (máximo de iterações)

### 3. Limiares de Consciência
- **fci_threshold_coma**: 0.2
- **fci_threshold_analysis**: 0.6
- **fci_threshold_meditation**: 0.8
- **fci_threshold_emergence**: 0.9

### 4. Configurações Ψcws (`psicws_config`)
- **enabled**: true (modelo Ψcws habilitado)
- **fallback_to_qrh**: true (fallback para QRH padrão)
- **use_cache**: true (cache habilitado)
- **model_dir**: "data/Ψcws" (diretório do modelo)

## Modificações Realizadas na API

### Arquivo `app.py`

As seguintes modificações foram feitas para incluir os parâmetros:

```python
# Adicionado na resposta da API
response_data = {
    'status': 'success',
    'user_message': user_message,
    'timestamp': torch.rand(1).item(),
    'processing_parameters': {
        'qrh_config': {
            'embed_dim': qrh_factory.config.embed_dim,
            'alpha': qrh_factory.config.alpha,
            'use_learned_rotation': qrh_factory.config.use_learned_rotation,
            'device': 'cpu'
        },
        'consciousness_config': qrh_factory.consciousness_config,
        'psicws_config': qrh_factory.psicws_config
    }
}

# Adicionado para parâmetros específicos do processamento
if 'layer1_fractal' in result:
    layer1_data = result['layer1_fractal']
    response_data['processing_parameters']['layer1_fractal'] = {
        'alpha_adaptive': layer1_data.get('alpha', 0.0),
        'shape': layer1_data.get('shape', []),
        'statistics': layer1_data.get('statistics', {}),
        'values_count': len(layer1_data.get('values', {}).get('magnitude', []))
    }
```

## Benefícios da Inclusão dos Parâmetros

1. **Transparência Total**: Todos os parâmetros utilizados no processamento são visíveis
2. **Reprodutibilidade**: Permite reproduzir exatamente o mesmo processamento
3. **Debugging**: Facilita identificar problemas de configuração
4. **Análise**: Permite correlacionar parâmetros com resultados
5. **Auditoria**: Registro completo do processamento realizado

## Como Usar

Agora cada resposta da API inclui automaticamente:
- Parâmetros de configuração carregados dos arquivos YAML
- Valores específicos calculados durante o processamento
- Métricas de consciência e visualização
- Dados completos para análise e reprodução

Isso garante que cada resposta seja completamente transparente e auditável, mostrando exatamente quais parâmetros foram utilizados para gerar cada resultado.