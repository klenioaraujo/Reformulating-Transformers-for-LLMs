# Exemplos de Curl para API ΨQRH

**Versão**: 1.0.0
**Endpoint Base**: `http://localhost:5000`

---

## 1. Geração Básica de Texto

### Request
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explique o conceito de transformada quaterniônica",
    "max_length": 200,
    "temperature": 0.7
  }'
```

### Response (200 OK)
```json
{
  "generated_text": "A transformada quaterniônica é uma generalização da transformada de Fourier para o domínio quaterniônico, permitindo representações 4D de sinais. No contexto de redes neurais, ela oferece rotações em espaços de alta dimensão preservando propriedades geométricas importantes.",
  "metadata": {
    "model": "psiqrh-gpt2-medium",
    "inference_time_ms": 234,
    "tokens_generated": 56,
    "spectral_alpha": 1.2,
    "quaternion_terms_count": 4
  },
  "quaternion_analysis": {
    "rotation_magnitude": 0.87,
    "phase_coherence": 0.92,
    "spectral_energy": 1234.56
  }
}
```

---

## 2. Modo Espectral Avançado

### Request
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Descreva a álgebra de Clifford em deep learning",
    "max_length": 300,
    "temperature": 0.8,
    "spectral_mode": "enhanced",
    "consciousness_metrics": true
  }'
```

### Response (200 OK)
```json
{
  "generated_text": "A álgebra de Clifford fornece uma estrutura matemática unificada para representar transformações geométricas em espaços multidimensionais...",
  "metadata": {
    "model": "psiqrh-gpt2-medium",
    "inference_time_ms": 456,
    "tokens_generated": 123,
    "spectral_alpha": 1.2,
    "spectral_mode": "enhanced"
  },
  "consciousness_metrics": {
    "fci": 0.347,
    "phi": 2.45,
    "integrated_information": 0.82,
    "fractal_depth": 3
  },
  "quaternion_analysis": {
    "rotation_magnitude": 1.12,
    "phase_coherence": 0.95,
    "spectral_energy": 2345.67,
    "hamilton_product_count": 8
  }
}
```

---

## 3. Controle Fino de Sampling

### Request
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Aplicações práticas de quatérnios em computer vision",
    "max_length": 250,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "num_beams": 4
  }'
```

### Response (200 OK)
```json
{
  "generated_text": "Quatérnios são amplamente utilizados em computer vision para representar rotações 3D de forma eficiente e numericamente estável. Principais aplicações incluem...",
  "metadata": {
    "model": "psiqrh-gpt2-medium",
    "inference_time_ms": 567,
    "tokens_generated": 89,
    "sampling_params": {
      "temperature": 0.9,
      "top_p": 0.95,
      "top_k": 50,
      "repetition_penalty": 1.2,
      "num_beams": 4
    }
  }
}
```

---

## 4. Análise de Energia Espectral

### Request
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A transformada de Fourier quaterniônica preserva energia.",
    "analysis_type": "spectral_energy"
  }'
```

### Response (200 OK)
```json
{
  "analysis": {
    "input_energy": 1234.56,
    "output_energy": 1245.23,
    "conservation_ratio": 1.009,
    "is_conserved": true,
    "tolerance": 0.05
  },
  "spectral_analysis": {
    "dominant_frequencies": [0.12, 0.34, 0.67],
    "energy_distribution": {
      "low_freq": 0.45,
      "mid_freq": 0.35,
      "high_freq": 0.20
    }
  }
}
```

---

## 5. Validação Matemática

### Request
```bash
curl -X POST http://localhost:5000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./models/trained",
    "validation_tests": [
      "energy_conservation",
      "unitarity",
      "numerical_stability",
      "quaternion_properties"
    ]
  }'
```

### Response (200 OK)
```json
{
  "validation_results": {
    "energy_conservation": {
      "passed": true,
      "input_energy": 1234.56,
      "output_energy": 1245.23,
      "ratio": 1.009
    },
    "unitarity": {
      "passed": true,
      "mean_magnitude": 0.998,
      "std_magnitude": 0.012
    },
    "numerical_stability": {
      "passed": true,
      "nan_count": 0,
      "inf_count": 0,
      "num_passes": 1000
    },
    "quaternion_properties": {
      "passed": true,
      "identity_valid": true,
      "inverse_valid": true
    }
  },
  "overall": {
    "all_passed": true,
    "tests_passed": 4,
    "tests_total": 4
  }
}
```

---

## 6. Benchmark Comparativo

### Request
```bash
curl -X POST http://localhost:5000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "psiqrh_model": "./models/psiqrh-trained",
    "baseline_model": "gpt2-medium",
    "test_prompts": [
      "Explique quatérnios",
      "O que é álgebra de Clifford?",
      "Transformadas espectrais em ML"
    ],
    "num_runs": 10
  }'
```

### Response (200 OK)
```json
{
  "benchmark_results": {
    "psiqrh": {
      "avg_inference_time_ms": 234.5,
      "avg_tokens_per_second": 234.5,
      "memory_usage_mb": 1523.45,
      "quality_score": 0.92
    },
    "baseline": {
      "avg_inference_time_ms": 156.2,
      "avg_tokens_per_second": 312.1,
      "memory_usage_mb": 1489.23,
      "quality_score": 0.79
    },
    "comparison": {
      "speed_difference_pct": -24.8,
      "memory_difference_pct": 2.3,
      "quality_improvement_pct": 16.5,
      "quaternion_terms_ratio": 2.4
    }
  },
  "detailed_results": [
    {
      "prompt": "Explique quatérnios",
      "psiqrh_response": "Quatérnios são números hipercomplexos...",
      "baseline_response": "Quatérnios são conceitos matemáticos...",
      "psiqrh_quality": 0.95,
      "baseline_quality": 0.82
    }
  ]
}
```

---

## 7. Configuração em Tempo Real

### Request
```bash
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{
    "action": "update",
    "config": {
      "qrh_layer": {
        "alpha": 1.5,
        "embed_dim": 128
      },
      "consciousness_processor": {
        "fci_threshold": 0.400
      }
    }
  }'
```

### Response (200 OK)
```json
{
  "status": "updated",
  "config": {
    "qrh_layer": {
      "alpha": 1.5,
      "embed_dim": 128,
      "use_learned_rotation": true
    },
    "consciousness_processor": {
      "fci_threshold": 0.400,
      "phi_integration": true,
      "fractal_depth": 3
    }
  },
  "message": "Configuração atualizada com sucesso"
}
```

---

## 8. Métricas de Sistema

### Request
```bash
curl -X GET http://localhost:5000/metrics \
  -H "Accept: application/json"
```

### Response (200 OK)
```json
{
  "system_metrics": {
    "uptime_seconds": 3600,
    "total_requests": 1234,
    "avg_response_time_ms": 234.5,
    "error_rate": 0.002,
    "cache_hit_rate": 0.67
  },
  "model_metrics": {
    "model_name": "psiqrh-gpt2-medium",
    "total_tokens_generated": 567890,
    "avg_tokens_per_request": 156,
    "quaternion_operations_count": 123456
  },
  "fft_cache_metrics": {
    "hits": 456,
    "misses": 234,
    "hit_rate": 0.66,
    "current_entries": 8,
    "max_entries": 10,
    "memory_usage_mb": 45.67
  }
}
```

---

## 9. Health Check

### Request
```bash
curl -X GET http://localhost:5000/health
```

### Response (200 OK)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": false,
  "version": "ΨQRH-v1.0.0",
  "timestamp": "2025-10-02T09:48:49.123456"
}
```

---

## 10. Batch Processing

### Request
```bash
curl -X POST http://localhost:5000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "Explique quatérnios",
      "O que é álgebra de Clifford?",
      "Transformadas espectrais"
    ],
    "max_length": 150,
    "temperature": 0.7
  }'
```

### Response (200 OK)
```json
{
  "results": [
    {
      "prompt": "Explique quatérnios",
      "generated_text": "Quatérnios são números hipercomplexos...",
      "metadata": {
        "tokens_generated": 45,
        "inference_time_ms": 123
      }
    },
    {
      "prompt": "O que é álgebra de Clifford?",
      "generated_text": "A álgebra de Clifford é uma extensão...",
      "metadata": {
        "tokens_generated": 52,
        "inference_time_ms": 145
      }
    },
    {
      "prompt": "Transformadas espectrais",
      "generated_text": "Transformadas espectrais decompõem sinais...",
      "metadata": {
        "tokens_generated": 38,
        "inference_time_ms": 112
      }
    }
  ],
  "batch_metadata": {
    "total_prompts": 3,
    "total_time_ms": 380,
    "avg_time_per_prompt_ms": 126.7
  }
}
```

---

## Headers Importantes

### Request Headers
```
Content-Type: application/json
Accept: application/json
X-API-Key: <optional-api-key>
X-Request-ID: <optional-uuid>
```

### Response Headers
```
Content-Type: application/json
X-Model-Version: ΨQRH-v1.0
X-Inference-Time: 234ms
X-Spectral-Alpha: 1.2
X-Request-ID: <uuid>
X-RateLimit-Remaining: 98
```

---

## Códigos de Status

| Código | Significado | Exemplo |
|--------|-------------|---------|
| 200 | OK | Requisição bem-sucedida |
| 400 | Bad Request | JSON inválido ou parâmetros faltando |
| 404 | Not Found | Endpoint não existe |
| 429 | Too Many Requests | Rate limit excedido |
| 500 | Internal Server Error | Erro no servidor |
| 503 | Service Unavailable | Modelo não carregado |

---

## Tratamento de Erros

### Erro 400 - Bad Request
```json
{
  "error": "Invalid parameters",
  "message": "Field 'prompt' is required",
  "status_code": 400,
  "timestamp": "2025-10-02T09:48:49.123456"
}
```

### Erro 429 - Rate Limit
```json
{
  "error": "Rate limit exceeded",
  "message": "Maximum 100 requests per minute",
  "retry_after_seconds": 45,
  "status_code": 429
}
```

### Erro 500 - Internal Server Error
```json
{
  "error": "Internal server error",
  "message": "Model inference failed",
  "details": "NaN detected in output",
  "status_code": 500,
  "request_id": "abc-123-def"
}
```

---

## Exemplos Avançados

### Streaming de Resposta (Server-Sent Events)

```bash
curl -N -X POST http://localhost:5000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explique álgebra de Clifford em detalhes",
    "max_length": 500,
    "temperature": 0.7
  }'
```

**Response (text/event-stream)**:
```
data: {"token": "A", "position": 0}

data: {"token": " álgebra", "position": 1}

data: {"token": " de", "position": 2}

data: {"token": " Clifford", "position": 3}

...

data: {"done": true, "total_tokens": 156}
```

---

## Script de Teste Completo

```bash
#!/bin/bash

# Teste completo da API ΨQRH

API_URL="http://localhost:5000"

echo "1. Health Check"
curl -s -X GET $API_URL/health | jq

echo -e "\n2. Geração Básica"
curl -s -X POST $API_URL/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explique quatérnios",
    "max_length": 100
  }' | jq

echo -e "\n3. Análise Espectral"
curl -s -X POST $API_URL/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Transformada quaterniônica",
    "analysis_type": "spectral_energy"
  }' | jq

echo -e "\n4. Métricas do Sistema"
curl -s -X GET $API_URL/metrics | jq

echo -e "\nTestes concluídos!"
```

---

## Conclusão

Esta documentação fornece exemplos completos de uso da API ΨQRH, incluindo:
- ✅ Geração de texto com parâmetros variados
- ✅ Análise espectral e validação matemática
- ✅ Benchmarking e métricas
- ✅ Configuração dinâmica
- ✅ Tratamento de erros
- ✅ Batch processing e streaming

**Próximos Passos**:
- Testar com diferentes modelos
- Otimizar parâmetros de inferência
- Monitorar métricas de performance
- Implementar rate limiting e autenticação

---

**Ω∞Ω** - Continuidade Garantida
**Assinatura**: ΨQRH-API-v1.0.0
