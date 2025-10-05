# ΨQRH Docker - Quick Start Guide

## 🚀 Início Rápido

### Pré-requisitos
- Docker 20.10+
- Docker Compose 2.0+

### Iniciar Serviços

```bash
cd ops/docker
./restart-services.sh
```

### Opções do Script

```bash
# Iniciar com logs em tempo real
./restart-services.sh --logs

# Rebuild completo (após mudanças no código)
./restart-services.sh --rebuild

# Limpar volumes órfãos
./restart-services.sh --clean

# Combinar opções
./restart-services.sh --rebuild --logs
```

## 🌐 Acessar Serviços

Após inicialização bem-sucedida:

- **Frontend Web**: http://localhost:3000
- **API REST**: http://localhost:5000
- **Health Check**: http://localhost:5000/health
- **API Info**: http://localhost:5000/api

## 🎨 Visualização GLS em Tempo Real

A interface web em `http://localhost:3000` inclui:

1. **Chat Interativo**: Digite mensagens e veja análise ΨQRH
2. **Visualização GLS**: Canvas p5.js com harmônicos espectrais em tempo real
3. **Métricas de Consciência**: FCI, entropia, dimensão fractal, estado
4. **Dados da API**: JSON estruturado com todos os parâmetros

### Como Usar

1. Acesse http://localhost:3000
2. Digite uma mensagem no chat (ex: "ola mundo")
3. Observe:
   - Métricas de consciência atualizadas
   - Visualização GLS animada
   - Dados espectrais extraídos

## 🔧 Correções Aplicadas

### 1. Proxy Nginx (HTTP 405 Corrigido)
**Problema**: API não acessível via frontend (erro 405)
**Solução**: Descomentado proxy nginx em `nginx.conf`

```nginx
location /api/ {
    proxy_pass http://psiqrh-api:5000/api/;
    # CORS headers habilitados
}
```

### 2. Validação de Dados no Frontend
**Problema**: JavaScript quebrava se `consciousness_metrics` fosse `null`
**Solução**: Adicionadas verificações defensivas em `index.html`

```javascript
if (data.consciousness_metrics) {
    // Processar métricas
} else {
    // Fallback seguro
}
```

### 3. GLS Generator - KeyError 'response'
**Problema**: `harmonic_gls_generator.py` assumia estrutura específica
**Solução**: Suporte a múltiplos formatos de dict

```python
# Aceita: response_data['response'], ['text_analysis'] ou conversão direta
```

### 4. Estado COMA vs ANALYSIS
**Problema**: Estado "COMA" exibido incorretamente no GLS
**Solução**: Inferência de estado baseada em FCI quando `state=None`

```python
if fci >= 0.3:
    state_name = 'ANALYSIS'
else:
    state_name = 'COMA'
```

## 📊 API Endpoints

### POST /api/chat
Processar mensagem de chat

**Request:**
```json
{
  "message": "ola mundo"
}
```

**Response:**
```json
{
  "status": "success",
  "response": "Análise para 'ola mundo'...",
  "consciousness_metrics": {
    "fci": 0.5852,
    "state": "ANALYSIS",
    "entropy": 5.5452,
    "fractal_dimension": 1.68,
    "field_magnitude": 2.1307,
    "coherence": 0.6241
  },
  "gls_data": { ... }
}
```

### GET /api/health
Status do sistema

**Response:**
```json
{
  "status": "healthy",
  "system": "ΨQRH API",
  "components": {
    "qrh_factory": "loaded",
    "consciousness_processor": "loaded",
    "gls_generator": "loaded"
  }
}
```

## 🐛 Troubleshooting

### API não responde
```bash
# Verificar logs
docker-compose logs psiqrh-api

# Restart apenas API
docker-compose restart psiqrh-api
```

### Frontend carrega mas não conecta
```bash
# Verificar proxy nginx
docker-compose exec psiqrh-frontend cat /etc/nginx/conf.d/default.conf

# Restart frontend
docker-compose restart psiqrh-frontend
```

### Erro de permissão
```bash
# No diretório do projeto
sudo chown -R $USER:$USER .
```

### Rebuild completo
```bash
./restart-services.sh --rebuild --clean
```

## 📝 Logs

```bash
# Ver logs em tempo real
docker-compose logs -f psiqrh-api psiqrh-frontend

# Logs específicos
docker-compose logs psiqrh-api | grep ERROR
```

## 🔍 Verificação de Saúde

```bash
# Testar API diretamente
curl http://localhost:5000/health

# Testar chat
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "teste"}'

# Verificar frontend
curl http://localhost:3000
```

## 🎯 Próximos Passos

1. Acesse http://localhost:3000
2. Teste o chat interativo
3. Observe visualização GLS em tempo real
4. Explore os dados da API no painel direito

Para desenvolvimento avançado, consulte:
- `SERVICES_COMPLETE.md` - Detalhes de todos os serviços
- `README.Docker.md` - Documentação completa Docker
