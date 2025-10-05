# 🐳 ΨQRH Docker Development Environment

## 🎉 Status: Totalmente Operacional ✅

Todos os serviços iniciam automaticamente com `make dev-up`.

---

## 🚀 Quick Start

```bash
# 1. Iniciar todos os serviços
make dev-up

# 2. Aguardar ~10 segundos para inicialização

# 3. Testar serviços
curl http://localhost:5000/health
```

---

## 📍 Serviços Disponíveis

### 🌐 Flask API - http://localhost:5000
**Status:** ✅ Auto-start habilitado

```bash
# Health check
curl http://localhost:5000/health

# Chat com análise de consciência
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello ΨQRH"}'
```

**Endpoints:**
- `GET /` - API info
- `GET /health` - Health check
- `POST /chat` - Consciência fractal
- `GET /metrics` - Métricas FCI

---

### 📓 Jupyter Notebook - http://localhost:8888
**Status:** ✅ Auto-start habilitado
**Token:** `dev123`

**URL de acesso:** http://localhost:8888/tree?token=dev123

```bash
# Abrir no browser
open http://localhost:8888/tree?token=dev123
```

---

### 📊 PostgreSQL - localhost:5432
**Status:** ✅ Auto-start + Schema inicializado

**Credenciais:**
- Database: `psiqrh_dev`
- User: `dev` / Password: `dev123`
- User: `psiqrh` / Password: `psiqrh123`

```bash
# Conectar
psql -h localhost -U dev -d psiqrh_dev

# Query via container
docker exec psiqrh-dev-db psql -U dev -d psiqrh_dev -c "SELECT * FROM consciousness_logs;"
```

**Tabelas criadas automaticamente:**
- `consciousness_logs` - Logs de processamento
- `consciousness_metrics_summary` - Agregações diárias
- `consciousness_sessions` - Sessões de conversação

---

### 🔴 Redis - localhost:6379
**Status:** ✅ Auto-start habilitado

```bash
# Ping
redis-cli ping

# Via container
docker exec psiqrh-dev-redis redis-cli ping
```

---

## 🧪 Teste Completo

Execute o script de testes para verificar todos os serviços:

```bash
# Testes manuais rápidos
curl http://localhost:5000/health           # Flask
curl http://localhost:8888?token=dev123     # Jupyter
redis-cli ping                              # Redis
psql -h localhost -U dev -d psiqrh_dev -c "\dt"  # PostgreSQL
```

**Resultado esperado:**
```
✅ Flask API: PASSED
✅ Jupyter: PASSED  
✅ PostgreSQL: PASSED
✅ Redis: PASSED
✅ ΨQRH Factory: PASSED
```

---

## 🛠️ Comandos Principais

```bash
# Iniciar
make dev-up

# Parar
make dev-down

# Rebuild (após mudanças)
make dev-build

# Logs
docker logs -f psiqrh-dev
docker logs -f psiqrh-dev-db

# Shell
docker exec -it psiqrh-dev bash

# Status
docker ps
```

---

## 📦 Arquivos de Configuração

- **Dockerfile:** `ops/docker/Dockerfile.dev`
- **Compose:** `ops/docker/docker-compose.dev.yml`
- **Entrypoint:** `ops/docker/entrypoint.dev.sh` (auto-start de serviços)
- **Init SQL:** `ops/docker/init-postgres.sql` (schema do banco)

---

## ✅ O que Inicia Automaticamente

Quando você executa `make dev-up`:

1. ✅ PostgreSQL com schema pré-criado
2. ✅ Redis
3. ✅ Aguarda dependências (DB + Redis)
4. ✅ Jupyter Notebook (porta 8888, token: dev123)
5. ✅ Flask API (porta 5000)
6. ✅ ΨQRH Factory + módulos de consciência

**Tempo de inicialização:** ~10 segundos

---

## 🐛 Troubleshooting

### Serviços não iniciam
```bash
docker logs psiqrh-dev
make dev-down && make dev-up
```

### PostgreSQL sem tabelas
```bash
docker exec psiqrh-dev-db psql -U dev -d psiqrh_dev -f /docker-entrypoint-initdb.d/init.sql
```

### Porta em uso
```bash
sudo lsof -i :5000  # Identificar processo
sudo kill -9 <PID>  # Matar processo
```

### Rebuild completo
```bash
make dev-down
docker volume rm psiqrh-dev-db-data
make dev-build
make dev-up
```

---

## 📊 Estrutura dos Serviços

```
┌─────────────────────────────────────────┐
│  psiqrh-dev (Main Container)            │
│  ├─ Jupyter Notebook :8888              │
│  ├─ Flask API        :5000              │
│  └─ ΨQRH Factory + Transformer          │
└─────────────────────────────────────────┘
           │              │
           ▼              ▼
    ┌──────────┐   ┌──────────┐
    │PostgreSQL│   │  Redis   │
    │  :5432   │   │  :6379   │
    └──────────┘   └──────────┘
```

---

## 🔐 Segurança

⚠️ **DESENVOLVIMENTO APENAS** - Não usar em produção sem:
1. Mudar senhas e tokens
2. Remover `debug=True`
3. Configurar HTTPS
4. Adicionar autenticação
5. Usar variáveis de ambiente

---

**Última atualização:** 2025-09-30  
**Versão ΨQRH:** 1.0.0  
**License:** GNU GPLv3
