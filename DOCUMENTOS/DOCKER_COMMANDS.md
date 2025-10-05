# 🐳 Docker Commands Guide - ΨQRH

Guia rápido de comandos Docker para gerenciar os ambientes de produção e desenvolvimento.

## 🚀 Quick Start Commands

### Gerenciamento Básico

```bash
# Parar TODOS os serviços (produção + desenvolvimento)
make stop-all

# Reiniciar apenas DESENVOLVIMENTO (para quando conflitos de porta)
make restart-dev

# Reiniciar apenas PRODUÇÃO
make restart-prod

# Ver status dos containers
make status
docker ps
```

## 📋 Comandos Disponíveis

### 🏭 Ambiente de Produção

```bash
# Iniciar produção
make start                  # Build e start
make docker-up              # Start sem build

# Parar produção
make stop                   # Para produção
make docker-down            # Alias para stop

# Restart produção
make restart                # Restart rápido (sem rebuild)
make restart-full           # Rebuild + restart
make restart-prod           # Para TUDO e inicia só produção

# Build
make docker-build           # Build imagens de produção

# Logs e shell
make docker-logs            # Ver logs em tempo real
make docker-shell           # Shell no container da API
```

**Portas de Produção:**
- Frontend: http://localhost:8080
- API: http://localhost:5000

---

### 🔬 Ambiente de Desenvolvimento

```bash
# Iniciar desenvolvimento
make dev-up                 # Start ambiente dev
make dev-build              # Build ambiente dev

# Parar desenvolvimento
make dev-down               # Para desenvolvimento

# Restart desenvolvimento
make dev-restart            # Restart dev (sem parar prod)
make restart-dev            # Para TUDO e inicia só dev

# Shell e ferramentas
make dev-shell              # Shell no container dev
make dev-jupyter            # Iniciar Jupyter notebook
make dev-api                # Rodar API no container dev
make dev-test               # Rodar testes

# Limpeza
make dev-clean              # Limpar ambiente dev completamente
```

**Portas de Desenvolvimento:**
- Frontend: http://localhost:3000 e http://localhost:8081
- API: http://localhost:5000 e http://localhost:8080
- Jupyter: http://localhost:8888 (token: `dev123`)
- PostgreSQL: localhost:5432
- Redis: localhost:6379

---

## ⚠️ Resolução de Conflitos de Porta

### Problema: "Bind for 0.0.0.0:5000 failed: port is already allocated"

Isso ocorre quando produção e desenvolvimento tentam usar a mesma porta.

**Solução:**

```bash
# Opção 1: Usar apenas desenvolvimento
make restart-dev

# Opção 2: Usar apenas produção
make restart-prod

# Opção 3: Parar tudo e escolher qual iniciar
make stop-all
# Depois:
make dev-up      # Para desenvolvimento
# OU
make docker-up   # Para produção
```

---

## 🔍 Comandos de Diagnóstico

```bash
# Ver containers rodando
docker ps

# Ver todos os containers (incluindo parados)
docker ps -a

# Ver logs de um container específico
docker logs psiqrh-dev
docker logs psiqrh-api
docker logs psiqrh-dev-frontend

# Ver logs em tempo real
docker logs -f psiqrh-dev

# Inspecionar container
docker inspect psiqrh-dev

# Ver uso de recursos
docker stats
```

---

## 🧹 Limpeza e Manutenção

```bash
# Limpar ambiente específico
make clean           # Limpar produção (remove volumes e imagens)
make dev-clean       # Limpar desenvolvimento

# Limpar tudo (CUIDADO!)
make stop-all
docker system prune -a --volumes

# Remover apenas volumes órfãos
docker volume prune

# Remover imagens não utilizadas
docker image prune -a
```

---

## 📊 Status e Monitoramento

```bash
# Status dos serviços
make status

# Ver portas em uso
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Ver consumo de recursos
docker stats --no-stream

# Ver redes
docker network ls

# Ver volumes
docker volume ls
```

---

## 🐛 Debug e Troubleshooting

### Entrar no container

```bash
# Desenvolvimento
make dev-shell
# OU
docker exec -it psiqrh-dev /bin/bash

# Produção
make docker-shell
# OU
docker exec -it psiqrh-api /bin/bash
```

### Executar comando específico

```bash
# No container de desenvolvimento
docker exec psiqrh-dev python3 psiqrh.py --interactive

# No container de produção
docker exec psiqrh-api python3 -m pytest tests/

# Ver variáveis de ambiente
docker exec psiqrh-dev env
```

### Rebuild forçado

```bash
# Rebuild sem cache
docker-compose -f ops/docker/docker-compose.dev.yml build --no-cache

# Rebuild e restart
make restart-full
```

---

## 🔄 Workflows Comuns

### 1. Desenvolvimento Normal

```bash
# 1. Iniciar ambiente dev
make restart-dev

# 2. Trabalhar no código...

# 3. Testar mudanças
make dev-test

# 4. Ver logs
make docker-logs

# 5. Quando terminar
make stop-all
```

### 2. Testar em Produção

```bash
# 1. Parar dev e iniciar prod
make restart-prod

# 2. Testar...

# 3. Ver logs
docker logs -f psiqrh-api

# 4. Voltar para dev
make restart-dev
```

### 3. Atualizar Dependências

```bash
# 1. Atualizar requirements.txt ou Dockerfile

# 2. Rebuild completo
make stop-all
make dev-build
make dev-up

# 3. Verificar
make dev-shell
pip list
```

### 4. Resetar Completamente

```bash
# Parar tudo
make stop-all

# Limpar tudo
make clean
make dev-clean

# Rebuild do zero
make dev-build
make restart-dev
```

---

## 🎯 Comandos Mais Usados (Cheat Sheet)

```bash
make restart-dev       # Reiniciar desenvolvimento
make stop-all          # Parar tudo
make dev-shell         # Shell no container
docker logs -f psiqrh-dev  # Ver logs
docker ps              # Ver containers rodando
make status            # Status dos serviços
```

---

## 📚 Referências

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Makefile Reference](./Makefile)
- Configurações:
  - Produção: `ops/docker/docker-compose.yml`
  - Desenvolvimento: `ops/docker/docker-compose.dev.yml`
  - Dockerfile prod: `ops/docker/Dockerfile`
  - Dockerfile dev: `ops/docker/Dockerfile.dev`

---

**Última atualização:** 2025-09-30
**Ambiente de desenvolvimento configurado com sucesso!** ✅