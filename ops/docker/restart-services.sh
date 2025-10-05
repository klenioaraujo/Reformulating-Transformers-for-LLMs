#!/bin/bash
# ΨQRH Docker Services - Restart Script
# Reinicia os serviços Docker com logs em tempo real

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔄 ΨQRH Docker Services - Restart"
echo "=================================="
echo ""

# Função para exibir status colorido
status() {
    echo -e "\033[1;32m✓\033[0m $1"
}

error() {
    echo -e "\033[1;31m✗\033[0m $1"
}

info() {
    echo -e "\033[1;34mℹ\033[0m $1"
}

# Verificar se docker e docker-compose estão instalados
if ! command -v docker &> /dev/null; then
    error "Docker não está instalado!"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    error "Docker Compose não está instalado!"
    exit 1
fi

# Parar containers existentes
info "Parando containers existentes..."
docker-compose down 2>/dev/null || true
status "Containers parados"

# Limpar volumes órfãos (opcional)
if [ "$1" == "--clean" ]; then
    info "Limpando volumes órfãos..."
    docker volume prune -f
    status "Volumes limpos"
fi

# Rebuild se solicitado
if [ "$1" == "--rebuild" ] || [ "$2" == "--rebuild" ]; then
    info "Reconstruindo imagens Docker..."
    docker-compose build --no-cache
    status "Imagens reconstruídas"
fi

# Iniciar serviços
info "Iniciando serviços ΨQRH..."
docker-compose up -d psiqrh-api psiqrh-frontend

# Aguardar inicialização
info "Aguardando inicialização dos serviços..."
sleep 3

# Verificar status
echo ""
echo "📊 Status dos Serviços:"
echo "======================="
docker-compose ps

# Verificar saúde da API
echo ""
info "Verificando saúde da API..."
sleep 2

API_HEALTH=$(curl -s http://localhost:5000/api/health 2>/dev/null || echo '{"status":"error"}')
if echo "$API_HEALTH" | grep -q '"status":"healthy"'; then
    status "API está saudável! 🚀"
else
    error "API não respondeu corretamente"
    echo "Response: $API_HEALTH"
fi

# Informações de acesso
echo ""
echo "🌐 Serviços Disponíveis:"
echo "========================"
echo "  • Frontend:  http://localhost:3000"
echo "  • API:       http://localhost:5000"
echo "  • Health:    http://localhost:5000/health"
echo "  • API Info:  http://localhost:5000/api"
echo ""

# Mostrar logs se solicitado
if [ "$1" == "--logs" ] || [ "$2" == "--logs" ] || [ "$3" == "--logs" ]; then
    info "Exibindo logs (Ctrl+C para sair)..."
    docker-compose logs -f psiqrh-api psiqrh-frontend
else
    echo "💡 Para ver logs em tempo real: ./restart-services.sh --logs"
    echo "💡 Para rebuild completo: ./restart-services.sh --rebuild"
    echo "💡 Para limpar volumes: ./restart-services.sh --clean"
fi

echo ""
status "Serviços ΨQRH iniciados com sucesso!"
