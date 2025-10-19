#!/bin/bash
# Script de Inicialização Rápida ΨQRH
# ===================================

echo "🚀 Iniciando Sistema ΨQRH..."

# Verificar se ambiente virtual existe
if [ ! -d "psiqrh_env" ]; then
    echo "⚠️ Ambiente virtual não encontrado. Execute setup_system.py primeiro."
    exit 1
fi

# Ativar ambiente virtual
source psiqrh_env/bin/activate

# Verificar instalação
python -c "from psiqrh import ΨQRHPipeline; print('✅ ΨQRH pronto!')"

echo ""
echo "🎯 Comandos disponíveis:"
echo "  make test              # Teste completo"
echo "  make train-physics-emergent  # Treinamento emergente"
echo "  python psiqrh.py --interactive  # Modo interativo"
echo "  python psiqrh.py "seu texto"     # Processar texto"
echo ""
echo "📚 Para mais opções: python psiqrh.py --help"
