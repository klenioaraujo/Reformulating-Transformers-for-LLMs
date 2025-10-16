#!/bin/bash

# Instalar dependências
pip install --break-system-packages transformers torch datasets

# Configurar variáveis de ambiente
export PYTHONPATH=$PYTHONPATH:$(pwd)/GLUE-baselines:$(pwd)/reformulated_transformers

# Baixar dados GLUE (se disponível)
echo "GLUE data download would go here - check GLUE-baselines for download instructions"