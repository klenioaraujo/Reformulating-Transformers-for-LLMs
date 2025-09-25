# Dockerfile para ΨQRH Reformulating Transformers
# Baseado em Python 3.11 - versão simplificada para máxima compatibilidade

FROM python:3.11-slim

# Metadados
LABEL maintainer="Klenio Araujo Padilha <klenioaraujo@gmail.com>"
LABEL description="ΨQRH Quaternionic Transformer Framework - Containerized Research Environment"
LABEL version="1.0"

# Criar diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências primeiro (para cache do Docker)
COPY requirements.txt .

# Instalar dependências Python (sem cache para economizar espaço)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY . .

# Criar diretórios necessários
RUN mkdir -p images logs reports __pycache__ && \
    chmod -R 755 .

# Variáveis de ambiente
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch

# Expor porta para visualizações web (se necessário)
EXPOSE 8080

# Script de entrada padrão - mostra versão do Python e framework
CMD python -c "import sys; print('Python version:', sys.version); print('ΨQRH Framework ready!'); import torch; print('PyTorch version:', torch.__version__)"

# Labels adicionais para documentação
LABEL project="Reformulating Transformers for LLMs"
LABEL framework="ΨQRH Quaternionic-Harmonic Framework"
LABEL license="GNU GPLv3"
LABEL repository="https://github.com/your-repo/reformulating-transformers"