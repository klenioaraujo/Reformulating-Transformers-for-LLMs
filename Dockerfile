FROM pytorch/pytorch:2.3-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy source code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Default command runs the benchmark
CMD ["python", "benchmark.py", "--model_type", "psiqrh", "--dataset", "wikitext-103", "--seq_len", "512", "--batch_size", "4", "--epochs", "3"]