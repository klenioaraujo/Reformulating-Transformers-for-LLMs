FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

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

# Default command runs the psiqrh script
CMD ["python", "psiqrh.py", "what color is the sky"]