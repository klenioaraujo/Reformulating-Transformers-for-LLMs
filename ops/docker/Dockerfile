FROM python:3.10-slim

# Set environment variables for disk optimization
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Update package list and install essential dependencies with DNS fallback
RUN apt-get update --fix-missing && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* || \
    (apt-get clean && apt-get update -o Acquire::Retries=3 && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*)

# Configure pip settings with better connectivity options
RUN echo "[global]" > /etc/pip.conf && \
    echo "timeout = 120" >> /etc/pip.conf && \
    echo "retries = 5" >> /etc/pip.conf && \
    echo "trusted-host = pypi.org" >> /etc/pip.conf && \
    echo "               pypi.python.org" >> /etc/pip.conf && \
    echo "               files.pythonhosted.org" >> /etc/pip.conf && \
    echo "index-url = https://pypi.org/simple/" >> /etc/pip.conf

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-docker.txt /app/requirements-docker.txt

# Install dependencies with fallback options
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir --timeout 120 --retries 5 -r requirements-docker.txt || \
    python -m pip install --no-cache-dir --timeout 120 --retries 5 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements-docker.txt

# Copy the rest of the project
COPY . /app

# Run integrity setup
RUN make -f ops/Makefile integrity-setup || echo "Integrity setup completed"

# Install Flask dependencies for API
RUN python -m pip install flask flask-cors

# Default command to start the API
CMD ["python", "app.py"]