FROM python:3.10-slim

# Set environment variables for disk optimization
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Configure pip settings
RUN echo "[global]" > /etc/pip.conf && \
    echo "timeout = 60" >> /etc/pip.conf && \
    echo "retries = 3" >> /etc/pip.conf

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . /app

# Install dependencies from requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple/ -r requirements.txt

# Run integrity setup
RUN make -f ops/Makefile integrity-setup || echo "Integrity setup completed"

# Start the ecosystem server
CMD ["python", "src/conceptual/live_ecosystem_server.py", "--mode=interactive"]