#!/bin/bash

# Build Docker image for ΨQRH reproduction
echo "Building ΨQRH Docker image..."

# Build the image
docker build -t psiqrh:latest .

echo "Docker image built successfully!"
echo ""
echo "To run the benchmark:"
echo "docker run --gpus all psiqrh:latest"
echo ""
echo "To run with custom parameters:"
echo "docker run --gpus all psiqrh:latest python benchmark.py --model_type baseline --epochs 5"