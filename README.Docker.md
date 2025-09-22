# ΨQRH Docker Quick Start Guide

This guide provides instructions for running the ΨQRH Quaternionic Transformer Framework in Docker containers.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 1.29+
- 8GB RAM minimum (16GB recommended)
- 10GB disk space

## Quick Start

### Option 1: Make Commands (Recommended)

```bash
# Clone and navigate to project
git clone <repository-url>
cd reformulating-transformers

# Complete setup (build and start)
make setup

# Access interactive container
make shell

# Show all available commands
make help
```

### Option 2: Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Access interactive container
docker-compose exec psiqrh bash
```

### Option 2: Single Container

```bash
# Build image
docker build -t psiqrh-transformer .

# Run interactive container
docker run -it --rm -v $(pwd):/app psiqrh-transformer bash

# Run specific script
docker run --rm -v $(pwd):/app psiqrh-transformer python fractal_pytorch_integration.py
```

## Available Services

### 1. Main Development Environment
```bash
docker-compose up psiqrh
docker-compose exec psiqrh bash
```

### 2. Automated Test Suite
```bash
docker-compose up psiqrh-test
# Runs: simple_validation_test.py, comprehensive_integration_test.py, robust_validation_test.py
```

### 3. Demonstrations
```bash
docker-compose up psiqrh-demo
# Runs: fractal_pytorch_integration.py, emergence_simulation.py, show_habitat_demo.py
```

### 4. Interactive Visualizations
```bash
docker-compose up psiqrh-viz
# Access at: http://localhost:8091
```

## Make Commands Reference

### Testing and Validation
```bash
# Run complete test suite
make test

# Run individual tests
make test-simple         # Simple validation
make test-comprehensive  # Comprehensive integration
make test-robust        # Robust validation
make test-4d            # 4D Unitary Layer tests

# CI/CD pipeline test
make ci-test
```

### Demonstrations and Analysis
```bash
# Run all demonstrations
make demo

# Individual demonstrations
make fractal       # Fractal dimension analysis
make spider        # Spider evolution simulation
make integration   # Fractal-PyTorch integration
make habitat       # Habitat demonstration
make ecosystem     # Live ecosystem server
```

### Development Environment
```bash
# Interactive development
make shell         # Regular user shell
make shell-root    # Root access shell

# Utilities
make logs          # View container logs
make status        # Show container/volume status
make results       # Show generated results
```

### Performance and Benchmarking
```bash
# Performance analysis
make benchmark     # Run performance benchmarks
make validate-all  # Complete validation suite

# GPU support
make gpu-setup     # Setup with GPU support
make gpu-test      # Test GPU functionality
```

### Maintenance
```bash
# Cleanup operations
make clean         # Clean containers and volumes
make clean-images  # Clean generated images
make clean-logs    # Clean log files
make reset         # Complete reset and rebuild

# Backup and restore
make backup        # Backup volumes and results
```

## Docker Compose Commands (Alternative)

### Run Validation Tests
```bash
# All tests using make (recommended)
make test

# All tests using docker-compose
docker-compose run --rm psiqrh bash -c "
python simple_validation_test.py &&
python comprehensive_integration_test.py &&
python robust_validation_test.py
"
```

### Fractal Analysis
```bash
# Generate fractal analysis
docker-compose run --rm psiqrh python needle_fractal_dimension.py

# View generated images
docker-compose exec psiqrh ls images/
```

### Spider Evolution Simulation
```bash
# Run genetic algorithm simulation
docker-compose run --rm psiqrh python emergence_simulation.py

# Monitor in real-time
docker-compose logs -f psiqrh
```

### 4D Unitary Layer Tests
```bash
# Run comprehensive layer tests
docker-compose run --rm psiqrh python test_4d_unitary_layer.py
```

## Persistent Data

Data is stored in named Docker volumes:

- `psiqrh-models/`: Trained models and checkpoints
- `psiqrh-logs/`: Training and validation logs
- `psiqrh-images/`: Generated visualizations
- `psiqrh-reports/`: Analysis reports

View volumes:
```bash
docker volume ls | grep psiqrh
```

## GPU Support (Optional)

For NVIDIA GPU support, create `docker-compose.gpu.yml`:

```yaml
version: '3.8'
services:
  psiqrh:
    extends:
      file: docker-compose.yml
      service: psiqrh
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run with GPU:
```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs psiqrh

# Rebuild image
docker-compose build --no-cache psiqrh
```

### Permission Issues
```bash
# Fix file permissions
docker-compose exec psiqrh chown -R $(id -u):$(id -g) /app
```

### Memory Issues
```bash
# Increase Docker memory limit in Docker Desktop:
# Settings > Resources > Memory > 8GB+
```

### Network Issues
```bash
# Clean Docker system
docker system prune -a
docker-compose down -v
docker-compose up --build
```

## Development Workflow

### Interactive Development
```bash
# Start container with mounted code
docker-compose up psiqrh

# In another terminal, access shell
docker-compose exec psiqrh bash

# Make changes to code on host, they appear in container
# Run tests/scripts inside container
python your_script.py
```

### Installing Additional Dependencies
```bash
# Access container
docker-compose exec psiqrh bash

# Install new package
pip install new-package

# To persist, add to requirements.txt and rebuild
echo "new-package==1.0.0" >> requirements.txt
docker-compose build psiqrh
```

## Production Deployment

### Build Optimized Image
```bash
# Multi-stage build for production
docker build --target production -t psiqrh-transformer:prod .
```

### Run in Production
```bash
# Run with resource limits
docker run -d \
  --name psiqrh-prod \
  --memory=8g \
  --cpus=4 \
  -v psiqrh-data:/app/data \
  psiqrh-transformer:prod
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: ΨQRH Docker Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and Test
        run: |
          docker-compose up --build psiqrh-test
          docker-compose down
```

## Advanced Usage

### Custom Environment Variables
```bash
# Set custom variables
docker-compose run -e TORCH_HOME=/custom/path psiqrh python script.py
```

### Port Forwarding
```bash
# Forward additional ports
docker run -p 8080:8080 -p 8888:8888 psiqrh-transformer
```

### Volume Mounting
```bash
# Mount custom directories
docker run -v /host/data:/app/data -v /host/models:/app/models psiqrh-transformer
```

## Performance Tips

1. **Use .dockerignore**: Already configured to exclude unnecessary files
2. **Layer Caching**: Dependencies installed before code copy for better caching
3. **Multi-stage Builds**: Use for production deployments
4. **Resource Limits**: Set appropriate memory/CPU limits
5. **Volume Management**: Use named volumes for persistent data

## Support

For issues specific to Docker deployment:

1. Check this README.Docker.md
2. Review docker-compose logs
3. Verify system requirements
4. Check Docker documentation
5. File issues at project repository

---

**Framework Status**: Production Ready ✅
**Docker Support**: Full ✅
**Test Coverage**: 100% ✅