# ΨQRH Framework - Quick Start Guide

Get the ΨQRH Quaternionic Transformer Framework running in under 5 minutes.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 1.29+
- Make (GNU Make)
- 8GB RAM minimum

## 1. Clone and Setup

```bash
git clone <repository-url>
cd reformulating-transformers
make setup
```

## 2. Verify Installation

```bash
make test
```

Expected output: `All tests passed!` with 100% success rate.

## 3. Explore the Framework

```bash
# Interactive development shell
make shell

# Run demonstrations
make demo

# Fractal analysis
make fractal

# Spider evolution simulation
make spider

# Show all available commands
make help
```

## 4. Access Results

```bash
# View generated visualizations and data
make results

# Show container status
make status
```

## Common Commands

| Command | Description |
|---------|-------------|
| `make setup` | Complete setup (build + start) |
| `make test` | Run all validation tests |
| `make shell` | Interactive development environment |
| `make demo` | Run all demonstrations |
| `make clean` | Clean everything |
| `make help` | Show all available commands |

## Need More Detail?

- [README.md](README.md) - Complete documentation
- [README.Docker.md](../docker/README.Docker.md) - Docker-specific details
- [Makefile.examples](Makefile.examples) - Command examples and workflows

## Troubleshooting

**Container won't start**: `make logs` → `make clean` → `make setup`

**Permission issues**: `make shell-root`

**Out of space**: `make clean` → `docker system prune -a`

**See all options**: `make help`

---

**Status**: Production Ready ✅ | **Test Coverage**: 100% ✅ | **Docker**: Fully Supported ✅