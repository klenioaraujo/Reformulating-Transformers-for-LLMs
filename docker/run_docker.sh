#!/bin/bash

# ΨQRH Docker Runner Script
# Convenience script for running ΨQRH framework in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Display help
show_help() {
    echo "ΨQRH Docker Runner Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build           Build the ΨQRH Docker image"
    echo "  up              Start all services with docker-compose"
    echo "  test            Run all validation tests"
    echo "  demo            Run demonstrations"
    echo "  shell           Open interactive shell in container"
    echo "  validation      Run specific validation script"
    echo "  fractal         Run fractal analysis"
    echo "  spider          Run spider evolution simulation"
    echo "  clean           Clean containers and volumes"
    echo "  logs            View container logs"
    echo "  status          Show container status"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build Docker image"
    echo "  $0 test                     # Run all tests"
    echo "  $0 validation simple        # Run simple validation"
    echo "  $0 shell                    # Open bash shell"
    echo "  $0 clean                    # Clean everything"
    echo ""
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Check if docker-compose is available
check_compose() {
    if ! command -v docker-compose > /dev/null 2>&1; then
        print_error "docker-compose not found. Please install docker-compose."
        exit 1
    fi
}

# Build Docker image
build_image() {
    print_info "Building ΨQRH Docker image..."
    docker build -t psiqrh-transformer .
    print_success "Image built successfully!"
}

# Start services
start_services() {
    print_info "Starting ΨQRH services..."
    docker-compose up --build -d
    print_success "Services started! Access with: docker-compose exec psiqrh bash"
}

# Run tests
run_tests() {
    print_info "Running ΨQRH validation tests..."
    docker-compose run --rm psiqrh bash -c "
        echo '=== ΨQRH Framework Test Suite ===' &&
        python simple_validation_test.py &&
        echo '--- Comprehensive Integration Test ---' &&
        python comprehensive_integration_test.py &&
        echo '--- Robust Validation Test ---' &&
        python robust_validation_test.py &&
        echo '=== All tests completed ==='
    "
    if [ $? -eq 0 ]; then
        print_success "All tests passed!"
    else
        print_error "Some tests failed!"
        exit 1
    fi
}

# Run demonstrations
run_demos() {
    print_info "Running ΨQRH demonstrations..."
    docker-compose run --rm psiqrh bash -c "
        echo '=== ΨQRH Framework Demonstrations ===' &&
        python fractal_pytorch_integration.py &&
        echo '--- Spider Evolution Simulation ---' &&
        python emergence_simulation.py &&
        echo '--- Habitat Demo ---' &&
        python show_habitat_demo.py &&
        echo '=== Demonstrations completed ==='
    "
    print_success "Demonstrations completed!"
}

# Open shell
open_shell() {
    print_info "Opening interactive shell..."
    docker-compose exec psiqrh bash
}

# Run specific validation
run_validation() {
    local validation_type=${1:-"simple"}
    case $validation_type in
        "simple")
            print_info "Running simple validation test..."
            docker-compose run --rm psiqrh python simple_validation_test.py
            ;;
        "comprehensive")
            print_info "Running comprehensive integration test..."
            docker-compose run --rm psiqrh python comprehensive_integration_test.py
            ;;
        "robust")
            print_info "Running robust validation test..."
            docker-compose run --rm psiqrh python robust_validation_test.py
            ;;
        *)
            print_error "Unknown validation type: $validation_type"
            print_info "Available types: simple, comprehensive, robust"
            exit 1
            ;;
    esac
}

# Run fractal analysis
run_fractal() {
    print_info "Running fractal dimension analysis..."
    docker-compose run --rm psiqrh python needle_fractal_dimension.py
    print_success "Fractal analysis completed! Check images/ directory"
}

# Run spider simulation
run_spider() {
    print_info "Running spider evolution simulation..."
    docker-compose run --rm psiqrh python emergence_simulation.py
    print_success "Spider simulation completed!"
}

# Clean containers and volumes
clean_all() {
    print_warning "This will remove all ΨQRH containers and volumes. Continue? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Cleaning containers and volumes..."
        docker-compose down -v
        docker system prune -f
        print_success "Cleanup completed!"
    else
        print_info "Cleanup cancelled."
    fi
}

# View logs
view_logs() {
    local service=${1:-"psiqrh"}
    print_info "Viewing logs for service: $service"
    docker-compose logs -f "$service"
}

# Show status
show_status() {
    print_info "Container status:"
    docker-compose ps
    echo ""
    print_info "Volume usage:"
    docker volume ls | grep psiqrh || echo "No ΨQRH volumes found"
}

# Main script logic
main() {
    check_docker
    check_compose

    case "${1:-help}" in
        "help" | "-h" | "--help")
            show_help
            ;;
        "build")
            build_image
            ;;
        "up")
            start_services
            ;;
        "test")
            run_tests
            ;;
        "demo")
            run_demos
            ;;
        "shell")
            open_shell
            ;;
        "validation")
            run_validation "$2"
            ;;
        "fractal")
            run_fractal
            ;;
        "spider")
            run_spider
            ;;
        "clean")
            clean_all
            ;;
        "logs")
            view_logs "$2"
            ;;
        "status")
            show_status
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"