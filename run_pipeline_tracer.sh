#!/bin/bash
# Pipeline Tracer Runner Script
# =============================
# Script para executar o tracer do pipeline ΨQRH com diferentes entradas

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LOG_DIR="debug_logs"
TRACER_SCRIPT="debug_pipeline_tracer.py"
TEST_SCRIPT="test_tracer_example.py"

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if scripts exist
check_scripts() {
    if [ ! -f "$TRACER_SCRIPT" ]; then
        print_error "Tracer script not found: $TRACER_SCRIPT"
        exit 1
    fi

    if [ ! -f "$TEST_SCRIPT" ]; then
        print_warning "Test script not found: $TEST_SCRIPT"
    fi

    print_success "All required scripts found"
}

# Run single trace
run_single_trace() {
    local text="$1"
    print_info "Running trace for: '$text'"
    python3 "$TRACER_SCRIPT" "$text" --output-dir "$LOG_DIR"
}

# Run test suite
run_test_suite() {
    print_info "Running test suite..."
    python3 "$TEST_SCRIPT"
}

# Run multiple traces
run_multiple_traces() {
    local texts=("a" "test" "hello world" "ΨQRH" "Qual a cor do céu?" "prove that √2 is irrational")

    print_info "Running multiple traces..."
    for text in "${texts[@]}"; do
        run_single_trace "$text"
        echo ""
    done
}

# Analyze logs
analyze_logs() {
    local latest_log=$(ls -t "$LOG_DIR"/pipeline_trace_*.jsonl 2>/dev/null | head -1)

    if [ -z "$latest_log" ]; then
        print_error "No log files found in $LOG_DIR"
        return 1
    fi

    print_info "Analyzing latest log: $(basename "$latest_log")"

    # Count steps
    local total_steps=$(wc -l < "$latest_log")
    local error_steps=$(grep -c '"error"' "$latest_log" || true)

    echo "📊 Log Analysis:"
    echo "   Total steps: $total_steps"
    echo "   Error steps: $error_steps"

    if [ "$error_steps" -gt 0 ]; then
        print_warning "Errors found in pipeline!"
        echo ""
        echo "🔍 Error details:"
        grep '"error"' "$latest_log" | jq -r '.step, .data.error' | paste - -
    else
        print_success "No errors found in pipeline"
    fi

    # Show pipeline flow
    echo ""
    echo "🔄 Pipeline Flow:"
    grep '"step"' "$latest_log" | jq -r '.step' | head -10
}

# Main menu
show_menu() {
    echo ""
    echo "🚀 ΨQRH Pipeline Tracer"
    echo "======================="
    echo "1. Run single trace (custom text)"
    echo "2. Run test suite"
    echo "3. Run multiple traces (predefined texts)"
    echo "4. Analyze latest logs"
    echo "5. View all log files"
    echo "6. Clean log directory"
    echo "7. Exit"
    echo ""
    read -p "Select option [1-7]: " choice

    case $choice in
        1)
            read -p "Enter text to trace: " text
            run_single_trace "$text"
            ;;
        2)
            run_test_suite
            ;;
        3)
            run_multiple_traces
            ;;
        4)
            analyze_logs
            ;;
        5)
            ls -la "$LOG_DIR"/*.jsonl 2>/dev/null || echo "No log files found"
            ;;
        6)
            rm -rf "$LOG_DIR"
            mkdir -p "$LOG_DIR"
            print_success "Log directory cleaned"
            ;;
        7)
            print_success "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            ;;
    esac
}

# Main execution
main() {
    print_info "ΨQRH Pipeline Tracer - Debug Tool"
    echo ""

    # Check requirements
    check_scripts

    # Create log directory
    mkdir -p "$LOG_DIR"

    # Interactive menu
    while true; do
        show_menu
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"