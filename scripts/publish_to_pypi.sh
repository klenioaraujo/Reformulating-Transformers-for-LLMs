#!/bin/bash
# Script to publish ΨQRH package to PyPI
#
# Usage:
#   ./scripts/publish_to_pypi.sh [test]
#
# If 'test' is provided, publishes to TestPyPI instead of production PyPI
#
# Prerequisites:
#   1. Create account at https://pypi.org (or https://test.pypi.org)
#   2. Generate API token at https://pypi.org/manage/account/token/
#   3. Configure token: ~/.pypirc or use environment variable
#
# Copyright (C) 2025 Klenio Araujo Padilha
# Licensed under GNU GPLv3 - see LICENSE file

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "ΨQRH PyPI Publication Script"
echo "=================================================="
echo ""

# Check if test mode
if [ "$1" == "test" ]; then
    echo -e "${YELLOW}Running in TEST mode${NC}"
    echo "Will publish to TestPyPI (test.pypi.org)"
    PYPI_URL="https://test.pypi.org/legacy/"
    MODE="test"
else
    echo -e "${GREEN}Running in PRODUCTION mode${NC}"
    echo "Will publish to PyPI (pypi.org)"
    PYPI_URL="https://upload.pypi.org/legacy/"
    MODE="production"
fi
echo ""

# Pre-flight checks
echo "Pre-flight checks:"
echo "-------------------"

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}✗ Not in virtual environment${NC}"
    echo "  Activate with: source .venv/bin/activate"
    exit 1
fi
echo -e "${GREEN}✓${NC} Virtual environment active"

# Check if build tools installed
if ! pip show build > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Installing build tools..."
    pip install build twine
fi
echo -e "${GREEN}✓${NC} Build tools installed"

# Check if twine installed
if ! pip show twine > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Installing twine..."
    pip install twine
fi
echo -e "${GREEN}✓${NC} Twine installed"

# Check for uncommitted changes
if [ -d ".git" ]; then
    if ! git diff-index --quiet HEAD --; then
        echo -e "${YELLOW}⚠${NC} Warning: You have uncommitted changes"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓${NC} No uncommitted changes"
    fi
fi

echo ""
echo "Building package:"
echo "-------------------"

# Clean old builds
if [ -d "dist" ]; then
    echo "Removing old dist/ directory..."
    rm -rf dist/
fi

if [ -d "build" ]; then
    echo "Removing old build/ directory..."
    rm -rf build/
fi

if [ -d "*.egg-info" ]; then
    echo "Removing old .egg-info directories..."
    rm -rf *.egg-info
fi

# Build package
echo "Building package..."
python -m build

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Package built successfully"

# Check package
echo ""
echo "Checking package:"
echo "-------------------"
twine check dist/*

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Package check failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Package checks passed"

# List files
echo ""
echo "Distribution files:"
echo "-------------------"
ls -lh dist/

echo ""
echo "=================================================="
if [ "$MODE" == "test" ]; then
    echo -e "${YELLOW}Ready to publish to TestPyPI${NC}"
else
    echo -e "${GREEN}Ready to publish to PyPI${NC}"
fi
echo "=================================================="
echo ""
read -p "Proceed with upload? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled"
    exit 0
fi

# Upload
echo ""
echo "Uploading package:"
echo "-------------------"

if [ "$MODE" == "test" ]; then
    twine upload --repository testpypi dist/*
else
    twine upload dist/*
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo -e "${GREEN}✓ Successfully published to $MODE PyPI!${NC}"
    echo "=================================================="
    echo ""
    if [ "$MODE" == "test" ]; then
        echo "Install with:"
        echo "  pip install --index-url https://test.pypi.org/simple/ psiqrh"
        echo ""
        echo "View at:"
        echo "  https://test.pypi.org/project/psiqrh/"
    else
        echo "Install with:"
        echo "  pip install psiqrh"
        echo ""
        echo "View at:"
        echo "  https://pypi.org/project/psiqrh/"
    fi
    echo ""
    echo "Next steps:"
    echo "1. Test the installation in a fresh environment"
    echo "2. Update README with installation instructions"
    echo "3. Create GitHub release with tag v1.0.0"
    echo "4. Update Zenodo record"
else
    echo ""
    echo -e "${RED}✗ Upload failed${NC}"
    echo ""
    echo "Common issues:"
    echo "1. Missing or invalid API token"
    echo "2. Package name already exists"
    echo "3. Version number already published"
    echo ""
    echo "Troubleshooting:"
    echo "- Configure token in ~/.pypirc"
    echo "- Check https://pypi.org/account/register/"
    exit 1
fi