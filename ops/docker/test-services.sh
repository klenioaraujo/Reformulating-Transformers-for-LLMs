#!/bin/bash
# ΨQRH Services Integration Test
# Tests all services: Flask, Jupyter, PostgreSQL, Redis

set -e

echo "🧪 Testing ΨQRH Development Services..."
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Test Flask API
echo -n "Testing Flask API (port 5000)... "
if curl -s http://localhost:5000/health | grep -q "healthy"; then
    echo -e "${GREEN}✅ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAILED${NC}"
    ((FAILED++))
fi

# Test Flask chat endpoint
echo -n "Testing Flask chat endpoint... "
RESPONSE=$(curl -s -X POST http://localhost:5000/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"test"}')
if echo "$RESPONSE" | grep -q "fci"; then
    echo -e "${GREEN}✅ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAILED${NC}"
    ((FAILED++))
fi

# Test Jupyter
echo -n "Testing Jupyter Notebook (port 8888)... "
if curl -s http://localhost:8888/tree?token=dev123 | grep -q "Jupyter"; then
    echo -e "${GREEN}✅ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAILED${NC}"
    ((FAILED++))
fi

# Test PostgreSQL
echo -n "Testing PostgreSQL (port 5432)... "
if docker exec psiqrh-dev-db psql -U dev -d psiqrh_dev -c "\dt" 2>/dev/null | grep -q "consciousness_logs"; then
    echo -e "${GREEN}✅ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAILED${NC}"
    ((FAILED++))
fi

# Test PostgreSQL data
echo -n "Testing PostgreSQL data integrity... "
COUNT=$(docker exec psiqrh-dev-db psql -U dev -d psiqrh_dev -t -c "SELECT COUNT(*) FROM consciousness_logs;" 2>/dev/null | tr -d ' ')
if [ "$COUNT" -ge "2" ]; then
    echo -e "${GREEN}✅ PASSED${NC} ($COUNT rows)"
    ((PASSED++))
else
    echo -e "${RED}❌ FAILED${NC} (expected >= 2 rows, got $COUNT)"
    ((FAILED++))
fi

# Test Redis
echo -n "Testing Redis (port 6379)... "
if docker exec psiqrh-dev-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
    echo -e "${GREEN}✅ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAILED${NC}"
    ((FAILED++))
fi

# Test ΨQRH Factory
echo -n "Testing ΨQRH Factory initialization... "
if docker exec psiqrh-dev python -c "from src.core.ΨQRH import QRHFactory; print('OK')" 2>/dev/null | grep -q "OK"; then
    echo -e "${GREEN}✅ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAILED${NC}"
    ((FAILED++))
fi

# Test ΨQRH Transformer import
echo -n "Testing ΨQRH Transformer import... "
if docker exec psiqrh-dev python -c "from src.architecture.psiqrh_transformer import PsiQRHTransformer; print('OK')" 2>/dev/null | grep -q "OK"; then
    echo -e "${GREEN}✅ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAILED${NC}"
    ((FAILED++))
fi

# Summary
echo ""
echo "========================================"
echo "Test Summary:"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo "  Total:  $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    echo ""
    echo "📍 Services Available:"
    echo "  • Flask API:        http://localhost:5000"
    echo "  • Jupyter Notebook: http://localhost:8888/tree?token=dev123"
    echo "  • PostgreSQL:       localhost:5432 (user: dev, db: psiqrh_dev)"
    echo "  • Redis:            localhost:6379"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Check logs with: docker logs psiqrh-dev${NC}"
    exit 1
fi