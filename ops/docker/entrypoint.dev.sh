#!/bin/bash
# ΨQRH Development Environment Entrypoint
# Initializes Jupyter, Flask API on container startup

set -e

echo "🚀 ΨQRH Development Environment Starting..."
echo "=============================================="

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL..."
until PGPASSWORD=dev123 psql -h psiqrh-dev-db -U dev -d psiqrh_dev -c '\q' 2>/dev/null; do
    echo "PostgreSQL not ready, waiting..."
    sleep 2
done
echo "✅ PostgreSQL is ready"

# Wait for Redis
echo "⏳ Waiting for Redis..."
until redis-cli -h psiqrh-dev-redis ping 2>/dev/null | grep -q PONG; do
    echo "Redis not ready, waiting..."
    sleep 1
done
echo "✅ Redis is ready"

# Start Jupyter Notebook
echo "📓 Starting Jupyter Notebook on port 8888..."
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser \
    --allow-root --NotebookApp.token='dev123' \
    --NotebookApp.password='' \
    --notebook-dir=/app &

JUPYTER_PID=$!
echo "✅ Jupyter started (PID: $JUPYTER_PID)"

# Wait for Jupyter to start
sleep 3

# Start Flask API
echo "🌐 Starting Flask API on port 5000..."
cd /app
python app.py &

FLASK_PID=$!
echo "✅ Flask API started (PID: $FLASK_PID)"

echo ""
echo "=============================================="
echo "🎉 ΨQRH Development Environment Ready!"
echo "=============================================="
echo ""
echo "📍 Available Services:"
echo "  • Jupyter Notebook: http://localhost:8888 (token: dev123)"
echo "  • Flask API:        http://localhost:5000"
echo "  • PostgreSQL:       localhost:5432 (user: dev, db: psiqrh_dev)"
echo "  • Redis:            localhost:6379"
echo ""
echo "💡 Quick Commands:"
echo "  • Test API:    curl http://localhost:5000/health"
echo "  • Enter shell: docker exec -it psiqrh-dev bash"
echo "  • View logs:   docker logs -f psiqrh-dev"
echo ""
echo "=============================================="

# Keep container running and monitor services
trap 'echo "Shutting down..."; kill $JUPYTER_PID $FLASK_PID 2>/dev/null; exit 0' SIGTERM SIGINT

# Monitor processes
while kill -0 $JUPYTER_PID 2>/dev/null && kill -0 $FLASK_PID 2>/dev/null; do
    sleep 10
done

echo "⚠️  Service stopped unexpectedly. Check logs."
exit 1