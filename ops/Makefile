# ΨQRH Makefile
# Local: ops/Makefile
# Executar com: make -f ops/Makefile <target>

.PHONY: docker-build docker-run docker-clean integrity-setup integrity-verify

# Docker image name
IMAGE_NAME = psiqrh-core

# Docker context is the project root
docker-build:
	docker build -t $(IMAGE_NAME) -f ops/docker/Dockerfile .

docker-run:
	docker run -it --rm -p 8000:8000 $(IMAGE_NAME)

docker-clean:
	docker rmi -f $(IMAGE_NAME) 2>/dev/null || true
	docker builder prune -f

# Integrity commands (run inside container or after build)
integrity-setup:
	@echo "⚠️  Run integrity-setup inside Docker or after installing dependencies"

integrity-verify:
	@echo "⚠️  Run integrity-verify inside Docker or after installing dependencies"

# Alias for compatibility
build: docker-build
