# ΨQRH — Makefile
# Run from project root: make build, make up, etc.

.PHONY: build up down test clean integrity

build:
	docker-compose -f ops/docker/docker-compose.yml build

up:
	docker-compose -f ops/docker/docker-compose.yml run --rm psiqrh

down:
	docker-compose -f ops/docker/docker-compose.yml down

test:
	docker-compose -f ops/docker/docker-compose.yml run --rm psiqrh-test

integrity:
	docker-compose -f ops/docker/docker-compose.yml run --rm psiqrh make -f ops/Makefile integrity-verify

clean:
	docker-compose -f ops/docker/docker-compose.yml down -v --rmi all
	docker builder prune -f