# ΨQRH — Makefile
# Run from project root: make build, make up, etc.

.PHONY: build up down test clean integrity convert-pdf Ψcws-stats demo-pdf-Ψcws list-Ψcws test-native-reader convert-wiki-topic list-wiki-topics convert-all-wiki-topics

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

# ΨQRH PDF to ΨCWS Conversion Commands
# Variables
ΨCWS_OUTPUT_DIR = data/Ψcws_cache

# Convert PDF to ΨCWS format: make convert-pdf PDF=path/to/file.pdf
convert-pdf:
	@if [ -z "$(PDF)" ]; then \
		echo "❌ Usage: make convert-pdf PDF=path/to/file.pdf"; \
		echo "📖 Example: make convert-pdf PDF=src/conceptual/models/d41d8cd98f00b204e9800998ecf8427e.pdf"; \
		exit 1; \
	fi
	@mkdir -p $(ΨCWS_OUTPUT_DIR)
	@echo "🔄 Converting $(PDF) to .Ψcws format using ΨQRH consciousness pipeline..."
	@python3 -c "\
import sys; \
sys.path.append('src'); \
from pathlib import Path; \
from conscience.conscious_wave_modulator import ConsciousWaveModulator; \
import hashlib; \
pdf_path = Path('$(PDF)'); \
config = {'cache_dir': '$(ΨCWS_OUTPUT_DIR)', 'embedding_dim': 256, 'sequence_length': 64, 'device': 'cpu'}; \
modulator = ConsciousWaveModulator(config); \
cwm_file = modulator.process_file(pdf_path); \
file_stat = pdf_path.stat(); \
hash_input = f'{pdf_path.absolute()}_{file_stat.st_mtime}'; \
file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]; \
output_path = Path('$(ΨCWS_OUTPUT_DIR)') / f'{file_hash}_{pdf_path.stem}.Ψcws'; \
cwm_file.save(output_path); \
print(f'✅ Generated: {output_path}'); \
print(f'🧠 Consciousness metrics: {cwm_file.spectral_data.consciousness_metrics}'); \
"

# Show CWM cache statistics and consciousness metrics
Ψcws-stats:
	@echo "📊 ΨQRH ΨCWS Cache Statistics:"
	@echo "Cache directory: $(ΨCWS_OUTPUT_DIR)"
	@echo "Number of .Ψcws files: $$(find $(ΨCWS_OUTPUT_DIR) -name '*.Ψcws' 2>/dev/null | wc -l)"
	@echo "Total cache size: $$(du -h $(ΨCWS_OUTPUT_DIR) 2>/dev/null | cut -f1 || echo '0B')"
	@find $(ΨCWS_OUTPUT_DIR) -name '*.Ψcws' -type f -exec ls -lht {} + 2>/dev/null | head -3 || echo "No .Ψcws files found"

# Demo: Convert test PDF and show stats
demo-pdf-Ψcws:
	@echo "🎬 ΨQRH PDF→ΨCWS Demo with d41d8cd98f00b204e9800998ecf8427e.pdf"
	@make convert-pdf PDF=src/conceptual/models/d41d8cd98f00b204e9800998ecf8427e.pdf
	@make Ψcws-stats

# List available .Ψcws files using native reader
list-Ψcws:
	@echo "📋 Listando arquivos .Ψcws disponíveis via leitura nativa:"
	@python3 -c "\
import sys; sys.path.append('src'); \
from conscience.psicws_native_reader import list_Ψcws_files; \
files = list_Ψcws_files(); \
print(f'Total: {len(files)} arquivos .Ψcws'); \
print('\\n📄 Arquivos encontrados:'); \
[print(f'  {i+1}. {f[\"original_name\"]} ({f[\"size_kb\"]} KB)\\n     Hash: {f[\"hash\"]}\\n     Modificado: {f[\"modified_time\"]}') for i, f in enumerate(files)]"

# Test native reader functionality
test-native-reader:
	@echo "🧪 Testando funcionalidade de leitura nativa .Ψcws:"
	@python3 test_native_reader.py

# Show consciousness analysis of .Ψcws cache
analyze-Ψcws-consciousness:
	@python3 analyze_consciousness.py

# List available Wikipedia topics for conversion
list-wiki-topics:
	@echo "📋 Available Wikipedia topics for conversion:"
	@python3 wiki_to_psicws_converter.py list

# Convert single Wikipedia topic to .Ψcws format
convert-wiki-topic:
	@if [ -z "$(TOPIC)" ]; then \
		echo "❌ Usage: make convert-wiki-topic TOPIC=consciousness"; \
		echo "📖 Available topics: run 'make list-wiki-topics'"; \
		exit 1; \
	fi
	@echo "🔄 Converting Wikipedia topic $(TOPIC) to .Ψcws format..."
	@python3 wiki_to_psicws_converter.py "$(TOPIC)"

# Convert all supported Wikipedia topics
convert-all-wiki-topics:
	@echo "🔄 Converting ALL Wikipedia topics to .Ψcws format..."
	@echo "⚠️  This will make multiple API requests to Wikipedia (may take 10+ minutes)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	@python3 wiki_to_psicws_converter.py all

# Demo: Convert a consciousness topic
demo-wiki-conversion:
	@echo "🎬 Demo: Converting Consciousness (Wikipedia topic)"
	@make convert-wiki-topic TOPIC=consciousness
	@make analyze-Ψcws-consciousness