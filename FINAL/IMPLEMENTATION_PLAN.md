# 🚀 Plano de Implementação: Correção do Mapeamento de Pesos Espectrais

## 🎯 Objetivo
Corrigir o gap entre conversão espectral e pipeline, garantindo que os pesos TREINADOS do GPT-2 sejam preservados no modelo ΨQRH via transformação física.

---

## 📋 Tarefas

### ✅ FASE 1: Diagnóstico (COMPLETO)
- [x] Analisar `SpectralModelConverter`
- [x] Identificar gap: pesos não são salvos/carregados
- [x] Documentar análise completa
- [x] Criar plano de correção

### 🔄 FASE 2: Implementação do Mapeador de Pesos

#### Tarefa 2.1: Criar `spectral_weight_mapper.py`
**Arquivo:** `src/utils/spectral_weight_mapper.py`

**Funções a implementar:**

1. **`quaternion_from_phase(theta: float) -> torch.Tensor`**
   ```python
   """
   Cria quaternion de rotação a partir de fase

   Args:
       theta: Fase em radianos [-π, π]

   Returns:
       q = [cos(θ/2), sin(θ/2), 0, 0] (rotação no eixo i)
   """
   ```

2. **`apply_quaternion_rotation(weight: Tensor, q: Tensor, alpha: float) -> Tensor`**
   ```python
   """
   Aplica rotação quaterniônica modulada por α

   Args:
       weight: Tensor de pesos (qualquer shape)
       q: Quaternion [w, x, y, z]
       alpha: Parâmetro espectral

   Returns:
       Peso transformado com mesma shape
   """
   ```

3. **`leech_project(weight: Tensor, block_size: int = 24) -> Tensor`**
   ```python
   """
   Projeta pesos no reticulado de Leech Λ₂₄

   Args:
       weight: Tensor de pesos
       block_size: Tamanho do bloco (24 para Leech)

   Returns:
       Peso projetado (mesma shape)
   """
   ```

4. **`map_layer_weights(source_weight: Tensor, alpha: float, theta: float) -> Tensor`**
   ```python
   """
   Mapeia peso de uma camada usando parâmetros espectrais

   Pipeline:
       source_weight → quaternion_rotation(θ) →
       modulate(α) → leech_project → psiqrh_weight

   Args:
       source_weight: Peso fonte (GPT-2)
       alpha: Parâmetro α da análise espectral
       theta: Fase θ da análise espectral

   Returns:
       Peso mapeado para ΨQRH
   """
   ```

5. **`map_spectral_to_state_dict(source_state_dict: Dict, spectral_params: Dict) -> Dict`**
   ```python
   """
   Mapeia state_dict completo usando parâmetros espectrais

   Args:
       source_state_dict: State dict do modelo fonte
       spectral_params: Parâmetros espectrais por camada
           {
               'layer_0': {'alpha': 1.4, 'theta': -0.5},
               'layer_1': {'alpha': 1.6, 'theta': 0.2},
               ...
           }

   Returns:
       State dict ΨQRH com pesos mapeados
   """
   ```

#### Tarefa 2.2: Atualizar `convert_model_spectral.py`
**Arquivo:** `scripts/convert_model_spectral.py`

**Mudanças:**

```python
# Adicionar import
from src.utils.spectral_weight_mapper import map_spectral_to_state_dict

def save_converted_model(
    converted_params: dict,
    output_dir: Path,
    source_info: dict
):
    # ... código atual (salva JSON) ...

    # ✅ ADICIONAR: Mapear e salvar state_dict
    print("\n💾 Mapeando pesos usando parâmetros espectrais...")

    # Verificar se temos source_model
    if 'source_model' in source_info and hasattr(source_info['source_model'], 'state_dict'):
        source_state_dict = source_info['source_model'].state_dict()

        # Mapear pesos
        psiqrh_state_dict = map_spectral_to_state_dict(
            source_state_dict,
            converted_params['converted_params']
        )

        # Salvar state_dict
        state_dict_path = output_dir / "pytorch_model.bin"
        torch.save(psiqrh_state_dict, state_dict_path)
        print(f"✅ State dict mapeado salvo: {state_dict_path}")
        print(f"   Número de tensores: {len(psiqrh_state_dict)}")

        # Calcular tamanho
        total_params = sum(t.numel() for t in psiqrh_state_dict.values())
        print(f"   Total de parâmetros: {total_params:,}")

    else:
        print("⚠️  Source model não disponível - state_dict não será salvo")
        print("   Apenas metadata espectral será salva")
```

**Atualizar função main():**

```python
def main():
    # ... código atual ...

    # Executar conversão
    try:
        report = converter.convert_model(source_model, ...)

        # ✅ Passar source_model para save_converted_model
        source_info = {
            'model_type': source_model.__class__.__name__,
            'source': args.source,
            'source_model': source_model  # ← ADICIONAR
        }

        save_converted_model(report, output_path, source_info)

    except Exception as e:
        # ... tratamento de erro ...
```

#### Tarefa 2.3: Atualizar `complete_spectral_pipeline.py`
**Arquivo:** `examples/complete_spectral_pipeline.py`

**Mudanças na função `_load_psiqrh_model()`:**

```python
def _load_psiqrh_model(self):
    """Carrega modelo ΨQRH convertido espectralmente"""

    print("\n🏗️  Carregando PsiQRHTransformer nativo...")

    # Carregar metadata espectral
    metadata_path = self.model_dir / "spectral_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"✅ Metadata espectral carregada")
        print(f"   D médio: {metadata.get('avg_fractal_dim', 'N/A'):.4f}")
        print(f"   α médio: {metadata.get('avg_alpha', 'N/A'):.4f}")
    else:
        print("⚠️  Metadata espectral não encontrada")
        metadata = {}

    # Carregar configuração
    config_path = self.model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Configuração padrão
        config = {
            'model': {
                'vocab_size': 50000,
                'd_model': 256,
                'n_layers': 6,
                'n_heads': 8,
                'dim_feedforward': 1024,
                'max_seq_length': 512
            }
        }

    # Criar modelo
    self.psiqrh_model = PsiQRHTransformer(
        vocab_size=config['model'].get('vocab_size', 50000),
        d_model=config['model'].get('d_model', 256),
        n_layers=config['model'].get('n_layers', 6),
        n_heads=config['model'].get('n_heads', 8),
        max_seq_length=config['model'].get('max_seq_length', 512)
    ).to(self.device)

    # ✅ ADICIONAR: Carregar pesos convertidos
    weights_path = self.model_dir / "pytorch_model.bin"
    if weights_path.exists():
        print(f"\n💾 Carregando pesos convertidos...")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.psiqrh_model.load_state_dict(state_dict, strict=False)
        print(f"✅ Pesos convertidos carregados do GPT-2")
        print(f"   Total de parâmetros: {sum(p.numel() for p in self.psiqrh_model.parameters()):,}")
    else:
        print("⚠️  pytorch_model.bin não encontrado")
        print("   Usando inicialização aleatória (SEM conhecimento do GPT-2)")
        print("   Para usar conhecimento convertido, execute:")
        print(f"   make convert-model SOURCE=gpt2 OUTPUT={self.model_dir}")

    print(f"✅ Modelo ΨQRH carregado")
```

---

## 🧪 Testes de Validação

### Teste 1: Preservação de Conhecimento
**Arquivo:** `tests/test_spectral_weight_mapping.py`

```python
import torch
from transformers import AutoModel, AutoTokenizer
from src.utils.spectral_model_converter import SpectralModelConverter
from src.utils.spectral_weight_mapper import map_spectral_to_state_dict
from src.architecture.psiqrh_transformer import PsiQRHTransformer

def test_knowledge_preservation():
    """
    Testa se conhecimento é preservado na conversão
    """
    # 1. Carregar GPT-2
    gpt2 = AutoModel.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 2. Converter
    converter = SpectralModelConverter()
    report = converter.convert_model(gpt2)

    # 3. Mapear pesos
    psiqrh_state_dict = map_spectral_to_state_dict(
        gpt2.state_dict(),
        report['converted_params']
    )

    # 4. Criar modelo ΨQRH
    psiqrh = PsiQRHTransformer(...)
    psiqrh.load_state_dict(psiqrh_state_dict, strict=False)

    # 5. Testar preservação
    test_text = "Hello world"
    input_ids = tokenizer(test_text, return_tensors="pt")['input_ids']

    with torch.no_grad():
        gpt2_out = gpt2(input_ids)['last_hidden_state']
        psiqrh_out = psiqrh(input_ids)

    # 6. Calcular similaridade
    similarity = torch.nn.functional.cosine_similarity(
        gpt2_out.flatten(),
        psiqrh_out.flatten(),
        dim=0
    ).item()

    print(f"Similaridade de saída: {similarity:.4f}")
    assert similarity > 0.7, "Conhecimento não foi preservado!"

    # 7. Validar energia
    gpt2_energy = torch.sum(gpt2_out ** 2).item()
    psiqrh_energy = torch.sum(psiqrh_out ** 2).item()
    energy_ratio = psiqrh_energy / (gpt2_energy + 1e-12)

    print(f"Razão de energia: {energy_ratio:.4f}")
    assert 0.9 <= energy_ratio <= 1.1, "Conservação de energia violada!"

    print("✅ Teste de preservação de conhecimento passou!")
```

### Teste 2: Pipeline End-to-End
**Arquivo:** `tests/test_spectral_pipeline_e2e.py`

```python
def test_pipeline_end_to_end():
    """
    Testa pipeline completo: converter → carregar → gerar texto
    """
    # 1. Converter modelo
    os.system("make convert-model SOURCE=gpt2 OUTPUT=./temp_models/gpt2_test")

    # 2. Executar pipeline
    pipeline = SpectralPipelineComplete("./temp_models/gpt2_test")

    # 3. Processar texto
    result = pipeline.process("Hello world")

    # 4. Verificar saída
    assert result['generated_text'].strip() != "", "Saída vazia!"
    assert len(result['generated_text']) > 5, "Saída muito curta!"

    # 5. Verificar métricas
    assert 'fci' in result['consciousness_metrics']
    assert 'alpha' in result

    print(f"✅ Pipeline E2E passou!")
    print(f"   Input: Hello world")
    print(f"   Output: {result['generated_text'][:50]}...")
    print(f"   FCI: {result['consciousness_metrics']['fci']}")
```

---

## 📊 Critérios de Sucesso

### ✅ Implementação Completa
- [ ] `spectral_weight_mapper.py` criado e testado
- [ ] `convert_model_spectral.py` atualizado
- [ ] `complete_spectral_pipeline.py` atualizado
- [ ] Testes unitários passando

### ✅ Validação Funcional
- [ ] Conversão salva `pytorch_model.bin`
- [ ] Pipeline carrega pesos convertidos
- [ ] Similaridade GPT-2 ↔ ΨQRH > 0.7
- [ ] Conservação de energia: 0.9 ≤ R ≤ 1.1

### ✅ Geração de Texto
- [ ] Input: "Hello world" → Output: texto coerente
- [ ] FCI > 0.0 (não mais sempre 0.0)
- [ ] Texto gerado > 10 caracteres (não apenas espaços)

---

## 🔄 Workflow de Desenvolvimento

### 1. Implementar Mapeador
```bash
# Criar arquivo
vim src/utils/spectral_weight_mapper.py

# Implementar funções:
# - quaternion_from_phase()
# - apply_quaternion_rotation()
# - leech_project()
# - map_layer_weights()
# - map_spectral_to_state_dict()

# Testar isoladamente
python3 -c "
from src.utils.spectral_weight_mapper import *
import torch
w = torch.randn(100, 100)
q = quaternion_from_phase(0.5)
w_rot = apply_quaternion_rotation(w, q, 1.5)
print(f'Shape: {w_rot.shape}')
print(f'Norm ratio: {torch.norm(w_rot) / torch.norm(w):.4f}')
"
```

### 2. Atualizar Conversão
```bash
# Editar convert_model_spectral.py
vim scripts/convert_model_spectral.py

# Testar conversão
make convert-model SOURCE=gpt2 OUTPUT=./temp_models/gpt2_test

# Verificar saída
ls -lh ./temp_models/gpt2_test/
# Deve mostrar: pytorch_model.bin (novo!)
```

### 3. Atualizar Pipeline
```bash
# Editar pipeline
vim examples/complete_spectral_pipeline.py

# Testar pipeline
python3 examples/complete_spectral_pipeline.py ./temp_models/gpt2_test

# Verificar saída
# Input: "Hello world"
# Output: DEVE ter texto real (não espaços)
```

### 4. Validar Completo
```bash
# Rodar testes
python3 tests/test_spectral_weight_mapping.py
python3 tests/test_spectral_pipeline_e2e.py

# Pipeline completo
make new-model SOURCE=gpt2 NAME=gpt2_validated

# Testar geração
python3 chat_with_model.py --model gpt2_validated
```

---

## 📝 Checklist Final

### Antes do Commit
- [ ] Código documentado (docstrings)
- [ ] Testes unitários passando
- [ ] Pipeline E2E funcional
- [ ] Geração de texto validada
- [ ] Documentação atualizada

### Antes do Deploy
- [ ] `make convert-model` salva `pytorch_model.bin`
- [ ] `complete_spectral_pipeline.py` carrega pesos
- [ ] Similaridade > 0.7
- [ ] Energia conservada (0.9-1.1)
- [ ] Texto gerado coerente

---

## 🎯 Resultado Final Esperado

```bash
# 1. Converter GPT-2
$ make convert-model SOURCE=gpt2 OUTPUT=./models/gpt2_psiqrh

📊 PASSO 1: Análise Espectral do Modelo Antigo
✅ transformer.h.0.attn.c_attn.weight: β=1.234, D=0.883, R²=0.956
✅ transformer.h.1.attn.c_attn.weight: β=1.456, D=0.772, R²=0.934
...
💾 Mapeando pesos usando parâmetros espectrais...
✅ State dict mapeado salvo: ./models/gpt2_psiqrh/pytorch_model.bin
   Total de parâmetros: 124,439,808

# 2. Executar pipeline
$ python3 examples/complete_spectral_pipeline.py ./models/gpt2_psiqrh

🏗️  Carregando PsiQRHTransformer nativo...
✅ Metadata espectral carregada
   D médio: 0.8835
   α médio: 1.4521
💾 Carregando pesos convertidos...
✅ Pesos convertidos carregados do GPT-2
   Total de parâmetros: 124,439,808

🧪 Testando com 3 entradas...

📝 Teste 1: "Hello world"
   ✅ Texto gerado: "Hello world! How can I help you today? I'm an AI..."
   📊 FCI: 0.78 (Estado: MEDITAÇÃO)
   ⚡ α: 1.450
   🌊 D: 0.883

📝 Teste 2: "Quantum physics is fascinating"
   ✅ Texto gerado: "Quantum physics is fascinating because it describes..."
   📊 FCI: 0.92 (Estado: EMERGÊNCIA)
   ⚡ α: 1.523
   🌊 D: 1.015

✅ Pipeline completo validado!
   Conhecimento do GPT-2 preservado via conversão espectral ΨQRH!
```

---

**Próximo passo:** Implementar `spectral_weight_mapper.py` conforme especificação acima.
