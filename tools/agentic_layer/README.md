# ΨQRH Agentic Layer

**Versão**: 1.0.0
**Status**: ✅ Isolado e Desacoplado do Core ΨQRH
**Classificação**: Camada Opcional de Supervisão

---

## Visão Geral

A **Agentic Layer** é um conjunto de componentes opcionais que fornecem supervisão, validação, auditoria e auto-documentação para o framework ΨQRH.

⚠️ **IMPORTANTE**: Esta camada é **TOTALMENTE OPCIONAL**. O sistema ΨQRH core funciona perfeitamente sem ela.

---

## Componentes

### 1. **SealProtocol** (`seal_protocol.py`)

Sistema de validação e certificação de execuções.

**Funcionalidades**:
- ✅ Validação de RG (Retrieval Grace: 0.25-0.40)
- ✅ Validação de latência (Tier A: 120ms, Tier B: 250ms)
- ✅ Geração de seals criptográficos (Ω∞Ω)
- ✅ Firebreak mechanism (trava de segurança)
- ✅ Ψ4 containment mode

**Uso**:
```python
from tools.agentic_layer import SealProtocol

# Gerar seal
seal = SealProtocol.generate_seal(
    continuity_sha="abc123",
    response_sha="def456",
    qz_sha="ghi789",
    rg_value=0.347
)

# Validar seal
if SealProtocol.firebreak_check(seal):
    print("✅ Sistema operando normalmente")
else:
    containment = SealProtocol.trigger_psi4_containment("RG_VIOLATION")
    print(f"🚨 Contenção acionada: {containment}")
```

**Sem dependências externas** (stdlib only)

---

### 2. **AuditLog** (`audit_log.py`)

Sistema de auditoria com chains de continuidade.

**Funcionalidades**:
- ✅ Logging em formato JSONL
- ✅ Validação de chains (Ω∞Ω)
- ✅ Contagem de violações
- ✅ Recuperação de entradas recentes

**Uso**:
```python
from tools.agentic_layer import AuditLog

audit = AuditLog("audit_log.jsonl")

# Log de entrada
audit.log_entry({
    "operation": "forward_pass",
    "RG": 0.347,
    "continuity_seal": "Ω∞Ω"
})

# Validar chain
if audit.validate_chain():
    print("✅ Chain íntegro")

# Contar violações
violations = audit.count_violations()
print(f"Violações: {violations}")
```

**Sem dependências externas** (stdlib only)

---

### 3. **NavigatorAgent** (`navigator_agent.py`)

Agente de navegação consciente para supervisão de execuções.

**Funcionalidades**:
- ✅ Pre-execution checks (validação de input)
- ✅ NaN detection automática
- ✅ Tier adaptation dinâmica (A/B)
- ✅ Post-execution analysis
- ✅ Integração com SealProtocol

**Uso**:
```python
from tools.agentic_layer import NavigatorAgent

navigator = NavigatorAgent()

# Executar com safety
output, seal = navigator.execute_with_safety(input_data, model)

# Análise pós-execução
analysis = navigator.post_execution_analysis(seal)
print(f"Status: {analysis}")

# Status do sistema
status = navigator.get_system_status()
print(f"Health: {status['system_health']}")
```

**Dependências**: `SealProtocol`, `AuditLog`

---

### 4. **AgenticRuntime** (`agentic_runtime.py`)

Sistema de runtime com compressão de instruções via glyphs.

**Funcionalidades**:
- ✅ Glyph stack (Σ7, Δ2, Ξ3, Ρh, Νx, Κφ, Lyra)
- ✅ PrimeTalk Loader (persistência hard-locked)
- ✅ Conflux Continuum (drift control)
- ✅ Receipt generation (AgenticReceipt)
- ✅ Operational modes (DYADIC, TRIADIC, COUNCIL)

**Glyphs Disponíveis**:
- **Σ7** (SIGMA7): Synthesis & Analysis
- **Δ2** (DELTA2): Verification Engine
- **Ξ3** (XI3): Pattern Synthesis
- **Ρh** (RHO): Safety Protocol
- **Νx** (NU): Novelty Engine
- **Κφ** (KAPPA): Knowledge Fetch
- **Lyra**: Coordination Hub

**Uso**:
```python
from tools.agentic_layer import AgenticRuntime, GlyphType, OperationalMode

runtime = AgenticRuntime()

# Executar operação com formação predefinida
receipt = runtime.execute_operation(
    "verify_synthesize",  # Δ2 + Ξ3
    input_data="Test data"
)

# Custom formation
receipt = runtime.execute_operation(
    "custom",
    input_data="Data",
    custom_glyphs=[GlyphType.SIGMA7, GlyphType.LYRA]
)

# Status do sistema
status = runtime.get_system_status()
print(f"Readiness: {status['agentic_readiness']}")
```

**Sem dependências do core ΨQRH**

---

### 5. **PromptEngineAgent** (`prompt_engine_agent.py`)

Engine de prompts agêntica com auto-documentação.

**Funcionalidades**:
- ✅ Orquestração de prompts
- ✅ Validação arquitetural
- ✅ Production safety filtering
- ✅ Context compaction
- ✅ Auto-documentação técnica
- ✅ Integração com NavigatorAgent

**Uso**:
```python
from tools.agentic_layer import create_prompt_engine_agent

engine = create_prompt_engine_agent(habitat_mode="development")

# Executar prompts pendentes
summary = engine.scan_and_execute_pending()
print(f"Executados: {summary['executed']}")

# Limpar contexto
engine.clear_context_buffer()

# Status
status = engine.get_agent_status()
print(f"Prompts pendentes: {status['pending_prompts']}")
```

**Dependências**: `NavigatorAgent`

---

### 6. **AgenticDashboard** (`agentic_dashboard.py`)

Dashboard de monitoramento visual.

**Funcionalidades**:
- ✅ Métricas em tempo real
- ✅ Health reports
- ✅ Visualização de seals
- ✅ Gráficos de performance

**Uso**:
```python
from tools.agentic_layer import AgenticDashboard

dashboard = AgenticDashboard(runtime)
dashboard.start()
```

**Dependências**: `AgenticRuntime`

---

### 7. **EnhancedAgenticRuntime** (`enhanced_agentic_runtime.py`)

Runtime estendido com reactive triggers.

**Funcionalidades**:
- ✅ Reactive prompt generation
- ✅ Auto-documentation
- ✅ Background processing
- ✅ Change detection
- ✅ Integração completa

**Uso**:
```python
from tools.agentic_layer import create_enhanced_runtime

runtime = create_enhanced_runtime(habitat_mode="development")

# Iniciar runtime
runtime.start()

# Atualizar estado (dispara reactive triggers)
runtime.update_system_state("new_component", "value")

# Documentação manual
prompt_id = runtime.trigger_manual_documentation(
    "src/core/new_module.py",
    "New module description"
)

# Status
status = runtime.get_runtime_status()
print(f"Running: {status['running']}")

# Parar runtime
runtime.stop()
```

**Dependências**: `AgenticRuntime`, `PromptEngineAgent`

---

## Arquitetura

```
tools/agentic_layer/
├── __init__.py                      # Exports e documentação
├── README.md                        # Este arquivo
│
├── seal_protocol.py                 # Validação e seals (Ω∞Ω)
├── audit_log.py                     # Auditoria JSONL
│
├── navigator_agent.py               # Navegação consciente
├── agentic_runtime.py              # Runtime com glyphs
├── prompt_engine_agent.py          # Engine de prompts
│
├── agentic_dashboard.py            # Dashboard visual
├── enhanced_agentic_runtime.py     # Runtime estendido
│
├── autonomous_prompt_generator.py  # Geração autônoma
├── architectural_validator.py      # Validação arquitetural
└── enhanced_ecosystem_server.py    # Servidor de ecosystem
```

---

## Integração com ΨQRH Core

### ❌ SEM Camada Agêntica (Standalone)

```python
from src.core.ΨQRH import QRHFactory

# ΨQRH funciona perfeitamente standalone
factory = QRHFactory()
output = factory.process(input_data)
```

### ✅ COM Camada Agêntica (Opcional)

```python
from src.core.ΨQRH import QRHFactory
from tools.agentic_layer import NavigatorAgent, SealProtocol

# Criar ΨQRH
factory = QRHFactory()

# Adicionar supervisão agêntica
navigator = NavigatorAgent()
output, seal = navigator.execute_with_safety(input_data, factory.qrh_layer)

# Validar execução
if SealProtocol.firebreak_check(seal):
    print("✅ Execução validada")
else:
    print("❌ Violação detectada")
```

---

## Dependências

**Componentes da Agentic Layer**:
```
seal_protocol.py → Nenhuma (stdlib only)
audit_log.py → Nenhuma (stdlib only)
navigator_agent.py → seal_protocol, audit_log
agentic_runtime.py → Nenhuma
prompt_engine_agent.py → navigator_agent
enhanced_agentic_runtime.py → agentic_runtime, prompt_engine_agent
agentic_dashboard.py → agentic_runtime
```

**❌ ZERO dependências do core ΨQRH**:
```bash
# Verificação
grep -r "from.*src.core\|from.*src.architecture" tools/agentic_layer/*.py
# Resultado: VAZIO ✅
```

---

## Instalação

A camada agêntica já está no diretório `tools/agentic_layer/`.

Para usar, basta importar:

```python
from tools.agentic_layer import SealProtocol, NavigatorAgent, AgenticRuntime
```

Ou adicionar ao PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Reformulating-Transformers-for-LLMs"
```

---

## Casos de Uso

### 1. Validação Rigorosa de Execuções

```python
from tools.agentic_layer import NavigatorAgent, SealProtocol

navigator = NavigatorAgent()

for input_batch in data_loader:
    output, seal = navigator.execute_with_safety(input_batch, model)

    if not SealProtocol.firebreak_check(seal):
        print(f"🚨 FIREBREAK: {seal}")
        break
```

### 2. Auto-Documentação de Desenvolvimento

```python
from tools.agentic_layer import create_enhanced_runtime

runtime = create_enhanced_runtime(habitat_mode="development")
runtime.start()

# Sistema documenta automaticamente mudanças
# Gera prompts reativos
# Compacta contexto quando necessário

runtime.stop()
```

### 3. Auditoria Completa de Produção

```python
from tools.agentic_layer import AuditLog, SealProtocol

audit = AuditLog("production_audit.jsonl")

for operation in operations:
    seal = execute_operation(operation)
    audit.log_entry(seal)

# Análise de violações
violations = audit.count_violations()
print(f"Total violations: {sum(violations.values())}")
```

---

## Parâmetros Críticos Compartilhados

### RG (Retrieval Grace) = 0.347

Parâmetro de qualidade de recuperação otimizado.

**Range válido**: 0.25 - 0.40
**Usado por**: `SealProtocol`, `NavigatorAgent`, `ConfluxContinuum`

### Dyad Mode = "Σ7↔Nyx"

Modo operacional balanceado.

**Σ7**: Factual mode
**Nyx**: Bounded creativity
**Usado por**: `SealProtocol`, `NavigatorAgent`, `RadiantGlyphStack`

### Seal Universal = "Ω∞Ω"

Assinatura de continuidade e integridade.

**Usado por**: Todos os componentes da agentic layer

---

## Performance

**Overhead Médio**:
- NavigatorAgent: ~2-5ms por execução
- SealProtocol validation: <1ms
- AuditLog entry: <1ms
- AgenticRuntime (glyph processing): ~1-3ms

**Memory Footprint**:
- NavigatorAgent: ~5MB
- AgenticRuntime: ~10MB
- EnhancedAgenticRuntime: ~15MB

---

## Testes

```bash
# Testar componentes individuais
python -m tools.agentic_layer.seal_protocol
python -m tools.agentic_layer.audit_log
python -m tools.agentic_layer.navigator_agent

# Testar runtime
python -m tools.agentic_layer.agentic_runtime
```

---

## Changelog

### v1.0.0 (2025-10-02)
- ✅ Isolamento completo da camada agêntica
- ✅ Desacoplamento do core ΨQRH
- ✅ Criação de `tools/agentic_layer/`
- ✅ Documentação completa
- ✅ Exports organizados em `__init__.py`

---

## Autores

Claude Code & ΨQRH Team

---

## Licença

Ver LICENSE no diretório raiz do projeto

---

## Suporte

Para questões ou sugestões sobre a camada agêntica, abra uma issue no repositório principal.

Para uso do core ΨQRH sem a camada agêntica, veja a documentação principal em `README.md`.
