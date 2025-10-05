"""
ΨQRH Agentic Layer - Sistema de Gerenciamento Agêntico
========================================================

Camada opcional de supervisão, validação e auto-documentação para o framework ΨQRH.

Este módulo fornece componentes agênticos que podem ser usados opcionalmente
para supervisionar e gerenciar o sistema ΨQRH core, mas NÃO são necessários
para o funcionamento básico do framework.

Componentes Principais
----------------------

1. **SealProtocol** (seal_protocol.py)
   - Validação de parâmetros (RG, latência, dyad mode)
   - Geração de seals criptográficos (Ω∞Ω)
   - Firebreak mechanism
   - Ψ4 containment mode

2. **AuditLog** (audit_log.py)
   - Logging de auditoria em formato JSONL
   - Validação de chains de continuidade
   - Contagem de violações

3. **NavigatorAgent** (navigator_agent.py)
   - Pre/post execution checks
   - NaN detection automática
   - Tier adaptation (A/B)
   - Integração com SealProtocol

4. **AgenticRuntime** (agentic_runtime.py)
   - Glyph-based instruction compression (Σ7, Δ2, Ξ3)
   - PrimeTalk Loader (persistência)
   - Conflux Continuum (drift control)
   - Receipt generation

5. **PromptEngineAgent** (prompt_engine_agent.py)
   - Orquestração de prompts
   - Auto-documentação
   - Context compaction
   - Production safety filtering

6. **AgenticDashboard** (agentic_dashboard.py)
   - Monitoramento visual
   - Métricas de sistema
   - Health reports

7. **EnhancedAgenticRuntime** (enhanced_agentic_runtime.py)
   - Runtime estendido com reactive triggers
   - Auto-documentation
   - Background processing

Uso Básico
----------

```python
# Importar componentes principais
from tools.agentic_layer.seal_protocol import SealProtocol
from tools.agentic_layer.navigator_agent import NavigatorAgent
from tools.agentic_layer.agentic_runtime import AgenticRuntime

# Usar NavigatorAgent para supervisão
navigator = NavigatorAgent()
output, seal = navigator.execute_with_safety(input_data, model)

# Validar seal
if SealProtocol.firebreak_check(seal):
    print("Sistema operando dentro dos limites")
else:
    containment = SealProtocol.trigger_psi4_containment("VIOLATION")
    print(f"Contenção acionada: {containment}")
```

Integração com ΨQRH Core
-------------------------

A camada agêntica é **OPCIONAL** e **DESACOPLADA** do core ΨQRH.

O sistema ΨQRH funciona completamente standalone sem estes componentes.

Para integrar:

```python
# ΨQRH Core standalone (SEM agentes)
from src.core.ΨQRH import QRHFactory

factory = QRHFactory()
output = factory.process(input_data)

# ΨQRH Core COM supervisão agêntica (OPCIONAL)
from tools.agentic_layer import NavigatorAgent

navigator = NavigatorAgent()
output, seal = navigator.execute_with_safety(input_data, factory.qrh_layer)
```

Dependências
------------

Componentes da camada agêntica:
- seal_protocol.py → Sem dependências externas (stdlib only)
- audit_log.py → Sem dependências externas (stdlib only)
- navigator_agent.py → seal_protocol, audit_log
- agentic_runtime.py → Sem dependências do core ΨQRH
- prompt_engine_agent.py → navigator_agent
- enhanced_agentic_runtime.py → agentic_runtime, prompt_engine_agent

NENHUMA dependência do core ΨQRH (src/core, src/architecture, src/conscience)

Versão
------
1.0.0 - Camada agêntica isolada e desacoplada

Autores
-------
Claude Code & ΨQRH Team

Licença
-------
Ver LICENSE no diretório raiz do projeto
"""

# Core components
from .seal_protocol import SealProtocol
from .audit_log import AuditLog

# Agentes
from .navigator_agent import NavigatorAgent
from .agentic_runtime import (
    AgenticRuntime,
    GlyphType,
    OperationalMode,
    AgenticReceipt,
    PrimeTalkLoader,
    RadiantGlyphStack,
    ConfluxContinuum
)
from .prompt_engine_agent import (
    PromptEngineAgent,
    PromptExecutionContext,
    create_prompt_engine_agent
)

# Enhanced components
from .enhanced_agentic_runtime import (
    EnhancedAgenticRuntime,
    ReactivePromptTrigger,
    create_enhanced_runtime
)

# Utilities
from .agentic_dashboard import AgenticDashboard

__version__ = "1.0.0"
__all__ = [
    # Core
    "SealProtocol",
    "AuditLog",

    # Agentes
    "NavigatorAgent",
    "AgenticRuntime",
    "PromptEngineAgent",

    # Runtime components
    "GlyphType",
    "OperationalMode",
    "AgenticReceipt",
    "PrimeTalkLoader",
    "RadiantGlyphStack",
    "ConfluxContinuum",

    # Enhanced
    "EnhancedAgenticRuntime",
    "ReactivePromptTrigger",

    # Dashboard
    "AgenticDashboard",

    # Factory functions
    "create_prompt_engine_agent",
    "create_enhanced_runtime",
]
