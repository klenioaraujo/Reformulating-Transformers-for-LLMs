# ΨQRH Framework - Análise de Fluxo de Processamento de Texto

## 📊 Testes Executados

### Teste 1: Entrada Aleatória
**Entrada**: `"SUYSUYUATYTYQTQTYAF"`

**Saída Completa**:
```
🚀 Inicializando ΨQRH Pipeline no dispositivo: cpu
🤖 SISTEMA DE TESTE HUMANO - ΨQRH FRAMEWORK
============================================================
📝 Session ID: psiqrh_chat_test_1758899275_71e512c0
📁 Logs: /tmp/tmpll1h2fhc/psiqrh_chat_test_logs
✅ Sistema de chat ΨQRH carregado
🧠 Processando: SUYSUYUATYTYQTQTYAF

✅ Resultado (cpu):
--------------------------------------------------
ΨQRH Framework resposta para: SUYSUYUATYTYQTQTYAF

Processado usando sistemas quaterniônicos e análise espectral.
--------------------------------------------------

📊 Metadados:
  - Tarefa: text-generation
  - Dispositivo: cpu
  - Entrada: 19 caracteres
  - Saída: 113 caracteres
```

### Teste 2: Entrada Natural
**Entrada**: `"Nove planetas?"`

**Saída Completa**:
```
🚀 Inicializando ΨQRH Pipeline no dispositivo: cpu
🤖 SISTEMA DE TESTE HUMANO - ΨQRH FRAMEWORK
============================================================
📝 Session ID: psiqrh_chat_test_1758899287_736254c1
📁 Logs: /tmp/tmpi149rnks/psiqrh_chat_test_logs
✅ Sistema de chat ΨQRH carregado
🧠 Processando: Nove planetas?

✅ Resultado (cpu):
--------------------------------------------------
ΨQRH Framework resposta para: Nove planetas?

Processado usando sistemas quaterniônicos e análise espectral.
--------------------------------------------------

📊 Metadados:
  - Tarefa: text-generation
  - Dispositivo: cpu
  - Entrada: 14 caracteres
  - Saída: 108 caracteres
```

## 🔄 Mapeamento do Fluxo de Execução

### 1. **Ponto de Entrada** (`psiqrh.py:340`)
```python
if __name__ == "__main__":
    sys.exit(main())
```

### 2. **Função Principal** (`psiqrh.py:164-237`)
```python
def main():
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()

    # Roteamento para processamento de texto único
    if args.text:
        return process_single_text(args.text, args.task, args.device, args.verbose)
```

### 3. **Processamento de Texto Único** (`psiqrh.py:313-337`)
```python
def process_single_text(text: str, task: str, device: Optional[str], verbose: bool = False) -> int:
    # 3.1 Inicialização do Pipeline
    pipeline = ΨQRHPipeline(task=task, device=device)

    # 3.2 Processamento da entrada
    result = pipeline(text)

    # 3.3 Exibição dos resultados
    if result['status'] == 'success':
        print(f"✅ Resultado ({result['device']}):")
        print(result['response'])
```

### 4. **Inicialização do Pipeline ΨQRH** (`psiqrh.py:40-93`)
```python
class ΨQRHPipeline:
    def __init__(self, task: str = "text-generation", device: Optional[str] = None):
        self.task = task
        self.device = self._detect_device(device)  # CPU detectado
        self.model = None
        self._initialize_model()  # Carrega HumanChatTest()
```

### 5. **Detecção de Dispositivo** (`psiqrh.py:56-66`)
```python
def _detect_device(self, device: Optional[str]) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"  # Retornado para ambos os testes
```

### 6. **Inicialização do Modelo** (`psiqrh.py:68-93`)
```python
def _initialize_model(self):
    print(f"🚀 Inicializando ΨQRH Pipeline no dispositivo: {self.device}")

    if self.task in ["text-generation", "chat"]:
        from src.core.human_chat_test import HumanChatTest
        self.model = HumanChatTest()  # Sistema de chat carregado
        print("✅ Sistema de chat ΨQRH carregado")
```

### 7. **Chamada do Pipeline** (`psiqrh.py:94-110`)
```python
def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
    if self.task in ["text-generation", "chat"]:
        return self._generate_text(text, **kwargs)  # Rota seguida
    elif self.task == "analysis":
        return self._analyze_text(text, **kwargs)
```

### 8. **Geração de Texto** (`psiqrh.py:106-127`)
```python
def _generate_text(self, text: str, **kwargs) -> Dict[str, Any]:
    try:
        # 8.1 Criação da resposta template
        response = f"ΨQRH Framework resposta para: {text}\n\nProcessado usando sistemas quaterniônicos e análise espectral."

        # 8.2 Retorno dos metadados
        return {
            'status': 'success',
            'response': response,
            'task': self.task,
            'device': self.device,
            'input_length': len(text),
            'output_length': len(response)
        }
```

## 📈 Análise Detalhada do Processamento

### **Entrada 1: "SUYSUYUATYTYQTQTYAF"**
- **Caracteres de entrada**: 19
- **Caracteres de saída**: 113 (template fixo + entrada específica)
- **Session ID**: `psiqrh_chat_test_1758899275_71e512c0`
- **Logs temporários**: `/tmp/tmpll1h2fhc/psiqrh_chat_test_logs`
- **Dispositivo**: CPU
- **Processamento**: Template simples + metadados
- **Resposta gerada**: "ΨQRH Framework resposta para: SUYSUYUATYTYQTQTYAF\n\nProcessado usando sistemas quaterniônicos e análise espectral."

### **Entrada 2: "Nove planetas?"**
- **Caracteres de entrada**: 14
- **Caracteres de saída**: 108 (template fixo + entrada específica)
- **Session ID**: `psiqrh_chat_test_1758899287_736254c1`
- **Logs temporários**: `/tmp/tmpi149rnks/psiqrh_chat_test_logs`
- **Dispositivo**: CPU
- **Processamento**: Template simples + metadados
- **Resposta gerada**: "ΨQRH Framework resposta para: Nove planetas?\n\nProcessado usando sistemas quaterniônicos e análise espectral."

## 🏗️ Arquitetura do Sistema

### **Componentes Envolvidos:**
1. **psiqrh.py**: Ponto de entrada CLI
2. **ΨQRHPipeline**: Classe principal de orquestração
3. **HumanChatTest**: Sistema de teste de chat básico
4. **Device Detection**: Auto-detecção de hardware (CPU/CUDA/MPS)

### **Fluxo de Dados:**
```
Entrada de Texto
       ↓
Parsing de Argumentos
       ↓
Inicialização do Pipeline
       ↓
Detecção de Dispositivo
       ↓
Carregamento do Modelo
       ↓
Processamento (_generate_text)
       ↓
Template de Resposta
       ↓
Formatação da Saída
       ↓
Exibição no Terminal
```

## 🔍 Observações

### **Comportamento Atual:**
- Sistema usa implementação de **template simples**
- Não há processamento quaterniônico **real** atualmente
- Análise espectral é **conceitual** na resposta
- Dispositivo é sempre **CPU** no ambiente atual

### **Metadados Gerados:**
- Status de execução (success/error)
- Tarefa executada (text-generation)
- Dispositivo utilizado (cpu)
- Comprimento da entrada e saída
- Mensagem de resposta formatada

### **Conclusão:**
O sistema atual funciona como um **wrapper CLI** para o framework ΨQRH, fornecendo uma interface unificada que pode ser expandida para integrar os componentes quaterniônicos e espectrais reais conforme o desenvolvimento avança.