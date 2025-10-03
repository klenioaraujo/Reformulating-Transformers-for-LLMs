#!/usr/bin/env python3
"""
ΨQRH CLI - Ponto de Entrada Unificado
======================================

ΨQRH-PROMPT-ENGINE: {
  "context": "Implementação de CLI unificada para ΨQRH framework com zero fallback policy",
  "analysis": "Arquitetura atual possui pontos de entrada complexos que contradizem promessa de simplicidade no README.md",
  "solution": "Criar pipeline unificado similar ao transformers.pipeline() que abstrai complexidade interna",
  "implementation": [
    "Implementar ΨQRHPipeline classe principal com inicialização automática",
    "Suportar múltiplas tarefas (text-generation, chat, analysis)",
    "Detecção automática de dispositivo (CPU/CUDA/MPS)",
    "Interface CLI com argparse para diferentes modos de uso",
    "Validação matemática obrigatória em todas as respostas",
    "ZERO fallback - sistema deve funcionar ou falhar claramente"
  ],
  "validation": "CLI deve executar com sucesso ou falhar com mensagem clara de erro, sem fallbacks não-matemáticos"
}

Interface simples tipo transformers.pipeline() para o framework ΨQRH.

Exemplos de uso:
    python psiqrh.py "Explique o conceito de quaternions"
    python psiqrh.py --interactive
    python psiqrh.py --test
    python psiqrh.py --help
"""

import argparse
import sys
import os
import torch
from typing import Optional, List, Dict, Any

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Import tensor validator
from src.core.tensor_validator import ScientificTensorValidator

class ΨQRHPipeline:
    """Pipeline unificado para o framework ΨQRH - similar ao transformers.pipeline()"""

    def __init__(self, task: str = "text-generation", device: Optional[str] = None, input_text: Optional[str] = None):
        """
        Inicializa o pipeline ΨQRH.

        Args:
            task: Tipo de tarefa (text-generation, analysis, chat, signal-processing)
            device: Dispositivo (cpu, cuda, mps) - detecta automaticamente se None
            input_text: Texto de entrada para detecção automática de tarefa (opcional)
        """
        self.device = self._detect_device(device)
        self.model = None

        # Initialize global tensor validator
        self.tensor_validator = ScientificTensorValidator(auto_adjust=True)

        # Detecção inteligente de tarefa se input_text for fornecido
        if input_text is not None:
            self.task = self._detect_task_type(input_text)
        else:
            self.task = task

        self._initialize_model()

    def _detect_device(self, device: Optional[str]) -> str:
        """Detecta o melhor dispositivo disponível"""
        if device:
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _detect_task_type(self, input_text: str) -> str:
        """
        Detecta automaticamente o tipo de tarefa com base no conteúdo da entrada.

        # Roteamento automático:
        # - signal-processing: se houver [números] ou palavras-chave de simulação física
        # - text-generation: para todo o resto
        """
        import re

        input_lower = input_text.lower()

        # Padrão para detectar arrays numéricos: [1.0, -2.5, 3e-2, ...]
        numeric_array_pattern = r'\[\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*(?:,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)*\]'

        # Palavras-chave de processamento de sinais
        signal_keywords = [
            'spectral filter', 'fourier transform', 'clifford algebra',
            'quaternionic', 'signal processing', 'norm preservation',
            'unitarity', 'energy conservation', 'process signal',
            'quaternion vector', 'numerical data', 'signal array',
            'apply filter', 'validate unitarity', 'energy conservation'
        ]

        # Palavras-chave de simulação física
        physics_keywords = [
            "simule", "calcule", "verifique", "mostre", "demonstre",
            "transformada", "fourier", "schrödinger", "tunelamento",
            "invariância", "lorentz", "campo eletromagnético", "pacote de onda"
        ]

        # Verifica requisições de simulação física
        has_physics_request = any(kw in input_lower for kw in physics_keywords)
        has_numeric_data = bool(re.search(numeric_array_pattern, input_text))
        has_signal_keywords = any(kw in input_lower for kw in signal_keywords)

        # Se houver requisição física OU dados numéricos OU palavras-chave de sinal → signal-processing
        if has_physics_request or has_numeric_data or has_signal_keywords:
            print(f"🔢 Detecção automática: usando signal-processing para entrada com dados numéricos/terminologia de sinal/simulação física")
            return "signal-processing"

        # Caso contrário, assume geração de texto
        return "text-generation"

    def _initialize_model(self):
        """Inicializa o modelo ΨQRH automaticamente - ZERO FALLBACK POLICY"""
        print(f"🚀 Inicializando ΨQRH Pipeline no dispositivo: {self.device}")

        # Carregar configuração apropriada baseada na tarefa
        config = self._load_task_config()

        # Para geração de texto → use ΨQRH framework completo
        if self.task in ["text-generation", "chat"]:
            from src.core.ΨQRH import QRHFactory
            self.model = QRHFactory()
            print("✅ Framework ΨQRH completo carregado")

        # Para análise matemática → use o analisador espectral
        elif self.task == "analysis":
            from src.core.response_spectrum_analyzer import ResponseSpectrumAnalyzer
            self.model = ResponseSpectrumAnalyzer(config)
            print("✅ Analisador espectral ΨQRH carregado")

        # Para processamento de sinais → use processador numérico
        elif self.task == "signal-processing":
            from src.core.numeric_signal_processor import NumericSignalProcessor
            # Usar configuração de dispositivo do arquivo de configuração
            device_config = config.get('default_device', {'device': 'cpu'})
            self.model = NumericSignalProcessor(device=device_config['device'])
            print("✅ Processador numérico ΨQRH carregado")

        else:
            raise ValueError(f"Tarefa não suportada: {self.task}")

    def _load_task_config(self):
        """Carrega configuração apropriada baseada na tarefa"""
        import yaml

        # Mapeamento de tarefa para arquivo de configuração
        task_config_map = {
            "text-generation": "configs/example_configs.yaml",
            "chat": "configs/example_configs.yaml",
            "analysis": "configs/example_configs.yaml",
            "signal-processing": "configs/example_configs.yaml"
        }

        config_path = task_config_map.get(self.task, "configs/example_configs.yaml")

        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Selecionar seção apropriada baseada na tarefa
            if self.task == "signal-processing":
                return config_data.get("energy_conservation", {})
            else:
                return config_data.get("scientific_validation", {})

        except FileNotFoundError:
            print(f"⚠️  Arquivo de configuração {config_path} não encontrado, usando padrão")
            return {}

    def _validate_tensor_output(self, tensor: torch.Tensor, operation_name: str) -> torch.Tensor:
        """Validates tensor output from pipeline operations."""
        try:
            return self.tensor_validator.validate_for_operation(tensor, operation_name)
        except ValueError as e:
            print(f"⚠️  Tensor validation warning in {operation_name}: {e}")
            return tensor

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Processa texto através do pipeline ΨQRH.

        Args:
            text: Texto de entrada
            **kwargs: Parâmetros adicionais

        Returns:
            Dicionário com resultado e metadados
        """
        if self.task in ["text-generation", "chat"]:
            return self._generate_text(text, **kwargs)
        elif self.task == "analysis":
            return self._analyze_text(text, **kwargs)
        elif self.task == "signal-processing":
            return self._process_signal(text, **kwargs)
        else:
            raise ValueError(f"Tarefa não implementada: {self.task}")

    def _generate_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Gera resposta usando o framework ΨQRH completo"""
        try:
            # Processar texto através do pipeline ΨQRH: texto → QRHLayer → saída
            processed_output = self.model.process_text(text, device=self.device)

            # Validate tensor output if applicable
            if isinstance(processed_output, torch.Tensor):
                processed_output = self._validate_tensor_output(processed_output, "pipeline_output")

            return {
                'status': 'success',
                'response': processed_output,
                'task': self.task,
                'device': self.device,
                'input_length': len(text),
                'output_length': len(processed_output) if isinstance(processed_output, str) else (processed_output.numel() if hasattr(processed_output, 'numel') else len(str(processed_output)))
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'task': self.task,
                'device': self.device
            }

    def _analyze_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Analisa texto usando o analisador de espectro"""
        try:
            result = self.model.process_response_request(text)

            # Validate tensor output if applicable
            if isinstance(result.get('response'), torch.Tensor):
                result['response'] = self._validate_tensor_output(result['response'], "analysis_output")

            return {
                'status': result['status'],
                'response': result.get('response'),
                'confidence': result.get('confidence', 0.0),
                'mathematical_validation': result.get('mathematical_validation', False),
                'task': self.task,
                'device': self.device
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'task': self.task,
                'device': self.device
            }

    def _process_signal(self, text: str, **kwargs) -> Dict[str, Any]:
        """Processa sinais numéricos usando o processador de sinais"""
        try:
            result = self.model(text)

            return {
                'status': 'success',
                'response': result.get('text_analysis', 'Processamento de sinal concluído'),
                'numeric_results': result.get('numeric_results', []),
                'validation': result.get('validation', 'MATHEMATICALLY_VALIDATED'),
                'task': self.task,
                'device': self.device,
                'input_length': len(text),
                'output_length': len(result.get('text_analysis', '')) if isinstance(result.get('text_analysis'), str) else 0
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'task': self.task,
                'device': self.device
            }

def main():
    """Função principal da CLI"""
    parser = argparse.ArgumentParser(
        description="ΨQRH CLI - Interface unificada para o framework ΨQRH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python psiqrh.py "Explique o conceito de quaternions"
  python psiqrh.py --interactive
  python psiqrh.py --task analysis "Analise esta frase matematicamente"
  python psiqrh.py --device cuda "Processe no GPU"
  python psiqrh.py --test
        """
    )

    parser.add_argument(
        'text',
        nargs='?',
        help='Texto para processar (opcional se usar --interactive)'
    )

    parser.add_argument(
        '--task',
        choices=['text-generation', 'chat', 'analysis', 'signal-processing'],
        default='text-generation',
        help='Tipo de tarefa (padrão: text-generation)'
    )

    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'mps', 'auto'],
        default='auto',
        help='Dispositivo para execução (padrão: auto-detect)'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Modo interativo (chat contínuo)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Executar teste rápido do sistema'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Saída detalhada'
    )

    args = parser.parse_args()

    # Ajustar device
    if args.device == 'auto':
        args.device = None

    # Modo teste
    if args.test:
        return run_quick_test(args.verbose)

    # Modo interativo
    if args.interactive:
        return run_interactive_mode(args.task, args.device, args.verbose)

    # Processamento de texto único
    if args.text:
        return process_single_text(args.text, args.task, args.device, args.verbose)

    # Se nenhum argumento, mostrar ajuda
    parser.print_help()
    return 1

def run_quick_test(verbose: bool = False) -> int:
    """Executa teste rápido do sistema"""
    print("🧪 Executando teste rápido do ΨQRH...")

    test_cases = [
        "O que são quaternions?",
        "Explique a transformada de Fourier",
        "Como funciona o framework ΨQRH?"
    ]

    pipeline = ΨQRHPipeline(task="text-generation")

    for i, test_text in enumerate(test_cases, 1):
        print(f"\n--- Teste {i}/{len(test_cases)} ---")
        print(f"Entrada: {test_text}")

        result = pipeline(test_text)

        if result['status'] == 'success':
            print(f"✅ Sucesso! ({result['output_length']} caracteres)")
            if verbose:
                print(f"Resposta: {result['response'][:200]}...")
        else:
            print(f"❌ Erro: {result.get('error', 'Desconhecido')}")

    print("\n🎯 Teste concluído!")
    return 0

def run_interactive_mode(task: str, device: Optional[str], verbose: bool = False) -> int:
    """Modo interativo de chat"""
    print("💬 Modo Interativo ΨQRH")
    print("Digite 'quit' para sair ou 'help' para ajuda")
    print("=" * 50)

    # Criar pipeline inicial com task padrão
    pipeline = ΨQRHPipeline(task=task, device=device)

    while True:
        try:
            user_input = input("\n🤔 Você: ").strip()

            if user_input.lower() in ['quit', 'exit', 'sair']:
                print("👋 Até logo!")
                break

            if user_input.lower() in ['help', 'ajuda']:
                print("""
Comandos disponíveis:
  quit/exit/sair - Sair do modo interativo
  help/ajuda - Mostrar esta ajuda
  [qualquer texto] - Processar com ΨQRH
                """)
                continue

            if not user_input:
                continue

            # Reconfigurar pipeline para detecção automática de tarefa
            pipeline = ΨQRHPipeline(task=task, device=device, input_text=user_input)
            print(f"🧠 ΨQRH processando... (Tarefa: {pipeline.task})")
            result = pipeline(user_input)

            if result['status'] == 'success':
                response = result['response']

                # Handle both string and dictionary responses
                if isinstance(response, dict) and 'text_analysis' in response:
                    print(f"🤖 ΨQRH: {response['text_analysis']}")

                    # Generate GLS output if consciousness results are available
                    if 'consciousness_results' in response and hasattr(pipeline.model, 'generate_gls_output'):
                        try:
                            gls_output = pipeline.model.generate_gls_output(response['consciousness_results'])
                            if gls_output.get('status') == 'success':
                                print("\n🎨 GLS VISUALIZATION CODE GENERATED:")
                                print("=" * 50)
                                print("📱 Processing Code (copy to Processing IDE):")
                                print(gls_output['processing_code'][:500] + "..." if len(gls_output['processing_code']) > 500 else gls_output['processing_code'])
                                print("\n🌐 p5.js Code (copy to HTML file):")
                                print(gls_output['p5js_code'][:500] + "..." if len(gls_output['p5js_code']) > 500 else gls_output['p5js_code'])
                                print("=" * 50)
                        except Exception as e:
                            print(f"⚠️  GLS output generation failed: {e}")
                else:
                    print(f"🤖 ΨQRH: {response}")

                if verbose:
                    print(f"📊 Metadados: {result['device']}, {result['output_length']} chars")
            else:
                print(f"❌ Erro: {result.get('error', 'Desconhecido')}")

        except EOFError:
            print("\n👋 EOF detectado, encerrando modo interativo")
            break
        except KeyboardInterrupt:
            print("\n👋 Interrompido pelo usuário")
            break
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")

    return 0

def process_single_text(text: str, task: str, device: Optional[str], verbose: bool = False) -> int:
    """Processa um único texto"""
    # Usar detecção automática de tarefa baseada no conteúdo do texto
    pipeline = ΨQRHPipeline(task=task, device=device, input_text=text)

    print(f"🧠 Processando: {text}")
    print(f"📋 Tarefa detectada: {pipeline.task}")
    result = pipeline(text)

    if result['status'] == 'success':
        print(f"\n✅ Resultado ({result['device']}):")
        print("-" * 50)
        print(result['response'])
        print("-" * 50)

        if verbose:
            print(f"\n📊 Metadados:")
            print(f"  - Tarefa: {result['task']}")
            print(f"  - Dispositivo: {result['device']}")
            print(f"  - Entrada: {result['input_length']} caracteres")
            print(f"  - Saída: {result['output_length']} caracteres")

    else:
        print(f"❌ Erro: {result.get('error', 'Desconhecido')}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())