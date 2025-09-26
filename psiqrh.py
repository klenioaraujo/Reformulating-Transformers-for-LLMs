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

class ΨQRHPipeline:
    """Pipeline unificado para o framework ΨQRH - similar ao transformers.pipeline()"""

    def __init__(self, task: str = "text-generation", device: Optional[str] = None):
        """
        Inicializa o pipeline ΨQRH.

        Args:
            task: Tipo de tarefa (text-generation, analysis, chat)
            device: Dispositivo (cpu, cuda, mps) - detecta automaticamente se None
        """
        self.task = task
        self.device = self._detect_device(device)
        self.model = None
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

    def _initialize_model(self):
        """Inicializa o modelo ΨQRH automaticamente - ZERO FALLBACK POLICY"""
        print(f"🚀 Inicializando ΨQRH Pipeline no dispositivo: {self.device}")

        # Para geração de texto → use ΨQRH framework completo
        if self.task in ["text-generation", "chat"]:
            from src.core.ΨQRH import QRHFactory
            self.model = QRHFactory()
            print("✅ Framework ΨQRH completo carregado")

        # Para análise matemática → use o analisador espectral
        elif self.task == "analysis":
            from src.core.response_spectrum_analyzer import ResponseSpectrumAnalyzer
            self.model = ResponseSpectrumAnalyzer()
            print("✅ Analisador espectral ΨQRH carregado")

        else:
            raise ValueError(f"Tarefa não suportada: {self.task}")

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
        else:
            raise ValueError(f"Tarefa não implementada: {self.task}")

    def _generate_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Gera resposta usando o framework ΨQRH completo"""
        try:
            # Processar texto através do pipeline ΨQRH: texto → QRHLayer → saída
            processed_output = self.model.process_text(text, device=self.device)

            return {
                'status': 'success',
                'response': processed_output,
                'task': self.task,
                'device': self.device,
                'input_length': len(text),
                'output_length': len(processed_output)
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
        choices=['text-generation', 'chat', 'analysis'],
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

            print("🧠 ΨQRH processando...")
            result = pipeline(user_input)

            if result['status'] == 'success':
                print(f"🤖 ΨQRH: {result['response']}")
                if verbose:
                    print(f"📊 Metadados: {result['device']}, {result['output_length']} chars")
            else:
                print(f"❌ Erro: {result.get('error', 'Desconhecido')}")

        except KeyboardInterrupt:
            print("\n👋 Interrompido pelo usuário")
            break
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")

    return 0

def process_single_text(text: str, task: str, device: Optional[str], verbose: bool = False) -> int:
    """Processa um único texto"""
    pipeline = ΨQRHPipeline(task=task, device=device)

    print(f"🧠 Processando: {text}")
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