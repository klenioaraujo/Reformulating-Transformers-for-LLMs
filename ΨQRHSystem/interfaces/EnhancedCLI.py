#!/usr/bin/env python3
"""
Ponto de entrada principal para o Sistema ΨQRH, com uma CLI Aprimorada.
"""

import argparse
import sys
from pathlib import Path

# Adicionar o diretório pai ao sys.path para permitir importações relativas
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ΨQRHSystem.core.Orchestrator import Orchestrator

class EnhancedCLI:
    """
    Classe que encapsula a Interface de Linha de Comando Aprimorada.
    """
    def __init__(self):
        try:
            print("=======================================")
            print("    Sistema de Raciocínio Quântico   ")
            print("      ΨQRH - Arquitetura Modular     ")
            print("=======================================")
            self.orchestrator = Orchestrator()
        except ImportError as e:
            print(f"\n❌ ERRO CRÍTICO: Falha ao importar um componente essencial: {e}")
            print("   Verifique se todas as dependências em 'src/' e 'tools/' estão acessíveis.")
            print("   Pode ser necessário ajustar o PYTHONPATH.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ ERRO CRÍTICO: Falha ao inicializar o Orquestrador: {e}")
            sys.exit(1)

    def process_text(self, text: str):
        """Processa um único texto e imprime o resultado."""
        result = self.orchestrator.process(text)
        self.print_result(result)

    def print_result(self, result: dict):
        """Formata e imprime o resultado do processamento."""
        print("\n--- Resposta ΨQRH ---")
        print(f"{result['generated_text']}")
        print("---------------------")

    def run_interactive_mode(self):
        """Inicia o modo interativo."""
        print("\nEntrando em modo interativo. Digite 'sair' para terminar.")
        while True:
            try:
                input_text = input("\n> ")
                if input_text.lower() in ['sair', 'exit', 'quit']:
                    break
                if not input_text:
                    continue
                self.process_text(input_text)
            except KeyboardInterrupt:
                print("\nSaindo...")
                break
            except Exception as e:
                print(f"\nOcorreu um erro durante o processamento: {e}")

def main():
    """
    Função principal que lida com a CLI, principalmente para o modo interativo.
    """
    parser = argparse.ArgumentParser(
        description="ΨQRH CLI - Interface de Linha de Comando para o Sistema ΨQRH.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "text",
        nargs='?',
        default=None,
        help="O texto de entrada para processar. Se não for fornecido, entra em modo interativo."
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Força o modo interativo."
    )

    args = parser.parse_args()
    cli = EnhancedCLI()

    if args.interactive or args.text is None:
        cli.run_interactive_mode()
    else:
        cli.process_text(args.text)

if __name__ == "__main__":
    main()
