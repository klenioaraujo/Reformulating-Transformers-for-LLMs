#!/usr/bin/env python3
"""
ΨQRH CLI Interface - Interface de linha de comando

Interface unificada para o sistema ΨQRH modular.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, project_root)

try:
    from ΨQRHSystem.configs.SystemConfig import SystemConfig
    from ΨQRHSystem.core.PipelineManager import PipelineManager
except ImportError:
    # Fallback para imports relativos
    try:
        from ..configs.SystemConfig import SystemConfig
        from ..core.PipelineManager import PipelineManager
    except ImportError:
        # Último fallback - adicionar caminho absoluto
        sys.path.insert(0, os.path.dirname(project_root))
        from ΨQRHSystem.configs.SystemConfig import SystemConfig
        from ΨQRHSystem.core.PipelineManager import PipelineManager


class ΨQRHCLI:
    """
    Interface de linha de comando para o sistema ΨQRH
    """

    def __init__(self):
        """
        Inicializa CLI
        """
        self.config = None
        self.pipeline = None

    def load_config(self, config_path: Optional[str] = None) -> SystemConfig:
        """
        Carrega configuração do sistema

        Args:
            config_path: Caminho para arquivo de configuração

        Returns:
            Configuração carregada
        """
        if config_path is None:
            # Procurar configuração padrão
            default_paths = [
                "config.yaml",
                "configs/system_config.yaml",
                "../config.yaml",
                "../configs/system_config.yaml"
            ]

            for path in default_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        if config_path and os.path.exists(config_path):
            print(f"📁 Carregando configuração: {config_path}")
            self.config = SystemConfig.from_yaml(config_path)
        else:
            print("📁 Usando configuração padrão")
            self.config = SystemConfig()

        return self.config

    def initialize_pipeline(self):
        """
        Inicializa pipeline ΨQRH
        """
        if self.config is None:
            self.load_config()

        print("🚀 Inicializando pipeline ΨQRH...")
        self.pipeline = PipelineManager(self.config)
        print("✅ Pipeline ΨQRH pronto!")

    def process_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Processa texto através do pipeline usando Sistema DCF com vocabulário GPT-2

        Args:
            text: Texto para processar
            **kwargs: Parâmetros adicionais

        Returns:
            Resultado do processamento com vocabulário GPT-2 selecionado
        """
        if self.pipeline is None:
            self.initialize_pipeline()

        print(f"🧠 Processando: '{text[:50]}...'")
        print("🎯 Usando Sistema DCF com vocabulário GPT-2 selecionado (regra arquitetural)")

        result = self.pipeline.process(text)

        # Exibir resultado
        self.display_result(result)

        return result

    def display_result(self, result: Dict[str, Any]):
        """
        Exibe resultado do processamento

        Args:
            result: Resultado para exibir
        """
        print("\n" + "="*60)
        print("🎯 RESULTADO ΨQRH")
        print("="*60)

        # Texto gerado
        if 'text' in result:
            print(f"📝 Texto: {result['text']}")

        # Métricas físicas
        if 'fractal_dim' in result:
            print(f"🔬 Dimensão Fractal: {result['fractal_dim']:.3f}")

        if 'energy_conserved' in result:
            status = "✅ CONSERVADA" if result['energy_conserved'] else "❌ VIOLADA"
            print(f"⚡ Energia: {status}")

        # Validações
        if 'validation' in result:
            validation = result['validation']
            if validation.get('validation_passed', False):
                print("✅ Validações: APROVADAS")
            else:
                print("❌ Validações: FALHARAM")

        # Status do pipeline
        if 'pipeline_state' in result:
            state = result['pipeline_state']
            print(f"🔧 Pipeline: {'ATIVO' if state.get('initialized', False) else 'INATIVO'}")

        print("="*60)

    def get_system_info(self) -> Dict[str, Any]:
        """
        Retorna informações do sistema

        Returns:
            Informações do sistema
        """
        if self.pipeline is None:
            self.initialize_pipeline()

        return self.pipeline.get_pipeline_status()

    def run_interactive_mode(self):
        """
        Executa modo interativo
        """
        print("\n🤖 MODO INTERATIVO ΨQRH")
        print("Digite 'sair' para encerrar")
        print("-" * 40)

        while True:
            try:
                user_input = input("\nVocê: ").strip()

                if user_input.lower() in ['sair', 'quit', 'exit']:
                    print("👋 Até logo!")
                    break

                if not user_input:
                    continue

                # Processar entrada
                result = self.process_text(user_input)

            except KeyboardInterrupt:
                print("\n👋 Interrompido pelo usuário")
                break
            except Exception as e:
                print(f"❌ Erro: {e}")

    def run_batch_processing(self, input_file: str, output_file: Optional[str] = None):
        """
        Processa lote de textos

        Args:
            input_file: Arquivo de entrada com textos
            output_file: Arquivo de saída (opcional)
        """
        if not os.path.exists(input_file):
            print(f"❌ Arquivo não encontrado: {input_file}")
            return

        print(f"📁 Processando lote: {input_file}")

        # Carregar textos
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        results = []
        for i, text in enumerate(texts, 1):
            print(f"\n--- Processando {i}/{len(texts)} ---")
            result = self.process_text(text)
            results.append({
                'input': text,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })

        # Salvar resultados
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Resultados salvos em: {output_file}")
        else:
            # Nome de arquivo automático
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"psiqrh_batch_results_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Resultados salvos em: {output_file}")


def main():
    """
    Função principal da CLI
    """
    parser = argparse.ArgumentParser(
        description="ΨQRH CLI - Sistema Físico Quântico-Fractal-Óptico",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python -m ΨQRHSystem.interfaces.CLI "Explique quaternions"
  python -m ΨQRHSystem.interfaces.CLI --interactive
  python -m ΨQRHSystem.interfaces.CLI --batch input.txt --output results.json
  python -m ΨQRHSystem.interfaces.CLI --info
  python -m ΨQRHSystem.interfaces.CLI --config my_config.yaml "teste"
        """
    )

    parser.add_argument(
        'text',
        nargs='?',
        help='Texto para processar'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Arquivo de configuração YAML'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Modo interativo'
    )

    parser.add_argument(
        '--batch',
        type=str,
        help='Arquivo de entrada para processamento em lote'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Arquivo de saída para resultados em lote'
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Exibir informações do sistema'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Saída em formato JSON'
    )

    args = parser.parse_args()

    # Inicializar CLI
    cli = ΨQRHCLI()

    try:
        # Carregar configuração
        if args.config:
            cli.load_config(args.config)
        else:
            cli.load_config()

        # Modo info
        if args.info:
            info = cli.get_system_info()
            if args.json:
                print(json.dumps(info, indent=2))
            else:
                print("\n🔬 INFORMAÇÕES DO SISTEMA ΨQRH")
                print("=" * 40)
                for key, value in info.items():
                    print(f"{key}: {value}")
            return

        # Modo interativo
        if args.interactive:
            cli.run_interactive_mode()
            return

        # Processamento em lote
        if args.batch:
            cli.run_batch_processing(args.batch, args.output)
            return

        # Processamento único
        if args.text:
            result = cli.process_text(args.text)

            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            parser.print_help()

    except Exception as e:
        print(f"❌ Erro na CLI: {e}")
        if not args.json:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()