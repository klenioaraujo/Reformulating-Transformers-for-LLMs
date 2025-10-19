#!/usr/bin/env python3
"""
Configuração do Sistema ΨQRH com Vocabulário Semântico

Este script configura o ΨQRHSystem para usar vocabulário semântico e modelo
semântico, apresentando informações do modelo durante a execução.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    from configs.SystemConfig import SystemConfig
    from core.PipelineManager import PipelineManager
    from interfaces.CLI import ΨQRHCLI
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print("📁 Tentando imports relativos...")
    try:
        from .config.SystemConfig import SystemConfig
        from .core.PipelineManager import PipelineManager
        from .interfaces.CLI import ΨQRHCLI
    except ImportError:
        print("❌ Não foi possível importar os módulos do ΨQRHSystem")
        sys.exit(1)


class SemanticSystemConfigurator:
    """
    Configurador do Sistema Semântico ΨQRH

    Configura vocabulário semântico, modelo semântico e exibe informações
    durante a execução, seguindo o formato do sistema legado.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o configurador com configuração opcional

        Args:
            config_path: Caminho para arquivo de configuração YAML
        """
        self.config_path = config_path
        self.semantic_vocab = None
        self.model_info = None
        self.token_count = 0

    def load_semantic_vocabulary(self, vocab_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Carrega vocabulário semântico

        Args:
            vocab_path: Caminho para arquivo de vocabulário

        Returns:
            Vocabulário semântico carregado
        """
        if vocab_path is None:
            # Procurar arquivos de vocabulário padrão
            default_paths = [
                "data/native_vocab.json",
                "../data/native_vocab.json",
                "dynamic_quantum_vocabulary.json",
                "../dynamic_quantum_vocabulary.json"
            ]

            for path in default_paths:
                if os.path.exists(path):
                    vocab_path = path
                    break

        if vocab_path and os.path.exists(vocab_path):
            print(f"📚 Carregando vocabulário semântico: {vocab_path}")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.semantic_vocab = json.load(f)
                self.token_count = len(self.semantic_vocab.get('tokens', {}))
        else:
            # Vocabulário padrão semântico
            print("📚 Usando vocabulário semântico padrão")
            self.semantic_vocab = {
                'tokens': {
                    'quantum': 0, 'consciousness': 1, 'fractal': 2, 'energy': 3,
                    'harmonic': 4, 'resonance': 5, 'coherence': 6, 'entanglement': 7,
                    'dimension': 8, 'field': 9, 'wave': 10, 'particle': 11,
                    'probability': 12, 'state': 13, 'transformation': 14,
                    'optical': 15, 'spectral': 16, 'temporal': 17, 'spatial': 18,
                    'geometric': 19, 'processing': 20, 'completed': 21, 'result': 22
                },
                'metadata': {
                    'type': 'semantic',
                    'size': 23,
                    'description': 'Vocabulário semântico para processamento quântico'
                }
            }
            self.token_count = len(self.semantic_vocab['tokens'])

        return self.semantic_vocab

    def configure_semantic_model(self) -> Dict[str, Any]:
        """
        Configura modelo semântico com informações detalhadas

        Returns:
            Informações do modelo configurado
        """
        self.model_info = {
            'name': 'ΨQRH Semantic Model',
            'type': 'semantic_quantum',
            'vocab_size': self.token_count,
            'embed_dim': 64,
            'num_layers': 3,
            'num_heads': 8,
            'hidden_dim': 128,
            'max_history': 10,
            'device': 'cpu',
            'vocab_type': 'semantic',
            'token_count': self.token_count,
            'description': 'Modelo semântico quântico-fractal com vocabulário semântico'
        }

        return self.model_info

    def display_system_info(self):
        """
        Exibe informações do sistema no formato do legado
        """
        print("\n" + "="*60)
        print("🔬 SISTEMA ΨQRH CONFIGURADO")
        print("="*60)

        if self.model_info:
            print(f"🧠 Modelo: {self.model_info['name']}")
            print(f"📊 Tipo: {self.model_info['type']}")
            print(f"🔢 Vocabulário: {self.model_info['vocab_type']}")
            print(f"📈 Tokens: {self.model_info['token_count']}")
            print(f"📐 Dimensão: {self.model_info['embed_dim']}")
            print(f"🏗️  Camadas: {self.model_info['num_layers']}")
            print(f"🎯 Cabeças: {self.model_info['num_heads']}")
            print(f"💾 Dispositivo: {self.model_info['device']}")

        print("="*60)

    def create_semantic_cli(self) -> ΨQRHCLI:
        """
        Cria CLI com configuração semântica

        Returns:
            Instância do CLI configurada
        """
        # Carregar configuração
        cli = ΨQRHCLI()

        # Carregar configuração do arquivo ou usar padrão
        if self.config_path and os.path.exists(self.config_path):
            cli.load_config(self.config_path)
        else:
            cli.load_config()

        # Configurar vocabulário semântico
        self.load_semantic_vocabulary()

        # Configurar modelo semântico
        self.configure_semantic_model()

        # Exibir informações do sistema
        self.display_system_info()

        return cli

    def process_text_semantic(self, text: str) -> Dict[str, Any]:
        """
        Processa texto usando o sistema semântico configurado

        Args:
            text: Texto para processar

        Returns:
            Resultado do processamento
        """
        # Criar CLI semântica
        cli = self.create_semantic_cli()

        # Processar texto
        print(f"\n🧠 Processando: '{text[:50]}...'")

        result = cli.process_text(text)

        # Adicionar informações do modelo ao resultado
        if self.model_info:
            result['model_info'] = self.model_info
            result['semantic_vocab_size'] = self.token_count

        return result


def main():
    """
    Função principal para demonstração do sistema semântico
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Configurador do Sistema ΨQRH Semântico",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python configure_semantic_system.py "Olá mundo quântico"
  python configure_semantic_system.py --config ../config.yaml "Teste semântico"
  python configure_semantic_system.py --vocab data/native_vocab.json "Processamento"
        """
    )

    parser.add_argument(
        'text',
        nargs='?',
        help='Texto para processar semanticamente'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Arquivo de configuração YAML'
    )

    parser.add_argument(
        '--vocab',
        type=str,
        help='Arquivo de vocabulário semântico JSON'
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Exibir apenas informações do sistema'
    )

    args = parser.parse_args()

    # Inicializar configurador
    configurator = SemanticSystemConfigurator(args.config)

    try:
        if args.info:
            # Apenas exibir informações
            configurator.load_semantic_vocabulary(args.vocab)
            configurator.configure_semantic_model()
            configurator.display_system_info()
        elif args.text:
            # Processar texto
            result = configurator.process_text_semantic(args.text)

            # Exibir resultado adicional
            print(f"\n📊 Informações do Modelo Semântico:")
            print(f"   🔢 Tokens no vocabulário: {result.get('semantic_vocab_size', 0)}")
            print(f"   🧠 Tipo de modelo: {result.get('model_info', {}).get('type', 'N/A')}")
        else:
            parser.print_help()

    except Exception as e:
        print(f"❌ Erro no sistema semântico: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()