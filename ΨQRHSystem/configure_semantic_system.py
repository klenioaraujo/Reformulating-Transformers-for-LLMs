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

# Adicionar diretório raiz do projeto ao path para acessar quantum_word_matrix
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:
    from configs.SystemConfig import SystemConfig
    from core.PipelineManager import PipelineManager
    from interfaces.CLI import ΨQRHCLI
    from quantum_word_matrix import QuantumWordMatrix
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print("📁 Tentando imports relativos...")
    try:
        from .config.SystemConfig import SystemConfig
        from .core.PipelineManager import PipelineManager
        from .interfaces.CLI import ΨQRHCLI
        from quantum_word_matrix import QuantumWordMatrix
    except ImportError:
        print("❌ Não foi possível importar os módulos do ΨQRHSystem")
        sys.exit(1)


class SemanticSystemConfigurator:
    """
    Configurador do Sistema Semântico ΨQRH

    Configura vocabulário semântico, modelo semântico e exibe informações
    durante a execução, seguindo o formato do sistema legado.
    """

    def __init__(self, config_path: Optional[str] = None, vocab_path: Optional[str] = None):
        """
        Inicializa o configurador com configuração opcional

        Args:
            config_path: Caminho para arquivo de configuração YAML
            vocab_path: Caminho para arquivo de vocabulário (suporta qualquer vocab via Makefile)
        """
        self.config_path = config_path
        self.vocab_path = vocab_path
        self.semantic_vocab = None
        self.model_info = None
        self.token_count = 0
        self.quantum_word_matrix = None

    def load_semantic_vocabulary(self, vocab_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Carrega vocabulário semântico - agora suporta qualquer vocabulário via quantum_word_matrix

        Args:
            vocab_path: Caminho para arquivo de vocabulário (suporta qualquer vocab)

        Returns:
            Vocabulário semântico carregado
        """
        # Usar vocab_path da instância se não fornecido
        if vocab_path is None:
            vocab_path = self.vocab_path

        # Procurar arquivos de vocabulário padrão (multi-vocab support)
        if vocab_path is None:
            # Verificar variável de ambiente do Makefile
            vocab_path = os.environ.get('SEMANTIC_VOCAB_PATH')

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
                vocab_data = json.load(f)

            # Suporte para diferentes formatos de vocabulário
            if 'token_to_id' in vocab_data and 'id_to_token' in vocab_data:
                # Formato GPT-2 nativo
                word_to_id = vocab_data['token_to_id']
                id_to_word = vocab_data['id_to_token']
                vocab_size = vocab_data.get('vocab_size', len(word_to_id))
                model_name = vocab_data.get('model_name', 'unknown')
            elif 'tokens' in vocab_data:
                # Formato legado semântico
                word_to_id = vocab_data['tokens']
                id_to_word = {v: k for k, v in word_to_id.items()}
                vocab_size = len(word_to_id)
                model_name = vocab_data.get('metadata', {}).get('type', 'semantic')
            else:
                raise ValueError(f"Formato de vocabulário não suportado em {vocab_path}")

            # Criar QuantumWordMatrix com o vocabulário carregado
            embed_dim = 64  # Pode ser configurável via Makefile
            device = 'cpu'  # Pode ser configurável

            self.quantum_word_matrix = QuantumWordMatrix(
                embed_dim=embed_dim,
                device=device,
                word_to_id=word_to_id,
                id_to_word=id_to_word
            )

            self.semantic_vocab = {
                'word_to_id': word_to_id,
                'id_to_word': id_to_word,
                'vocab_size': vocab_size,
                'model_name': model_name,
                'metadata': {
                    'type': 'multi_vocab_quantum',
                    'size': vocab_size,
                    'description': f'Vocabulário quântico multi-modelo de {model_name}',
                    'source': vocab_path
                }
            }
            self.token_count = vocab_size

        else:
            # Fallback: Vocabulário padrão semântico
            print("📚 Usando vocabulário semântico padrão (fallback)")
            word_to_id = {
                'quantum': 0, 'consciousness': 1, 'fractal': 2, 'energy': 3,
                'harmonic': 4, 'resonance': 5, 'coherence': 6, 'entanglement': 7,
                'dimension': 8, 'field': 9, 'wave': 10, 'particle': 11,
                'probability': 12, 'state': 13, 'transformation': 14,
                'optical': 15, 'spectral': 16, 'temporal': 17, 'spatial': 18,
                'geometric': 19, 'processing': 20, 'completed': 21, 'result': 22
            }
            id_to_word = {v: k for k, v in word_to_id.items()}

            self.quantum_word_matrix = QuantumWordMatrix(
                embed_dim=64,
                device='cpu',
                word_to_id=word_to_id,
                id_to_word=id_to_word
            )

            self.semantic_vocab = {
                'word_to_id': word_to_id,
                'id_to_word': id_to_word,
                'vocab_size': len(word_to_id),
                'model_name': 'semantic_fallback',
                'metadata': {
                    'type': 'semantic_fallback',
                    'size': len(word_to_id),
                    'description': 'Vocabulário semântico de fallback'
                }
            }
            self.token_count = len(word_to_id)

        return self.semantic_vocab

    def configure_semantic_model(self) -> Dict[str, Any]:
        """
        Configura modelo semântico com QuantumWordMatrix e multi-vocab support

        Returns:
            Informações do modelo configurado
        """
        vocab_name = self.semantic_vocab.get('model_name', 'unknown') if self.semantic_vocab else 'unknown'

        self.model_info = {
            'name': f'ΨQRH Multi-Vocab Semantic Model ({vocab_name})',
            'type': 'multi_vocab_quantum_semantic',
            'vocab_size': self.token_count,
            'embed_dim': 64,
            'num_layers': 3,
            'num_heads': 8,
            'hidden_dim': 128,
            'max_history': 10,
            'device': 'cpu',
            'vocab_type': 'multi_vocab_quantum',
            'token_count': self.token_count,
            'vocab_name': vocab_name,
            'quantum_word_matrix': self.quantum_word_matrix is not None,
            'description': f'Modelo semântico quântico multi-vocab com {self.token_count} tokens de {vocab_name}'
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
            print(f"🔮 Quantum Word Matrix: {'✅ Ativo' if self.model_info.get('quantum_word_matrix') else '❌ Inativo'}")
            if self.semantic_vocab and 'metadata' in self.semantic_vocab:
                print(f"📚 Fonte do Vocab: {self.semantic_vocab['metadata'].get('source', 'N/A')}")

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
            result['quantum_word_matrix'] = self.quantum_word_matrix is not None
            result['vocab_metadata'] = self.semantic_vocab.get('metadata', {}) if self.semantic_vocab else {}

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
        help='Arquivo de vocabulário (suporta qualquer vocab: GPT-2, semantic, etc.)'
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
            print(f"\n📊 Informações do Modelo Semântico Multi-Vocab:")
            print(f"   🔢 Tokens no vocabulário: {result.get('semantic_vocab_size', 0)}")
            print(f"   🧠 Tipo de modelo: {result.get('model_info', {}).get('type', 'N/A')}")
            print(f"   🔮 Quantum Word Matrix: {'✅ Ativo' if result.get('quantum_word_matrix') else '❌ Inativo'}")
            vocab_meta = result.get('vocab_metadata', {})
            if vocab_meta:
                print(f"   📚 Fonte do Vocab: {vocab_meta.get('source', 'N/A')}")
                print(f"   🏷️  Tipo do Vocab: {vocab_meta.get('type', 'N/A')}")
        else:
            parser.print_help()

    except Exception as e:
        print(f"❌ Erro no sistema semântico: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()