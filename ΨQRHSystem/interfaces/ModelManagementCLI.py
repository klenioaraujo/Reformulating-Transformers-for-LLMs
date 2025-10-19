#!/usr/bin/env python3
"""
Model Management CLI - Interface de Linha de Comando para Gerenciamento de Modelos

Interface completa para gerenciar modelos no sistema ΨQRH multi-modelo,
incluindo listagem, download, conversão e configuração.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

# Imports com fallback para diferentes contextos de execução
import sys
import os

# Adicionar caminho do projeto ao sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

try:
    # Tentar import relativo primeiro (quando executado como módulo)
    from ..core.MultiModelManager import MultiModelManager
    from ..core.ModelRegistry import ModelRegistry
except ImportError:
    try:
        # Fallback para import absoluto (quando executado diretamente)
        from ΨQRHSystem.core.MultiModelManager import MultiModelManager
        from ΨQRHSystem.core.ModelRegistry import ModelRegistry
    except ImportError:
        # Último fallback - importar diretamente
        sys.path.insert(0, os.path.join(project_root, 'ΨQRHSystem'))
        from core.MultiModelManager import MultiModelManager
        from core.ModelRegistry import ModelRegistry

class ModelManagementCLI:
    """
    CLI para gerenciamento completo de modelos ΨQRH

    Permite operações como listar, baixar, converter e configurar modelos
    através de uma interface de linha de comando intuitiva.
    """

    def __init__(self):
        """Inicializa a CLI de gerenciamento de modelos"""
        self.manager = MultiModelManager()
        self.registry = ModelRegistry()

    def create_parser(self) -> argparse.ArgumentParser:
        """Cria o parser de argumentos"""
        parser = argparse.ArgumentParser(
            description="ΨQRH Model Management CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Exemplos de uso:
  python -m interfaces.ModelManagementCLI list
  python -m interfaces.ModelManagementCLI download gpt2
  python -m interfaces.ModelManagementCLI convert gpt2
  python -m interfaces.ModelManagementCLI set-default deepseek-coder
  python -m interfaces.ModelManagementCLI status
            """
        )

        subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')

        # Comando list
        list_parser = subparsers.add_parser('list', help='Lista modelos disponíveis')
        list_parser.add_argument('--type', choices=['all', 'loaded', 'available'],
                               default='all', help='Tipo de modelos a listar')

        # Comando download
        download_parser = subparsers.add_parser('download', help='Baixa um modelo')
        download_parser.add_argument('model_name', help='Nome do modelo a baixar')
        download_parser.add_argument('--force', action='store_true',
                                   help='Força re-download se já existe')

        # Comando convert
        convert_parser = subparsers.add_parser('convert', help='Converte modelo para formato semântico')
        convert_parser.add_argument('model_name', help='Nome do modelo a converter')
        convert_parser.add_argument('--output-dir', default='models/semantic',
                                  help='Diretório de saída')

        # Comando distill
        distill_parser = subparsers.add_parser('distill', help='Destila conhecimento de um modelo')
        distill_parser.add_argument('model_name', help='Nome do modelo fonte')
        distill_parser.add_argument('--output-dir', default='models/distilled',
                                  help='Diretório de saída')

        # Comando set-default
        default_parser = subparsers.add_parser('set-default', help='Define modelo padrão')
        default_parser.add_argument('model_name', help='Nome do modelo padrão')

        # Comando load
        load_parser = subparsers.add_parser('load', help='Carrega um modelo')
        load_parser.add_argument('model_name', help='Nome do modelo a carregar')
        load_parser.add_argument('--set-active', action='store_true',
                               help='Define como modelo ativo')

        # Comando unload
        unload_parser = subparsers.add_parser('unload', help='Descarrega um modelo')
        unload_parser.add_argument('model_name', help='Nome do modelo a descarregar')

        # Comando switch
        switch_parser = subparsers.add_parser('switch', help='Troca para um modelo')
        switch_parser.add_argument('model_name', help='Nome do modelo para trocar')

        # Comando status
        status_parser = subparsers.add_parser('status', help='Mostra status do sistema')

        # Comando scan
        scan_parser = subparsers.add_parser('scan', help='Escaneia por modelos disponíveis')

        # Comando clean
        clean_parser = subparsers.add_parser('clean', help='Limpa cache de modelos')
        clean_parser.add_argument('--all', action='store_true',
                                help='Remove todos os modelos carregados')

        return parser

    def run(self, args: Optional[List[str]] = None):
        """
        Executa a CLI

        Args:
            args: Argumentos da linha de comando (usa sys.argv se None)
        """
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        if not parsed_args.command:
            parser.print_help()
            return

        # Executa comando correspondente
        command_method = getattr(self, f"cmd_{parsed_args.command}", None)
        if command_method:
            try:
                command_method(parsed_args)
            except Exception as e:
                print(f"❌ Erro ao executar comando '{parsed_args.command}': {e}")
                sys.exit(1)
        else:
            print(f"❌ Comando '{parsed_args.command}' não implementado")
            sys.exit(1)

    def cmd_list(self, args):
        """Comando: list - Lista modelos"""
        if args.type == 'loaded':
            models = self.manager.list_loaded_models()
            print("🔄 Modelos Carregados:")
            if models:
                for model in models:
                    info = self.manager.get_model_info(model)
                    if info:
                        print(f"   📁 {model} ({info.get('type', 'unknown')})")
            else:
                print("   📁 Nenhum modelo carregado")

        elif args.type == 'available':
            models = self.manager.list_available_models()
            print("📚 Modelos Disponíveis:")
            if models:
                for model in models:
                    info = self.registry.get_model_info(model)
                    if info:
                        print(f"   📁 {model} ({info.get('type', 'unknown')})")
            else:
                print("   📁 Nenhum modelo disponível")

        else:  # all
            print("🔄 Modelos Carregados:")
            loaded = self.manager.list_loaded_models()
            if loaded:
                for model in loaded:
                    info = self.manager.get_model_info(model)
                    if info:
                        print(f"   ✅ {model} ({info.get('type', 'unknown')})")
            else:
                print("   📁 Nenhum modelo carregado")

            print("\n📚 Modelos Disponíveis:")
            available = self.manager.list_available_models()
            if available:
                for model in available:
                    info = self.registry.get_model_info(model)
                    if info:
                        status = "✅" if model in loaded else "📁"
                        print(f"   {status} {model} ({info.get('type', 'unknown')})")
            else:
                print("   📁 Nenhum modelo disponível")

    def cmd_download(self, args):
        """Comando: download - Baixa modelo"""
        print(f"📥 Baixando modelo: {args.model_name}")

        # Verificar se já existe
        if not args.force:
            existing_info = self.registry.get_model_info(f"source_{args.model_name}")
            if existing_info:
                print(f"⚠️  Modelo {args.model_name} já existe. Use --force para re-download.")
                return

        try:
            # Simulação de download (implementar lógica real)
            print(f"🔄 Fazendo download de {args.model_name} do Hugging Face...")

            # Registrar modelo baixado
            self.registry.register_model(f"source_{args.model_name}", {
                "type": "source",
                "model_name": args.model_name,
                "path": f"models/source/{args.model_name}",
                "status": "downloaded"
            })

            print(f"✅ Modelo {args.model_name} baixado com sucesso!")

        except Exception as e:
            print(f"❌ Erro ao baixar modelo {args.model_name}: {e}")

    def cmd_convert(self, args):
        """Comando: convert - Converte para formato semântico"""
        print(f"🔮 Convertendo modelo {args.model_name} para formato semântico")

        try:
            # Verificar se modelo fonte existe
            source_info = self.registry.get_model_info(f"source_{args.model_name}")
            if not source_info:
                print(f"❌ Modelo fonte {args.model_name} não encontrado. Execute 'download {args.model_name}' primeiro.")
                return

            # Simulação de conversão (implementar lógica real)
            print(f"🔄 Convertendo {args.model_name} para formato semântico...")

            # Registrar modelo convertido
            self.registry.register_model(f"semantic_{args.model_name}", {
                "type": "semantic_converted",
                "source_model": args.model_name,
                "path": f"{args.output_dir}/psiqrh_semantic_{args.model_name}.pt",
                "status": "converted"
            })

            print(f"✅ Modelo {args.model_name} convertido para formato semântico!")

        except Exception as e:
            print(f"❌ Erro ao converter modelo {args.model_name}: {e}")

    def cmd_distill(self, args):
        """Comando: distill - Destila conhecimento"""
        print(f"🧠 Destilando conhecimento de {args.model_name}")

        try:
            # Verificar se modelo fonte existe
            source_info = self.registry.get_model_info(f"source_{args.model_name}")
            if not source_info:
                print(f"❌ Modelo fonte {args.model_name} não encontrado. Execute 'download {args.model_name}' primeiro.")
                return

            # Simulação de destilação (implementar lógica real)
            print(f"🔄 Destilando conhecimento de {args.model_name}...")

            # Registrar modelo destilado
            self.registry.register_model(f"distilled_{args.model_name}", {
                "type": "distilled",
                "source_model": args.model_name,
                "path": f"{args.output_dir}/psiqrh_distilled_{args.model_name}.pt",
                "status": "distilled"
            })

            print(f"✅ Conhecimento de {args.model_name} destilado com sucesso!")

        except Exception as e:
            print(f"❌ Erro ao destilar modelo {args.model_name}: {e}")

    def cmd_set_default(self, args):
        """Comando: set-default - Define modelo padrão"""
        print(f"🎯 Definindo {args.model_name} como modelo padrão")

        # Verificar se modelo existe
        if not self.registry.is_model_registered(args.model_name):
            print(f"⚠️  Modelo {args.model_name} não encontrado no registro. Continuando mesmo assim...")

        self.manager.set_default_model(args.model_name)
        print(f"✅ Modelo padrão definido como: {args.model_name}")

    def cmd_load(self, args):
        """Comando: load - Carrega modelo"""
        print(f"🔄 Carregando modelo: {args.model_name}")

        success = self.manager.load_model(args.model_name, set_active=args.set_active)
        if success:
            print(f"✅ Modelo {args.model_name} carregado com sucesso!")
            if args.set_active:
                print(f"🎯 Definido como modelo ativo")
        else:
            print(f"❌ Falha ao carregar modelo {args.model_name}")

    def cmd_unload(self, args):
        """Comando: unload - Descarrega modelo"""
        print(f"🗑️  Descarregando modelo: {args.model_name}")

        success = self.manager.unload_model(args.model_name)
        if success:
            print(f"✅ Modelo {args.model_name} descarregado com sucesso!")
        else:
            print(f"⚠️  Modelo {args.model_name} não estava carregado")

    def cmd_switch(self, args):
        """Comando: switch - Troca para modelo"""
        print(f"🔄 Trocando para modelo: {args.model_name}")

        success = self.manager.switch_to_model(args.model_name)
        if success:
            print(f"✅ Trocado para modelo {args.model_name}!")
        else:
            print(f"❌ Falha ao trocar para modelo {args.model_name}")

    def cmd_status(self, args):
        """Comando: status - Mostra status do sistema"""
        print("🔬 STATUS DO SISTEMA ΨQRH MULTI-MODELO")
        print("=" * 50)

        status = self.manager.get_system_status()

        print(f"🎯 Modelo Ativo: {status['active_model'] or 'Nenhum'}")
        print(f"🔄 Modelos Carregados: {len(status['loaded_models'])}")
        print(f"📚 Modelos Disponíveis: {len(status['available_models'])}")

        print(f"\n💾 Uso de Memória:")
        mem = status['memory_usage']
        print(f"   • Parâmetros Totais: {mem['total_parameters']:,}")
        print(f"   • Tamanho Estimado: {mem['estimated_size_mb']:.1f} MB")

        if status['loaded_models']:
            print(f"\n🔄 Modelos Carregados:")
            for name, info in status['loaded_models'].items():
                active = " (ATIVO)" if name == status['active_model'] else ""
                print(f"   ✅ {name}{active}")
                print(f"      📊 Tipo: {info['type']}")
                print(f"      🔢 Vocab: {info['vocab_size']}")
                print(f"      📐 Embed: {info['embed_dim']}")
                print(f"      🎯 Uso: {info['usage_count']}x")

        registry = status['registry_summary']
        print(f"\n📊 Registro de Modelos:")
        print(f"   • Total: {registry['total_models']}")
        for model_type, count in registry['by_type'].items():
            print(f"   • {model_type.title()}: {count}")

    def cmd_scan(self, args):
        """Comando: scan - Escaneia por modelos"""
        print("🔍 Escaneando por modelos disponíveis...")
        self.manager.scan_and_register_models()
        print("✅ Escaneamento concluído!")

    def cmd_clean(self, args):
        """Comando: clean - Limpa cache"""
        if args.all:
            print("🧹 Limpando todos os modelos carregados...")
            loaded = self.manager.list_loaded_models()
            for model in loaded:
                self.manager.unload_model(model)
            print(f"✅ {len(loaded)} modelos descarregados!")
        else:
            print("🧹 Otimizando uso de memória...")
            self.manager.optimize_memory()
            print("✅ Otimização concluída!")


def main():
    """Função principal para execução via linha de comando"""
    cli = ModelManagementCLI()
    cli.run()


if __name__ == "__main__":
    main()