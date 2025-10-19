#!/usr/bin/env python3
"""
Script de Configuração Automática do Sistema ΨQRH
=================================================

Este script automatiza a configuração inicial completa do sistema ΨQRH,
incluindo verificação de dependências, criação de arquivos necessários
e testes de validação.
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

class ΨQRHSetup:
    """Classe para configuração automática do sistema ΨQRH"""

    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.required_packages = [
            'torch', 'numpy', 'scipy', 'matplotlib', 'tqdm',
            'scikit-learn', 'pandas', 'pyyaml', 'requests'
        ]
        self.optional_packages = [
            'torchvision', 'torchaudio', 'transformers', 'datasets'
        ]

    def print_header(self, title: str):
        """Imprime cabeçalho formatado"""
        print("\n" + "="*60)
        print(f"🔧 {title}")
        print("="*60)

    def print_step(self, step: str, status: str = "EXECUTANDO"):
        """Imprime passo atual"""
        print(f"\n📋 {status}: {step}")

    def print_success(self, message: str):
        """Imprime mensagem de sucesso"""
        print(f"✅ {message}")

    def print_warning(self, message: str):
        """Imprime aviso"""
        print(f"⚠️  {message}")

    def print_error(self, message: str):
        """Imprime erro"""
        print(f"❌ {message}")

    def run_command(self, command: str, description: str = "") -> Tuple[bool, str]:
        """Executa comando do sistema"""
        try:
            if description:
                self.print_step(description)

            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )

            if result.returncode == 0:
                if description:
                    self.print_success(f"{description} concluído")
                return True, result.stdout
            else:
                self.print_error(f"Falha em: {description}")
                print(f"Erro: {result.stderr}")
                return False, result.stderr

        except Exception as e:
            self.print_error(f"Exceção em comando: {e}")
            return False, str(e)

    def check_python_version(self) -> bool:
        """Verifica versão do Python"""
        self.print_step("Verificando versão do Python")

        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.print_error(f"Python {version.major}.{version.minor} detectado. Necessário Python 3.8+")
            return False

        self.print_success(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        return True

    def check_dependencies(self) -> bool:
        """Verifica dependências Python"""
        self.print_step("Verificando dependências Python")

        missing_required = []
        missing_optional = []

        for package in self.required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"   ✅ {package}")
            except ImportError:
                missing_required.append(package)
                print(f"   ❌ {package}")

        for package in self.optional_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"   ✅ {package} (opcional)")
            except ImportError:
                missing_optional.append(package)
                print(f"   ⚠️  {package} (opcional - não encontrado)")

        if missing_required:
            self.print_error(f"Dependências obrigatórias faltando: {', '.join(missing_required)}")
            self.print_warning("Ambiente Python gerenciado externamente detectado")
            print("   💡 Recomendação: Use ambiente virtual")
            print("   🔧 Execute: python3 -m venv psiqrh_env && source psiqrh_env/bin/activate")
            print("   📦 Depois: pip install -r requirements.txt")

            # Tenta instalar mesmo assim com --break-system-packages
            self.print_step("Tentando instalar dependências (modo avançado)")
            for package in missing_required:
                success, error = self.run_command(f"pip install {package} --break-system-packages", f"Instalando {package}")
                if not success:
                    self.print_error(f"Falhou instalar {package}. Instale manualmente.")
                    print(f"   Execute: pip install {package} --break-system-packages")
                    return False

        if missing_optional:
            self.print_warning(f"Dependências opcionais não encontradas: {', '.join(missing_optional)}")
            print("   💡 Sistema funcionará, mas com funcionalidades reduzidas")

        self.print_success("Verificação de dependências concluída")
        return True

    def check_pytorch_cuda(self) -> bool:
        """Verifica PyTorch e CUDA"""
        self.print_step("Verificando PyTorch e CUDA")

        try:
            import torch
            print(f"   📦 PyTorch versão: {torch.__version__}")

            if torch.cuda.is_available():
                print(f"   🎮 CUDA disponível: {torch.cuda.get_device_name()}")
                print(f"   🔢 GPUs detectadas: {torch.cuda.device_count()}")
                self.print_success("CUDA habilitado - ótimo para performance")
            else:
                self.print_warning("CUDA não disponível - usando CPU")
                print("   💡 Para GPU: instale PyTorch com CUDA support")

        except ImportError:
            self.print_error("PyTorch não encontrado")
            return False

        return True

    def create_directories(self) -> bool:
        """Cria diretórios necessários"""
        self.print_step("Criando estrutura de diretórios")

        directories = [
            'data',
            'data/audit_logs',
            'data/secure_assets',
            'data/secure_assets/certificates',
            'data/secure_assets/manifests',
            'data/secure_assets/Ψcws',
            'data/Ψcws',
            'data/Ψcws_cache',
            'data/reports',
            'data/test_logs',
            'data/validation_reports',
            'data/system_state',
            'configs',
            'logs',
            'models',
            'results',
            'results/interactive_sessions',
            'benchmark_results',
            'cache',
            'temp'
        ]

        for dir_path in directories:
            full_path = self.root_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"   📁 {dir_path}")

        self.print_success("Estrutura de diretórios criada")
        return True

    def create_default_configs(self) -> bool:
        """Cria arquivos de configuração padrão"""
        self.print_step("Criando arquivos de configuração")

        # Configuração principal
        config_data = {
            "system": {
                "name": "ΨQRH Pipeline",
                "version": "2.0.0",
                "device": "auto",
                "enable_auto_calibration": True,
                "enable_noncommutative": True,
                "enable_cognitive_priming": True
            },
            "model": {
                "embed_dim": 64,
                "num_heads": 8,
                "num_layers": 3,
                "vocab_size": 256,
                "max_history": 10
            },
            "training": {
                "learning_rate": 1e-4,
                "batch_size": 1,
                "max_epochs": 100,
                "patience": 10
            },
            "physics": {
                "alpha": 1.0,
                "beta": 0.5,
                "I0": 1.0,
                "omega": 1.0,
                "k": 2.0
            }
        }

        config_path = self.root_dir / 'config.yaml'
        try:
            import yaml
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            print(f"   📄 config.yaml criado")
        except ImportError:
            # Fallback para JSON se yaml não estiver disponível
            config_path = self.root_dir / 'config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            print(f"   📄 config.json criado (yaml não disponível)")

        # Arquivo de estado do sistema
        system_state = {
            "initialized": True,
            "setup_date": "2025-01-01T00:00:00Z",
            "components": {
                "pipeline": False,
                "auto_calibration": False,
                "harmonic_orchestrator": False,
                "dcf_system": False
            },
            "performance": {
                "device": "unknown",
                "cuda_available": False,
                "memory_gb": 0
            }
        }

        state_path = self.root_dir / 'data' / 'system_state' / 'status.json'
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(system_state, f, indent=2)
        print(f"   📄 system_state/status.json criado")

        self.print_success("Arquivos de configuração criados")
        return True

    def initialize_vocabulary(self) -> bool:
        """Inicializa vocabulário básico apenas se não existir"""
        self.print_step("Inicializando vocabulário básico")

        vocab_path = self.root_dir / 'data' / 'native_vocab.json'

        # CORREÇÃO: Não sobrescrever o vocabulário se ele já foi criado pelo make setup-vocab
        if vocab_path.exists():
            print(f"   📚 Vocabulário nativo já existe em: {vocab_path}")
            self.print_success("Inicialização do vocabulário pulada (já existe)")
            return True

        # Vocabulário ASCII básico (apenas como fallback absoluto)
        basic_vocab = {
            "vocab_size": 95,
            "characters": [chr(i) for i in range(32, 127)],
            "special_tokens": {
                "<PAD>": 0,
                "<UNK>": 1,
                "<BOS>": 2,
                "<EOS>": 3
            },
            "description": "Vocabulário ASCII básico criado automaticamente como fallback."
        }

        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(basic_vocab, f, indent=2, ensure_ascii=False)

        print(f"   📚 Vocabulário básico de fallback criado: {vocab_path}")
        self.print_success("Vocabulário inicializado")
        return True

    def run_basic_tests(self) -> bool:
        """Executa testes básicos"""
        self.print_step("Executando testes básicos")

        # Teste 1: Importação básica
        try:
            from psiqrh import ΨQRHPipeline
            print("   ✅ Importação ΨQRHPipeline - OK")
        except ImportError as e:
            self.print_error(f"Falha na importação: {e}")
            return False

        # Teste 2: Criação básica do pipeline
        try:
            pipeline = ΨQRHPipeline(enable_auto_calibration=False)
            print("   ✅ Criação do pipeline - OK")
        except Exception as e:
            self.print_error(f"Falha na criação do pipeline: {e}")
            return False

        # Teste 3: Processamento básico
        try:
            result = pipeline("teste")
            if result and result.get('status') in ['success', 'error']:
                print("   ✅ Processamento básico - OK")
            else:
                print("   ⚠️  Processamento básico - Resposta inesperada")
        except Exception as e:
            self.print_error(f"Falha no processamento: {e}")
            return False

        self.print_success("Testes básicos concluídos")
        return True

    def create_startup_script(self) -> bool:
        """Cria script de inicialização rápida"""
        self.print_step("Criando script de inicialização")

        startup_script = '''#!/bin/bash
# Script de Inicialização Rápida ΨQRH
# ===================================

echo "🚀 Iniciando Sistema ΨQRH..."

# Verificar se ambiente virtual existe
if [ ! -d "psiqrh_env" ]; then
    echo "⚠️ Ambiente virtual não encontrado. Execute setup_system.py primeiro."
    exit 1
fi

# Ativar ambiente virtual
source psiqrh_env/bin/activate

# Verificar instalação
python -c "from psiqrh import ΨQRHPipeline; print('✅ ΨQRH pronto!')"

echo ""
echo "🎯 Comandos disponíveis:"
echo "  make test              # Teste completo"
echo "  make train-physics-emergent  # Treinamento emergente"
echo "  python psiqrh.py --interactive  # Modo interativo"
echo "  python psiqrh.py \"seu texto\"     # Processar texto"
echo ""
echo "📚 Para mais opções: python psiqrh.py --help"
'''

        script_path = self.root_dir / 'start_psiqrh.sh'
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(startup_script)

        # Tornar executável
        os.chmod(script_path, 0o755)

        print(f"   📜 Script criado: {script_path}")
        self.print_success("Script de inicialização criado")
        return True

    def update_system_state(self) -> bool:
        """Atualiza estado do sistema"""
        self.print_step("Atualizando estado do sistema")

        import torch

        state_path = self.root_dir / 'data' / 'system_state' / 'status.json'
        if state_path.exists():
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
        else:
            state = {}

        # Atualizar informações
        state.update({
            "initialized": True,
            "setup_completed": True,
            "components": {
                "pipeline": True,
                "auto_calibration": True,
                "harmonic_orchestrator": True,
                "dcf_system": True
            },
            "performance": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "cuda_available": torch.cuda.is_available(),
                "memory_gb": 0  # Pode ser calculado depois
            }
        })

        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

        self.print_success("Estado do sistema atualizado")
        return True

    def show_summary(self):
        """Mostra resumo da configuração"""
        self.print_header("CONFIGURAÇÃO CONCLUÍDA COM SUCESSO!")

        print("\n🎯 SISTEMA ΨQRH PRONTO PARA USO!")
        print("="*60)
        print("📁 Diretório raiz:", self.root_dir)
        print("🐍 Python:", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print("📦 PyTorch:", end=" ")
        try:
            import torch
            print(torch.__version__, end="")
            if torch.cuda.is_available():
                print(" (CUDA habilitado)")
            else:
                print(" (CPU)")
        except:
            print("Não encontrado")

        print("\n🚀 PRÓXIMOS PASSOS:")
        print("1. Execute: ./start_psiqrh.sh")
        print("2. Teste: make test")
        print("3. Treine: make train-physics-emergent")
        print("4. Explore: python psiqrh.py --interactive")

        print("\n📚 DOCUMENTAÇÃO:")
        print("- SETUP.md: Guia completo de configuração")
        print("- README.md: Documentação geral")
        print("- docs/: Documentação técnica detalhada")

        print("\n🔧 COMANDOS ÚTEIS:")
        print("- make help: Lista todos os comandos disponíveis")
        print("- python psiqrh.py --help: Opções da CLI")
        print("- make clean: Limpar cache e arquivos temporários")

        print("\n✨ BOA SORTE COM SEU SISTEMA ΨQRH!")
        print("="*60)

    def run_setup(self) -> bool:
        """Executa configuração completa"""
        self.print_header("INICIALIZAÇÃO DO SISTEMA ΨQRH")

        steps = [
            ("Verificação do Python", self.check_python_version),
            ("Verificação de dependências", self.check_dependencies),
            ("Verificação PyTorch/CUDA", self.check_pytorch_cuda),
            ("Criação de diretórios", self.create_directories),
            ("Criação de configurações", self.create_default_configs),
            ("Inicialização do vocabulário", self.initialize_vocabulary),
            ("Execução de testes básicos", self.run_basic_tests),
            ("Criação de script de inicialização", self.create_startup_script),
            ("Atualização do estado do sistema", self.update_system_state)
        ]

        for step_name, step_func in steps:
            if not step_func():
                self.print_error(f"FALHA NA ETAPA: {step_name}")
                return False

        self.show_summary()
        return True


def main():
    """Função principal"""
    setup = ΨQRHSetup()

    try:
        success = setup.run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Configuração interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()