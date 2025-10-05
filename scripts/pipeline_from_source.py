#!/usr/bin/env python3
"""
ΨQRH Pipeline Orchestrator - Pipeline de Ponta a Ponta

Orquestra todo o processo de aquisição, conversão, treinamento e certificação de modelos.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
Project: https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs
"""

import argparse
import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Carregar configuração do diretório temporário
def get_temp_dir_config():
    """Carrega configuração do diretório temporário do arquivo de configuração."""
    try:
        import yaml
        config_path = project_root / "configs" / "qrh_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            temp_dir = config.get('pipeline_config', {}).get('temp_model_dir')
            if temp_dir:
                return Path(temp_dir)
    except Exception as e:
        print(f"⚠️  Erro ao carregar configuração, usando padrão: {e}")

    # Fallback para diretório no dispositivo com espaço (/dev/sda2)
    return Path("/media/padilha/2e63c2e1-be13-459f-883e-f364306d4d66/temp_models")

TEMP_BASE_DIR = get_temp_dir_config()


def log_step(step: str, message: str):
    """Log formatado para cada etapa do pipeline."""
    print(f"\n{'='*60}")
    print(f"🚀 {step}")
    print(f"{'='*60}")
    print(f"📝 {message}")
    print(f"{'='*60}")


def error_exit(message: str):
    """Saída de erro padronizada."""
    print(f"\n❌ ERRO: {message}")
    sys.exit(1)


def run_command(cmd: list, description: str):
    """Executa um comando com tratamento de erro."""
    print(f"\n▶️  Executando: {description}")
    print(f"   Comando: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ Saída: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erro: {e.stderr.strip() if e.stderr else 'Comando falhou'}")
        error_exit(f"Falha em: {description}")


def acquire_model(source: str) -> Path:
    """
    Adquire o modelo da fonte especificada.

    Args:
        source: Caminho local, ID do Hugging Face ou URL git

    Returns:
        Path: Diretório temporário com o modelo adquirido
    """
    log_step("ETAPA 1: Aquisição do Modelo", f"Fonte: {source}")

    # Usar diretório configurado ou fallback para tempfile
    try:
        if TEMP_BASE_DIR.exists():
            temp_dir = TEMP_BASE_DIR / f"psiqrh_source_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="psiqrh_source_"))
    except (PermissionError, OSError) as e:
        print(f"⚠️  Erro ao acessar diretório configurado, usando temp padrão: {e}")
        temp_dir = Path(tempfile.mkdtemp(prefix="psiqrh_source_"))
    print(f"📁 Diretório temporário: {temp_dir}")

    # Verificar se é um caminho local
    if Path(source).exists():
        print(f"📂 Usando modelo local: {source}")
        if Path(source).is_dir():
            # Copiar diretório inteiro
            shutil.copytree(source, temp_dir / "model")
            return temp_dir / "model"
        else:
            # Copiar arquivo único
            shutil.copy2(source, temp_dir / "model.pt")
            return temp_dir

    # Verificar se é um ID do Hugging Face
    if not source.startswith(('http://', 'https://', 'git@')):
        print(f"🤗 Tentando baixar do Hugging Face via URL: {source}")

        # Usar URL direta do Hugging Face para download via curl
        hf_url = f"https://huggingface.co/{source}/resolve/main/pytorch_model.bin"

        try:
            # Criar diretório temporário para download
            download_dir = temp_dir / "download"
            download_dir.mkdir(parents=True, exist_ok=True)

            download_path = download_dir / "pytorch_model.bin"

            # Baixar usando curl
            run_command(
                ["curl", "-L", "-o", str(download_path), hf_url],
                f"Baixando modelo do Hugging Face: {source}"
            )

            # Verificar se o download foi bem-sucedido
            if download_path.exists() and download_path.stat().st_size > 0:
                print(f"✅ Download concluído: {download_path}")

                # Tentar baixar config.json também
                config_url = f"https://huggingface.co/{source}/resolve/main/config.json"
                config_path = download_dir / "config.json"

                try:
                    run_command(
                        ["curl", "-L", "-o", str(config_path), config_url],
                        f"Baixando configuração do modelo"
                    )
                except Exception as e:
                    print(f"⚠️  Não foi possível baixar config.json: {e}")

                return download_dir
            else:
                print(f"⚠️  Download falhou, tentando como URL direta")
                # Continuar para tentar como URL

        except Exception as e:
            print(f"⚠️  Falha no download via curl: {e}")
            # Continuar para tentar como URL

    # Verificar se é uma URL (HTTP/HTTPS)
    if source.startswith(('http://', 'https://')):
        print(f"🌐 Baixando modelo via URL: {source}")
        try:
            # Criar diretório temporário para download
            download_dir = temp_dir / "download"
            download_dir.mkdir(parents=True, exist_ok=True)

            # Extrair nome do arquivo da URL
            filename = source.split('/')[-1]
            if not filename or '.' not in filename:
                filename = "model_download.bin"

            download_path = download_dir / filename

            # Baixar usando curl
            run_command(
                ["curl", "-L", "-o", str(download_path), source],
                f"Baixando modelo da URL: {source}"
            )

            # Verificar se o download foi bem-sucedido
            if download_path.exists() and download_path.stat().st_size > 0:
                print(f"✅ Download concluído: {download_path}")
                return download_dir
            else:
                error_exit(f"Download falhou ou arquivo vazio: {source}")

        except Exception as e:
            error_exit(f"Falha ao baixar da URL: {e}")

    # Verificar se é uma URL git
    if source.startswith('git@'):
        print(f"🔗 Clonando repositório git: {source}")
        try:
            run_command(
                ["git", "clone", source, str(temp_dir / "model")],
                f"Clonando repositório: {source}"
            )
            return temp_dir / "model"
        except Exception as e:
            error_exit(f"Falha ao clonar repositório: {e}")

    error_exit(f"Fonte não reconhecida ou inválida: {source}")


def convert_to_psiqrh_format(source_dir: Path, use_spectral: bool = True, model_name: str = None) -> Path:
    """
    Converte o modelo para o formato ΨQRH.

    Args:
        source_dir: Diretório com o modelo original
        use_spectral: Se True, usa conversão espectral física (padrão)
        model_name: Nome descritivo para o modelo (opcional)

    Returns:
        Path: Diretório do modelo convertido
    """
    log_step("ETAPA 2: Conversão para Formato ΨQRH", f"Origem: {source_dir}")

    # Criar nome para o modelo convertido
    if model_name:
        # Sanitizar o nome (remover caracteres especiais)
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
        # Remover underscores consecutivos e finais/iniciais
        safe_name = re.sub(r'_+', '_', safe_name)  # Múltiplos underscores → um único
        safe_name = safe_name.strip('_')  # Remover do início/fim
        final_model_name = f"psiqrh_{safe_name}"
    else:
        # Usar timestamp como fallback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_name = f"psiqrh_converted_{timestamp}"

    output_dir = project_root / "models" / final_model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Nome do modelo: {final_model_name}")
    print(f"📁 Diretório de saída: {output_dir}")

    # NOVO: Tentar conversão espectral física primeiro
    if use_spectral:
        converter_script = project_root / "scripts" / "convert_model_spectral.py"

        if converter_script.exists():
            print("🔬 Usando Conversão Espectral Física (análise de pesos)")
            print("   Método: Espectro de potência + Lei de potência + Correção de Leech")

            try:
                # Determinar source path (pode ser modelo HF ou diretório local)
                # Se source_dir contém pytorch_model.bin, usar diretório
                # Senão, tentar interpretar como nome HF
                if (source_dir / "pytorch_model.bin").exists():
                    source_arg = str(source_dir)
                else:
                    # Tentar extrair nome do modelo de config.json
                    config_path = source_dir / "config.json"
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            source_arg = config.get('_name_or_path', str(source_dir))
                        except:
                            source_arg = str(source_dir)
                    else:
                        source_arg = str(source_dir)

                result = subprocess.run(
                    [
                        "python3", str(converter_script),
                        "--source", source_arg,
                        "--output", str(output_dir),
                        "--use-leech",
                        "--validate-energy"
                    ],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    print(f"✅ Conversão Espectral concluída: {output_dir}")
                    print(result.stdout)
                    return output_dir
                else:
                    print(f"⚠️  Conversão Espectral falhou, tentando método genérico")
                    print(f"   Erro: {result.stderr}")

            except Exception as e:
                print(f"⚠️  Conversão Espectral falhou, tentando método genérico: {e}")
        else:
            print(f"⚠️  Conversor espectral não encontrado: {converter_script}")

    # Fallback: Método genérico (conversão simples sem física)
    print("🔄 Usando método genérico de conversão...")

    # Copiar todos os arquivos do diretório de origem
    for file in source_dir.glob("*"):
        if file.is_file():
            shutil.copy2(file, output_dir / file.name)
            print(f"  📄 Copiado: {file.name}")

    # Se houver subdiretório "download", copiar arquivos de lá também
    download_subdir = source_dir / "download"
    if download_subdir.exists():
        for file in download_subdir.glob("*"):
            if file.is_file():
                shutil.copy2(file, output_dir / file.name)
                print(f"  📄 Copiado (download): {file.name}")

    # Criar config.json básico se não existir
    config_path = output_dir / "config.json"
    if not config_path.exists():
        config = {
            "model_type": "psiqrh_converted",
            "source": str(source_dir),
            "converted_at": datetime.now().isoformat(),
            "alpha": 0.5,
            "normalization": "spectral",
            "conversion_method": "generic"
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ⚙️  Criado: config.json")

    print(f"✅ Modelo convertido (método genérico): {output_dir}")
    return output_dir


def train_model(model_dir: Path, use_complete: bool = True):
    """
    Executa treinamento/fine-tuning no modelo convertido.

    Args:
        model_dir: Diretório do modelo convertido
        use_complete: Se True, usa PsiQRHTransformerComplete (física rigorosa)
    """
    log_step("ETAPA 3: Treinamento/Fine-Tuning", f"Modelo: {model_dir}")

    # Priorizar train_psiqrh_native.py (mais moderno e suporta --use_complete)
    train_script = project_root / "train_psiqrh_native.py"

    if not train_script.exists():
        # Fallback para train_spectral.py
        train_script = project_root / "train_spectral.py"
        if not train_script.exists():
            print("⚠️  Script de treinamento não encontrado, pulando treinamento")
            return

    # Preparar comando de treinamento
    cmd = ["python3", str(train_script), "--output_dir", str(model_dir)]

    # Adicionar flag --use_complete se disponível
    if use_complete and train_script.name == "train_psiqrh_native.py":
        cmd.append("--use_complete")
        print("🌟 Usando PsiQRHTransformerComplete (Física Rigorosa)")
        print("   Features: Fractal Embedding, Spectral Attention, SO(4), Optical Probe")

    # Executar treinamento com parâmetros padrão
    try:
        run_command(
            cmd,
            f"Executando treinamento/fine-tuning"
        )
    except Exception as e:
        print(f"⚠️  Treinamento falhou, continuando sem treinamento: {e}")


def integrate_and_certify(model_dir: Path):
    """
    Integra o modelo com o sistema e executa certificação.

    Args:
        model_dir: Diretório do modelo treinado
    """
    log_step("ETAPA 4: Integração e Certificação", f"Modelo: {model_dir}")

    model_name = model_dir.name

    # Descobrir modelo
    run_command(
        ["make", "model-discover"],
        "Descobrindo modelos no sistema"
    )

    # Tentar certificar modelo (pode falhar se o modelo não for compatível)
    certification_success = False
    try:
        run_command(
            ["make", "model-certify", f"MODEL={model_name}"],
            f"Certificando modelo: {model_name}"
        )
        certification_success = True
        print("✅ Certificação bem-sucedida!")
    except Exception as e:
        print(f"⚠️  Certificação falhou: {e}")
        print("💡 O modelo não será ativado automaticamente.")

    # Ativar modelo SOMENTE se certificação for bem-sucedida
    if certification_success:
        print("\n🎯 Ativando modelo certificado automaticamente...")
        try:
            run_command(
                ["make", "model-set-active", f"MODEL={model_name}"],
                f"Ativando modelo: {model_name}"
            )
            print("✅ Modelo certificado e ativado com sucesso!")
        except Exception as e:
            print(f"⚠️  Ativação falhou: {e}")
    else:
        print("\n⚠️  Modelo NÃO foi ativado (certificação falhou)")
        print("💡 Para ativar manualmente: make model-set-active MODEL=" + model_name)


def start_chat_session():
    """Inicia a sessão de chat interativo."""
    log_step("ETAPA 5: Sessão de Chat Interativo", "Iniciando interface de chat")

    print("\n🎉 Pipeline concluído com sucesso!")
    print("💬 Iniciando sessão de chat com o modelo recém-preparado...")
    print("="*60)

    # Executar chat interativo mesmo com modelo não certificado
    try:
        subprocess.run(["python3", "psiqrh.py", "--interactive"], check=True)
    except subprocess.CalledProcessError:
        print("\n⚠️  Chat interativo falhou devido à certificação")
        print("💡 Tentando modo de teste de eco...")
        try:
            subprocess.run(["python3", "psiqrh.py", "--test-echo"], check=True)
        except subprocess.CalledProcessError:
            print("\n❌ Não foi possível iniciar sessão de chat")
            print("📋 Modelo foi preparado mas não está certificado")
            print("💡 Execute manualmente: python3 psiqrh.py --interactive")


def main():
    """Função principal do pipeline."""
    parser = argparse.ArgumentParser(
        description="ΨQRH Pipeline Orchestrator - Pipeline completo de ponta a ponta"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Fonte do modelo: caminho local, ID do Hugging Face ou URL git"
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nome descritivo para o modelo (ex: 'gpt2_qa', 'bert_sentiment'). Se não fornecido, usa timestamp."
    )

    parser.add_argument(
        "--use-complete",
        action="store_true",
        default=True,
        help="Usar PsiQRHTransformerComplete com física rigorosa (padrão: True)"
    )

    parser.add_argument(
        "--no-complete",
        dest="use_complete",
        action="store_false",
        help="Não usar implementação completa (usar PsiQRHTransformer padrão)"
    )

    parser.add_argument(
        "--use-spectral",
        action="store_true",
        default=True,
        help="Usar conversão espectral física (padrão: True)"
    )

    parser.add_argument(
        "--no-spectral",
        dest="use_spectral",
        action="store_false",
        help="Não usar conversão espectral (usar método genérico)"
    )

    args = parser.parse_args()

    print("🚀 ΨQRH Pipeline de Ponta a Ponta")
    print("="*60)
    print(f"📦 Fonte: {args.source}")
    if args.name:
        print(f"🏷️  Nome: {args.name}")
    if args.use_complete:
        print("🌟 Implementação: PsiQRHTransformerComplete (Física Rigorosa)")
    else:
        print("📋 Implementação: PsiQRHTransformer (Padrão)")
    if args.use_spectral:
        print("🔬 Conversão: Espectral (FFT + Lei de Potência + Leech)")
    else:
        print("🔄 Conversão: Genérica")
    print("="*60)
    print("Este processo é totalmente automatizado e pode levar um tempo considerável.")
    print("="*60)

    # Etapa 1: Aquisição
    source_dir = acquire_model(args.source)

    # Etapa 2: Conversão
    converted_dir = convert_to_psiqrh_format(source_dir, use_spectral=args.use_spectral, model_name=args.name)

    # Etapa 3: Treinamento
    train_model(converted_dir, use_complete=args.use_complete)

    # Etapa 4: Integração e Certificação
    integrate_and_certify(converted_dir)

    # Etapa 5: Chat Interativo
    start_chat_session()


if __name__ == "__main__":
    main()