#!/usr/bin/env python3
"""
Conversão Espectral de Modelos para ΨQRH
==========================================

Script standalone para converter modelos tradicionais (GPT-2, BERT, etc.)
para ΨQRH usando análise espectral física.

Usage:
    python3 convert_model_spectral.py --source gpt2 --output ./models/gpt2_psiqrh
    python3 convert_model_spectral.py --source ./path/to/model --output ./models/converted
    python3 convert_model_spectral.py --source bert-base-uncased --use-leech --validate-energy

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import argparse
import sys
import json
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.spectral_model_converter import SpectralModelConverter, save_conversion_report
from src.utils.spectral_weight_mapper import map_spectral_to_state_dict, validate_energy_preservation
from src.utils.embedding_spectral_converter import (
    convert_gpt2_embedding_to_psiqrh,
    save_psiqrh_embedding
)


def load_source_model(source: str, device: str = 'cpu'):
    """
    Carrega modelo da fonte (local apenas - sistema autônomo ΨQRH)

    Args:
        source: Fonte do modelo (path local)
        device: Dispositivo para carregar

    Returns:
        Modelo carregado
    """
    print(f"📦 Carregando modelo de: {source}")

    # Sistema autônomo ΨQRH: apenas carregamento local
    print("   Sistema autônomo ΨQRH - sem dependências externas")

    # Tentar carregar de arquivo local
    source_path = Path(source)
    if source_path.exists():
        print("   Tentando carregar de arquivo local...")

        # Se for diretório, procurar por pytorch_model.bin
        if source_path.is_dir():
            model_file = source_path / "pytorch_model.bin"
            if model_file.exists():
                print(f"   ✅ Encontrado: {model_file}")
                model_state = torch.load(model_file, map_location=device)
                # TODO: Reconstruir modelo baseado em config.json
                return model_state, None
        else:
            # Arquivo único
            print(f"   ✅ Carregando: {source_path}")
            model_state = torch.load(source_path, map_location=device)
            return model_state, None

    raise ValueError(f"❌ Não foi possível carregar modelo de: {source}")


def save_converted_model(
    converted_params: dict,
    output_dir: Path,
    source_info: dict
):
    """
    Salva modelo convertido em formato compatível com ΨQRH.

    Args:
        converted_params: Parâmetros convertidos
        output_dir: Diretório de saída
        source_info: Informações do modelo fonte
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salvar parâmetros convertidos
    params_file = output_dir / "converted_params.json"
    with open(params_file, 'w') as f:
        json.dump(converted_params, f, indent=2, default=str)
    print(f"✅ Parâmetros salvos: {params_file}")

    # Salvar configuração para ΨQRH
    config = {
        "model_type": "PsiQRHTransformerComplete",
        "source_model": source_info.get('model_type', 'unknown'),
        "framework": "ΨQRH",
        "version": "2.0.0",
        "conversion_method": "spectral_analysis",
        "avg_fractal_dim": converted_params.get('avg_fractal_dim', 1.5),
        "avg_alpha": converted_params.get('avg_alpha', 1.5),
        "n_layers_analyzed": converted_params.get('n_layers_analyzed', 0)
    }

    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Configuração salva: {config_file}")

    # Salvar relatório de conversão
    report_file = output_dir / "conversion_report.json"
    with open(report_file, 'w') as f:
        json.dump(converted_params, f, indent=2, default=str)
    print(f"✅ Relatório salvo: {report_file}")

    # ✅ ADICIONAR: Converter embedding layer espectralmente
    print("\n🔄 Convertendo embedding layer do GPT-2...")

    if 'source_model' in source_info and hasattr(source_info['source_model'], 'state_dict'):
        source_model = source_info['source_model']
        source_state_dict = source_model.state_dict()

        # 1. Converter embedding espectralmente
        # Procurar embedding layer (pode ser wte.weight, transformer.wte.weight, etc.)
        embedding_key = None
        for key in source_state_dict.keys():
            if 'wte.weight' in key or 'word_embeddings.weight' in key or 'embedding.weight' in key:
                embedding_key = key
                break

        if embedding_key:
            print(f"   • Encontrado embedding: {embedding_key}")
            gpt2_embedding = source_state_dict[embedding_key]
            print(f"   • Shape: {gpt2_embedding.shape}")

            # Converter para quaterniônico
            psi_embedding, embedding_metadata = convert_gpt2_embedding_to_psiqrh(
                gpt2_embedding,
                verbose=True
            )

            # Salvar embedding quaterniônico (sem tokenizer - sistema autônomo)
            save_psiqrh_embedding(
                psi_embedding,
                embedding_metadata,
                output_dir
            )

        else:
            print(f"   ⚠️  Embedding layer não encontrado no modelo")

        # 2. Mapear pesos usando transformações quaterniônicas
        print("\n💾 Mapeando pesos usando parâmetros espectrais...")

        psiqrh_state_dict = map_spectral_to_state_dict(
            source_state_dict,
            converted_params['converted_params']
        )

        # Substituir embedding clássico por quaterniônico
        if embedding_key and embedding_key in psiqrh_state_dict:
            # Flatten quaternion embedding [V, d/4, 4] → [V, d]
            psi_emb_flat = psi_embedding.reshape(psi_embedding.shape[0], -1)
            psiqrh_state_dict[embedding_key] = psi_emb_flat
            print(f"   ✅ Embedding quaterniônico inserido em {embedding_key}")

            # Weight tying: copiar para lm_head se existir
            lm_head_key = None
            for key in psiqrh_state_dict.keys():
                if 'lm_head.weight' in key or 'decoder.weight' in key:
                    lm_head_key = key
                    break

            if lm_head_key:
                psiqrh_state_dict[lm_head_key] = psi_emb_flat.clone()
                print(f"   ✅ Weight tying: {lm_head_key} compartilha embedding")

        # Validar preservação de energia
        validation = validate_energy_preservation(
            source_state_dict,
            psiqrh_state_dict,
            tolerance=0.1
        )

        # Salvar state_dict transformado
        state_dict_path = output_dir / "pytorch_model.bin"
        torch.save(psiqrh_state_dict, state_dict_path)
        print(f"\n✅ State dict mapeado salvo: {state_dict_path}")
        print(f"   Número de tensores: {len(psiqrh_state_dict)}")

        # Calcular tamanho
        total_params = sum(t.numel() for t in psiqrh_state_dict.values())
        total_size_mb = sum(t.element_size() * t.numel() for t in psiqrh_state_dict.values()) / (1024**2)
        print(f"   Total de parâmetros: {total_params:,}")
        print(f"   Tamanho: {total_size_mb:.2f} MB")
        print(f"   Razão de energia média: {validation['mean_energy_ratio']:.4f}")

        # Salvar metadados de validação
        validation_file = output_dir / "weight_mapping_validation.json"
        with open(validation_file, 'w') as f:
            json.dump(validation, f, indent=2)
        print(f"✅ Validação salva: {validation_file}")

    else:
        print("⚠️  Source model não disponível - state_dict não será salvo")
        print("   Apenas metadata espectral será salva")


def main():
    parser = argparse.ArgumentParser(
        description="Conversão Espectral de Modelos para ΨQRH",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Argumentos principais
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Fonte do modelo (nome HF, path local, URL)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Diretório de saída para modelo convertido"
    )

    # Parâmetros de conversão
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.1,
        help="Valor mínimo de α"
    )

    parser.add_argument(
        "--alpha-max",
        type=float,
        default=3.0,
        help="Valor máximo de α"
    )

    parser.add_argument(
        "--lambda-coupling",
        type=float,
        default=1.0,
        help="Constante de acoplamento λ"
    )

    parser.add_argument(
        "--use-leech",
        action="store_true",
        default=True,
        help="Usar correção topológica com Rede de Leech"
    )

    parser.add_argument(
        "--no-leech",
        dest="use_leech",
        action="store_false",
        help="Desabilitar correção de Leech"
    )

    parser.add_argument(
        "--validate-energy",
        action="store_true",
        default=True,
        help="Validar conservação de energia"
    )

    parser.add_argument(
        "--no-validate-energy",
        dest="validate_energy",
        action="store_false",
        help="Desabilitar validação energética"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo para processamento"
    )

    parser.add_argument(
        "--target-architecture",
        type=str,
        default="PsiQRHTransformerComplete",
        choices=["PsiQRHTransformer", "PsiQRHTransformerComplete"],
        help="Arquitetura alvo ΨQRH"
    )

    args = parser.parse_args()

    print("="*70)
    print("🚀 CONVERSÃO ESPECTRAL: Modelo → ΨQRH")
    print("="*70)
    print(f"📦 Fonte: {args.source}")
    print(f"📁 Saída: {args.output}")
    print(f"🎯 Arquitetura: {args.target_architecture}")
    print(f"🔧 Correção Leech: {'✅ Habilitada' if args.use_leech else '❌ Desabilitada'}")
    print(f"⚡ Validação Energia: {'✅ Habilitada' if args.validate_energy else '❌ Desabilitada'}")
    print("="*70)

    # Carregar modelo fonte
    try:
        source_model, tokenizer = load_source_model(args.source, args.device)
    except Exception as e:
        print(f"\n❌ ERRO ao carregar modelo: {e}")
        sys.exit(1)

    # Criar conversor
    print("\n🔧 Inicializando Spectral Converter...")
    converter = SpectralModelConverter(
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        lambda_coupling=args.lambda_coupling,
        use_leech_correction=args.use_leech,
        validate_energy=args.validate_energy
    )

    # Executar conversão
    print("\n🔬 Executando Conversão Física (5 passos)...")
    try:
        if isinstance(source_model, dict):
            # Se for state_dict, converter diretamente usando spectral analysis
            print("🔄 Convertendo state_dict usando análise espectral...")

            # Criar conversor
            converter = SpectralModelConverter(
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                lambda_coupling=args.lambda_coupling,
                use_leech_correction=args.use_leech,
                validate_energy=args.validate_energy
            )

            # Converter state_dict
            report = converter.convert_state_dict(
                source_model,
                target_architecture=args.target_architecture
            )
        else:
            # Modelo completo
            report = converter.convert_model(
                source_model,
                target_architecture=args.target_architecture
            )

    except Exception as e:
        print(f"\n❌ ERRO durante conversão: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Salvar modelo convertido
    print("\n💾 Salvando Modelo Convertido...")
    output_path = Path(args.output)

    try:
        source_info = {
            'model_type': source_model.__class__.__name__ if hasattr(source_model, '__class__') else 'unknown',
            'source': args.source,
            'source_model': source_model  # ← modelo fonte
        }

        save_converted_model(report, output_path, source_info)

    except Exception as e:
        print(f"\n❌ ERRO ao salvar: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Resumo final
    print("\n" + "="*70)
    print("✅ CONVERSÃO CONCLUÍDA COM SUCESSO!")
    print("="*70)
    print(f"📊 Dimensão Fractal Média: {report['avg_fractal_dim']:.4f}")
    print(f"⚡ Alpha Médio: {report['avg_alpha']:.4f}")
    print(f"📊 Camadas Convertidas: {report['n_layers_analyzed']}")
    print(f"📁 Modelo salvo em: {output_path}")
    print("="*70)

    print("\n💡 Próximos passos:")
    print(f"   1. Treinar: python3 train_psiqrh_native.py --output_dir {output_path} --use_complete")
    print(f"   2. Validar: python3 validate_training_output.py --model_dir {output_path}")
    print(f"   3. Certificar: make model-certify MODEL={output_path.name}")
    print(f"   4. Ativar: make model-set-active MODEL={output_path.name}")


if __name__ == "__main__":
    main()
