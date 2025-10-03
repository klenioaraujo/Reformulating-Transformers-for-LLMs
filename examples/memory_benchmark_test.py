#!/usr/bin/env python3
"""
ΨQRH Memory and Parameter Benchmark Tool
=========================================

Flexible command-line tool for benchmarking memory usage and parameter efficiency
of ΨQRH Transformer vs standard Transformer architectures.

Features:
- Command-line configurable model parameters
- Automatic device detection
- Dynamic comparative reporting
- Pure architecture-agnostic testing

Usage examples:
  python3 memory_benchmark_test.py
  python3 memory_benchmark_test.py --d_model 512 --n_layers 6
  python3 memory_benchmark_test.py --device cuda --batch_size 32
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import gc
import sys
import os
import argparse

# Adicionar o diretório src ao path para importar módulos locais
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def parse_arguments():
    """Parse command line arguments for flexible benchmarking."""
    parser = argparse.ArgumentParser(
        description="ΨQRH Memory and Parameter Benchmark Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model Architecture Arguments
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension (embedding size)')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--vocab_size', type=int, default=5000,
                       help='Vocabulary size')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                       help='Feed-forward dimension')

    # Test Configuration Arguments
    parser.add_argument('--seq_len', type=int, default=64,
                       help='Input sequence length')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for testing')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for testing (auto detects best option)')

    return parser.parse_args()


def setup_model(model_type: str, device: str, args) -> nn.Module:
    """
    Instancia um modelo (ΨQRH ou padrão) e move para o dispositivo de teste.

    Args:
        model_type: 'psiqrh', 'psiqrh_complete', 'rotational_psiqrh', ou 'standard'
        device: 'cpu' ou 'cuda'
        args: Command line arguments with model configuration

    Returns:
        Modelo instanciado no dispositivo
    """
    if model_type == 'psiqrh':
        try:
            from src.architecture.psiqrh_transformer import PsiQRHTransformer
            model = PsiQRHTransformer(
                vocab_size=args.vocab_size,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                max_seq_length=args.seq_len
            )
            print(f"✅ ΨQRH Transformer criado com sucesso")
            print(f"   Config: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
        except ImportError as e:
            print(f"❌ Erro ao importar ΨQRH Transformer: {e}")
            # Fallback para um modelo simplificado
            model = create_fallback_psiqrh_model(args.vocab_size, args.d_model, args.n_layers, args.n_heads)
            print(f"⚠️  Usando modelo fallback para ΨQRH")

    elif model_type == 'psiqrh_complete':
        try:
            from src.core.fractal_quantum_embedding import PsiQRHTransformerComplete
            model = PsiQRHTransformerComplete(
                vocab_size=args.vocab_size,
                embed_dim=min(args.d_model // 2, 128),  # Embed dim typically smaller
                quaternion_dim=4,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                n_rotations=4,
                dropout=0.1,
                max_seq_len=args.seq_len,
                use_leech_correction=False
            )
            print(f"✅ ΨQRH Transformer Complete criado com sucesso (Física Rigorosa)")
            print(f"   Config: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
            print(f"   Features: Fractal Embedding, Spectral Attention, SO(4), Optical Probe")
        except ImportError as e:
            print(f"❌ Erro ao importar ΨQRH Transformer Complete: {e}")
            # Fallback para modelo padrão
            from src.architecture.psiqrh_transformer import PsiQRHTransformer
            model = PsiQRHTransformer(
                vocab_size=args.vocab_size,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                max_seq_length=args.seq_len
            )
            print(f"⚠️  Usando ΨQRH padrão como fallback")

    elif model_type == 'standard':
        model = create_standard_transformer(args.vocab_size, args.d_model, args.n_layers, args.n_heads, args.dim_feedforward)
        print(f"✅ Standard Transformer criado com sucesso")
        print(f"   Config: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")

    elif model_type == 'rotational_psiqrh':
        try:
            from test_rotational_quaternion import create_rotational_psiqrh_transformer
            model = create_rotational_psiqrh_transformer(
                vocab_size=args.vocab_size,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                max_seq_length=args.seq_len
            )
            print(f"✅ Rotational ΨQRH Transformer criado com sucesso")
            print(f"   Config: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
        except ImportError as e:
            print(f"❌ Erro ao importar Rotational ΨQRH Transformer: {e}")
            # Fallback para modelo padrão
            model = create_standard_transformer(args.vocab_size, args.d_model, args.n_layers, args.n_heads, args.dim_feedforward)
            print(f"⚠️  Usando Standard Transformer como fallback")

    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

    # Mover para dispositivo
    model = model.to(device)
    model.eval()  # Modo de avaliação para consistência

    return model


def create_standard_transformer(vocab_size: int, d_model: int, n_layers: int, n_heads: int, dim_feedforward: int) -> nn.Module:
    """Cria um Transformer padrão para comparação."""

    class StandardTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers, n_heads, dim_feedforward):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)

            # Camadas de atenção multi-head
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                for _ in range(n_layers)
            ])

            # Feed-forward layers
            self.ff_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, d_model)
                ) for _ in range(n_layers)
            ])

            self.layer_norms_1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.layer_norms_2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

            self.output_layer = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.embedding(x)

            for i in range(len(self.attention_layers)):
                # Self-attention
                attn_out, _ = self.attention_layers[i](x, x, x)
                x = self.layer_norms_1[i](x + attn_out)

                # Feed-forward
                ff_out = self.ff_layers[i](x)
                x = self.layer_norms_2[i](x + ff_out)

            return self.output_layer(x)

    return StandardTransformer(vocab_size, d_model, n_layers, n_heads, dim_feedforward)


def create_fallback_psiqrh_model(vocab_size: int, d_model: int, n_layers: int, n_heads: int) -> nn.Module:
    """
    Cria um modelo ΨQRH simplificado como fallback.
    Esta implementação simula a estrutura básica do ΨQRH.
    """
    class FallbackPsiQRH(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers, n_heads):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)

            # Camadas de transformação quaterniônica simulada
            self.quaternion_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),  # Simula operações quaterniônicas
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model)
                ) for _ in range(n_layers)
            ])

            # Camadas de consciência fractal simulada
            self.consciousness_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.Tanh(),
                    nn.Linear(d_model * 2, d_model)
                ) for _ in range(n_layers // 2)
            ])

            self.output_layer = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.embedding(x)

            # Aplicar camadas quaterniônicas
            for layer in self.quaternion_layers:
                x = layer(x) + x  # Residual connection

            # Aplicar camadas de consciência
            for layer in self.consciousness_layers:
                x = layer(x) + x  # Residual connection

            return self.output_layer(x)

    return FallbackPsiQRH(vocab_size, d_model, n_layers, n_heads)


def analyze_model(model: nn.Module, device: str, batch_size: int, seq_len: int, vocab_size: int) -> Dict[str, float]:
    """
    Analisa parâmetros e uso de memória do modelo.

    Args:
        model: Modelo a ser analisado
        device: Dispositivo de teste
        batch_size: Tamanho do batch para teste
        seq_len: Comprimento da sequência para teste
        vocab_size: Tamanho do vocabulário

    Returns:
        Dicionário com métricas de análise
    """
    results = {}

    print("\n📊 ANÁLISE DO MODELO:")

    # 1. Contagem de parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"- Parâmetros Totais: {total_params:,}")
    print(f"- Parâmetros Treináveis: {trainable_params:,}")

    results['total_params'] = total_params
    results['trainable_params'] = trainable_params

    # 2. Medição de memória
    if device == 'cuda' and torch.cuda.is_available():
        # Medição GPU
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        # Criar tensor de entrada com índices válidos (range mais conservador)
        input_tensor = torch.randint(0, vocab_size - 100, (batch_size, seq_len), device=device)

        # Executar forward pass
        with torch.no_grad():
            model(input_tensor)

        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)

        print(f"- Pico de Memória (GPU): {peak_memory_mb:.2f} MB")
        results['peak_memory_mb'] = peak_memory_mb

    else:
        # Medição CPU - usar psutil como alternativa ao memory_profiler
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 ** 2)  # MB

            # Criar tensor de entrada com índices válidos (range mais conservador)
            input_tensor = torch.randint(0, vocab_size - 100, (batch_size, seq_len), device=device)

            # Executar forward pass
            with torch.no_grad():
                model(input_tensor)

            memory_after = process.memory_info().rss / (1024 ** 2)  # MB
            peak_memory_mb = memory_after - memory_before

            print(f"- Pico de Memória (CPU): {peak_memory_mb:.2f} MB")
            results['peak_memory_mb'] = peak_memory_mb

        except ImportError:
            print("⚠️  psutil não disponível - pulando medição de memória CPU")
            results['peak_memory_mb'] = 0.0

    # 3. Análise de camadas (opcional)
    layer_analysis = analyze_model_layers(model)
    results.update(layer_analysis)

    return results


def analyze_model_layers(model: nn.Module) -> Dict[str, Any]:
    """
    Analisa a distribuição de parâmetros por tipo de camada.
    """
    layer_stats = {}

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                layer_type = type(module).__name__
                layer_stats[f"layer_{layer_type}"] = layer_stats.get(f"layer_{layer_type}", 0) + params

    return layer_stats


def main():
    """
    Executa a matriz de testes (2 modelos x 2 dispositivos).
    """
    args = parse_arguments()

    # Auto-detect device if 'auto' is specified
    if args.device == 'auto':
        selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        selected_device = args.device

    print("🧠 ΨQRH MEMORY AND PARAMETER BENCHMARK TOOL")
    print("=" * 60)
    print(f"📋 Configuration:")
    print(f"   Model: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"   Test: batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f"   Device: {selected_device}")
    print("=" * 60)

    # Matriz de teste
    model_types = ['standard', 'psiqrh', 'psiqrh_complete', 'rotational_psiqrh']
    devices = [selected_device]

    results = {}

    for device in devices:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING ON DEVICE: {device.upper()}")
        print(f"{'='*60}")

        for model_type in model_types:
            print(f"\n🔍 MODEL: {model_type.upper()}")
            print("-" * 40)

            try:
                # Setup do modelo
                model = setup_model(model_type, device, args)

                # Análise
                model_results = analyze_model(model, device, args.batch_size, args.seq_len, args.vocab_size)
                results[f"{model_type}_{device}"] = model_results

                # Cleanup
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"❌ Erro ao testar {model_type} em {device}: {e}")
                import traceback
                traceback.print_exc()
                results[f"{model_type}_{device}"] = {'error': str(e)}

    # Relatório comparativo
    print("\n" + "="*60)
    print("📈 COMPARATIVE REPORT")
    print("="*60)

    generate_comparative_report(results)

    return results


def generate_comparative_report(results: Dict[str, Dict[str, float]]):
    """
    Gera relatório comparativo entre os modelos com métricas dinâmicas.
    """
    print("\n📊 EFFICIENCY COMPARISON:")

    # Encontrar resultados válidos
    valid_results = {}
    for key, result in results.items():
        if 'error' not in result and 'total_params' in result:
            valid_results[key] = result

    if len(valid_results) < 2:
        print("⚠️  Insufficient data for comparison")
        return

    # Calcular métricas comparativas
    standard_cpu = valid_results.get('standard_cpu')
    psiqrh_cpu = valid_results.get('psiqrh_cpu')
    psiqrh_complete_cpu = valid_results.get('psiqrh_complete_cpu')
    rotational_psiqrh_cpu = valid_results.get('rotational_psiqrh_cpu')
    standard_gpu = valid_results.get('standard_gpu')
    psiqrh_gpu = valid_results.get('psiqrh_gpu')
    psiqrh_complete_gpu = valid_results.get('psiqrh_complete_gpu')
    rotational_psiqrh_gpu = valid_results.get('rotational_psiqrh_gpu')

    def calculate_efficiency_metrics(standard, psiqrh):
        """Calculate efficiency metrics between standard and psiqrh models."""
        if not standard or not psiqrh:
            return None

        param_ratio = psiqrh['total_params'] / standard['total_params']
        param_change_percent = (1 - param_ratio) * 100

        memory_ratio = psiqrh['peak_memory_mb'] / standard['peak_memory_mb'] if standard['peak_memory_mb'] > 0 else 0
        memory_change_percent = (1 - memory_ratio) * 100 if memory_ratio > 0 else 0

        return {
            'param_ratio': param_ratio,
            'param_change_percent': param_change_percent,
            'memory_ratio': memory_ratio,
            'memory_change_percent': memory_change_percent
        }

    # CPU Metrics
    if standard_cpu and psiqrh_cpu:
        cpu_metrics = calculate_efficiency_metrics(standard_cpu, psiqrh_cpu)

        print(f"\n💻 CPU:")
        print(f"   Parameters: Standard={standard_cpu['total_params']:,} | ΨQRH={psiqrh_cpu['total_params']:,}")

        if cpu_metrics['param_change_percent'] > 0:
            print(f"   ✅ Parameter Efficiency: {cpu_metrics['param_change_percent']:.2f}% REDUCTION")
        else:
            print(f"   ❌ Parameter Inefficiency: {-cpu_metrics['param_change_percent']:.2f}% INCREASE")

        if cpu_metrics['memory_ratio'] > 0:
            if cpu_metrics['memory_change_percent'] > 0:
                print(f"   ✅ Memory Efficiency: {cpu_metrics['memory_change_percent']:.2f}% REDUCTION")
            else:
                print(f"   ❌ Memory Inefficiency: {-cpu_metrics['memory_change_percent']:.2f}% INCREASE")

    # ΨQRH Complete Comparison
    if standard_cpu and psiqrh_complete_cpu:
        complete_metrics = calculate_efficiency_metrics(standard_cpu, psiqrh_complete_cpu)

        print(f"\n🌟 ΨQRH Complete (Física Rigorosa):")
        print(f"   Parameters: Standard={standard_cpu['total_params']:,} | ΨQRH Complete={psiqrh_complete_cpu['total_params']:,}")

        if complete_metrics['param_change_percent'] > 0:
            print(f"   ✅ Parameter Efficiency: {complete_metrics['param_change_percent']:.2f}% REDUCTION")
        else:
            print(f"   ❌ Parameter Inefficiency: {-complete_metrics['param_change_percent']:.2f}% INCREASE")

        if complete_metrics['memory_ratio'] > 0:
            if complete_metrics['memory_change_percent'] > 0:
                print(f"   ✅ Memory Efficiency: {complete_metrics['memory_change_percent']:.2f}% REDUCTION")
            else:
                print(f"   ❌ Memory Inefficiency: {-complete_metrics['memory_change_percent']:.2f}% INCREASE")

    # Rotational ΨQRH Comparison
    if standard_cpu and rotational_psiqrh_cpu:
        rotational_metrics = calculate_efficiency_metrics(standard_cpu, rotational_psiqrh_cpu)

        print(f"\n🔄 Rotational ΨQRH:")
        print(f"   Parameters: Standard={standard_cpu['total_params']:,} | Rotational ΨQRH={rotational_psiqrh_cpu['total_params']:,}")

        if rotational_metrics['param_change_percent'] > 0:
            print(f"   ✅ Parameter Efficiency: {rotational_metrics['param_change_percent']:.2f}% REDUCTION")
        else:
            print(f"   ❌ Parameter Inefficiency: {-rotational_metrics['param_change_percent']:.2f}% INCREASE")

        if rotational_metrics['memory_ratio'] > 0:
            if rotational_metrics['memory_change_percent'] > 0:
                print(f"   ✅ Memory Efficiency: {rotational_metrics['memory_change_percent']:.2f}% REDUCTION")
            else:
                print(f"   ❌ Memory Inefficiency: {-rotational_metrics['memory_change_percent']:.2f}% INCREASE")

    # GPU Metrics
    if standard_gpu and psiqrh_gpu:
        gpu_metrics = calculate_efficiency_metrics(standard_gpu, psiqrh_gpu)

        print(f"\n🎮 GPU:")
        print(f"   Parameters: Standard={standard_gpu['total_params']:,} | ΨQRH={psiqrh_gpu['total_params']:,}")

        if gpu_metrics['param_change_percent'] > 0:
            print(f"   ✅ Parameter Efficiency: {gpu_metrics['param_change_percent']:.2f}% REDUCTION")
        else:
            print(f"   ❌ Parameter Inefficiency: {-gpu_metrics['param_change_percent']:.2f}% INCREASE")

        if gpu_metrics['memory_ratio'] > 0:
            if gpu_metrics['memory_change_percent'] > 0:
                print(f"   ✅ Memory Efficiency: {gpu_metrics['memory_change_percent']:.2f}% REDUCTION")
            else:
                print(f"   ❌ Memory Inefficiency: {-gpu_metrics['memory_change_percent']:.2f}% INCREASE")

    # Overall Assessment
    print(f"\n🎯 ASSESSMENT:")
    if standard_cpu and psiqrh_cpu:
        cpu_metrics = calculate_efficiency_metrics(standard_cpu, psiqrh_cpu)
        param_ratio = cpu_metrics['param_ratio']

        print(f"\n   Standard ΨQRH:")
        if param_ratio > 2.5:
            print(f"   ❌ CRITICAL INEFFICIENCY: ΨQRH has {param_ratio:.1f}x more parameters")
        elif param_ratio > 1.8:
            print(f"   ⚠️  MODERATE INEFFICIENCY: ΨQRH has {param_ratio:.1f}x more parameters")
        elif param_ratio > 1.3:
            print(f"   🔶 ACCEPTABLE EFFICIENCY: ΨQRH has {param_ratio:.1f}x more parameters")
        else:
            print(f"   ✅ EXCELLENT EFFICIENCY: ΨQRH has {param_ratio:.1f}x more parameters")

    if standard_cpu and psiqrh_complete_cpu:
        complete_metrics = calculate_efficiency_metrics(standard_cpu, psiqrh_complete_cpu)
        complete_param_ratio = complete_metrics['param_ratio']

        print(f"\n   ΨQRH Complete (Física Rigorosa):")
        if complete_param_ratio > 2.5:
            print(f"   ❌ CRITICAL INEFFICIENCY: ΨQRH Complete has {complete_param_ratio:.1f}x more parameters")
        elif complete_param_ratio > 1.8:
            print(f"   ⚠️  MODERATE INEFFICIENCY: ΨQRH Complete has {complete_param_ratio:.1f}x more parameters")
        elif complete_param_ratio > 1.3:
            print(f"   🔶 ACCEPTABLE EFFICIENCY: ΨQRH Complete has {complete_param_ratio:.1f}x more parameters")
        elif complete_param_ratio > 1.0:
            print(f"   🔶 GOOD EFFICIENCY: ΨQRH Complete has {complete_param_ratio:.1f}x more parameters")
        else:
            print(f"   ✅ EXCELLENT EFFICIENCY: ΨQRH Complete has {complete_param_ratio:.1f}x fewer parameters")

    if standard_cpu and rotational_psiqrh_cpu:
        rotational_metrics = calculate_efficiency_metrics(standard_cpu, rotational_psiqrh_cpu)
        rotational_param_ratio = rotational_metrics['param_ratio']

        print(f"\n   Rotational ΨQRH:")
        if rotational_param_ratio > 2.5:
            print(f"   ❌ CRITICAL INEFFICIENCY: Rotational ΨQRH has {rotational_param_ratio:.1f}x more parameters")
        elif rotational_param_ratio > 1.8:
            print(f"   ⚠️  MODERATE INEFFICIENCY: Rotational ΨQRH has {rotational_param_ratio:.1f}x more parameters")
        elif rotational_param_ratio > 1.3:
            print(f"   🔶 ACCEPTABLE EFFICIENCY: Rotational ΨQRH has {rotational_param_ratio:.1f}x more parameters")
        elif rotational_param_ratio > 1.0:
            print(f"   🔶 GOOD EFFICIENCY: Rotational ΨQRH has {rotational_param_ratio:.1f}x more parameters")
        else:
            print(f"   ✅ EXCELLENT EFFICIENCY: Rotational ΨQRH has {rotational_param_ratio:.1f}x fewer parameters")


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Benchmark concluído com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro durante o benchmark: {e}")
        import traceback
        traceback.print_exc()