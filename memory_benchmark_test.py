#!/usr/bin/env python3
"""
Memory Benchmark Test - Diagnóstico de Eficiência do ΨQRH Transformer
====================================================================

Script para quantificar e identificar a origem do consumo excessivo de memória
e parâmetros do ΨQRH Transformer. Compara sistematicamente o modelo ΨQRH
contra um Transformer padrão em ambientes de CPU e GPU.

Problema identificado: ΨQRH consome ~388% mais memória e tem ~5x mais
parâmetros (215M vs 44M) que o baseline.

Hipótese: A ineficiência é significativamente pior em CPU do que em GPU.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import gc
import sys
import os

# Adicionar o diretório src ao path para importar módulos locais
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def setup_model(model_type: str, device: str) -> nn.Module:
    """
    Instancia um modelo (ΨQRH ou padrão) e move para o dispositivo de teste.

    Args:
        model_type: 'psiqrh' ou 'standard'
        device: 'cpu' ou 'cuda'

    Returns:
        Modelo instanciado no dispositivo
    """
    # Configuração otimizada para CPU
    vocab_size = 5000   # Vocabulário menor
    d_model = 256       # Dimensão menor
    n_layers = 4        # Menos camadas
    n_heads = 4         # Menos heads
    seq_len = 64        # Sequência menor

    if model_type == 'psiqrh':
        try:
            from src.architecture.psiqrh_transformer import PsiQRHTransformer
            model = PsiQRHTransformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                max_seq_length=seq_len
            )
            print(f"✅ ΨQRH Transformer criado com sucesso")
        except ImportError as e:
            print(f"❌ Erro ao importar ΨQRH Transformer: {e}")
            # Fallback para um modelo simplificado
            model = create_fallback_psiqrh_model(vocab_size, d_model, n_layers, n_heads)
            print(f"⚠️  Usando modelo fallback para ΨQRH")

    elif model_type == 'standard':
        model = create_standard_transformer(vocab_size, d_model, n_layers, n_heads)
        print(f"✅ Standard Transformer criado com sucesso")

    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

    # Mover para dispositivo
    model = model.to(device)
    model.eval()  # Modo de avaliação para consistência

    return model


def create_standard_transformer(vocab_size: int, d_model: int, n_layers: int, n_heads: int) -> nn.Module:
    """Cria um Transformer padrão para comparação."""

    class StandardTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers, n_heads):
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
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model)
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

    return StandardTransformer(vocab_size, d_model, n_layers, n_heads)


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


def analyze_model(model: nn.Module, device: str, batch_size: int = 8, seq_len: int = 128) -> Dict[str, float]:
    """
    Analisa parâmetros e uso de memória do modelo.

    Args:
        model: Modelo a ser analisado
        device: Dispositivo de teste
        batch_size: Tamanho do batch para teste
        seq_len: Comprimento da sequência para teste

    Returns:
        Dicionário com métricas de análise
    """
    # Usar o vocab_size correto do modelo
    vocab_size = getattr(model, 'vocab_size', 5000)
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
        # Usar o vocab_size correto do modelo
        model_vocab_size = getattr(model, 'vocab_size', vocab_size)
        input_tensor = torch.randint(0, model_vocab_size - 1, (batch_size, seq_len), device=device)

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
            # Usar o vocab_size correto do modelo
            model_vocab_size = getattr(model, 'vocab_size', vocab_size)
            input_tensor = torch.randint(0, model_vocab_size - 1, (batch_size, seq_len), device=device)

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
    print("🧠 MEMORY BENCHMARK TEST - ΨQRH vs Standard Transformer")
    print("=" * 60)

    # Configurações de teste otimizadas para CPU
    batch_size = 4  # Batch menor para evitar problemas de memória
    seq_len = 64    # Sequência menor

    # Matriz de teste
    model_types = ['standard', 'psiqrh']
    devices = ['cpu']

    print(f"⚠️  CUDA não disponível - testando apenas CPU com configuração otimizada")
    print(f"   Batch size: {batch_size}, Seq len: {seq_len}")

    results = {}

    for device in devices:
        print(f"\n{'='*60}")
        print(f"🧪 TESTANDO NO DISPOSITIVO: {device.upper()}")
        print(f"{'='*60}")

        for model_type in model_types:
            print(f"\n🔍 MODELO: {model_type.upper()}")
            print("-" * 40)

            try:
                # Setup do modelo
                model = setup_model(model_type, device)

                # Análise
                model_results = analyze_model(model, device, batch_size, seq_len)
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
    print("📈 RELATÓRIO COMPARATIVO")
    print("="*60)

    generate_comparative_report(results)

    return results


def generate_comparative_report(results: Dict[str, Dict[str, float]]):
    """
    Gera relatório comparativo entre os modelos.
    """
    print("\n📊 COMPARAÇÃO DE EFICIÊNCIA:")

    # Encontrar resultados válidos
    valid_results = {}
    for key, result in results.items():
        if 'error' not in result and 'total_params' in result:
            valid_results[key] = result

    if len(valid_results) < 2:
        print("⚠️  Dados insuficientes para comparação")
        return

    # Calcular métricas comparativas
    standard_cpu = valid_results.get('standard_cpu')
    psiqrh_cpu = valid_results.get('psiqrh_cpu')
    standard_gpu = valid_results.get('standard_gpu')
    psiqrh_gpu = valid_results.get('psiqrh_gpu')

    if standard_cpu and psiqrh_cpu:
        param_ratio_cpu = psiqrh_cpu['total_params'] / standard_cpu['total_params']
        memory_ratio_cpu = psiqrh_cpu['peak_memory_mb'] / standard_cpu['peak_memory_mb'] if standard_cpu['peak_memory_mb'] > 0 else 0

        print(f"\n💻 CPU:")
        print(f"   Parâmetros: ΨQRH = {standard_cpu['total_params']:,} vs {psiqrh_cpu['total_params']:,}")
        print(f"   Ratio Parâmetros: {param_ratio_cpu:.2f}x")
        if memory_ratio_cpu > 0:
            print(f"   Ratio Memória: {memory_ratio_cpu:.2f}x")

    if standard_gpu and psiqrh_gpu:
        param_ratio_gpu = psiqrh_gpu['total_params'] / standard_gpu['total_params']
        memory_ratio_gpu = psiqrh_gpu['peak_memory_mb'] / standard_gpu['peak_memory_mb'] if standard_gpu['peak_memory_mb'] > 0 else 0

        print(f"\n🎮 GPU:")
        print(f"   Parâmetros: ΨQRH = {standard_gpu['total_params']:,} vs {psiqrh_gpu['total_params']:,}")
        print(f"   Ratio Parâmetros: {param_ratio_gpu:.2f}x")
        if memory_ratio_gpu > 0:
            print(f"   Ratio Memória: {memory_ratio_gpu:.2f}x")

    # Análise de impacto
    print(f"\n🎯 DIAGNÓSTICO:")
    if param_ratio_cpu > 3:
        print(f"   ❌ ΨQRH tem {param_ratio_cpu:.1f}x mais parâmetros - INEFICIÊNCIA CRÍTICA")
    elif param_ratio_cpu > 2:
        print(f"   ⚠️  ΨQRH tem {param_ratio_cpu:.1f}x mais parâmetros - INEFICIÊNCIA MODERADA")
    else:
        print(f"   ✅ ΨQRH tem {param_ratio_cpu:.1f}x mais parâmetros - EFICIÊNCIA ACEITÁVEL")


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Benchmark concluído com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro durante o benchmark: {e}")
        import traceback
        traceback.print_exc()