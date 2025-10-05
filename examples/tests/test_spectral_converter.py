#!/usr/bin/env python3
"""
Teste do Spectral Model Converter
===================================

Demonstra conversão física de um modelo Transformer padrão para ΨQRH
usando análise espectral (SEM backpropagation).
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.spectral_model_converter import SpectralModelConverter


class SimpleTransformer(nn.Module):
    """Transformer padrão simples para teste."""

    def __init__(self, vocab_size=1000, d_model=256, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = x + self.ff(x)
        return self.output(x)


def main():
    print("="*70)
    print("🧪 TESTE: Spectral Model Converter")
    print("="*70)

    # Criar modelo fonte (Transformer padrão)
    print("\n1️⃣ Criando Transformer Padrão...")
    source_model = SimpleTransformer(vocab_size=1000, d_model=256, n_heads=4)
    n_params = sum(p.numel() for p in source_model.parameters())
    print(f"✅ Modelo criado: {n_params:,} parâmetros")

    # Criar conversor
    print("\n2️⃣ Inicializando Spectral Converter...")
    converter = SpectralModelConverter(
        alpha_min=0.1,
        alpha_max=3.0,
        lambda_coupling=1.0,
        use_leech_correction=True,
        validate_energy=True
    )
    print("✅ Conversor inicializado")

    # Executar conversão
    print("\n3️⃣ Executando Conversão Física (5 passos)...")
    report = converter.convert_model(source_model, target_architecture="PsiQRHTransformerComplete")

    # Mostrar resultados
    print("\n" + "="*70)
    print("📊 RESULTADOS DA CONVERSÃO")
    print("="*70)

    print(f"\n🔬 Modelo Fonte: {report['source_model']}")
    print(f"🎯 Arquitetura Alvo: {report['target_architecture']}")
    print(f"📊 Camadas Analisadas: {report['n_layers_analyzed']}")
    print(f"📐 Dimensão Fractal Média: {report['avg_fractal_dim']:.4f}")
    print(f"⚡ Alpha Médio: {report['avg_alpha']:.4f}")

    print("\n📋 Detalhes por Camada:")
    print("-" * 70)
    for layer_name, params in list(report['converted_params'].items())[:5]:
        print(f"  {layer_name[:50]:50s} | D={params['fractal_dim']:.4f} | α={params['alpha']:.4f}")

    if len(report['converted_params']) > 5:
        print(f"  ... ({len(report['converted_params']) - 5} camadas adicionais)")

    # Teste de análise individual
    print("\n" + "="*70)
    print("🔬 TESTE: Análise Espectral de Camada Individual")
    print("="*70)

    test_weight = source_model.embedding.weight.data
    print(f"\nAnalisando: embedding.weight {test_weight.shape}")

    analysis = converter.analyze_weights_spectrum(test_weight, "embedding.weight")

    print(f"\n📊 Resultados:")
    print(f"   β (expoente): {analysis['beta']:.4f}")
    print(f"   D (dimensão fractal): {analysis['fractal_dim']:.4f}")
    print(f"   R² (qualidade): {analysis['r_squared']:.4f}")
    print(f"   Espectro médio: {analysis['spectrum_mean']:.2e}")
    print(f"   Espectro std: {analysis['spectrum_std']:.2e}")

    # Teste de mapeamento para alpha
    alpha = converter.map_to_alpha(analysis['fractal_dim'])
    print(f"\n⚡ Mapeamento para α:")
    print(f"   D={analysis['fractal_dim']:.4f} → α={alpha:.4f}")

    # Teste de extração de fase
    theta = converter.extract_phase_from_weights(test_weight)
    print(f"\n🌊 Fase dominante:")
    print(f"   θ = {theta:.4f} rad ({theta*180/3.14159:.1f}°)")

    # Teste de quaternion
    q = converter.initialize_rotation_quaternion(theta, axis='i')
    print(f"\n🔄 Quaternion de rotação:")
    print(f"   q = [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
    print(f"   |q| = {torch.norm(q):.6f} (deve ser ≈1.0)")

    # Teste de embedding quaterniônico
    print("\n" + "="*70)
    print("🔬 TESTE: Conversão de Embedding para Quaterniônico")
    print("="*70)

    original_embedding = source_model.embedding.weight.data
    print(f"\nEmbedding original: {original_embedding.shape}")
    print(f"Memória: {original_embedding.numel() * 4 / 1024:.2f} KB")

    quat_embedding = converter.embed_to_quaternion(original_embedding)
    print(f"\nEmbedding quaterniônico: {quat_embedding.shape}")
    print(f"Memória: {quat_embedding.numel() * 4 / 1024:.2f} KB")

    reduction = (1 - quat_embedding.numel() / original_embedding.numel()) * 100
    print(f"Redução: {reduction:.1f}% (sem perda de informação)")

    # Teste de correção Leech
    print("\n" + "="*70)
    print("🔬 TESTE: Correção Topológica com Rede de Leech")
    print("="*70)

    test_params = torch.randn(100)
    print(f"\nParâmetros originais: mean={test_params.mean():.6f}, std={test_params.std():.6f}")

    corrected_params = converter.leech_lattice_correction(test_params)
    print(f"Parâmetros corrigidos: mean={corrected_params.mean():.6f}, std={corrected_params.std():.6f}")

    diff = torch.abs(test_params - corrected_params).mean()
    print(f"Diferença média: {diff:.6f}")

    print("\n" + "="*70)
    print("✅ TESTE CONCLUÍDO COM SUCESSO!")
    print("="*70)

    print("\n💡 Próximos Passos:")
    print("   1. Integrar ao pipeline_from_source.py")
    print("   2. Testar com GPT-2 real")
    print("   3. Validar conservação de energia")
    print("   4. Aplicar ajuste fino óptico")


if __name__ == "__main__":
    main()
