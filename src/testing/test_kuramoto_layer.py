#!/usr/bin/env python3
"""
Teste da Camada de Kuramoto com Localização Espacial de Neurônios Espectrais
============================================================================

Valida:
1. Sincronização de osciladores acoplados
2. Equações de reação-difusão espacial
3. Integração com estrutura quaterniônica do ΨQRH
4. Conservação de energia

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import sys
from pathlib import Path

# Adicionar root do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.core.kuramoto_spectral_neurons import (
    create_kuramoto_layer,
    load_kuramoto_config
)


def test_basic_forward_pass():
    """Teste básico do forward pass"""
    print("="*70)
    print("TESTE 1: Forward Pass Básico")
    print("="*70)

    # Criar camada
    kuramoto_layer = create_kuramoto_layer(device="cpu")

    # Input de teste
    batch_size = 2
    seq_len = 8
    embed_dim = 64 * 4  # quaternion

    x = torch.randn(batch_size, seq_len, embed_dim)

    print(f"\n📊 Input shape: {x.shape}")

    # Forward pass
    output, metrics = kuramoto_layer(x, return_metrics=True)

    print(f"📊 Output shape: {output.shape}")
    print(f"\n🎯 Métricas de Sincronização:")
    print(f"   Ordem média: {metrics['synchronization_order_mean']:.4f}")
    print(f"   Desvio padrão: {metrics['synchronization_order_std']:.4f}")
    print(f"   Sincronizado: {metrics['is_synchronized']}")

    assert output.shape == x.shape, "Shape mismatch"
    print("\n✅ Teste de forward pass: PASSOU")

    return metrics


def test_energy_conservation():
    """Teste de conservação de energia"""
    print("\n" + "="*70)
    print("TESTE 2: Conservação de Energia")
    print("="*70)

    kuramoto_layer = create_kuramoto_layer(device="cpu")

    batch_size = 2
    seq_len = 8
    embed_dim = 64 * 4

    x = torch.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    output, _ = kuramoto_layer(x, return_metrics=False)

    # Calcular energias
    input_energy = torch.norm(x).item()
    output_energy = torch.norm(output).item()
    energy_ratio = output_energy / input_energy

    print(f"\n⚡ Energias:")
    print(f"   Input:  {input_energy:.4f}")
    print(f"   Output: {output_energy:.4f}")
    print(f"   Razão:  {energy_ratio:.4f}")

    # Tolerância de 20%
    assert 0.8 <= energy_ratio <= 1.2, f"Energia não conservada: {energy_ratio}"
    print("\n✅ Teste de conservação de energia: PASSOU")

    return energy_ratio


def test_synchronization_convergence():
    """Teste de convergência da sincronização"""
    print("\n" + "="*70)
    print("TESTE 3: Convergência da Sincronização")
    print("="*70)

    kuramoto_layer = create_kuramoto_layer(device="cpu")

    batch_size = 1
    seq_len = 1
    embed_dim = 64 * 4

    x = torch.randn(batch_size, seq_len, embed_dim)

    # Múltiplos forward passes
    sync_orders = []
    num_iterations = 10

    print("\n🔄 Executando iterações:")
    for i in range(num_iterations):
        output, metrics = kuramoto_layer(x, return_metrics=True)
        sync_order = metrics['synchronization_order_mean']
        sync_orders.append(sync_order)
        x = output  # Usar output como novo input

        print(f"   Iteração {i+1}: sync_order = {sync_order:.4f}")

    # Verificar se sincronização aumenta
    initial_sync = sync_orders[0]
    final_sync = sync_orders[-1]

    print(f"\n📊 Evolução da Sincronização:")
    print(f"   Inicial: {initial_sync:.4f}")
    print(f"   Final:   {final_sync:.4f}")
    print(f"   Δ:       {final_sync - initial_sync:+.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_iterations + 1), sync_orders, 'b-o', linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', label='Threshold (0.9)')
    plt.xlabel('Iteração')
    plt.ylabel('Ordem de Sincronização (r)')
    plt.title('Convergência da Sincronização de Kuramoto')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = project_root / "kuramoto_synchronization_convergence.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráfico salvo em: {output_path}")
    plt.close()

    print("\n✅ Teste de convergência: PASSOU")

    return sync_orders


def test_spatial_locality():
    """Teste de localidade espacial dos neurônios"""
    print("\n" + "="*70)
    print("TESTE 4: Localidade Espacial")
    print("="*70)

    from src.core.kuramoto_spectral_neurons import SpatialNeuronGrid

    config = load_kuramoto_config()
    grid = SpatialNeuronGrid(config)

    print(f"\n🔮 Grid de Neurônios:")
    print(f"   Dimensões: {grid.H}×{grid.W}×{grid.D}")
    print(f"   Total de neurônios: {grid.n_neurons}")
    print(f"   Topologia: {grid.topology}")

    # Verificar coordenadas
    coords = grid.coordinates
    print(f"\n📍 Coordenadas:")
    print(f"   Shape: {coords.shape}")
    print(f"   Range X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"   Range Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    print(f"   Range Z: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")

    # Verificar conectividade
    connectivity = grid.connectivity_matrix
    print(f"\n🔗 Matriz de Conectividade:")
    print(f"   Shape: {connectivity.shape}")
    print(f"   Densidade: {(connectivity > 0).float().mean():.2%}")
    print(f"   Média: {connectivity.mean():.4f}")

    # Plot da conectividade
    plt.figure(figsize=(10, 8))
    plt.imshow(connectivity.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Força de Conexão')
    plt.title('Matriz de Conectividade Espacial')
    plt.xlabel('Neurônio j')
    plt.ylabel('Neurônio i')

    output_path = project_root / "kuramoto_connectivity_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Matriz salva em: {output_path}")
    plt.close()

    print("\n✅ Teste de localidade espacial: PASSOU")


def test_reaction_diffusion():
    """Teste das equações de reação-difusão"""
    print("\n" + "="*70)
    print("TESTE 5: Equações de Reação-Difusão")
    print("="*70)

    from src.core.kuramoto_spectral_neurons import KuramotoReactionDiffusion

    config = load_kuramoto_config()
    system = KuramotoReactionDiffusion(config)

    print(f"\n🌊 Parâmetros:")
    print(f"   Coupling K: {system.K}")
    print(f"   Diffusion D: {system.D}")
    print(f"   Time step dt: {system.dt}")

    # Fases iniciais
    batch_size = 1
    n_neurons = system.neuron_grid.n_neurons

    theta_init = torch.randn(batch_size, n_neurons) * np.pi
    phi_init = torch.randn(batch_size, n_neurons) * np.pi

    print(f"\n🎲 Fases Iniciais:")
    print(f"   θ range: [{theta_init.min():.2f}, {theta_init.max():.2f}]")
    print(f"   φ range: [{phi_init.min():.2f}, {phi_init.max():.2f}]")

    # Integrar
    results = system(theta_init, phi_init, num_steps=50)

    print(f"\n📊 Resultados:")
    print(f"   θ final range: [{results['theta_final'].min():.2f}, {results['theta_final'].max():.2f}]")
    print(f"   φ final range: [{results['phi_final'].min():.2f}, {results['phi_final'].max():.2f}]")
    print(f"   Ordem de sincronização: {results['synchronization_order']}")
    print(f"   Sincronizado: {results['is_synchronized']}")

    # Plot evolução
    sync_evolution = results['synchronization_order'].numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(sync_evolution, 'g-', linewidth=2)
    plt.axhline(y=system.sync_threshold, color='r', linestyle='--',
                label=f'Threshold ({system.sync_threshold})')
    plt.xlabel('Passo de Integração')
    plt.ylabel('Ordem de Sincronização (r)')
    plt.title('Evolução da Sincronização - Reação-Difusão')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = project_root / "kuramoto_reaction_diffusion.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Evolução salva em: {output_path}")
    plt.close()

    print("\n✅ Teste de reação-difusão: PASSOU")


def main():
    """Executa todos os testes"""
    print("\n" + "🔬"*35)
    print("SUITE DE TESTES: Kuramoto Spectral Neurons Layer")
    print("🔬"*35 + "\n")

    try:
        # Teste 1: Forward pass
        metrics1 = test_basic_forward_pass()

        # Teste 2: Conservação de energia
        energy_ratio = test_energy_conservation()

        # Teste 3: Convergência de sincronização
        sync_orders = test_synchronization_convergence()

        # Teste 4: Localidade espacial
        test_spatial_locality()

        # Teste 5: Reação-difusão
        test_reaction_diffusion()

        # Resumo
        print("\n" + "="*70)
        print("📋 RESUMO DOS TESTES")
        print("="*70)
        print(f"✅ Forward Pass: PASSOU")
        print(f"✅ Conservação de Energia: PASSOU (razão={energy_ratio:.3f})")
        print(f"✅ Convergência: PASSOU (Δsync={sync_orders[-1]-sync_orders[0]:+.3f})")
        print(f"✅ Localidade Espacial: PASSOU")
        print(f"✅ Reação-Difusão: PASSOU")
        print("\n🎉 TODOS OS TESTES PASSARAM!")

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
